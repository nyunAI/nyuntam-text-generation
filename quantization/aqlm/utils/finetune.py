# Parts of this code are taken from https://github.com/nyunAI/AQLM/blob/pv-tuning/finetune_fsdp.py

from text_generation.quantization.aqlm.config import AQLMConfig

# quantization/aqlm/AQLM
from AQLM.finetune_fsdp import (
    prepare_training_dataset,
    _load_state,
    _save_state,
    _save_model,
    load_base_model,
    load_dequantized_model,
    compute_loss_on_batch,
    compute_validation_perplexities,
)
from AQLM.src.aq import QuantizedWeight
from AQLM.src.aq_ops import master_rank_first, one_rank_at_a_time
from AQLM.src.datautils import (
    get_loaders,
)
from AQLM.src.pv_utils import (
    get_original_named_parameters_from_fsdp_module,
    split_quantized_weights_between_ranks,
    YourQuantizedWeightIsInAnotherRank,
)
from AQLM.src.pv_optimizer import StraightThroughAdamW

import torch
import torch.distributed
import torch.utils
import torch.utils.data
import logging
import transformers
from tqdm import tqdm
from contextlib import nullcontext


try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

logger = logging.getLogger(__name__)


def finetune_quantized(config: AQLMConfig):
    """Finetunes an AQLM quantized model with PV-Tuning."""

    assert torch.cuda.is_available() and torch.distributed.is_available()

    if (
        not torch.distributed.is_initialized()
    ):  ## tokenizer initializes torch.distributed
        torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    args = config.finetune_config

    if not config.overwrite_or_run_all and not config.overwrite_or_run_finetune:
        logger.info("Skipping finetuning")
        return

    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)

    args.load_dtype = (
        getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    )
    args.amp_dtype = (
        getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None
    )
    args.code_dtype = (
        getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    )
    args.master_dtype = getattr(torch, args.master_dtype)
    if args.straight_through_buffer_dtype is not None:
        args.straight_through_buffer_dtype = getattr(
            torch, args.straight_through_buffer_dtype
        )
    else:
        args.straight_through_buffer_dtype = args.master_dtype

    if args.save_every_steps is not None:
        assert (
            args.save is not None
        ), f"save_every_steps={args.save_every_steps}, but save path not specified"
    if args.keep_best_model:
        assert args.save is not None, f"keep_best_model requires save path"
        assert (
            args.eval_every_steps is not None
        ), f"keep_best_model requires eval_every_steps"
        assert args.eval_datasets is not None, f"keep_best_model requires eval_datasets"

    if args.wandb and rank == 0:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")}
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with master_rank_first(local=True):
        dataset = prepare_training_dataset(args, tokenizer)

    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=transformers.default_data_collator,
    )
    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
        )
        for dataset_name in args.eval_datasets
    }

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        base_model = load_base_model(args, device)
        dequantized_model, named_quantized_params = load_dequantized_model(args, device)
        if rank == 0:
            logger.info("Wrapped model:")
            logger.info(dequantized_model)
            for name, param in dequantized_model.named_parameters():
                logger.info(f"{name}, {param.shape}, {param.dtype}")
        named_dequantized_params = get_original_named_parameters_from_fsdp_module(
            dequantized_model
        )
        assert all(name in named_dequantized_params for name in named_quantized_params)

        if world_size > 1:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=True
            )
        for quantized_weight in named_quantized_params.values():
            if (
                isinstance(quantized_weight, QuantizedWeight)
                or quantized_weight.__class__.__name__
                == "QuantizedWeight"  ## TODO: remove this after testing
            ):
                quantized_weight.to(device)
            else:
                assert (
                    isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)
                    or quantized_weight.__class__.__name__
                    == "YourQuantizedWeightIsInAnotherRank"  ## TODO: remove this after testing
                )

    optimizer = StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=(
            dict(
                lr=args.code_lr,
                betas=(args.code_beta1, args.code_beta2),
                lamb=args.lamb,
                debias=args.debias,
                amsgrad=args.amsgrad,
                compute_dtype=args.master_dtype,
                exp_avg_dtype=(
                    torch.float16 if args.code_adam_16bit else args.master_dtype
                ),
                exp_avg_sq_dtype=(
                    torch.bfloat16 if args.code_adam_16bit else args.master_dtype
                ),
                v_hat_max_dtype=(
                    torch.float16 if args.code_adam_16bit else args.master_dtype
                ),
            )
            if args.update_codes
            else None
        ),
        update_codebooks_and_scales=(
            dict(
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                lamb=args.lamb,
                debias=args.debias,
                amsgrad=args.amsgrad,
                compute_dtype=args.master_dtype,
                exp_avg_dtype=args.master_dtype,
                exp_avg_sq_dtype=args.master_dtype,
                v_hat_max_dtype=args.master_dtype,
            )
            if args.update_codebooks_and_scales
            else None
        ),
        update_non_quantized_parameters=(
            dict(
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                lamb=args.lamb,
                debias=args.debias,
                amsgrad=args.amsgrad,
                compute_dtype=args.master_dtype,
                exp_avg_dtype=args.master_dtype,
                exp_avg_sq_dtype=args.master_dtype,
                v_hat_max_dtype=args.master_dtype,
            )
            if args.update_non_quantized_parameters
            else None
        ),
        delta_decay=args.delta_decay,
        max_code_change_per_step=args.max_code_change_per_step,
        force_code_update=args.force_code_update,
        code_trust_ratio=args.code_trust_ratio,
        beam_size=args.beam_size,
        straight_through_buffer_dtype=args.straight_through_buffer_dtype,
        verbose=args.verbose_optimizer,
    )
    del named_quantized_params

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )

    _load_state(args, metadata, dequantized_model, optimizer)
    torch.distributed.barrier()

    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata["current_epoch"]:
            continue  # skip finished epochs
        sampler.set_epoch(current_epoch)

        batch_iter = (
            tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}")
            if rank == 0
            else train_dataloader
        )
        for batch_index, batch in enumerate(batch_iter):
            if batch_index <= metadata["microbatches_since_epoch_start"]:
                continue  # skip batches processed before checkpoint
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss_on_batch(
                batch, base_model, dequantized_model, amp_dtype=args.amp_dtype
            )  ## add support for DPO

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1
            if metadata["grad_steps_accumulated"] < grad_accumulation_steps:
                with (
                    dequantized_model.no_sync() if args.minimize_sync else nullcontext()
                ):
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                optimizer.step()
                optimizer.zero_grad()
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1

                if (
                    args.print_every_steps
                    and metadata["total_optimizer_steps"] % args.print_every_steps == 0
                ):
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata["loss_numerator"], metadata["loss_denominator"]],
                        dtype=torch.float64,
                        device=device,
                    )

                    torch.distributed.all_reduce(
                        loss_numerator_and_denominator,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                    loss_numerator, loss_denominator = (
                        loss_numerator_and_denominator.tolist()
                    )
                    metadata["aggregated_loss"] = loss_numerator / loss_denominator
                    metadata["loss_numerator"] = metadata["loss_denominator"] = 0
                    if rank == 0:
                        logger.info(
                            f"epoch {metadata['current_epoch']}\tbatch {batch_index}",
                            f"\t| total updates = {metadata['total_optimizer_steps']}",
                            f"\tloss = {metadata['aggregated_loss']:.9f}",
                        )

                if (
                    args.eval_every_steps
                    and metadata["total_optimizer_steps"] % args.eval_every_steps == 0
                ):
                    perplexity_scores = compute_validation_perplexities(
                        args, dequantized_model, eval_datasets
                    )
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f"perplexity_{dataset_name}"] = perplexity
                    metric_name = metadata["early_stop_on"]
                    if (
                        perplexity_scores[metric_name]
                        < metadata["best_eval_perplexity"]
                    ):
                        if rank == 0:
                            logger.info(
                                f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}"
                            )
                        metadata["best_eval_perplexity"] = perplexity_scores[
                            args.eval_datasets[0]
                        ]
                        metadata["best_step"] = metadata["total_optimizer_steps"]
                        if args.keep_best_model:
                            _save_model(args, dequantized_model, optimizer)
                if args.wandb and rank == 0:
                    wandb.log(metadata, step=metadata["total_microbatches"])
                if (
                    args.save_every_steps
                    and metadata["total_optimizer_steps"] % args.save_every_steps == 0
                ):
                    _save_state(args, metadata, dequantized_model, optimizer)

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1

    _save_state(args, metadata, dequantized_model, optimizer)
    logger.info("Finished training")
