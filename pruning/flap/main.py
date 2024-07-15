from text_generation.core.job import LMJob

# nyuntam
from nyuntam.algorithm import Algorithm

# pruning/flap/FLAP
from FLAP.lib.layerwrapper import BiasGPT
from FLAP.lib.prune import (
    metrics,
    find_layers,
    prepare_calibration_input,
    compress,
    cal_remove_neuron,
)

# nyuntam-adapt
from nyuntam_adapt.tasks import CausalLLM
from nyuntam_adapt.tasks.params import AdaptParams, create_instance

import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, asdict, fields

import logging

logger = logging.getLogger(__name__)

# wandb offline
os.environ["WANDB_MODE"] = "offline"


def free(times=2):
    for _ in range(times):
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class BaseParams:
    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in fields(cls)})


@dataclass
class FlapArgs(BaseParams):

    seed: int = 0
    nsamples: int = 2048
    pruning_ratio: float = 0.2
    remove_heads: int = 8
    metrics: str = "WIFV"
    structure: str = "AL-AM"
    unstr: bool = False
    """If True, only mask without real pruning. Defaults to False."""
    eval: bool = False  # not supporting eval

    def __post_init__(self):
        assert self.metrics in ["IFV", "WIFV", "WIFN", "N/A"]
        assert self.structure in ["UL-UM", "UL-MM", "AL-MM", "AL-AM", "N/A"]
        assert self.pruning_ratio >= 0 and self.pruning_ratio <= 1


class Pruner:
    def __init__(self, args, job: LMJob, **kwargs):
        self.args = args
        self.job = job
        self.kw = kwargs
        self.output_dir = self.job.user_dir.output

        self.device_to_use = self.job.environment.cuda_device_ids[0]

    def prune(self):
        np.random.seed(self.args.seed)
        torch.random.manual_seed(self.args.seed)
        dtype = eval(f'torch.{self.kw.get("dtype", "float16")}')
        logger.info("Loading model..")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.job.model.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        device = torch.device(f"cuda:{self.device_to_use}")
        for i in range(32):
            self.model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
                torch.empty(
                    self.model.model.layers[i].self_attn.o_proj.out_features,
                    dtype=dtype,
                    device=self.model.model.layers[i].self_attn.o_proj.weight.device,
                )
            )
            self.model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                torch.empty(
                    self.model.model.layers[i].mlp.down_proj.out_features,
                    dtype=dtype,
                    device=self.model.model.layers[i].mlp.down_proj.weight.device,
                )
            )
        torch.nn.init.zeros_(self.model.model.layers[i].self_attn.o_proj.bias)
        torch.nn.init.zeros_(self.model.model.layers[i].mlp.down_proj.bias)

        self.model.seqlen = 128

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.job.model.model_path)

        self.model.eval()
        device = self.model.hf_device_map.get("lm_head", self.model.device)
        logger.info("Pruning model...")
        self.prune_flap(
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.job.dataset,
            device=device,
        )

        try:
            torch.save(self.model, self.output_dir / "wds.pt")
        except Exception as e:
            # TODO: fix torch.save when pruning on multiple GPUs
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.warn(
                f"couldn't save with torch.save, saved with save_pretrained instead"
            )
        logger.info("Model pruned")

        del self.model
        del self.tokenizer

    def prune_flap(
        self, args, model, tokenizer, device=torch.device("cuda:0"), dataset=None
    ):
        # method adapted from FLAP.lib.prune
        use_cache = model.config.use_cache
        model.config.use_cache = False
        dataloader = dataset.get_flap_dataloader(
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )
        layers = model.model.layers

        attn_metric_list, mlp_metric_list = [], []
        attn_baseline_inp_list, mlp_baseline_inp_list = [], []
        attn_mask, mlp_mask = [], []

        # Split into sub-problems, separate statistics for each module
        for i in tqdm(range(len(layers)), desc="Processing layers"):
            layer = layers[i]
            subset = {}
            subset.update({"self_attn.o_proj": find_layers(layer)["self_attn.o_proj"]})
            subset.update({"mlp.down_proj": find_layers(layer)["mlp.down_proj"]})

            if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev),
                    position_ids.to(dev),
                )

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            for h in handles:
                h.remove()

            for name in subset:
                if name == "self_attn.o_proj":
                    W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    if args.structure == "UL-UM":
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                        thresh = torch.sort(W_metric.cuda())[0][
                            int(args.pruning_ratio * layer.self_attn.num_heads)
                        ].cpu()
                        W_mask = W_metric >= thresh
                        attn_mask.append(W_mask)
                    elif args.structure == "UL-MM":
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                        thresh = torch.sort(W_metric.cuda())[0][
                            args.remove_heads // len(layers)
                        ].cpu()
                        W_mask = W_metric >= thresh
                        attn_mask.append(W_mask)
                    else:
                        attn_metric_list.append(W_metric.cpu())
                    attn_baseline_inp_list.append(
                        wrapped_layers[name].baseline_inp.type(torch.half)
                    )
                else:
                    W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                    if args.structure == "UL-UM":
                        thresh = torch.sort(W_metric.cuda())[0][
                            int(W_metric.numel() * args.pruning_ratio)
                        ].cpu()
                        W_mask = W_metric >= thresh
                        mlp_mask.append(W_mask)
                    elif args.structure == "UL-MM":
                        thresh = torch.sort(W_metric.cuda())[0][
                            cal_remove_neuron(args, model)
                        ].cpu()
                        W_mask = W_metric >= thresh
                        mlp_mask.append(W_mask)
                    else:
                        mlp_metric_list.append(W_metric.cpu())
                    mlp_baseline_inp_list.append(
                        wrapped_layers[name].baseline_inp.type(torch.half)
                    )
                wrapped_layers[name].free()

            inps, outs = (
                outs,
                inps,
            )  # Use the original output as input to the next layer
            torch.cuda.empty_cache()

        def standarlization(x):
            return (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(
                x, axis=1, keepdim=True
            )

        if args.structure in ["AL-MM", "AL-AM"]:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = standarlization(attn_metric)
            attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)

            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)

            if args.structure == "AL-MM":
                sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
                attn_thres = sorted_attn[-int(args.remove_heads)]
                attn_mask = attn_metric > attn_thres  # 1 means retain

                sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
                mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
                mlp_mask = mlp_metric > mlp_thres
            else:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(indices)
                compression_weight[indices < attn_metric.numel()] = 512.0 / 3
                threshold = sorted_prune[
                    torch.argmin(
                        torch.abs(
                            torch.cumsum(compression_weight, 0)
                            - torch.sum(compression_weight) * (1 - args.pruning_ratio)
                        )
                    )
                ]
                attn_mask = attn_metric > threshold
                mlp_mask = mlp_metric > threshold
        else:
            attn_mask = torch.stack(attn_mask)
            mlp_mask = torch.stack(mlp_mask)

        for idx in range(len(layers)):
            if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
                compress(
                    model.model.layers[idx],
                    attn_mask[idx],
                    None,
                    attn_baseline_inp_list[idx],
                    None,
                    model.hf_device_map[f"model.layers.{idx}"],
                    unstr=args.unstr,
                )
            else:
                compress(
                    model.model.layers[idx],
                    attn_mask[idx],
                    None,
                    attn_baseline_inp_list[idx],
                    None,
                    device,
                    unstr=args.unstr,
                )

            if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
                compress(
                    model.model.layers[idx],
                    None,
                    mlp_mask[idx],
                    None,
                    mlp_baseline_inp_list[idx],
                    model.hf_device_map[f"model.layers.{idx}"],
                    unstr=args.unstr,
                )
            else:
                compress(
                    model.model.layers[idx],
                    None,
                    mlp_mask[idx],
                    None,
                    mlp_baseline_inp_list[idx],
                    device,
                    unstr=args.unstr,
                )

        model.config.use_cache = use_cache
        torch.cuda.empty_cache()


class FlapPruner(Algorithm):
    def __init__(self, job: LMJob, **kwargs):
        self.job = job
        self.args = FlapArgs.from_dict(kwargs)
        logger.info(f"Experiment arguments: {self.args}")
        self.kw = kwargs
        self.output_dir = self.job.user_dir.output

        self.set_pruner(self.args, self.job, self.kw)
        self.adapter = None
        if self.kw.get("to_finetune", True):
            self.set_adapter()

    def set_adapter(self):
        """Factory method to set the adapter object."""

        self.adapt_params = create_instance(AdaptParams, self.kw)
        # value updates
        self.adapt_params.DATASET_ARGS.CUSTOM_DATASET_PATH = str(
            self.job.dataset.dataset_name_or_path
        )
        self.adapt_params.DATASET_ARGS.FORMAT_STRING = self.job.dataset.format_string
        self.adapt_params.LOGGING_PATH = str(self.job.user_dir.logs)
        self.adapt_params.MODEL_ARGS.MODEL_PATH = self.job.model.model_name
        self.adapt_params.MODEL_ARGS.LOCAL_MODEL_PATH = str(
            (self.job.user_dir.output).absolute()
        )
        self.adapt_params.OUTPUT_DIR = str(self.job.user_dir.output.absolute())
        self.adapt_params.cuda_id = ",".join(
            map(str, self.job.environment.cuda_device_ids)
        )

        self.adapter = CausalLLM(**asdict(self.adapt_params))

    def set_pruner(self, args, job, kwargs):
        self.pruner = Pruner(args=args, job=job, kwargs=kwargs)

    def export_scripts(self):
        # check if quant(4bit/8bit)
        bnb_config = None
        if self.adapt_params.BNB_CONFIG.USE_4BIT.load_in_4bit:
            # loading in 4bit
            bnb_config = asdict(self.adapt_params.BNB_CONFIG.USE_4BIT)
        elif self.adapt_params.BNB_CONFIG.USE_8BIT.load_in_8bit:
            # loading in 8bit
            bnb_config = asdict(self.adapt_params.BNB_CONFIG.USE_8BIT)

        if bnb_config is not None:
            import json

            with open(self.job.user_dir.output / "bnbconfig.json", "w") as f:
                json.dump(bnb_config, f, indent=4)

        export_script = f"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from pathlib import Path
CURR = Path(__file__).parent.absolute()
print(CURR)
# To load in quant (4bit/8bit)
bnb_path = CURR / "bnbconfig.json"
bnb_path = bnb_path if bnb_path.exists() else None

if bnb_path is not None:
    with open(bnb_path) as f:
        bnbconfig = BitsAndBytesConfig(**json.load(f))

if bnb_path:
    model = AutoModelForCausalLM.from_pretrained("{self.job.model.model_name}", state_dict=torch.load("merged_model_state_dict.pth", map_location="cpu"), ignore_mismatched_sizes=True, quantization_config=bnbconfig, device_map="cpu")
else:
    # loads without quant (if not finetuned with quant)
    model = AutoModelForCausalLM.from_pretrained("{self.job.model.model_name}", state_dict=torch.load("merged_model_state_dict.pth", map_location="cpu"), ignore_mismatched_sizes=True, device_map="cpu")

model.to("cuda:0")

# forward pass
tokenizer = AutoTokenizer.from_pretrained("{self.job.model.model_name}")
inputs = tokenizer("Question:\nWhat is the meaning of life?\nAnswer:", return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.batch_decode(outputs)[0])

"""
        with open(self.job.user_dir.output / "run.py", "w") as f:
            f.write(export_script)

    def compress_model(self):
        assert self.pruner is not None
        self.pruner.prune()
        del self.pruner
        free()

        if self.adapter is not None:
            logger.info("finetuning")
            self.adapter.adapt_model()
            self.export_scripts()
        del self.adapter
        free()
        return None, None
