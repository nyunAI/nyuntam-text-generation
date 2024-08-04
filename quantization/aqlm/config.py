from dataclasses import dataclass, field

# ============ Configs ============

@dataclass
class CalibrationConfig:
    model_path: str = "meta-llama/Meta-Llama-3-8B"
    dataset_path: str = "pajama"

    nsamples: int = 2048
    val_size: int = 256
    model_seqlen: int = 4096
    num_codebooks: int = 1
    nbits_per_codebook: int = 16
    out_group_size: int = 1
    in_group_size: int = 8
    beam_size: int = 1
    relative_mse_tolerance: float = 0.01
    max_epochs: int = 100
    finetune_lr: float = 1e-4
    finetune_adam_beta1: float = 0.90
    finetune_adam_beta2: float = 0.999
    finetune_keep_best: bool = True
    finetune_batch_size: int = 64
    local_batch_size: int = 4
    finetune_max_epochs: int = 25
    finetune_early_stop: int = 3
    offload_activations: bool = True
    trust_remote_code: bool = True
    save: str = "llama3_outputs/quantized_model/meta-llama_meta-llama-3-8b"
    resume: bool = False




@dataclass
class AQLMConfig:
    calibration_config: CalibrationConfig = field(default_factory=CalibrationConfig)