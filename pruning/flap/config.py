from dataclasses import dataclass


@dataclass
class FlapConfig:

    # add all the arguments here in sorted order
    eval: bool = False
    gqa_groups: int = 4
    head_dim: int = 128
    hidden_dim: int = 4096
    metrics: str = "WIFV"
    nsamples: int = 1024
    pruning_ratio: float = 0.2
    remove_heads: int = -1
    seed: int = 0
    start_pruning_layer_idx: int = 22
    structure: str = "AL-AM"

    def __post_init__(self):
        assert self.metrics in [
            "IFV",
            "WIFV",
            "WIFN",
            "N/A",
        ], f"Invalid metrics: {self.metrics}. Supported metrics are ['IFV', 'WIFV', 'WIFN', 'N/A']"
        assert (
            self.pruning_ratio >= 0 and self.pruning_ratio <= 1
        ), f"Invalid pruning_ratio: {self.pruning_ratio}. It should be in (0, 1)"
        assert (
            self.remove_heads >= -1
        ), f"Invalid remove_heads: {self.remove_heads}. It should be greater than or equal to -1"
        assert self.structure in [
            "AL-AM"
        ], f"Invalid structure: {self.structure}. Supported structures are ['AL-AM']"
        assert (
            self.start_pruning_layer_idx >= 0
        ), f"Invalid start_pruning_layer_idx: {self.start_pruning_layer_idx}. It should be greater than or equal to 0"
