# nyuntam
from algorithm import Algorithm

# ===================================
#           quantization
# ===================================


def _import_AutoAWQ() -> Algorithm:
    from .quantisation.autoawq import AutoAWQ

    return AutoAWQ


def __getattr__(name: str) -> Algorithm:

    # quantization
    if name == "AutoAWQ":
        return _import_AutoAWQ()

    else:
        raise AttributeError(f"Unsupported algorithm: {name}")
