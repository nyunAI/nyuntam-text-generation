# nyuntam
from nyuntam.algorithm import Algorithm

# ===================================
#           quantization
# ===================================


def _import_AutoAWQ() -> Algorithm:
    from .quantisation.autoawq import AutoAWQ

    return AutoAWQ


# ===================================
#               pruning
# ===================================


def _import_Flap() -> Algorithm:
    from .pruning.flap import FlapPruner

    return FlapPruner


# ===================================
#               engine
# ===================================


def _import_TensorRTLLM() -> Algorithm:
    from .engines.tensorrt_llm import TensorRTLLM

    return TensorRTLLM


def __getattr__(name: str) -> Algorithm:

    # quantization
    if name == "AutoAWQ":
        return _import_AutoAWQ()

    # pruning
    elif name == "FlapPruner":
        return _import_Flap()

    # engine
    elif name == "TensorRTLLM":
        return _import_TensorRTLLM()

    else:
        raise AttributeError(f"Unsupported algorithm: {name}")
