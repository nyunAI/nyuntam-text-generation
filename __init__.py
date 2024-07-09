# nyuntam
from algorithm import Algorithm

# ===================================
#           quantization
# ===================================


def _import_AutoAWQ() -> Algorithm:
    from .quantisation.autoawq import AutoAWQ

    return AutoAWQ


# ===================================
#               engine
# ===================================


def _import_TensorRTLLM():
    from .engines.tensorrt_llm import TensorRTLLM

    return TensorRTLLM


def __getattr__(name: str) -> Algorithm:

    # quantization
    if name == "AutoAWQ":
        return _import_AutoAWQ()

    # engine
    elif name == "TensorRTLLM":
        return _import_TensorRTLLM()

    else:
        raise AttributeError(f"Unsupported algorithm: {name}")
