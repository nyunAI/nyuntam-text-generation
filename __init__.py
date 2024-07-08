# nyuntam
from algorithm import Algorithm


def __getattr__(name: str) -> Algorithm:

    raise AttributeError(f"Unsupported algorithm: {name}")
