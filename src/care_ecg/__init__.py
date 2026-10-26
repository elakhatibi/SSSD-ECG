"""CARE-ECG: causal ECG representation, inference, and grounded explanation."""

from .config import CAREConfig
from .types import PipelineOutput

__all__ = ["CAREConfig", "CAREECGPipeline", "PipelineOutput"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # Keep non-neural utilities importable in lightweight environments.
    if name == "CAREECGPipeline":
        from .pipeline import CAREECGPipeline

        return CAREECGPipeline
    raise AttributeError(name)
