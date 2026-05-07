"""
CoseNiche: Context-aware Spatial Expression Niche Foundation Model

A foundation model for spatial transcriptomics that leverages spatial 
neighborhood context to learn robust spot-level representations.
"""

__version__ = "0.1.0"
__author__ = "CoseNiche Authors"

from .config import CoseNicheConfig
from .model import CoseNicheModel

__all__ = [
    "CoseNicheConfig",
    "CoseNicheModel",
    "__version__",
]


