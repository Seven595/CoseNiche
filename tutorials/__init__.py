"""
CoseNiche Tutorials Module

This module contains downstream analysis tutorials for the CoseNiche model:
- Deconvolution: Spatial deconvolution to infer cell type composition
- Attention Analysis: Gene-gene interaction networks from self-attention
- Spatial Communication: Cell-cell communication from spatial attention
"""

__version__ = "0.1.0"
__author__ = "CoseNiche Team"

from pathlib import Path

TUTORIAL_DIR = Path(__file__).parent
DECONV_DIR = TUTORIAL_DIR / "deconvolution"
ATTENTION_DIR = TUTORIAL_DIR / "attention_analysis"
SPATIAL_DIR = TUTORIAL_DIR / "spatial_communication"

__all__ = [
    "TUTORIAL_DIR",
    "DECONV_DIR",
    "ATTENTION_DIR",
    "SPATIAL_DIR",
]
