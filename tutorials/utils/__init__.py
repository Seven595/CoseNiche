"""
Shared utility functions for CoseNiche tutorials.

This module provides common utilities for:
- I/O operations (loading/saving data and configs)
- Gene filtering and normalization
- Visualization and plotting
- Logging and progress tracking
"""

from .io import (
    load_config,
    save_results,
    load_h5ad_with_truth,
    ensure_dir,
    get_project_root,
    get_relative_path
)
from .gene_filtering import (
    clean_symbol,
    normalize_celltype,
    normalize_pdac_celltype,
    filter_genes_by_quality,
    build_id2symbol_mapping,
    ensure_unique_var_names,
    get_common_genes,
    preprocess_celltype_column
)
from .visualization import (
    setup_plotting_style,
    save_figure,
    get_nature_colors,
    create_colormap_from_colors
)
from .logger import (
    setup_logger,
    log_system_info,
    log_parameters,
    create_progress_callback
)

__version__ = '0.1.0'

__all__ = [
    # IO
    "load_config",
    "save_results",
    "load_h5ad_with_truth",
    "ensure_dir",
    "get_project_root",
    "get_relative_path",
    
    # Gene filtering
    "clean_symbol",
    "normalize_celltype",
    "normalize_pdac_celltype",
    "filter_genes_by_quality",
    "build_id2symbol_mapping",
    "ensure_unique_var_names",
    "get_common_genes",
    "preprocess_celltype_column",
    
    # Visualization
    "setup_plotting_style",
    "save_figure",
    "get_nature_colors",
    "create_colormap_from_colors",
    
    # Logging
    "setup_logger",
    "log_system_info",
    "log_parameters",
    "create_progress_callback",
]
