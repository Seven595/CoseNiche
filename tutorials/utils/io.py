"""
I/O utilities for CoseNiche tutorials.

Handles file loading, saving, and configuration management.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pickle
import numpy as np
import pandas as pd
import scanpy as sc


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
        
    Returns
    -------
    config : dict
        Configuration dictionary
        
    Examples
    --------
    >>> config = load_config('config_pdac.yaml')
    >>> print(config['data_dir'])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(data: Any, output_path: Union[str, Path], 
                 file_format: str = 'auto') -> None:
    """
    Save results to file with automatic format detection.
    
    Parameters
    ----------
    data : Any
        Data to save (numpy array, pandas DataFrame, dict, etc.)
    output_path : str or Path
        Output file path
    file_format : str, optional
        File format ('auto', 'npy', 'csv', 'pkl', 'json')
        If 'auto', format is inferred from file extension
        
    Examples
    --------
    >>> save_results(embeddings, 'embeddings.npy')
    >>> save_results(proportions, 'proportions.csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer format from extension if auto
    if file_format == 'auto':
        ext = output_path.suffix.lower()
        format_map = {
            '.npy': 'npy',
            '.csv': 'csv',
            '.pkl': 'pkl',
            '.pickle': 'pkl',
            '.json': 'json',
            '.h5ad': 'h5ad'
        }
        file_format = format_map.get(ext, 'pkl')
    
    # Save based on format
    if file_format == 'npy':
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        np.save(output_path, data)
    
    elif file_format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=True)
        elif isinstance(data, np.ndarray):
            pd.DataFrame(data).to_csv(output_path, index=False)
        else:
            raise TypeError(f"Cannot save type {type(data)} as CSV")
    
    elif file_format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    elif file_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif file_format == 'h5ad':
        if not isinstance(data, sc.AnnData):
            raise TypeError("h5ad format requires AnnData object")
        data.write_h5ad(output_path)
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_h5ad_with_truth(h5ad_path: Union[str, Path], 
                         truth_csv: Optional[Union[str, Path]] = None,
                         truth_column: str = "Region") -> sc.AnnData:
    """
    Load h5ad file and optionally merge with ground truth labels.
    
    Parameters
    ----------
    h5ad_path : str or Path
        Path to h5ad file
    truth_csv : str or Path, optional
        Path to CSV file containing ground truth labels
    truth_column : str, optional
        Column name in CSV containing labels (default: "Region")
        
    Returns
    -------
    adata : AnnData
        Loaded AnnData object with ground truth in obs['ground_truth']
        
    Examples
    --------
    >>> adata = load_h5ad_with_truth('data.h5ad', 'truth.csv')
    >>> print(adata.obs['ground_truth'].value_counts())
    """
    h5ad_path = Path(h5ad_path)
    
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")
    
    adata = sc.read_h5ad(h5ad_path)
    
    # Load ground truth if provided
    if truth_csv is not None:
        truth_csv = Path(truth_csv)
        if not truth_csv.exists():
            raise FileNotFoundError(f"Truth CSV not found: {truth_csv}")
        
        df_truth = pd.read_csv(truth_csv)
        
        if truth_column not in df_truth.columns:
            raise ValueError(f"Column '{truth_column}' not found in truth CSV")
        
        # Add to obs
        adata.obs['ground_truth'] = df_truth[truth_column].values
    
    return adata


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Parameters
    ----------
    directory : str or Path
        Directory path
        
    Returns
    -------
    directory : Path
        Directory path as Path object
        
    Examples
    --------
    >>> output_dir = ensure_dir('./results')
    >>> output_dir = ensure_dir(Path('./plots'))
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """
    Get the root directory of the CoseNiche project.
    
    Returns
    -------
    root : Path
        Project root directory
        
    Examples
    --------
    >>> root = get_project_root()
    >>> data_dir = root / 'data'
    """
    # Assuming this file is in CoseNiche/tutorials/utils/
    return Path(__file__).parent.parent.parent


def get_relative_path(absolute_path: Union[str, Path], 
                     base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Convert absolute path to relative path.
    
    Parameters
    ----------
    absolute_path : str or Path
        Absolute path to convert
    base_dir : str or Path, optional
        Base directory for relative path (default: project root)
        
    Returns
    -------
    relative_path : Path
        Relative path
        
    Examples
    --------
    >>> rel_path = get_relative_path('/home/user/project/data/file.h5ad')
    >>> print(rel_path)  # data/file.h5ad
    """
    absolute_path = Path(absolute_path).resolve()
    
    if base_dir is None:
        base_dir = get_project_root()
    else:
        base_dir = Path(base_dir).resolve()
    
    try:
        return absolute_path.relative_to(base_dir)
    except ValueError:
        # If not relative to base_dir, return absolute path
        return absolute_path
