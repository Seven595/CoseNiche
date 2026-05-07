"""
Gene filtering and normalization utilities for CoseNiche tutorials.

Handles gene name cleaning, cell type normalization, and quality filtering.
"""

import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc


def clean_symbol(gene_name: str) -> str:
    """
    Clean and normalize gene symbol.
    
    Removes version suffixes (e.g., '.1', '.2') and whitespace.
    
    Parameters
    ----------
    gene_name : str
        Original gene name
        
    Returns
    -------
    cleaned : str
        Cleaned gene name
        
    Examples
    --------
    >>> clean_symbol('ENSG00000001.1')
    'ENSG00000001'
    >>> clean_symbol(' TP53 ')
    'TP53'
    """
    # Remove whitespace
    gene_name = str(gene_name).strip()
    
    # Remove version suffix (e.g., .1, .2)
    gene_name = re.sub(r'\.\d+$', '', gene_name)
    
    return gene_name


def normalize_celltype(celltype_name: str) -> str:
    """
    Normalize cell type name.
    
    Removes numeric suffixes and standardizes formatting.
    Common for PDAC datasets with numbered cell type variants.
    
    Parameters
    ----------
    celltype_name : str
        Original cell type name
        
    Returns
    -------
    normalized : str
        Normalized cell type name
        
    Examples
    --------
    >>> normalize_celltype('T cell.151')
    'T cell'
    >>> normalize_celltype('Macrophage_2')
    'Macrophage'
    """
    # Remove trailing dot and numbers (e.g., '.151', '.150')
    cleaned = re.sub(r'\.\d+$', '', str(celltype_name))
    
    # Remove trailing underscore and numbers (e.g., '_1', '_2')
    cleaned = re.sub(r'_\d+$', '', cleaned)
    
    # Trim whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def filter_genes_by_quality(adata: sc.AnnData,
                            min_counts: int = 1,
                            min_cells: int = 3,
                            inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Filter genes based on quality metrics.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    min_counts : int, optional
        Minimum total counts per gene (default: 1)
    min_cells : int, optional
        Minimum number of cells expressing the gene (default: 3)
    inplace : bool, optional
        If True, modify adata in place (default: True)
        
    Returns
    -------
    adata : AnnData or None
        Filtered AnnData (if inplace=False)
        
    Examples
    --------
    >>> adata = filter_genes_by_quality(adata, min_counts=10, min_cells=5)
    >>> print(f"Kept {adata.n_vars} genes")
    """
    if not inplace:
        adata = adata.copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if not inplace:
        return adata


def build_id2symbol_mapping(adata: sc.AnnData,
                           id_column: str = 'gene_ids',
                           symbol_column: Optional[str] = None,
                           prefer_symbol: bool = True) -> Dict[str, str]:
    """
    Build mapping from gene IDs to gene symbols.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    id_column : str, optional
        Column name containing gene IDs (default: 'gene_ids')
    symbol_column : str, optional
        Column name containing gene symbols (default: None, use var_names)
    prefer_symbol : bool, optional
        If True, prefer symbols over IDs when available (default: True)
        
    Returns
    -------
    id2symbol : dict
        Mapping from gene IDs to symbols
        
    Examples
    --------
    >>> mapping = build_id2symbol_mapping(adata)
    >>> symbol = mapping.get('ENSG00000141510', 'TP53')
    """
    id2symbol = {}
    
    # Get gene IDs
    if id_column in adata.var.columns:
        gene_ids = adata.var[id_column].values
    else:
        # Use var_names as IDs
        gene_ids = adata.var_names.values
    
    # Get gene symbols
    if symbol_column is not None and symbol_column in adata.var.columns:
        gene_symbols = adata.var[symbol_column].values
    else:
        # Use var_names as symbols
        gene_symbols = adata.var_names.values
    
    # Build mapping
    for gene_id, symbol in zip(gene_ids, gene_symbols):
        gene_id = clean_symbol(gene_id)
        symbol = clean_symbol(symbol)
        
        if prefer_symbol and symbol:
            id2symbol[gene_id] = symbol
        elif gene_id:
            id2symbol[gene_id] = gene_id
    
    return id2symbol


def ensure_unique_var_names(adata: sc.AnnData, inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Ensure gene names are unique.
    
    Keeps first occurrence when duplicates exist.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    inplace : bool, optional
        If True, modify adata in place (default: True)
        
    Returns
    -------
    adata : AnnData or None
        AnnData with unique gene names (if inplace=False)
        
    Examples
    --------
    >>> adata = ensure_unique_var_names(adata)
    >>> assert adata.var_names.is_unique
    """
    if not inplace:
        adata = adata.copy()
    
    if not adata.var_names.is_unique:
        # Get unique genes (keep first occurrence)
        var_index = pd.Index(adata.var_names)
        first_pos = ~var_index.duplicated(keep="first")
        keep_cols = np.where(first_pos)[0]
        
        # Subset to unique genes
        adata._inplace_subset_var(keep_cols)
        
        print(f"[Gene filtering] Kept {len(keep_cols)} unique genes from {len(var_index)} total")
    
    if not inplace:
        return adata


def get_common_genes(adata1: sc.AnnData, 
                    adata2: sc.AnnData,
                    ensure_unique: bool = True) -> pd.Index:
    """
    Get common genes between two AnnData objects.
    
    Parameters
    ----------
    adata1 : AnnData
        First AnnData object
    adata2 : AnnData
        Second AnnData object
    ensure_unique : bool, optional
        If True, ensure gene names are unique first (default: True)
        
    Returns
    -------
    common_genes : pd.Index
        Index of common gene names
        
    Examples
    --------
    >>> common = get_common_genes(adata_sp, adata_sc)
    >>> print(f"Found {len(common)} common genes")
    """
    if ensure_unique:
        ensure_unique_var_names(adata1, inplace=True)
        ensure_unique_var_names(adata2, inplace=True)
    
    common = adata1.var_names.intersection(adata2.var_names)
    
    if len(common) == 0:
        raise ValueError("No common genes found between datasets")
    
    return common


def normalize_pdac_celltype(celltype_name: str) -> str:
    """
    Normalize PDAC cell type names.
    
    Specifically handles PDAC dataset conventions.
    Removes numeric suffixes like '.151', '.150'.
    
    Parameters
    ----------
    celltype_name : str
        Original PDAC cell type name
        
    Returns
    -------
    normalized : str
        Normalized cell type name
        
    Examples
    --------
    >>> normalize_pdac_celltype('Acinar.151')
    'Acinar'
    >>> normalize_pdac_celltype('Cancer clone A.150')
    'Cancer clone A'
    """
    return normalize_celltype(celltype_name)


def preprocess_celltype_column(adata: sc.AnnData,
                              celltype_col: str = 'cell_type',
                              normalize_func = normalize_celltype,
                              backup_original: bool = True,
                              verbose: bool = True) -> sc.AnnData:
    """
    Preprocess cell type annotations.
    
    Applies normalization and optionally backs up original labels.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    celltype_col : str, optional
        Column name containing cell types (default: 'cell_type')
    normalize_func : callable, optional
        Normalization function (default: normalize_celltype)
    backup_original : bool, optional
        If True, save original labels to '{celltype_col}_original' (default: True)
    verbose : bool, optional
        If True, print statistics (default: True)
        
    Returns
    -------
    adata : AnnData
        AnnData with normalized cell types
        
    Examples
    --------
    >>> adata = preprocess_celltype_column(adata)
    >>> print(adata.obs['cell_type'].value_counts())
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_col}' not found in adata.obs")
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"Preprocessing cell type column: {celltype_col}")
        print("=" * 80)
    
    # Backup original
    if backup_original:
        adata.obs[f'{celltype_col}_original'] = adata.obs[celltype_col].copy()
    
    # Statistics before
    original_unique = adata.obs[celltype_col].nunique()
    if verbose:
        print(f"\nOriginal cell types: {original_unique}")
    
    # Apply normalization
    adata.obs[celltype_col] = adata.obs[celltype_col].apply(normalize_func)
    
    # Statistics after
    normalized_unique = adata.obs[celltype_col].nunique()
    if verbose:
        print(f"Normalized cell types: {normalized_unique}")
        print(f"Merged {original_unique - normalized_unique} duplicate types")
        
        print("\nCell type distribution:")
        print("-" * 80)
        ct_counts = adata.obs[celltype_col].value_counts()
        for i, (ct, count) in enumerate(ct_counts.items(), 1):
            pct = count / len(adata) * 100
            print(f"{i:2d}. {ct:45s}: {count:5d} cells ({pct:5.1f}%)")
        print("=" * 80 + "\n")
    
    return adata
