"""
Data Preprocessing Module for CoseNiche

Handles normalization, HVG selection, and expression binning.
"""

import logging
from typing import Optional, Dict, Union
import warnings

import numpy as np
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def _digitize(x: np.ndarray, bins: np.ndarray, side: str = "both") -> np.ndarray:
    """
    Digitize expression values into bins.
    
    Args:
        x: Input array to digitize
        bins: Bin edges
        side: Which side to include ("both", "left", "right")
        
    Returns:
        Digitized array
    """
    assert x.ndim == 1 and bins.ndim == 1, "x and bins must be 1D arrays"
    
    if side == "one":
        return np.digitize(x, bins)
    if side == "both":
        left_digits = np.digitize(x, bins, right=False)
        right_digits = np.digitize(x, bins, right=True)
        
        rands = np.random.rand(len(x))
        digits = np.where(rands < 0.5, left_digits, right_digits)
    else:
        raise ValueError(f"Invalid side: {side}")
    
    return digits


class Preprocessor:
    """
    Preprocessor for spatial transcriptomics data.
    
    Handles:
    - Filtering genes and cells
    - Normalization (total count)
    - Log transformation
    - HVG selection
    - Expression binning
    
    Args:
        use_key: Key in adata.layers to use (None for adata.X)
        filter_gene_by_counts: Minimum counts to keep a gene
        filter_cell_by_counts: Minimum counts to keep a cell
        normalize_total: Target sum for normalization
        result_normed_key: Key to store normalized data
        log1p: Whether to apply log1p transformation
        result_log1p_key: Key to store log-transformed data
        subset_hvg: Number of HVGs to select
        hvg_use_key: Key to use for HVG calculation
        hvg_flavor: HVG calculation method
    """
    
    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        Process AnnData object.
        
        Args:
            adata: Input AnnData object (modified in place)
            batch_key: Optional batch key for batch-aware HVG selection
            
        Returns:
            Dictionary with processing info
        """
        info = {"status": "success", "warnings": []}
        
        # Get initial data
        if self.use_key is not None and self.use_key in adata.layers:
            key_to_process = self.use_key
        else:
            key_to_process = None
            
        # Filter genes
        if self.filter_gene_by_counts:
            try:
                initial_genes = adata.n_vars
                sc.pp.filter_genes(adata, min_counts=self.filter_gene_by_counts)
                info["genes_filtered"] = initial_genes - adata.n_vars
                logger.info(f"Filtered {info['genes_filtered']} genes (< {self.filter_gene_by_counts} counts)")
            except Exception as e:
                info["warnings"].append(f"Gene filtering failed: {e}")
                
        # Filter cells
        if self.filter_cell_by_counts:
            try:
                initial_cells = adata.n_obs
                sc.pp.filter_cells(adata, min_counts=self.filter_cell_by_counts)
                info["cells_filtered"] = initial_cells - adata.n_obs
                logger.info(f"Filtered {info['cells_filtered']} cells (< {self.filter_cell_by_counts} counts)")
            except Exception as e:
                info["warnings"].append(f"Cell filtering failed: {e}")
        
        # Normalize
        if self.normalize_total:
            try:
                sc.pp.normalize_total(adata, target_sum=self.normalize_total)
                if self.result_normed_key:
                    adata.layers[self.result_normed_key] = adata.X.copy()
                logger.info(f"Normalized to target sum {self.normalize_total}")
            except Exception as e:
                info["warnings"].append(f"Normalization failed: {e}")
        
        # Log transform
        if self.log1p:
            try:
                sc.pp.log1p(adata)
                if self.result_log1p_key:
                    adata.layers[self.result_log1p_key] = adata.X.copy()
                logger.info("Applied log1p transformation")
            except Exception as e:
                info["warnings"].append(f"Log transformation failed: {e}")
        
        # HVG selection
        if self.subset_hvg:
            try:
                self._process_hvg(adata, batch_key, key_to_process)
                logger.info(f"Selected {self.subset_hvg} HVGs")
            except Exception as e:
                info["warnings"].append(f"HVG selection failed: {e}")
                self._fallback_to_expression_ranking(adata, self.subset_hvg)
        
        return info

    def _process_hvg(self, adata: AnnData, batch_key: Optional[str], key_to_process: Optional[str]):
        """Process highly variable gene selection."""
        hvg_params = {
            "n_top_genes": self.subset_hvg,
            "flavor": self.hvg_flavor,
            "batch_key": batch_key,
        }
        
        if key_to_process and key_to_process in adata.layers:
            hvg_params["layer"] = key_to_process
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                sc.pp.highly_variable_genes(adata, **hvg_params)
                adata._inplace_subset_var(adata.var["highly_variable"])
            except Exception as e:
                logger.warning(f"HVG selection failed with {self.hvg_flavor}: {e}")
                self._try_alternative_hvg_methods(adata, hvg_params)

    def _try_alternative_hvg_methods(self, adata: AnnData, hvg_params: Dict):
        """Try alternative HVG methods if primary fails."""
        alternative_flavors = ["seurat", "cell_ranger"]
        
        for flavor in alternative_flavors:
            try:
                hvg_params["flavor"] = flavor
                sc.pp.highly_variable_genes(adata, **hvg_params)
                adata._inplace_subset_var(adata.var["highly_variable"])
                logger.info(f"HVG selection succeeded with {flavor}")
                return
            except Exception:
                continue
        
        # Final fallback
        self._fallback_to_expression_ranking(adata, hvg_params["n_top_genes"])

    def _fallback_to_expression_ranking(self, adata: AnnData, n_top_genes: int):
        """Select genes by mean expression if HVG fails."""
        logger.warning("Using expression ranking as HVG fallback")
        
        import scipy.sparse as sp
        
        if sp.issparse(adata.X):
            gene_means = np.asarray(adata.X.mean(axis=0)).flatten()
        else:
            gene_means = np.mean(adata.X, axis=0)
        
        n_genes = min(n_top_genes, len(gene_means))
        top_indices = np.argsort(gene_means)[-n_genes:]
        
        adata._inplace_subset_var(adata.var_names[top_indices])
        logger.info(f"Selected {n_genes} genes by mean expression")

    @staticmethod
    def check_logged(adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if data appears to be log-transformed.
        
        Args:
            adata: AnnData object
            obs_key: Optional observation key to check specific subset
            
        Returns:
            True if data appears logged (max < 30)
        """
        import scipy.sparse as sp
        
        data = adata.X
        if sp.issparse(data):
            max_val = data.max()
        else:
            max_val = np.max(data)
        
        return max_val < 30


class GeneVocab:
    """
    Gene vocabulary for mapping gene names to indices.
    """
    
    def __init__(self, gene_list: list, special_tokens: Optional[list] = None):
        """
        Initialize vocabulary.
        
        Args:
            gene_list: List of gene names
            special_tokens: Optional list of special tokens to prepend
        """
        self.special_tokens = special_tokens or ["<pad>", "<cls>", "<mask>"]
        
        self.gene2idx = {}
        self.idx2gene = {}
        
        # Add special tokens
        for idx, token in enumerate(self.special_tokens):
            self.gene2idx[token] = idx
            self.idx2gene[idx] = token
        
        # Add genes
        offset = len(self.special_tokens)
        for idx, gene in enumerate(gene_list):
            if gene not in self.gene2idx:
                self.gene2idx[gene] = idx + offset
                self.idx2gene[idx + offset] = gene
    
    def __len__(self):
        return len(self.gene2idx)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.gene2idx.get(key, self.gene2idx.get("<pad>", 0))
        elif isinstance(key, int):
            return self.idx2gene.get(key, "<unk>")
        else:
            raise KeyError(f"Invalid key type: {type(key)}")
    
    def encode(self, genes: list) -> list:
        """Encode gene names to indices."""
        return [self[g] for g in genes]
    
    def decode(self, indices: list) -> list:
        """Decode indices to gene names."""
        return [self[i] for i in indices]
    
    @classmethod
    def from_json(cls, path: str) -> "GeneVocab":
        """Load vocabulary from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Assume it's gene2idx mapping
            genes = list(data.keys())
            # Remove special tokens from gene list
            special_tokens = ["<pad>", "<cls>", "<mask>", "[PAD]", "[CLS]", "[MASK]"]
            genes = [g for g in genes if g not in special_tokens]
        else:
            genes = data
        
        return cls(genes)
    
    def save(self, path: str):
        """Save vocabulary to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.gene2idx, f, indent=2)


