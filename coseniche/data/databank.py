"""
SpatialDataBank: Data management for spatial transcriptomics

Handles loading, caching, and serving of spatial transcriptomics data.
"""

import os
import json
import logging
import datetime
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import h5py
import anndata as ad
import scanpy as sc
from torch.utils.data import DataLoader

from .preprocessor import Preprocessor, GeneVocab
from .dataset import MemoryEfficientSpotDataset, optimized_collate_fn

logger = logging.getLogger(__name__)


class SpatialDataBank:
    """
    Data management system for spatial transcriptomics datasets.
    
    Handles:
    - Loading and preprocessing of h5ad files
    - Caching preprocessed data to HDF5
    - Spatial neighbor computation
    - Efficient data serving for training/inference
    
    Args:
        dataset_paths: List of paths to h5ad files
        cache_dir: Directory for caching preprocessed data
        config: Configuration object with preprocessing parameters
        force_rebuild: Whether to force rebuild cache
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        cache_dir: str,
        config: Any,
        force_rebuild: bool = False
    ):
        self.dataset_paths = dataset_paths
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.force_rebuild = force_rebuild
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            "datasets": [],
            "dataset_indices": [],
            "total_spots": 0,
            "created_at": datetime.datetime.now().isoformat(),
        }
        
        # File handles (lazy loaded)
        self._file_handles = {}
        self._neighbor_handles = {}
        
        # Vocabulary
        self._vocab = None
        
        # Initialize datasets
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize and cache all datasets."""
        logger.info(f"Initializing {len(self.dataset_paths)} datasets...")
        
        cumulative_count = 0
        all_genes = set()
        
        for idx, path in enumerate(self.dataset_paths):
            if not os.path.exists(path):
                logger.warning(f"Dataset not found: {path}")
                continue
            
            dataset_name = Path(path).stem
            dataset_cache_dir = self.cache_dir / dataset_name
            dataset_cache_dir.mkdir(exist_ok=True)
            
            # Check if cache exists
            spots_file = dataset_cache_dir / "spots.h5"
            metadata_file = dataset_cache_dir / "dataset_metadata.json"
            
            if spots_file.exists() and metadata_file.exists() and not self.force_rebuild:
                # Load from cache
                logger.info(f"Loading cached data for {dataset_name}")
                with open(metadata_file) as f:
                    dataset_metadata = json.load(f)
            else:
                # Process dataset
                logger.info(f"Processing dataset: {dataset_name}")
                dataset_metadata = self._process_dataset(
                    path, dataset_name, dataset_cache_dir, cumulative_count
                )
            
            if dataset_metadata is None:
                continue
            
            # Update global metadata
            n_spots = dataset_metadata.get("n_spots", 0)
            start_idx = cumulative_count
            end_idx = cumulative_count + n_spots
            
            self.metadata["datasets"].append({
                "name": dataset_name,
                "path": path,
                "cache_dir": str(dataset_cache_dir),
                "n_spots": n_spots,
                "n_genes": dataset_metadata.get("n_genes", 0),
            })
            
            self.metadata["dataset_indices"].append({
                "dataset_idx": idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })
            
            # Update gene set
            genes = dataset_metadata.get("genes", [])
            all_genes.update(genes)
            
            cumulative_count = end_idx
        
        self.metadata["total_spots"] = cumulative_count
        self.metadata["total_genes"] = len(all_genes)
        self.metadata["all_genes"] = sorted(list(all_genes))
        
        # Save global metadata
        metadata_path = self.cache_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Initialized {len(self.metadata['datasets'])} datasets, "
                   f"{cumulative_count} total spots, {len(all_genes)} unique genes")
    
    def _process_dataset(
        self,
        path: str,
        dataset_name: str,
        cache_dir: Path,
        start_idx: int
    ) -> Optional[Dict]:
        """Process a single dataset and cache it."""
        try:
            # Load data
            adata = ad.read_h5ad(path)
            logger.info(f"Loaded {dataset_name}: {adata.shape}")
            
            # Preprocess
            preprocessor = Preprocessor(
                filter_gene_by_counts=getattr(self.config, 'filter_gene_by_counts', False),
                filter_cell_by_counts=getattr(self.config, 'filter_cell_by_genes', False),
                normalize_total=1e4,
                result_normed_key="X_normed",
                log1p=True,
                result_log1p_key="X_log1p",
                subset_hvg=getattr(self.config, 'subset_hvg', 2000),
            )
            
            preprocessor(adata)
            
            # Prepare gene vocabulary
            vocab = self._get_or_create_vocab(adata.var_names.tolist())
            
            # Cache spot data
            spots_file = cache_dir / "spots.h5"
            n_spots = self._cache_spots(adata, spots_file, vocab, start_idx)
            
            # Get metadata
            metadata = {
                "n_spots": n_spots,
                "n_genes": adata.n_vars,
                "genes": adata.var_names.tolist(),
                "has_spatial": "spatial" in adata.obsm,
            }
            
            # Extract platform/organ info if available
            if "platform" in adata.obs.columns:
                metadata["platforms"] = adata.obs["platform"].unique().tolist()
            if "organ" in adata.obs.columns:
                metadata["organs"] = adata.obs["organ"].unique().tolist()
            
            # Save metadata
            with open(cache_dir / "dataset_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_or_create_vocab(self, genes: List[str]) -> GeneVocab:
        """Get or create gene vocabulary."""
        vocab_path = self.config.vocab_file if hasattr(self.config, 'vocab_file') else None
        
        if vocab_path and os.path.exists(vocab_path):
            return GeneVocab.from_json(vocab_path)
        
        # Create new vocabulary
        return GeneVocab(genes)
    
    def _cache_spots(
        self,
        adata: ad.AnnData,
        spots_file: Path,
        vocab: GeneVocab,
        global_offset: int
    ) -> int:
        """Cache spot data to HDF5 file."""
        n_spots = adata.n_obs
        
        with h5py.File(spots_file, 'w') as f:
            f.attrs['n_spots'] = n_spots
            f.attrs['global_offset'] = global_offset
            
            spots_group = f.create_group('spots')
            
            # Get expression data
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            
            # Get normalized/log data if available
            X_normed = adata.layers.get('X_normed', X)
            if sp.issparse(X_normed):
                X_normed = X_normed.toarray()
            
            X_log = adata.layers.get('X_log1p', np.log1p(X_normed))
            if sp.issparse(X_log):
                X_log = X_log.toarray()
            
            # Gene IDs
            gene_names = adata.var_names.tolist()
            gene_ids = np.array([vocab[g] for g in gene_names], dtype=np.int64)
            
            # Platform/organ IDs
            platform_ids = self._encode_metadata(adata.obs.get('platform', None))
            organ_ids = self._encode_metadata(adata.obs.get('organ', None))
            
            for i in range(n_spots):
                global_idx = global_offset + i
                spot_group = spots_group.create_group(str(global_idx))
                
                # Store gene IDs and values
                expr = X_log[i] if len(X_log.shape) > 1 else X_log
                nonzero_mask = expr > 0
                
                # Option 1: Store all genes
                spot_group.create_dataset('gene_ids', data=gene_ids, compression='gzip')
                spot_group.create_dataset('values', data=expr.astype(np.float32), compression='gzip')
                spot_group.create_dataset('raw_normed_values', data=X_normed[i].astype(np.float32), compression='gzip')
                
                # Store metadata
                spot_group.attrs['platform_id'] = platform_ids[i] if platform_ids is not None else 0
                spot_group.attrs['organ_id'] = organ_ids[i] if organ_ids is not None else 0
                spot_group.attrs['local_idx'] = i
        
        return n_spots
    
    def _encode_metadata(self, series) -> Optional[np.ndarray]:
        """Encode categorical metadata to integer IDs."""
        if series is None:
            return None
        
        categories = series.unique().tolist()
        cat_to_id = {cat: idx for idx, cat in enumerate(categories)}
        return np.array([cat_to_id[v] for v in series], dtype=np.int32)
    
    def get_spot_data(self, global_idx: int, include_raw: bool = False) -> Optional[Dict]:
        """
        Get data for a single spot.
        
        Args:
            global_idx: Global spot index
            include_raw: Whether to include raw expression values
            
        Returns:
            Dictionary with spot data
        """
        # Find dataset
        dataset_idx = None
        local_idx = None
        
        for idx_info in self.metadata["dataset_indices"]:
            if idx_info["start_idx"] <= global_idx < idx_info["end_idx"]:
                dataset_idx = idx_info["dataset_idx"]
                local_idx = global_idx - idx_info["start_idx"]
                break
        
        if dataset_idx is None:
            return None
        
        # Get file handle
        dataset_info = self.metadata["datasets"][dataset_idx]
        spots_file = Path(dataset_info["cache_dir"]) / "spots.h5"
        
        if str(spots_file) not in self._file_handles:
            self._file_handles[str(spots_file)] = h5py.File(spots_file, 'r')
        
        f = self._file_handles[str(spots_file)]
        
        try:
            spot_group = f['spots'][str(global_idx)]
            
            data = {
                'gene_ids': spot_group['gene_ids'][:],
                'values': spot_group['values'][:],
                'raw_normed_values': spot_group['raw_normed_values'][:],
                'platform_id': spot_group.attrs.get('platform_id', 0),
                'organ_id': spot_group.attrs.get('organ_id', 0),
                'global_idx': global_idx,
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load spot {global_idx}: {e}")
            return None
    
    def get_neighbors_for_spot(self, global_idx: int) -> List[int]:
        """
        Get neighbor indices for a spot.
        
        Args:
            global_idx: Global spot index
            
        Returns:
            List of neighbor global indices
        """
        # Find dataset
        dataset_idx = None
        for idx_info in self.metadata["dataset_indices"]:
            if idx_info["start_idx"] <= global_idx < idx_info["end_idx"]:
                dataset_idx = idx_info["dataset_idx"]
                break
        
        if dataset_idx is None:
            return []
        
        # Check for neighbors file
        neighbors_dir = self.cache_dir / "neighbors"
        index_file = neighbors_dir / "neighbors_index.json"
        
        if not index_file.exists():
            return []
        
        # Load neighbor index
        with open(index_file) as f:
            neighbor_files = json.load(f)
        
        if str(dataset_idx) not in neighbor_files:
            return []
        
        neighbors_file = neighbor_files[str(dataset_idx)]
        
        # Open neighbor file
        if neighbors_file not in self._neighbor_handles:
            self._neighbor_handles[neighbors_file] = h5py.File(neighbors_file, 'r')
        
        f = self._neighbor_handles[neighbors_file]
        
        try:
            neighbors = f['neighbors'][str(global_idx)][:]
            # Filter out padding (-1)
            return [int(n) for n in neighbors if n >= 0]
        except Exception:
            return []
    
    def prepare_vocabulary(self) -> GeneVocab:
        """Get gene vocabulary."""
        if self._vocab is None:
            vocab_path = getattr(self.config, 'vocab_file', None)
            if vocab_path and os.path.exists(vocab_path):
                self._vocab = GeneVocab.from_json(vocab_path)
            else:
                self._vocab = GeneVocab(self.metadata.get("all_genes", []))
        return self._vocab
    
    def get_data_split(
        self,
        validation_split: float = 0.1,
        split: str = "train",
        random_seed: int = 42
    ) -> List[int]:
        """
        Get train/validation split indices.
        
        Args:
            validation_split: Fraction for validation
            split: "train" or "val"
            random_seed: Random seed for reproducibility
            
        Returns:
            List of global indices
        """
        np.random.seed(random_seed)
        
        all_indices = list(range(self.metadata["total_spots"]))
        np.random.shuffle(all_indices)
        
        n_val = int(len(all_indices) * validation_split)
        
        if split == "val":
            return all_indices[:n_val]
        else:
            return all_indices[n_val:]
    
    def get_data_loader(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        validation_split: float = 0.1,
        is_training: Optional[bool] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader for training/inference.
        
        Args:
            split: "train" or "val"
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            validation_split: Fraction for validation
            is_training: Override training mode
            
        Returns:
            PyTorch DataLoader
        """
        indices = self.get_data_split(validation_split, split)
        
        if is_training is None:
            is_training = (split == "train")
        
        dataset = MemoryEfficientSpotDataset(
            indices=indices,
            databank=self,
            max_seq_len=getattr(self.config, 'max_seq_len', 1700),
            mask_ratio=getattr(self.config, 'mask_ratio', 0.4),
            mask_value=getattr(self.config, 'mask_value', -1),
            pad_value=getattr(self.config, 'pad_value', -2),
            is_training=is_training
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=optimized_collate_fn,
            drop_last=False,
            **kwargs
        )
    
    def _get_dataset_by_idx(self, dataset_idx: int) -> Tuple[ad.AnnData, Dict]:
        """Load a dataset by index (for preprocessing)."""
        dataset_info = self.metadata["datasets"][dataset_idx]
        adata = ad.read_h5ad(dataset_info["path"])
        return adata, dataset_info
    
    def preprocess_dataset_spots(self, dataset_idx: int) -> str:
        """Preprocess and cache a single dataset's spots."""
        adata, dataset_info = self._get_dataset_by_idx(dataset_idx)
        
        cache_dir = Path(dataset_info["cache_dir"])
        spots_file = cache_dir / "spots.h5"
        
        start_idx = self.metadata["dataset_indices"][dataset_idx]["start_idx"]
        vocab = self.prepare_vocabulary()
        
        self._cache_spots(adata, spots_file, vocab, start_idx)
        
        return str(spots_file)
    
    def cleanup_resources(self):
        """Close all open file handles."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass
        self._file_handles.clear()
        
        for handle in self._neighbor_handles.values():
            try:
                handle.close()
            except:
                pass
        self._neighbor_handles.clear()
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_resources()


