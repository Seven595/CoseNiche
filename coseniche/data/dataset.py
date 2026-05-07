"""
Dataset classes for CoseNiche

Memory-efficient dataset implementation for spatial transcriptomics data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BatchSkipException(Exception):
    """Exception raised when a batch should be skipped."""
    pass


class MemoryEfficientSpotDataset(Dataset):
    """
    Memory-efficient spatial transcriptomics dataset.
    
    Loads data on-demand from pre-processed HDF5 cache files.
    
    Args:
        indices: Global indices of spots to include
        databank: SpatialDataBank instance
        max_seq_len: Maximum sequence length
        mask_ratio: Ratio of tokens to mask (training only)
        mask_value: Value for masked positions
        pad_value: Value for padding positions
        pad_token: Padding token string
        cls_token: CLS token string
        max_prefetch: Maximum items to cache
        is_training: Whether in training mode
    """
    
    def __init__(
        self,
        indices: List[int],
        databank: 'SpatialDataBank',
        max_seq_len: int = 1024,
        mask_ratio: float = 0.15,
        mask_value: int = -1,
        pad_value: int = -2,
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        max_prefetch: int = 1000,
        is_training: bool = True
    ):
        self.indices = indices
        self.databank = databank
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.max_prefetch = max_prefetch
        self.is_training = is_training
        
        # LRU cache for spots
        self._spot_cache = OrderedDict()
        self._neighbor_cache = OrderedDict()
        
        # Get vocabulary
        self.vocab = databank.prepare_vocabulary()
        
        logger.info(f"Initialized dataset with {len(indices)} spots")

    def _update_cache(self, key: Any, value: Any, cache_dict: OrderedDict):
        """Update LRU cache."""
        if key in cache_dict:
            cache_dict.move_to_end(key)
        else:
            cache_dict[key] = value
            if len(cache_dict) > self.max_prefetch:
                cache_dict.popitem(last=False)

    def _get_spot_data(self, global_idx: int) -> Dict:
        """Get spot data with caching."""
        if global_idx in self._spot_cache:
            self._spot_cache.move_to_end(global_idx)
            return self._spot_cache[global_idx]
        
        spot_data = self.databank.get_spot_data(global_idx, include_raw=True)
        self._update_cache(global_idx, spot_data, self._spot_cache)
        return spot_data

    def _get_neighbors(self, global_idx: int) -> List[int]:
        """Get neighbor indices with caching."""
        if global_idx in self._neighbor_cache:
            self._neighbor_cache.move_to_end(global_idx)
            return self._neighbor_cache[global_idx]
        
        neighbors = self.databank.get_neighbors_for_spot(global_idx)
        self._update_cache(global_idx, neighbors, self._neighbor_cache)
        return neighbors

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with spot data and neighbor information
        """
        global_idx = self.indices[idx]
        
        # Get center spot data
        center_data = self._get_spot_data(global_idx)
        
        if center_data is None or 'gene_ids' not in center_data:
            raise IndexError(f"Invalid spot data for index {global_idx}")
        
        # Get neighbors
        neighbors = self._get_neighbors(global_idx)
        
        # Collect neighbor data
        neighbor_data_list = []
        for n_idx in neighbors:
            n_data = self._get_spot_data(n_idx)
            if n_data is not None:
                neighbor_data_list.append({
                    'global_idx': n_idx,
                    'data': n_data
                })
        
        return {
            'center_idx': global_idx,
            'center_data': center_data,
            'neighbors': neighbor_data_list,
            'platform_id': center_data.get('platform_id', 0),
            'organ_id': center_data.get('organ_id', 0),
        }


def pad_sequences_dual(
    gene_ids_list: List[np.ndarray],
    values_list: List[np.ndarray],
    max_len: int,
    pad_token_id: int,
    pad_value: float,
    include_zero_gene: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad gene ID and value sequences to uniform length.
    
    Args:
        gene_ids_list: List of gene ID arrays
        values_list: List of expression value arrays
        max_len: Maximum sequence length
        pad_token_id: ID for padding token
        pad_value: Value for padding positions
        include_zero_gene: Whether to include zero-expression genes
        
    Returns:
        Tuple of (padded_genes, padded_values, attention_mask)
    """
    batch_size = len(gene_ids_list)
    
    # Initialize outputs
    padded_genes = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
    padded_values = np.full((batch_size, max_len), pad_value, dtype=np.float32)
    attention_mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, (genes, vals) in enumerate(zip(gene_ids_list, values_list)):
        genes = np.asarray(genes)
        vals = np.asarray(vals)
        
        if not include_zero_gene:
            # Filter out zero-expression genes
            nonzero_mask = vals > 0
            if nonzero_mask.sum() > 0:
                genes = genes[nonzero_mask]
                vals = vals[nonzero_mask]
        
        # Truncate if necessary
        seq_len = min(len(genes), max_len)
        
        if seq_len > 0:
            padded_genes[i, :seq_len] = genes[:seq_len]
            padded_values[i, :seq_len] = vals[:seq_len]
            attention_mask[i, :seq_len] = True
    
    return (
        torch.from_numpy(padded_genes),
        torch.from_numpy(padded_values),
        torch.from_numpy(attention_mask)
    )


def optimized_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    
    Efficiently batches samples with variable-length sequences.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Batched dictionary with tensors
    """
    if not batch:
        raise BatchSkipException("Empty batch")
    
    # Filter out None samples
    batch = [b for b in batch if b is not None and b.get('center_data') is not None]
    
    if not batch:
        raise BatchSkipException("All samples in batch are invalid")
    
    batch_size = len(batch)
    
    # Collect all spots (center + neighbors)
    all_spot_data = []
    center_indices = []
    batch_to_spots_map = []
    platform_ids = []
    organ_ids = []
    
    spot_offset = 0
    
    for sample in batch:
        center_data = sample['center_data']
        neighbors = sample.get('neighbors', [])
        
        # Add center spot
        center_indices.append(spot_offset)
        all_spot_data.append(center_data)
        platform_ids.append(sample.get('platform_id', 0))
        organ_ids.append(sample.get('organ_id', 0))
        spot_offset += 1
        
        # Track neighbor range
        neighbor_start = spot_offset
        
        # Add neighbors
        for n in neighbors:
            if n.get('data') is not None:
                all_spot_data.append(n['data'])
                platform_ids.append(n['data'].get('platform_id', 0))
                organ_ids.append(n['data'].get('organ_id', 0))
                spot_offset += 1
        
        neighbor_end = spot_offset
        batch_to_spots_map.append((neighbor_start, neighbor_end))
    
    # Determine maximum sequence length
    max_seq_len = max(
        len(d.get('gene_ids', [])) for d in all_spot_data
    )
    max_seq_len = min(max_seq_len, 1700)  # Cap at reasonable maximum
    
    if max_seq_len == 0:
        raise BatchSkipException("All sequences have zero length")
    
    # Pad sequences
    gene_ids_list = [np.asarray(d.get('gene_ids', [])) for d in all_spot_data]
    values_list = [np.asarray(d.get('values', d.get('raw_normed_values', []))) for d in all_spot_data]
    
    # Get pad token ID (assume 0 if not available)
    pad_token_id = 0
    
    genes, values, attention_mask = pad_sequences_dual(
        gene_ids_list, values_list, max_seq_len,
        pad_token_id=pad_token_id,
        pad_value=-2.0,
        include_zero_gene=True
    )
    
    # Also get raw normed values if available
    raw_normed_list = [np.asarray(d.get('raw_normed_values', d.get('values', []))) for d in all_spot_data]
    _, raw_normed_values, _ = pad_sequences_dual(
        gene_ids_list, raw_normed_list, max_seq_len,
        pad_token_id=pad_token_id,
        pad_value=-2.0,
        include_zero_gene=True
    )
    
    return {
        'flat': {
            'genes': genes,
            'values': values,
            'raw_normed_values': raw_normed_values,
            'padding_attention_mask': attention_mask,
        },
        'structure': {
            'center_indices': center_indices,
            'batch_to_spots_map': batch_to_spots_map,
            'platform_ids': torch.tensor(platform_ids, dtype=torch.long),
            'organ_ids': torch.tensor(organ_ids, dtype=torch.long),
        }
    }


class RetryOnSkipDataLoader:
    """
    Wrapper that retries on BatchSkipException.
    
    Automatically gets next batch when current batch is skipped.
    """
    
    def __init__(self, dataloader, max_retries: int = 50):
        self.dataloader = dataloader
        self.max_retries = max_retries

    def __iter__(self):
        iterator = iter(self.dataloader)
        retry_count = 0
        
        while True:
            try:
                batch = next(iterator)
                retry_count = 0
                yield batch
            except StopIteration:
                break
            except BatchSkipException as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.warning(f"Max retries ({self.max_retries}) reached, stopping")
                    break
                continue
            except Exception as e:
                logger.error(f"Error in dataloader: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    break
                continue

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def sampler(self):
        return getattr(self.dataloader, "sampler", None)

    @property
    def batch_size(self):
        return getattr(self.dataloader, "batch_size", None)


