#!/usr/bin/env python
"""
CoseNiche Data Preprocessing Script

Preprocesses spatial transcriptomics data and computes spatial neighbors.

Usage:
    python preprocess.py --h5ad_file data.h5ad --cache_dir ./cache --max_neighbors 6
"""

import os
import sys
import argparse
import logging
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import h5py
import scipy.sparse as sp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coseniche import CoseNicheConfig
from coseniche.data import SpatialDataBank

logger = logging.getLogger('coseniche.preprocess')


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocess.log')
        ]
    )
    return logging.getLogger('coseniche.preprocess')


def compute_spatial_neighbors(
    spatial_coords: np.ndarray,
    max_neighbors: int,
    use_gpu: bool = True,
    block_size: int = 5000
) -> tuple:
    """
    Compute spatial neighbors using KNN.
    
    Args:
        spatial_coords: Spatial coordinates [N, 2]
        max_neighbors: Maximum number of neighbors
        use_gpu: Whether to use GPU acceleration
        block_size: Block size for batch processing
        
    Returns:
        Tuple of (neighbors_dict, distances_dict)
    """
    n_spots = spatial_coords.shape[0]
    
    # Ensure 2D coordinates
    if spatial_coords.shape[1] > 2:
        spatial_coords = spatial_coords[:, :2]
    
    logger.info(f"Computing {max_neighbors} neighbors for {n_spots} spots")
    
    all_neighbors = {}
    all_distances = {}
    
    # GPU acceleration
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        coords_gpu = torch.tensor(spatial_coords, dtype=torch.float32, device=device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        n_blocks = (n_spots + block_size - 1) // block_size
        
        for i in range(n_blocks):
            start_i = i * block_size
            end_i = min(start_i + block_size, n_spots)
            
            for j in range(start_i, end_i):
                point = coords_gpu[j].view(1, -1)
                diffs = coords_gpu - point
                sq_dists = torch.sum(diffs * diffs, dim=1)
                sq_dists[j] = float('inf')  # Exclude self
                
                _, nearest_indices = torch.topk(sq_dists, k=max_neighbors, largest=False)
                nearest_indices = nearest_indices.cpu().numpy()
                nearest_dists = torch.sqrt(sq_dists[nearest_indices]).cpu().numpy()
                
                all_neighbors[j] = nearest_indices
                all_distances[j] = nearest_dists
            
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                logger.info(f"Processed block {i+1}/{n_blocks}")
    else:
        logger.info("Using CPU")
        for j in range(n_spots):
            dists = np.sqrt(np.sum((spatial_coords - spatial_coords[j])**2, axis=1))
            dists[j] = np.inf
            
            nearest_indices = np.argsort(dists)[:max_neighbors]
            nearest_dists = dists[nearest_indices]
            
            all_neighbors[j] = nearest_indices
            all_distances[j] = nearest_dists
            
            if (j + 1) % 1000 == 0:
                logger.info(f"Processed {j+1}/{n_spots} spots")
    
    return all_neighbors, all_distances


def save_neighbors_to_hdf5(
    neighbors_file: str,
    dataset_name: str,
    max_neighbors: int,
    start_idx: int,
    end_idx: int,
    all_neighbors: dict,
    all_distances: dict
):
    """Save neighbor relationships to HDF5 file."""
    n_spots = end_idx - start_idx
    
    with h5py.File(neighbors_file, 'w') as f:
        f.attrs['dataset_name'] = dataset_name
        f.attrs['max_neighbors'] = max_neighbors
        f.attrs['start_idx'] = start_idx
        f.attrs['end_idx'] = end_idx
        f.attrs['n_spots'] = n_spots
        f.attrs['created_at'] = datetime.datetime.now().isoformat()
        
        neighbors_group = f.create_group('neighbors')
        distances_group = f.create_group('distances')
        
        for local_idx in range(n_spots):
            global_idx = local_idx + start_idx
            
            if local_idx in all_neighbors:
                spot_neighbors = all_neighbors[local_idx].tolist()
                spot_distances = all_distances[local_idx].tolist()
            else:
                spot_neighbors = []
                spot_distances = []
            
            # Pad to fixed length
            padded_neighbors = spot_neighbors + [-1] * (max_neighbors - len(spot_neighbors))
            padded_distances = spot_distances + [0.0] * (max_neighbors - len(spot_distances))
            
            # Convert to global indices
            global_neighbors = [n + start_idx if n >= 0 else -1 for n in padded_neighbors]
            
            neighbors_group.create_dataset(
                str(global_idx),
                data=np.array(global_neighbors, dtype=np.int32),
                compression="gzip"
            )
            
            distances_group.create_dataset(
                str(global_idx),
                data=np.array(padded_distances, dtype=np.float32),
                compression="gzip"
            )
    
    logger.info(f"Saved neighbors to {neighbors_file}")


def process_dataset_neighbors(
    databank: SpatialDataBank,
    dataset_idx: int,
    cache_dir: str,
    max_neighbors: int,
    force_rebuild: bool = False,
    use_gpu: bool = True
) -> str:
    """Process spatial neighbors for a dataset."""
    import anndata as ad
    
    dataset_info = databank.metadata["datasets"][dataset_idx]
    dataset_name = dataset_info["name"]
    
    neighbors_dir = Path(cache_dir) / "neighbors"
    neighbors_dir.mkdir(exist_ok=True)
    
    neighbors_file = neighbors_dir / f"neighbors_{dataset_name}_n{max_neighbors}.h5"
    
    if neighbors_file.exists() and not force_rebuild:
        logger.info(f"Neighbors file exists: {neighbors_file}")
        return str(neighbors_file)
    
    logger.info(f"Computing neighbors for {dataset_name}")
    
    # Load data
    adata = ad.read_h5ad(dataset_info["path"])
    
    start_idx = databank.metadata["dataset_indices"][dataset_idx]["start_idx"]
    end_idx = databank.metadata["dataset_indices"][dataset_idx]["end_idx"]
    
    if 'spatial' not in adata.obsm:
        logger.error(f"No spatial coordinates in {dataset_name}")
        return None
    
    spatial_coords = adata.obsm['spatial']
    
    # Compute neighbors
    all_neighbors, all_distances = compute_spatial_neighbors(
        spatial_coords, max_neighbors, use_gpu
    )
    
    # Save
    save_neighbors_to_hdf5(
        str(neighbors_file), dataset_name, max_neighbors,
        start_idx, end_idx, all_neighbors, all_distances
    )
    
    return str(neighbors_file)


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(
        description="CoseNiche Data Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--h5ad_file', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Cache directory')
    parser.add_argument('--max_neighbors', type=int, default=6,
                        help='Maximum number of spatial neighbors')
    parser.add_argument('--force_rebuild', action='store_true',
                        help='Force rebuild cache')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--subset_hvg', type=int, default=2000,
                        help='Number of highly variable genes')
    parser.add_argument('--filter_gene_by_counts', type=int, default=50,
                        help='Minimum counts to keep a gene')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("CoseNiche Data Preprocessing")
    logger.info("=" * 60)
    
    # Create config
    config = CoseNicheConfig(
        max_neighbors=args.max_neighbors,
        subset_hvg=args.subset_hvg,
        filter_gene_by_counts=args.filter_gene_by_counts,
    )
    
    # Initialize databank
    logger.info(f"Processing: {args.h5ad_file}")
    
    databank = SpatialDataBank(
        dataset_paths=[args.h5ad_file],
        cache_dir=args.cache_dir,
        config=config,
        force_rebuild=args.force_rebuild
    )
    
    # Process neighbors
    neighbor_files = {}
    
    for dataset_idx in range(len(databank.metadata["datasets"])):
        neighbors_file = process_dataset_neighbors(
            databank, dataset_idx, args.cache_dir,
            args.max_neighbors, args.force_rebuild,
            use_gpu=not args.no_gpu
        )
        
        if neighbors_file:
            neighbor_files[str(dataset_idx)] = neighbors_file
    
    # Save neighbor index
    neighbors_dir = Path(args.cache_dir) / "neighbors"
    neighbors_dir.mkdir(exist_ok=True)
    
    index_file = neighbors_dir / "neighbors_index.json"
    with open(index_file, 'w') as f:
        json.dump(neighbor_files, f, indent=2)
    
    logger.info(f"Neighbor index saved to {index_file}")
    
    # Cleanup
    databank.cleanup_resources()
    
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Total spots: {databank.metadata['total_spots']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


