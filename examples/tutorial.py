"""
CoseNiche Tutorial - Embedding Extraction Example

This script demonstrates how to use CoseNiche to extract spot embeddings
from spatial transcriptomics data.

Example usage:
    python tutorial.py --h5ad_path /path/to/data.h5ad \
                       --model_path /path/to/model.safetensors \
                       --output_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap

from coseniche import CoseNicheConfig, CoseNicheModel
from coseniche.data import SpatialDataBank
from coseniche.utils import set_seed, setup_logger


def main():
    parser = argparse.ArgumentParser(description='CoseNiche Tutorial')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--output_dir', type=str, default='./tutorial_output',
                        help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=7,
                        help='Number of clusters for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger('tutorial')
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # =========================================================================
    # Step 1: Load and explore data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading data")
    logger.info("=" * 60)
    
    adata = sc.read_h5ad(args.h5ad_path)
    logger.info(f"Loaded data: {adata.shape[0]} spots, {adata.shape[1]} genes")
    
    if 'spatial' in adata.obsm:
        logger.info(f"Spatial coordinates: {adata.obsm['spatial'].shape}")
    
    # Check for ground truth labels
    ground_truth_key = None
    for key in ['cell_type', 'cluster', 'annotation', 'ground_truth']:
        if key in adata.obs.columns:
            ground_truth_key = key
            n_classes = adata.obs[key].nunique()
            logger.info(f"Found ground truth: '{key}' with {n_classes} classes")
            break
    
    # =========================================================================
    # Step 2: Create CoseNiche configuration
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Setting up CoseNiche")
    logger.info("=" * 60)
    
    config = CoseNicheConfig(
        d_model=512,
        num_heads=8,
        gene_encoder_layers=6,
        decoder_layers=4,
        max_seq_len=1700,
        max_neighbors=6,
        inference=True,
    )
    
    # =========================================================================
    # Step 3: Preprocess data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Preprocessing data")
    logger.info("=" * 60)
    
    cache_dir = os.path.join(args.output_dir, 'cache')
    
    databank = SpatialDataBank(
        dataset_paths=[args.h5ad_path],
        cache_dir=cache_dir,
        config=config,
        force_rebuild=False
    )
    
    logger.info(f"Preprocessed {databank.metadata['total_spots']} spots")
    
    # =========================================================================
    # Step 4: Load model and extract embeddings
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Extracting embeddings")
    logger.info("=" * 60)
    
    if args.model_path and os.path.exists(args.model_path):
        # Load pretrained model
        model = CoseNicheModel(config)
        model.load_pretrained(args.model_path)
        model = model.to(device)
        model.eval()
        
        logger.info("Loaded pretrained model")
        
        # Extract embeddings
        from torch.utils.data import DataLoader
        from coseniche.data import MemoryEfficientSpotDataset, optimized_collate_fn
        from coseniche.utils import ensure_tensor
        
        all_indices = list(range(databank.metadata['total_spots']))
        
        dataset = MemoryEfficientSpotDataset(
            indices=all_indices,
            databank=databank,
            max_seq_len=config.max_seq_len,
            mask_ratio=0,
            is_training=False
        )
        
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False,
            num_workers=0, collate_fn=optimized_collate_fn
        )
        
        embeddings_list = []
        
        with torch.no_grad():
            for batch in loader:
                flat = batch['flat']
                struct = batch['structure']
                
                genes = ensure_tensor(flat['genes'], device)
                values = ensure_tensor(flat['raw_normed_values'], device)
                mask = ensure_tensor(flat['padding_attention_mask'], device)
                platform_ids = ensure_tensor(struct.get('platform_ids'), device)
                organ_ids = ensure_tensor(struct.get('organ_ids'), device)
                
                outputs = model(
                    genes=genes,
                    input_values=values,
                    padding_attention_mask=mask,
                    center_indices=struct.get('center_indices'),
                    batch_to_global_map=struct.get('batch_to_spots_map'),
                    platform_ids=platform_ids,
                    organ_ids=organ_ids,
                )
                
                embeddings_list.append(outputs['center_cls'].cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
        logger.info(f"Extracted embeddings: {embeddings.shape}")
        
    else:
        # Use PCA on expression data as fallback
        logger.info("No model provided, using PCA on expression data")
        
        from sklearn.decomposition import PCA
        
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(X)
        logger.info(f"PCA embeddings: {embeddings.shape}")
    
    # =========================================================================
    # Step 5: Clustering and evaluation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Clustering analysis")
    logger.info("=" * 60)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Add to adata
    adata.obs['coseniche_cluster'] = cluster_labels.astype(str)
    adata.obsm['X_coseniche'] = embeddings
    
    # Evaluate if ground truth available
    if ground_truth_key:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_labels = le.fit_transform(adata.obs[ground_truth_key])
        
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        
        logger.info(f"Clustering metrics:")
        logger.info(f"  ARI: {ari:.4f}")
        logger.info(f"  NMI: {nmi:.4f}")
    
    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Visualization")
    logger.info("=" * 60)
    
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    adata.obsm['X_umap_coseniche'] = umap_coords
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # UMAP colored by cluster
    scatter = axes[0].scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=cluster_labels, cmap='tab10', s=5, alpha=0.7
    )
    axes[0].set_title('UMAP - CoseNiche Clusters')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Spatial plot colored by cluster
    if 'spatial' in adata.obsm:
        spatial = adata.obsm['spatial']
        scatter2 = axes[1].scatter(
            spatial[:, 0], spatial[:, 1],
            c=cluster_labels, cmap='tab10', s=5, alpha=0.7
        )
        axes[1].set_title('Spatial - CoseNiche Clusters')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].invert_yaxis()
        plt.colorbar(scatter2, ax=axes[1])
    
    # Ground truth if available
    if ground_truth_key:
        colors = le.transform(adata.obs[ground_truth_key])
        scatter3 = axes[2].scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=colors, cmap='tab20', s=5, alpha=0.7
        )
        axes[2].set_title(f'UMAP - Ground Truth ({ground_truth_key})')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
    
    plt.tight_layout()
    
    fig_path = os.path.join(args.output_dir, 'visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to {fig_path}")
    
    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 7: Saving results")
    logger.info("=" * 60)
    
    # Save embeddings
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    
    # Save updated adata
    adata.write(os.path.join(args.output_dir, 'annotated_data.h5ad'))
    
    # Save metrics
    metrics = {
        'n_spots': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'n_clusters': args.n_clusters,
    }
    if ground_truth_key:
        metrics['ari'] = float(ari)
        metrics['nmi'] = float(nmi)
    
    import json
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Tutorial complete!")
    
    # Cleanup
    databank.cleanup_resources()


if __name__ == "__main__":
    main()


