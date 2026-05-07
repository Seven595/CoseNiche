#!/usr/bin/env python
"""
CoseNiche Embedding Extraction Script

Extract spot embeddings from pretrained CoseNiche model.

Usage:
    python extract_embeddings.py \
        --model_path model.safetensors \
        --h5ad_path data.h5ad \
        --cache_dir ./cache \
        --output_dir ./embeddings \
        --device cuda:0
"""

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coseniche import CoseNicheConfig, CoseNicheModel
from coseniche.data import SpatialDataBank, MemoryEfficientSpotDataset, optimized_collate_fn
from coseniche.utils import set_seed, setup_logger, ensure_tensor, to_numpy_pack

logger = logging.getLogger('coseniche.extract')


def load_model(
    model_path: str,
    config: CoseNicheConfig,
    device: torch.device,
    gene_vocab_path: Optional[str] = None,
    metadata_path: Optional[str] = None
) -> CoseNicheModel:
    """
    Load pretrained model.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Target device
        gene_vocab_path: Optional path to gene vocabulary
        metadata_path: Optional path to metadata JSON
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_path}")
    
    # Load checkpoint
    if model_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    
    # Load metadata if available
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update config from metadata
        for field in ['num_platforms', 'num_organs', 'num_diseases']:
            if field in metadata:
                setattr(config, field, metadata[field])
                logger.info(f"Set {field} = {metadata[field]}")
    
    # Infer config from model weights
    for key, value in state_dict.items():
        clean_key = key.replace('module.', '')
        
        if 'platform_embedding.weight' in clean_key or 'plat_emb.weight' in clean_key:
            config.num_platforms = value.shape[0]
            logger.info(f"Inferred num_platforms = {value.shape[0]}")
        
        if 'organ_embedding.weight' in clean_key or 'organ_emb.weight' in clean_key:
            config.num_organs = value.shape[0]
            logger.info(f"Inferred num_organs = {value.shape[0]}")
    
    # Create model
    model = CoseNicheModel(config)
    
    # Clean up state dict
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '')
        clean_state_dict[clean_key] = v
    
    # Load weights
    model.load_state_dict(clean_state_dict, strict=False)
    logger.info("Model weights loaded")
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_embeddings(
    model: CoseNicheModel,
    data_loader: DataLoader,
    device: torch.device,
    config: CoseNicheConfig
) -> Dict:
    """
    Extract embeddings from all spots.
    
    Args:
        model: CoseNiche model
        data_loader: Data loader
        device: Target device
        config: Configuration
        
    Returns:
        Dictionary with embeddings and other outputs
    """
    logger.info("Starting embedding extraction...")
    
    embeddings = []
    center_indices_list = []
    reconstructed_expr_list = []
    center_latent_list = []
    input_genes_list = []
    
    # For attention scores (optional)
    context_attention_list = []
    decoder_attention_list = []
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for batch_idx, batch in enumerate(data_loader):
            try:
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
                
                flat_data = batch['flat']
                structure_data = batch['structure']
                
                genes = ensure_tensor(flat_data['genes'], device)
                values = ensure_tensor(flat_data['raw_normed_values'], device)
                padding_mask = ensure_tensor(flat_data['padding_attention_mask'], device)
                
                center_indices = structure_data.get('center_indices')
                platform_ids = ensure_tensor(structure_data.get('platform_ids'), device)
                organ_ids = ensure_tensor(structure_data.get('organ_ids'), device)
                batch_to_spots_map = structure_data.get('batch_to_spots_map')
                
                # Forward pass
                outputs = model(
                    genes=genes,
                    input_values=values,
                    padding_attention_mask=padding_mask,
                    center_indices=center_indices,
                    batch_to_global_map=batch_to_spots_map,
                    platform_ids=platform_ids,
                    organ_ids=organ_ids,
                )
                
                # Extract outputs
                center_cls = outputs.get("center_cls")
                if center_cls is not None:
                    embeddings.append(center_cls.cpu().numpy())
                
                reconstructed = outputs.get("reconstructed_expr")
                if reconstructed is not None:
                    reconstructed_expr_list.append(reconstructed.cpu().numpy())
                
                input_genes = outputs.get("input_genes")
                if input_genes is not None:
                    input_genes_list.append(input_genes.cpu().numpy())
                
                # Track indices
                if center_indices is not None:
                    center_indices_list.extend(center_indices)
                
                # Optional: attention scores
                ctx_attn = outputs.get("context_attention_scores")
                if ctx_attn:
                    context_attention_list.append(to_numpy_pack(ctx_attn))
                
                # Cleanup
                del outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Aggregate results
    results = {}
    
    if embeddings:
        results['embeddings'] = np.vstack(embeddings)
        logger.info(f"Extracted embeddings: {results['embeddings'].shape}")
    
    if reconstructed_expr_list:
        results['reconstructed_expr'] = np.vstack(reconstructed_expr_list)
    
    if input_genes_list:
        results['input_genes'] = np.vstack(input_genes_list)
    
    if center_indices_list:
        results['center_indices'] = center_indices_list
    
    if context_attention_list:
        results['context_attention_scores'] = context_attention_list
    
    if decoder_attention_list:
        results['decoder_attention_scores'] = decoder_attention_list
    
    return results


def preprocess_single_h5ad(
    h5ad_path: str,
    cache_dir: str,
    config: CoseNicheConfig,
    force_rebuild: bool = False,
    max_neighbors: int = 6
) -> SpatialDataBank:
    """
    Preprocess a single h5ad file.
    
    Args:
        h5ad_path: Path to h5ad file
        cache_dir: Cache directory
        config: Configuration
        force_rebuild: Whether to force rebuild
        max_neighbors: Maximum neighbors
        
    Returns:
        SpatialDataBank instance
    """
    logger.info(f"Preprocessing: {h5ad_path}")
    
    databank = SpatialDataBank(
        dataset_paths=[h5ad_path],
        cache_dir=cache_dir,
        config=config,
        force_rebuild=force_rebuild
    )
    
    return databank


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(
        description="CoseNiche Embedding Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Cache directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for embeddings')
    
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to gene vocabulary')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata JSON')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (use small values to avoid OOM)')
    parser.add_argument('--max_neighbors', type=int, default=6,
                        help='Maximum spatial neighbors')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force_rebuild', action='store_true',
                        help='Force rebuild cache')
    
    args = parser.parse_args()
    
    # Setup
    setup_logger('coseniche.extract')
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config
    config = CoseNicheConfig(
        inference=True,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        vocab_file=args.vocab_path,
    )
    
    # Preprocess data
    databank = preprocess_single_h5ad(
        args.h5ad_path,
        args.cache_dir,
        config,
        args.force_rebuild,
        args.max_neighbors
    )
    
    # Load model
    model = load_model(
        args.model_path,
        config,
        device,
        args.vocab_path,
        args.metadata_path
    )
    
    # Create data loader
    all_indices = list(range(databank.metadata['total_spots']))
    
    dataset = MemoryEfficientSpotDataset(
        indices=all_indices,
        databank=databank,
        max_seq_len=config.max_seq_len,
        mask_ratio=0,  # No masking for inference
        is_training=False
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=optimized_collate_fn
    )
    
    # Extract embeddings
    results = extract_embeddings(model, data_loader, device, config)
    
    # Save results
    if 'embeddings' in results:
        np.save(
            os.path.join(args.output_dir, 'embeddings.npy'),
            results['embeddings']
        )
        logger.info(f"Saved embeddings: {results['embeddings'].shape}")
    
    if 'reconstructed_expr' in results:
        np.save(
            os.path.join(args.output_dir, 'reconstructed_expr.npy'),
            results['reconstructed_expr']
        )
    
    if 'input_genes' in results:
        np.save(
            os.path.join(args.output_dir, 'input_genes.npy'),
            results['input_genes']
        )
    
    if 'context_attention_scores' in results:
        with open(os.path.join(args.output_dir, 'context_attention.pkl'), 'wb') as f:
            pickle.dump(results['context_attention_scores'], f)
    
    # Cleanup
    databank.cleanup_resources()
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Extraction complete!")


if __name__ == "__main__":
    main()


