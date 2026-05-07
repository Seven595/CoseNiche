#!/usr/bin/env python3
"""
CoseNiche Spatial Deconvolution

This script performs cell type deconvolution on spatial transcriptomics data
using CoseNiche embeddings. It learns a mapping matrix between spatial spots
and single-cell reference data to infer cell type compositions.

The deconvolution pipeline implements a GraphST-inspired architecture that:
1. Constructs a spatial neighborhood graph from tissue coordinates
2. Learns an optimal mapping matrix between spots and reference cells
3. Uses reconstruction loss to match spatial embeddings
4. Applies contrastive loss to preserve spatial relationships
5. Estimates cell type proportions from the learned mapping

Main Features
-------------
- GraphST-style mapping matrix learning with softmax normalization
- Multiple loss functions: MSE, cosine similarity, correlation, combined
- Numerically stable contrastive loss with log-sum-exp trick
- Automatic spatial graph construction from coordinates
- Cell type proportion inference and quality metrics
- Extensive logging and progress tracking

Algorithm Overview
------------------
Given spatial spot embeddings H_s and single-cell embeddings H_c, the method:

1. Spatial Graph: Build k-nearest neighbor graph G from spot coordinates
2. Mapping Matrix: Learn M ∈ R^(n_spots × n_cells) via softmax(logits)
3. Reconstruction: Minimize ||M @ H_c - H_s||^2 (or alternatives)
4. Contrastive: Preserve spatial neighbors via InfoNCE loss
5. Inference: Cell type proportions = M @ one_hot(cell_types)

Usage Examples
--------------
Using configuration file (recommended):
    python 1_deconvolution.py --config config_pdac.yaml

Direct parameter specification:
    python 1_deconvolution.py \\
        --st-data data/spatial.h5ad \\
        --sc-data data/sc_reference.h5ad \\
        --st-embeddings embeddings/spatial.npy \\
        --sc-embeddings embeddings/sc_ref.npy \\
        --output-dir results/deconv \\
        --epochs 2000 \\
        --lr 0.005 \\
        --loss-type combined

References
----------
.. [1] Long, Y., et al. (2023). "Spatially informed clustering, integration, 
   and deconvolution of spatial transcriptomics with GraphST."
   Nature Communications, 14, 1155.

Author: CoseNiche Team
License: MIT
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from tqdm import tqdm
import scanpy as sc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_config,
    save_results,
    ensure_dir,
    setup_logger,
    log_parameters,
    preprocess_celltype_column,
    normalize_pdac_celltype,
    ensure_unique_var_names,
    get_common_genes
)


# =============================================================================
# Utility Functions
# =============================================================================

def to_numpy(X):
    """Convert sparse or dense matrix to numpy array."""
    if issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def l2_normalize_rows(x, eps=1e-12):
    """L2 normalize each row of matrix."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def build_spatial_graph(adata: sc.AnnData, n_neighbors: int = 3) -> np.ndarray:
    """
    Build spatial adjacency graph from spot coordinates using k-nearest neighbors.
    
    Constructs an undirected graph where each spot is connected to its k nearest
    spatial neighbors based on Euclidean distance. The resulting adjacency matrix
    is symmetric and stored in adata.obsm['adj'].
    
    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data. Must contain:
        - adata.obsm['spatial'] : (n_spots, 2) array of (x, y) coordinates
    n_neighbors : int, default=3
        Number of nearest neighbors to connect for each spot.
        Typical values: 3-10 depending on tissue density.
        
    Returns
    -------
    adj : np.ndarray, shape (n_spots, n_spots)
        Symmetric binary adjacency matrix where adj[i,j]=1 if spots i and j
        are neighbors (within k-nearest or vice versa), 0 otherwise.
        Self-loops are excluded (diagonal is 0).
        
    Raises
    ------
    ValueError
        If adata.obsm['spatial'] is missing or has invalid shape.
        
    Notes
    -----
    The function also stores intermediate results in adata.obsm:
    - 'distance_matrix' : Full pairwise distance matrix
    - 'graph_neigh' : Directed k-NN graph (before symmetrization)
    - 'adj' : Final symmetric adjacency matrix
    
    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.visium_sge()
    >>> adj = build_spatial_graph(adata, n_neighbors=6)
    >>> print(f"Graph density: {adj.sum() / adj.size:.4f}")
    """
    if "spatial" not in adata.obsm:
        raise ValueError(
            "Missing adata.obsm['spatial']. Cannot build spatial graph. "
            "Please ensure spatial coordinates are available."
        )
    
    pos = np.asarray(adata.obsm["spatial"])
    n_spots = pos.shape[0]
    
    # Compute distance matrix
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(pos, pos, metric="euclidean")
    adata.obsm['distance_matrix'] = distance_matrix
    
    # Find k nearest neighbors
    interaction = np.zeros((n_spots, n_spots), dtype=np.float32)
    for i in range(n_spots):
        distances = distance_matrix[i, :]
        nearest_indices = np.argsort(distances)
        
        # Skip self (index 0) and take next k neighbors
        for t in range(1, n_neighbors + 1):
            j = nearest_indices[t]
            interaction[i, j] = 1.0
    
    adata.obsm["graph_neigh"] = interaction
    
    # Create symmetric adjacency matrix
    adj = interaction + interaction.T
    adj = np.where(adj > 1, 1.0, adj).astype(np.float32)
    adata.obsm["adj"] = adj
    
    return adj


def get_spatial_graph(adata: sc.AnnData, 
                     neigh_key: str = "graph_neigh",
                     adj_key: str = "adj",
                     n_neighbors: int = 3) -> np.ndarray:
    """
    Get or build spatial adjacency graph.
    
    Parameters
    ----------
    adata : AnnData
        Spatial data
    neigh_key : str
        Key for neighbor matrix in obsm
    adj_key : str
        Key for adjacency matrix in obsm
    n_neighbors : int
        Number of neighbors if building new graph
        
    Returns
    -------
    adj_binary : np.ndarray
        Binary adjacency matrix with self-loops
    """
    # Try to use existing adjacency matrix
    if adj_key in adata.obsm and adata.obsm[adj_key] is not None:
        adj = np.asarray(adata.obsm[adj_key], dtype=np.float32)
    elif neigh_key in adata.obsm and adata.obsm[neigh_key] is not None:
        # Build from neighbor matrix
        inter = np.asarray(adata.obsm[neigh_key], dtype=np.float32)
        adj = inter + inter.T
        adj = np.where(adj > 1, 1.0, adj).astype(np.float32)
        adata.obsm[adj_key] = adj
    else:
        # Build new graph
        adj = build_spatial_graph(adata, n_neighbors=n_neighbors)
    
    # Binarize and add self-loops
    adj_binary = (adj > 0).astype(np.float32)
    np.fill_diagonal(adj_binary, 1.0)
    
    return adj_binary


# =============================================================================
# Deconvolution Model
# =============================================================================

class MappingMatrixEncoder(torch.nn.Module):
    """
    Learnable mapping matrix from spatial spots to single cells.
    
    The mapping matrix M has shape (n_spots, n_cells), where M[i,j]
    represents the contribution of cell j to spot i.
    """
    
    def __init__(self, n_spots: int, n_cells: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((n_spots, n_cells), dtype=torch.float32)
        )
        torch.nn.init.xavier_uniform_(self.weight)
    
    def forward(self):
        return self.weight


def stable_contrastive_loss(pred_sp, emb_sp, graph_neigh, tau=1.0):
    """
    Compute numerically stable contrastive loss with log-sum-exp trick.
    
    Implements an InfoNCE-style contrastive loss that encourages predicted
    embeddings to be similar to spatial neighbors while being dissimilar to
    non-neighbors. Uses the log-sum-exp trick for numerical stability.
    
    The loss for each spot i is:
        L_i = -log( sum_j∈N(i) exp(sim(i,j)/τ) / sum_k≠i exp(sim(i,k)/τ) )
    
    where N(i) are spatial neighbors of spot i, and sim is cosine similarity.
    
    Parameters
    ----------
    pred_sp : torch.Tensor, shape (n_spots, d)
        Predicted spatial embeddings from mapping matrix
    emb_sp : torch.Tensor, shape (n_spots, d)
        True spatial embeddings from CoseNiche model
    graph_neigh : torch.Tensor, shape (n_spots, n_spots)
        Binary spatial adjacency matrix where graph_neigh[i,j]=1 if
        spots i and j are neighbors, 0 otherwise
    tau : float, default=1.0
        Temperature parameter for softmax. Lower values (e.g., 0.1) make
        the model more discriminative; higher values (e.g., 10.0) soften
        the distribution. Typical range: [0.1, 2.0]
        
    Returns
    -------
    loss : torch.Tensor, scalar
        Mean contrastive loss across all spots
        
    Notes
    -----
    - Uses log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    - Automatically excludes self-similarity from denominator
    - Handles edge cases where a spot has no neighbors (loss = 0)
    - All embeddings are L2-normalized before computing similarity
    
    References
    ----------
    .. [1] Oord, A. et al. (2018). "Representation Learning with Contrastive 
       Predictive Coding." arXiv:1807.03748
    """
    # Cosine similarity
    pred_norm = F.normalize(pred_sp, p=2, dim=1)
    emb_norm = F.normalize(emb_sp, p=2, dim=1)
    similarity = torch.matmul(pred_norm, emb_norm.T) / tau  # (n_spots, n_spots)
    
    # Log-sum-exp trick for numerical stability
    sim_max = similarity.max(dim=1, keepdim=True)[0]
    sim_stable = similarity - sim_max
    
    # Denominator: all samples except self
    exp_sim = torch.exp(sim_stable)
    denominator = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_stable))
    
    # Numerator: neighbors
    numerator = (exp_sim * graph_neigh).sum(dim=1)
    
    # Avoid log(0)
    ratio = torch.clamp(numerator / (denominator + 1e-8), min=1e-8, max=1.0)
    loss = -torch.log(ratio).mean()
    
    return loss


def graphst_deconvolution(
    adata: sc.AnnData,
    adata_sc: sc.AnnData,
    st_embeddings: np.ndarray,
    sc_embeddings: np.ndarray,
    n_neighbors: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 1000,
    lam_recon: float = 100.0,
    lam_contrast: float = 1.0,
    tau: float = 1.0,
    loss_type: str = "cosine",
    seed: int = 41,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    log_every: int = 10,
    save_best: bool = True,
    logger = None
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Perform spatial deconvolution using GraphST-inspired embedding-based approach.
    
    Learns an optimal mapping matrix M between spatial spots and single-cell
    reference data by minimizing reconstruction and contrastive losses. The
    mapping matrix is softmax-normalized to ensure valid probabilistic
    interpretation as cell-to-spot assignment probabilities.
    
    Algorithm Steps
    ---------------
    1. Construct spatial k-NN graph from spot coordinates
    2. Initialize learnable mapping matrix M (n_spots × n_cells)
    3. For each epoch:
       a. Compute softmax-normalized mapping: P = softmax(M)
       b. Reconstruct spatial embeddings: H_pred = P @ H_cells
       c. Compute reconstruction loss: L_recon = loss(H_pred, H_spatial)
       d. Compute contrastive loss: L_contrast = InfoNCE(H_pred, H_spatial, graph)
       e. Update M via gradient descent: M ← M - lr * ∇(L_recon + L_contrast)
    4. Extract cell type proportions from final P
    
    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data. Must contain:
        - adata.obsm['spatial'] : (n_spots, 2) spatial coordinates
        - adata.X : (n_spots, n_genes) expression matrix (optional)
    adata_sc : AnnData
        Single-cell reference data. Must contain:
        - adata_sc.obs['cell_type'] : cell type annotations
        - adata_sc.X : (n_cells, n_genes) expression matrix (optional)
    st_embeddings : np.ndarray, shape (n_spots, d)
        Pre-computed spatial embeddings from CoseNiche model.
        Should be L2-normalized or will be normalized internally.
    sc_embeddings : np.ndarray, shape (n_cells, d)
        Pre-computed single-cell embeddings from CoseNiche model.
        Should be L2-normalized or will be normalized internally.
    n_neighbors : int, default=8
        Number of nearest spatial neighbors for graph construction.
        Larger values increase spatial smoothness. Typical range: 3-15.
    lr : float, default=0.001
        Learning rate for AdamW optimizer. Recommended: [0.001, 0.01]
        Decrease if training is unstable (NaN/Inf losses).
    weight_decay : float, default=0.0
        L2 regularization weight for optimizer. Set to small value
        (e.g., 1e-4) if overfitting occurs.
    epochs : int, default=1000
        Number of training epochs. Typical range: 1000-5000.
        Monitor loss curves to determine convergence.
    lam_recon : float, default=100.0
        Weight for reconstruction loss component. Higher values prioritize
        accurate embedding reconstruction. Typical range: 10-500.
    lam_contrast : float, default=1.0
        Weight for contrastive loss component. Higher values enforce
        stronger spatial coherence. Typical range: 0.1-10.
    tau : float, default=1.0
        Temperature parameter for contrastive loss softmax.
        Lower values (0.1-0.5) increase discrimination.
        Higher values (1-2) allow more flexibility.
    loss_type : {'mse', 'cosine', 'correlation', 'combined'}, default='cosine'
        Type of reconstruction loss:
        - 'mse' : Mean squared error (fast, stable)
        - 'cosine' : 1 - cosine similarity (direction-focused)
        - 'correlation' : 1 - Pearson correlation (scale-invariant)
        - 'combined' : Weighted combination (best quality, slower)
    seed : int, default=41
        Random seed for reproducibility (affects initialization)
    device : torch.device, optional
        PyTorch device for computation. If None, automatically selects
        'cuda' if available, otherwise 'cpu'.
    verbose : bool, default=True
        If True, display tqdm progress bar with loss statistics
    log_every : int, default=10
        Logging interval in epochs. Detailed statistics are logged
        every `log_every` epochs.
    save_best : bool, default=True
        If True, restore model to best state (lowest total loss) after
        training completes.
    logger : logging.Logger, optional
        Logger instance for detailed progress messages. If None, only
        prints essential information.
        
    Returns
    -------
    adata : AnnData
        Updated spatial data with new fields in .obsm:
        - 'map_matrix' : (n_spots, n_cells) mapping probabilities
        - 'pred_embeddings' : (n_spots, d) reconstructed embeddings
        - 'cell_type_proportions' : (n_spots, n_types) DataFrame with
          proportion of each cell type at each spot
    adata_sc : AnnData
        Unmodified single-cell reference data
        
    Raises
    ------
    ValueError
        - If embedding dimensions don't match (st and sc must have same d)
        - If number of embeddings doesn't match adata sizes
        - If spatial coordinates are missing
        - If unsupported loss_type is specified
    RuntimeError
        - If loss becomes NaN or Inf during training
        
    Notes
    -----
    - All embeddings are L2-normalized before optimization
    - Gradient clipping (max norm = 1.0) is applied for stability
    - Best model state (lowest loss) is saved and restored if save_best=True
    - Cell type proportions are computed by aggregating mapping probabilities
      for cells of the same type: prop[i, t] = sum_j M[i,j] * (cell_j == t)
    
    Performance Tips
    ----------------
    - Start with loss_type='mse' for fast prototyping
    - Use loss_type='combined' for best results on final runs
    - Increase lam_recon if reconstruction quality is poor
    - Increase lam_contrast if spatial patterns are not smooth
    - Reduce lr if loss oscillates or increases during training
    - Increase epochs if loss is still decreasing at end of training
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import numpy as np
    >>> from coseniche import extract_embeddings
    >>> 
    >>> # Load data
    >>> adata_spatial = sc.read_h5ad("spatial.h5ad")
    >>> adata_sc = sc.read_h5ad("sc_reference.h5ad")
    >>> 
    >>> # Extract embeddings (assuming model is loaded)
    >>> st_emb = extract_embeddings(model, adata_spatial)
    >>> sc_emb = extract_embeddings(model, adata_sc)
    >>> 
    >>> # Run deconvolution
    >>> adata_spatial, adata_sc = graphst_deconvolution(
    ...     adata_spatial, adata_sc,
    ...     st_emb, sc_emb,
    ...     epochs=2000,
    ...     lr=0.005,
    ...     loss_type='combined'
    ... )
    >>> 
    >>> # Access results
    >>> proportions = adata_spatial.obsm['cell_type_proportions']
    >>> print(proportions.head())
    
    References
    ----------
    .. [1] Long, Y., et al. (2023). "Spatially informed clustering, integration, 
       and deconvolution of spatial transcriptomics with GraphST."
       Nature Communications, 14, 1155.
    .. [2] Cable, D. M., et al. (2022). "Robust decomposition of cell type 
       mixtures in spatial transcriptomics." Nature Biotechnology, 40, 517-526.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if logger:
        logger.info(f"Using device: {device}")
        logger.info(f"ST embeddings shape: {st_embeddings.shape}")
        logger.info(f"SC embeddings shape: {sc_embeddings.shape}")
    
    # Validate dimensions
    n_spots, d_st = st_embeddings.shape
    n_cells, d_sc = sc_embeddings.shape
    
    if d_st != d_sc:
        raise ValueError(f"Embedding dimensions don't match: ST={d_st}, SC={d_sc}")
    
    if n_spots != adata.n_obs:
        raise ValueError(f"ST embedding count mismatch: {n_spots} vs {adata.n_obs}")
    
    if n_cells != adata_sc.n_obs:
        raise ValueError(f"SC embedding count mismatch: {n_cells} vs {adata_sc.n_obs}")
    
    # Get spatial graph
    G_np = get_spatial_graph(adata, n_neighbors=n_neighbors)
    if G_np.shape != (n_spots, n_spots):
        raise ValueError(f"Graph shape mismatch: {G_np.shape} vs ({n_spots}, {n_spots})")
    
    # Convert to tensors
    Hs = torch.tensor(st_embeddings, dtype=torch.float32, device=device)
    Hc = torch.tensor(sc_embeddings, dtype=torch.float32, device=device)
    G = torch.tensor(G_np, dtype=torch.float32, device=device)
    
    # L2 normalize embeddings
    Hs = F.normalize(Hs, p=2, eps=1e-12, dim=1)
    Hc = F.normalize(Hc, p=2, eps=1e-12, dim=1)
    
    # Initialize model
    model = MappingMatrixEncoder(n_spots, n_cells).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    iterator = range(1, epochs + 1)
    best_loss = np.inf
    best_state = None
    
    if verbose:
        iterator = tqdm(iterator, desc="Deconvolution")
    
    if logger:
        logger.info("Starting mapping matrix learning...")
    
    for epoch in iterator:
        model.train()
        
        # Forward pass
        M_logits = model()  # (n_spots, n_cells)
        map_probs = F.softmax(M_logits, dim=1)  # Normalize across cells
        
        # Reconstruct spatial embeddings
        pred_sp = torch.matmul(map_probs, Hc)  # (n_spots, d)
        
        # Normalize for loss computation
        pred_sp_norm = F.normalize(pred_sp, p=2, dim=1)
        Hs_norm = F.normalize(Hs, p=2, dim=1)
        
        # Reconstruction loss
        if loss_type == "mse":
            loss_recon = F.mse_loss(pred_sp_norm, Hs_norm)
        elif loss_type == "cosine":
            cos_sim = F.cosine_similarity(pred_sp_norm, Hs_norm, dim=1).mean()
            loss_recon = 1.0 - cos_sim
        elif loss_type == "correlation":
            pred_centered = pred_sp_norm - pred_sp_norm.mean(dim=1, keepdim=True)
            hs_centered = Hs_norm - Hs_norm.mean(dim=1, keepdim=True)
            corr = F.cosine_similarity(pred_centered, hs_centered, dim=1).mean()
            loss_recon = 1.0 - corr
        elif loss_type == "combined":
            mse_loss = F.mse_loss(pred_sp_norm, Hs_norm)
            cos_sim = F.cosine_similarity(pred_sp_norm, Hs_norm, dim=1).mean()
            cos_loss = 1.0 - cos_sim
            pred_centered = pred_sp_norm - pred_sp_norm.mean(dim=1, keepdim=True)
            hs_centered = Hs_norm - Hs_norm.mean(dim=1, keepdim=True)
            corr = F.cosine_similarity(pred_centered, hs_centered, dim=1).mean()
            corr_loss = 1.0 - corr
            loss_recon = 0.5 * mse_loss + 0.3 * cos_loss + 0.2 * corr_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Contrastive loss
        loss_contrast = stable_contrastive_loss(pred_sp_norm, Hs_norm, G, tau=tau)
        
        # Total loss
        loss = lam_recon * loss_recon + lam_contrast * loss_contrast
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            if logger:
                logger.error(f"Epoch {epoch}: Loss is NaN/Inf, stopping training")
            break
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            if save_best:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        
        # Logging
        if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            with torch.no_grad():
                # Compute additional statistics
                logits_mean = M_logits.mean().item()
                logits_std = M_logits.std().item()
                
                # Entropy of mapping probabilities
                eps = 1e-12
                probs_safe = torch.clamp(map_probs, min=eps)
                entropy = -(probs_safe * probs_safe.log()).sum(dim=1).mean().item()
                
                # Gradient norm
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 
                              for p in model.parameters() if p.grad is not None) ** 0.5
            
            if isinstance(iterator, tqdm):
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{loss_recon.item():.4f}",
                    "contrast": f"{loss_contrast.item():.4f}",
                    "entropy": f"{entropy:.3f}",
                    "||g||": f"{grad_norm:.2e}",
                })
            elif logger and epoch % (log_every * 10) == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Recon: {loss_recon.item():.4f} | "
                    f"Contrast: {loss_contrast.item():.4f} | "
                    f"Entropy: {entropy:.3f}"
                )
    
    if logger:
        logger.info("Mapping matrix learning completed!")
    
    # Extract results
    with torch.no_grad():
        if best_state is not None:
            model.load_state_dict(best_state)
        
        model.eval()
        M_logits = model()
        map_probs = F.softmax(M_logits, dim=1)
        map_matrix = map_probs.detach().cpu().numpy()
        pred_sp_final = torch.matmul(map_probs, Hc).detach().cpu().numpy()
    
    # Save to AnnData
    adata.obsm["map_matrix"] = map_matrix
    adata.obsm["pred_embeddings"] = pred_sp_final
    
    # Compute cell type proportions
    if "cell_type" in adata_sc.obs.columns:
        cell_types = adata_sc.obs['cell_type'].values
        unique_types = np.unique(cell_types)
        
        ct_proportions = np.zeros((n_spots, len(unique_types)))
        for i, ct in enumerate(unique_types):
            mask = (cell_types == ct)
            ct_proportions[:, i] = map_matrix[:, mask].sum(axis=1)
        
        # Save as DataFrame
        ct_df = pd.DataFrame(
            ct_proportions,
            index=adata.obs_names,
            columns=unique_types
        )
        adata.obsm['cell_type_proportions'] = ct_df
        
        if logger:
            logger.info("\n" + "=" * 80)
            logger.info("Average cell type proportions:")
            logger.info("-" * 80)
            mean_props = ct_df.mean().sort_values(ascending=False)
            for ct, prop in mean_props.items():
                logger.info(f"{ct:45s}: {prop:.4f} ({prop*100:.2f}%)")
            logger.info("=" * 80)
    
    return adata, adata_sc


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="CoseNiche spatial deconvolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--config', type=str, help='Configuration YAML file')
    parser.add_argument('--st-data', type=str, help='Spatial transcriptomics h5ad file')
    parser.add_argument('--sc-data', type=str, help='Single-cell reference h5ad file')
    parser.add_argument('--st-embeddings', type=str, help='Spatial embeddings .npy file')
    parser.add_argument('--sc-embeddings', type=str, help='Single-cell embeddings .npy file')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    # Model arguments
    parser.add_argument('--n-neighbors', type=int, default=6, help='Number of spatial neighbors')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lam-recon', type=float, default=100.0, help='Reconstruction loss weight')
    parser.add_argument('--lam-contrast', type=float, default=1.0, help='Contrastive loss weight')
    parser.add_argument('--loss-type', type=str, default='mse', 
                       choices=['mse', 'cosine', 'correlation', 'combined'],
                       help='Reconstruction loss type')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for contrastive loss')
    parser.add_argument('--seed', type=int, default=41, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--log-every', type=int, default=10, help='Logging interval')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        
        # Override with config values if not specified in command line
        if not args.st_data and 'st_data' in config:
            args.st_data = config['st_data']
        if not args.sc_data and 'sc_data' in config:
            args.sc_data = config['sc_data']
        if not args.st_embeddings and 'st_embeddings' in config:
            args.st_embeddings = config['st_embeddings']
        if not args.sc_embeddings and 'sc_embeddings' in config:
            args.sc_embeddings = config['sc_embeddings']
        
        # Update other parameters from config
        for key in ['n_neighbors', 'epochs', 'lr', 'lam_recon', 'lam_contrast', 
                   'loss_type', 'tau', 'seed', 'output_dir']:
            if key in config:
                setattr(args, key, config[key])
    
    # Validate required arguments
    if not args.st_data or not args.sc_data:
        raise ValueError("Must provide --st-data and --sc-data (or --config)")
    if not args.st_embeddings or not args.sc_embeddings:
        raise ValueError("Must provide --st-embeddings and --sc-embeddings (or --config)")
    
    # Setup output directory and logger
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger('deconvolution', level='INFO', 
                         log_file=output_dir / 'deconvolution.log')
    
    logger.info("=" * 80)
    logger.info("CoseNiche Spatial Deconvolution")
    logger.info("=" * 80)
    
    # Log parameters
    log_parameters(logger, vars(args), "Configuration")
    
    # Load data
    logger.info("\nLoading spatial transcriptomics data...")
    adata = sc.read_h5ad(args.st_data)
    logger.info(f"  Spots: {adata.n_obs}, Genes: {adata.n_vars}")
    
    logger.info("\nLoading single-cell reference data...")
    adata_sc = sc.read_h5ad(args.sc_data)
    logger.info(f"  Cells: {adata_sc.n_obs}, Genes: {adata_sc.n_vars}")
    
    # Preprocess cell types (if PDAC dataset)
    if 'cell_type' in adata_sc.obs.columns:
        logger.info("\nPreprocessing cell type annotations...")
        adata_sc = preprocess_celltype_column(
            adata_sc, 
            normalize_func=normalize_pdac_celltype,
            verbose=False
        )
        logger.info(f"  Unique cell types: {adata_sc.obs['cell_type'].nunique()}")
    
    # Load embeddings
    logger.info("\nLoading embeddings...")
    st_embeddings = np.load(args.st_embeddings)
    sc_embeddings = np.load(args.sc_embeddings)
    logger.info(f"  ST embeddings: {st_embeddings.shape}")
    logger.info(f"  SC embeddings: {sc_embeddings.shape}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")
    
    # Run deconvolution
    logger.info("\n" + "=" * 80)
    logger.info("Starting deconvolution...")
    logger.info("=" * 80)
    
    adata, adata_sc = graphst_deconvolution(
        adata=adata,
        adata_sc=adata_sc,
        st_embeddings=st_embeddings,
        sc_embeddings=sc_embeddings,
        n_neighbors=args.n_neighbors,
        lr=args.lr,
        epochs=args.epochs,
        lam_recon=args.lam_recon,
        lam_contrast=args.lam_contrast,
        tau=args.tau,
        loss_type=args.loss_type,
        seed=args.seed,
        device=device,
        verbose=True,
        log_every=args.log_every,
        logger=logger
    )
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving results...")
    logger.info("=" * 80)
    
    st_output = output_dir / "deconvolution_result.h5ad"
    sc_output = output_dir / "sc_reference.h5ad"
    
    adata.write_h5ad(st_output)
    adata_sc.write_h5ad(sc_output)
    
    logger.info(f"  Spatial data: {st_output}")
    logger.info(f"  SC reference: {sc_output}")
    
    # Save cell type proportions as CSV
    if 'cell_type_proportions' in adata.obsm:
        ct_csv = output_dir / "cell_type_proportions.csv"
        adata.obsm['cell_type_proportions'].to_csv(ct_csv)
        logger.info(f"  Cell type proportions: {ct_csv}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Deconvolution completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
