"""
CoseNiche Transformer Layers and Modules

This module contains the building blocks for the CoseNiche model:
- SpatialAwareTransformerLayer: Transformer layer with optional spatial cross-attention
- CrossAttentionLayer: Cross-attention layer for decoder
- ConditionedLayerNormHead: FiLM-based conditioning head
- Discriminator: Bilinear discriminator for contrastive learning
"""

import logging
from typing import Optional, List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class AvgReadoutMask(nn.Module):
    """
    Masked average readout with L2 normalization.
    
    Computes weighted average of embeddings based on mask and normalizes output.
    
    Args:
        emb: Embeddings [B, N, H]
        mask: Boolean or float mask [B, N], True/1.0 = valid position
        
    Returns:
        L2-normalized averaged embeddings [B, H]
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != emb.dtype:
            mask_ = mask.to(dtype=emb.dtype)
        else:
            mask_ = mask
        vsum = torch.bmm(mask_.unsqueeze(1), emb).squeeze(1)  # [B, H]
        row_sum = mask_.sum(dim=1, keepdim=True).clamp_min(1e-6)
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class Discriminator(nn.Module):
    """
    Bilinear discriminator for contrastive learning.
    
    Computes bilinear scores between center representation and positive/negative samples.
    
    Args:
        n_h: Hidden dimension
    """
    
    def __init__(self, n_h: int):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        nn.init.xavier_uniform_(self.f_k.weight)
        if self.f_k.bias is not None:
            nn.init.zeros_(self.f_k.bias)

    def forward(
        self,
        c: torch.Tensor,
        h_pl: torch.Tensor,
        h_mi: torch.Tensor,
        s_bias1: Optional[torch.Tensor] = None,
        s_bias2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            c: Center representations [B, H]
            h_pl: Positive samples [B, Np, H]
            h_mi: Negative samples [B, Nn, H]
            s_bias1: Optional bias for positive scores
            s_bias2: Optional bias for negative scores
            
        Returns:
            Logits [B, Np+Nn]
        """
        if h_pl.numel() == 0 and h_mi.numel() == 0:
            return c.new_zeros(c.size(0), 0)
        
        logits_list = []
        
        if h_pl.numel() > 0:
            c_pl = c.unsqueeze(1).expand(-1, h_pl.size(1), -1)
            sc_1 = self.f_k(h_pl, c_pl).squeeze(-1)
            if s_bias1 is not None:
                sc_1 = sc_1 + s_bias1
            logits_list.append(sc_1)
            
        if h_mi.numel() > 0:
            c_mi = c.unsqueeze(1).expand(-1, h_mi.size(1), -1)
            sc_2 = self.f_k(h_mi, c_mi).squeeze(-1)
            if s_bias2 is not None:
                sc_2 = sc_2 + s_bias2
            logits_list.append(sc_2)
            
        logits = torch.cat(logits_list, dim=1) if logits_list else c.new_zeros(c.size(0), 0)
        return logits


class ConditionedLayerNormHead(nn.Module):
    """
    FiLM-based conditioning head for platform and organ adaptation.
    
    Maps platform and organ embeddings to layer-wise (gamma, beta) parameters
    for feature-wise linear modulation (FiLM).
    
    Args:
        d_model: Model dimension
        plat_emb_dim: Platform embedding dimension
        organ_emb_dim: Organ embedding dimension
        hidden: Hidden layer dimension (0 for direct projection)
        dropout: Dropout rate
        num_platforms: Number of platforms
        num_organs: Number of organs
    """
    
    def __init__(
        self,
        d_model: int,
        plat_emb_dim: int = 64,
        organ_emb_dim: int = 64,
        hidden: int = 0,
        dropout: float = 0.0,
        num_platforms: int = 5,
        num_organs: int = 43
    ):
        super().__init__()
        self.d_model = d_model
        
        self.plat_emb = nn.Embedding(num_platforms, plat_emb_dim)
        self.organ_emb = nn.Embedding(num_organs, organ_emb_dim)
        
        in_dim = plat_emb_dim + organ_emb_dim
        layers = []
        if hidden and hidden > 0:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2 * d_model)
            ]
        else:
            layers += [nn.Linear(in_dim, 2 * d_model)]
        self.head = nn.Sequential(*layers)

        # Initialize for stable training
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        platform_ids: torch.Tensor,
        organ_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            platform_ids: Platform IDs [B]
            organ_ids: Organ IDs [B]
            
        Returns:
            gamma: Scale parameters [B, 1, D]
            beta: Shift parameters [B, 1, D]
        """
        p = self.plat_emb(platform_ids)  # [B, plat_emb_dim]
        o = self.organ_emb(organ_ids)    # [B, organ_emb_dim]
        h = torch.cat([p, o], dim=-1)    # [B, plat_emb_dim + organ_emb_dim]
        gb = self.head(h)                # [B, 2*d_model]
        
        D = self.d_model
        gamma, beta = gb[..., :D], gb[..., D:]
        
        # Initialize gamma close to 1, beta close to 0 for stability
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        
        return gamma.unsqueeze(1), beta.unsqueeze(1)


class CrossAttentionLayer_Spatial(nn.Module):
    """
    Spatial cross-attention layer.
    
    Lightweight cross-attention for spatial neighbor integration.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        center_cond_gamma_beta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: Query tensor [B, Lq, D]
            key_value: Key/Value tensor [B, Lkv, D]
            key_padding_mask: Padding mask [B, Lkv], True = padding
            center_cond_gamma_beta: Optional FiLM parameters (gamma, beta)
            return_attention: Whether to return attention weights
        """
        q = self.q_ln(query)
        kv = self.kv_ln(key_value)
        
        if center_cond_gamma_beta is not None:
            gamma, beta = center_cond_gamma_beta
            q = q * gamma + beta

        if return_attention:
            attn_output, attn_weights = self.multihead_attn(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask,
                average_attn_weights=True
            )
            return attn_output, attn_weights
        else:
            attn_output, _ = self.multihead_attn(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask
            )
            return attn_output


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer with feedforward network.
    
    Full cross-attention layer with pre-norm, residual connections, and FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        all_cond_gamma_beta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: Query tensor [B, Lq, D]
            key_value: Key/Value tensor [B, Lkv, D]
            key_padding_mask: Padding mask [B, Lkv], True = padding
            all_cond_gamma_beta: Optional FiLM parameters (gamma, beta)
            return_attention: Whether to return attention weights
        """
        q = self.norm1(query)
        if all_cond_gamma_beta is not None:
            gamma, beta = all_cond_gamma_beta
            q = q * gamma + beta

        if return_attention:
            attn_output, attn_weights = self.multihead_attn(
                query=q,
                key=key_value,
                value=key_value,
                key_padding_mask=key_padding_mask,
                average_attn_weights=True
            )
        else:
            attn_output, _ = self.multihead_attn(
                query=q,
                key=key_value,
                value=key_value,
                key_padding_mask=key_padding_mask
            )

        x = query + self.dropout1(attn_output)

        y = self.norm2(x)
        if all_cond_gamma_beta is not None:
            gamma, beta = all_cond_gamma_beta
            y = y * gamma + beta

        y = self.linear2(self.dropout(self.activation(self.linear1(y))))
        out = x + self.dropout2(y)

        if return_attention:
            return out, attn_weights
        return out


class CrossAttentionTransformer(nn.Module):
    """Stack of cross-attention layers."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, key_value, key_padding_mask)
        return query


class SpatialAwareTransformerLayer(nn.Module):
    """
    Transformer layer with optional spatial cross-attention.
    
    This is the core encoder layer that combines:
    1. Self-attention over gene tokens
    2. Optional spatial cross-attention over neighbor spots
    3. Feedforward network
    
    All components support FiLM conditioning.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        enable_spatial_attention: bool = False
    ):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.enable_spatial_attention = enable_spatial_attention
        if enable_spatial_attention:
            self.spatial_cross_attention = CrossAttentionLayer_Spatial(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        spatial_data: Optional[Dict] = None,
        all_cond_gamma_beta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        center_cond_gamma_beta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Args:
            src: Input tensor [B, L, D]
            src_key_padding_mask: Padding mask [B, L], True = padding
            spatial_data: Spatial neighbor information
            all_cond_gamma_beta: FiLM parameters for all spots
            center_cond_gamma_beta: FiLM parameters for center spots
            return_attention: Whether to return attention scores
        """
        # 1) Pre-norm + optional FiLM conditioning for self-attention
        x = self.norm1(src)
        if all_cond_gamma_beta is not None:
            gamma, beta = all_cond_gamma_beta
            x = x * gamma + beta

        # Get center indices if available
        center_idx = spatial_data.get('center_indices', None) if spatial_data else None
        attention_scores = {}
        
        if return_attention:
            attn_out, self_attn_weights = self.self_attention(
                x, x, x,
                key_padding_mask=src_key_padding_mask,
                average_attn_weights=True
            )
            if center_idx is not None:
                attention_scores['center_self_attention'] = self_attn_weights[center_idx]
        else:
            attn_out, _ = self.self_attention(
                x, x, x,
                key_padding_mask=src_key_padding_mask,
                average_attn_weights=True
            )

        src = src + self.dropout1(attn_out)

        # 2) Spatial cross-attention (if enabled)
        if self.enable_spatial_attention and spatial_data is not None:
            if return_attention:
                spatial_residual, spatial_attn_scores = self._apply_spatial_attention(
                    src, spatial_data, center_cond_gamma_beta, return_attention=True
                )
                attention_scores['spatial_cross_attention'] = spatial_attn_scores
            else:
                spatial_residual = self._apply_spatial_attention(
                    src, spatial_data, center_cond_gamma_beta
                )
            src = src + self.dropout2(spatial_residual)

        # 3) FFN with pre-norm + optional FiLM conditioning
        y = self.norm2(src)
        if all_cond_gamma_beta is not None:
            gamma, beta = all_cond_gamma_beta
            y = y * gamma + beta

        y = self.linear2(self.dropout(self.activation(self.linear1(y))))
        src = src + self.dropout3(y)

        if return_attention:
            return src, attention_scores
        return src

    @torch.no_grad()
    def _get_lengths_tensor(
        self,
        batch_lengths: List[int],
        device: torch.device,
        dtype: torch.dtype = torch.long
    ) -> Tensor:
        return torch.tensor(batch_lengths, device=device, dtype=dtype)

    def _normalize_neighbors(
        self,
        batch_to_spots_map: Optional[List[Tuple[int, int]]],
        center_indices: Tensor,
        num_spots: int
    ) -> List[List[int]]:
        """Convert neighbor mapping to per-center neighbor lists."""
        if batch_to_spots_map is None:
            return [[] for _ in range(center_indices.numel())]

        neighbors_per_center: List[List[int]] = [[] for _ in range(center_indices.numel())]

        if isinstance(batch_to_spots_map, list) and len(batch_to_spots_map) > 0:
            if isinstance(batch_to_spots_map[0], tuple):
                for i, (start, end) in enumerate(batch_to_spots_map):
                    start = max(0, int(start))
                    end = min(num_spots, int(end))
                    if end > start:
                        neighbors_per_center[i] = list(range(start, end))

        return neighbors_per_center

    def _apply_spatial_attention(
        self,
        src: torch.Tensor,
        spatial_data: Dict,
        center_cond_gamma_beta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict]]]:
        """Apply spatial cross-attention to aggregate neighbor information."""
        device = src.device
        dtype = src.dtype
        B, Lmax, D = src.shape

        center_indices = spatial_data.get('center_indices')
        batch_to_spots_map = spatial_data.get('batch_to_spots_map')
        batch_lengths = spatial_data.get('batch_lengths')
        genes_ctx = spatial_data.get('genes_ctx')

        residual = torch.zeros_like(src)

        if center_indices is None or batch_to_spots_map is None or batch_lengths is None:
            return (residual, None) if return_attention else residual

        if not torch.is_tensor(center_indices):
            center_indices = torch.tensor(center_indices, device=device, dtype=torch.long)
        else:
            center_indices = center_indices.to(device=device, dtype=torch.long)

        if center_indices.numel() == 0:
            return (residual, None) if return_attention else residual

        neighbors_per_center = self._normalize_neighbors(
            batch_to_spots_map, center_indices, B
        )

        lengths = self._get_lengths_tensor(batch_lengths, device=device)
        lengths = torch.clamp(lengths, min=0, max=Lmax)

        raw_neighbor_max_len = max(
            (sum(int(lengths[n_idx].item()) for n_idx in neighs) for neighs in neighbors_per_center),
            default=0
        )
        if raw_neighbor_max_len == 0:
            return (residual, None) if return_attention else residual

        neighbor_max_len = min(raw_neighbor_max_len, 4000)
        C = len(center_indices)
        q_max = int(lengths[center_indices].max().item()) if C > 0 else 0
        
        if q_max == 0:
            return (residual, None) if return_attention else residual

        # Prepare batched tensors
        query_batch = torch.zeros((C, q_max, D), device=device, dtype=dtype)
        kv_batch = torch.zeros((C, neighbor_max_len, D), device=device, dtype=dtype)
        kv_pad_mask = torch.ones((C, neighbor_max_len), device=device, dtype=torch.bool)

        kv_gene_ids_batch = None
        query_gene_ids_batch = None
        if return_attention and genes_ctx is not None:
            kv_gene_ids_batch = torch.full((C, neighbor_max_len), -1, device=device, dtype=torch.long)
            query_gene_ids_batch = torch.full((C, q_max), -1, device=device, dtype=torch.long)

        kv_source_slices = []
        q_valid_lengths = torch.zeros(C, dtype=torch.long, device=device)

        for i, c_idx in enumerate(center_indices.tolist()):
            q_len = int(lengths[c_idx].item())
            q_valid_lengths[i] = q_len
            
            if q_len > 0:
                query_batch[i, :q_len] = src[c_idx, :q_len]
                if query_gene_ids_batch is not None and genes_ctx is not None:
                    query_gene_ids_batch[i, :q_len] = genes_ctx[c_idx, :q_len]

            slices_i = []
            write_ptr = 0
            for n_idx in neighbors_per_center[i]:
                k_len = int(lengths[n_idx].item())
                if k_len > 0 and write_ptr < neighbor_max_len:
                    take = min(k_len, neighbor_max_len - write_ptr)
                    kv_batch[i, write_ptr:write_ptr+take] = src[n_idx, :take]
                    kv_pad_mask[i, write_ptr:write_ptr+take] = False
                    slices_i.append({
                        'spot_idx': n_idx,
                        'start': write_ptr,
                        'end': write_ptr + take,
                        'is_center': False
                    })
                    if kv_gene_ids_batch is not None:
                        kv_gene_ids_batch[i, write_ptr:write_ptr+take] = genes_ctx[n_idx, :take]
                    write_ptr += take
            kv_source_slices.append(slices_i)

        # Compute spatial attention
        spatial_result = self.spatial_cross_attention(
            query=query_batch,
            key_value=kv_batch,
            key_padding_mask=kv_pad_mask,
            center_cond_gamma_beta=center_cond_gamma_beta,
            return_attention=return_attention
        )
        
        if return_attention:
            spatial_out, spatial_attn_weights = spatial_result
        else:
            spatial_out = spatial_result
            spatial_attn_weights = None

        # Write back to residual
        for i, c_idx in enumerate(center_indices.tolist()):
            q_len = int(lengths[c_idx].item())
            if q_len > 0:
                residual[c_idx, :q_len] = spatial_out[i, :q_len]

        if return_attention:
            packaged = {
                'weights': spatial_attn_weights,
                'kv_source_slices': kv_source_slices,
                'q_valid_lengths': q_valid_lengths,
                'neighbor_max_len': neighbor_max_len,
                'q_max': q_max,
                'batch_to_spots_map': batch_to_spots_map
            }
            if kv_gene_ids_batch is not None:
                packaged['kv_gene_ids'] = kv_gene_ids_batch
            if query_gene_ids_batch is not None:
                packaged['query_gene_ids'] = query_gene_ids_batch
            
            # Add auxiliary alignment fields
            for key in ['batch_local_to_global', 'centers_global_indices', 'neighbors_local_rows_per_center']:
                if key in spatial_data:
                    packaged[key] = spatial_data[key]
            
            return residual, packaged
        
        return residual


