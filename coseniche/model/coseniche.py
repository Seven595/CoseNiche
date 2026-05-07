"""
CoseNiche: Context-aware Spatial Expression Niche Foundation Model

Main model class implementing the full architecture for spatial transcriptomics analysis.
"""

import os
import logging
from typing import Optional, List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import (
    SpatialAwareTransformerLayer,
    CrossAttentionLayer,
    ConditionedLayerNormHead,
    Discriminator,
    AvgReadoutMask,
)

logger = logging.getLogger(__name__)


class CoseNicheModel(nn.Module):
    """
    CoseNiche: Context-aware Spatial Expression Niche Foundation Model.
    
    A transformer-based model for spatial transcriptomics that incorporates:
    - Gene-level encoding with expression-weighted embeddings
    - Spatial cross-attention for neighborhood context
    - Platform and organ conditioning via FiLM
    - Cross-attention decoder for expression reconstruction
    
    Args:
        config: CoseNicheConfig instance with model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Core dimensions
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.inference = config.inference

        # Spatial attention settings
        self.use_spatial_attention = getattr(config, 'use_spatial_attention', True)
        self.spatial_attention_layers = getattr(config, 'spatial_attention_layers', [2, 5])
        
        # CLS pooling mode
        self.cls_pooling_mode = getattr(config, 'cls_pooling_mode', 'avg')

        # Gene embeddings
        self._init_gene_embeddings(config)

        # Bottleneck reconstruction: CLS -> all genes
        self.num_genes = self.gene_embedding_layer.num_embeddings
        self.cls2genes = nn.Linear(self.d_model, self.num_genes, bias=True)
        nn.init.xavier_uniform_(self.cls2genes.weight)
        if self.cls2genes.bias is not None:
            nn.init.zeros_(self.cls2genes.bias)

        # Gene projection to model dimension
        self.gene_projection = nn.Linear(self.gene_embedding_dim, config.d_model)

        # Contrastive learning components
        self.contrast_proj = nn.Linear(self.d_model, self.d_model)
        nn.init.xavier_uniform_(self.contrast_proj.weight)
        if self.contrast_proj.bias is not None:
            nn.init.zeros_(self.contrast_proj.bias)

        self.readout_act = nn.GELU()
        self.avg_readout = AvgReadoutMask()
        self.disc_D = Discriminator(n_h=self.d_model)
        self.contrast_loss_weight = getattr(config, "contrast_loss_weight", 0.1)

        # Context encoder layers
        encoder_layers = []
        for layer_idx in range(config.gene_encoder_layers):
            enable_spatial = (layer_idx in self.spatial_attention_layers) and self.use_spatial_attention
            encoder_layers.append(
                SpatialAwareTransformerLayer(
                    d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.d_model * 2,
                    dropout=config.dropout,
                    enable_spatial_attention=enable_spatial
                )
            )
        self.context_encoder_layers = nn.ModuleList(encoder_layers)

        # Cross-attention decoder
        self.cross_attention_decoder = nn.ModuleList([
            CrossAttentionLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * 2,
                dropout=config.dropout
            ) for _ in range(config.decoder_layers)
        ])

        # Reconstruction head
        self.norm_before_head = nn.LayerNorm(config.d_model)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )

        # Conditioning head (FiLM)
        self.cond_head = ConditionedLayerNormHead(
            d_model=config.d_model,
            hidden=0,
            dropout=config.dropout,
            num_platforms=getattr(config, 'num_platforms', 5),
            num_organs=getattr(config, 'num_organs', 43)
        )

        self._init_weights()

    def _init_gene_embeddings(self, config):
        """Initialize gene embeddings (pretrained or random)."""
        if hasattr(config, "pretrained_gene_embeddings_path") and config.pretrained_gene_embeddings_path:
            if os.path.exists(config.pretrained_gene_embeddings_path):
                pretrained_weights = torch.load(config.pretrained_gene_embeddings_path, map_location='cpu')
                self.gene_embedding_dim = pretrained_weights.size(1)
                self.gene_embedding_layer = nn.Embedding.from_pretrained(
                    pretrained_weights, freeze=False, padding_idx=0
                )
                logger.info(f"Loaded pretrained gene embeddings: {pretrained_weights.shape}")
                return
        
        # Random initialization fallback
        logger.info("Using randomly initialized gene embeddings")
        self.gene_embedding_dim = self.d_model
        num_genes = 59483  # Default vocabulary size
        self.gene_embedding_layer = nn.Embedding(
            num_embeddings=num_genes,
            embedding_dim=self.gene_embedding_dim,
            padding_idx=0
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_pretrained(self, model_path: str, strict: bool = True):
        """
        Load pretrained model weights.
        
        Args:
            model_path: Path to model checkpoint (.pt, .pth, or .safetensors)
            strict: Whether to strictly enforce state dict matching
        """
        logger.info(f"Loading model from: {model_path}")
        
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
        
        # Clean up DDP prefixes
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace('module.', '')
            clean_state_dict[clean_key] = v
        
        self.load_state_dict(clean_state_dict, strict=strict)
        logger.info("Model weights loaded successfully")

    def forward(
        self,
        genes: torch.Tensor,
        input_values: torch.Tensor,
        padding_attention_mask: torch.Tensor,
        center_indices: Optional[List[int]] = None,
        batch_to_global_map: Optional[List[Tuple[int, int]]] = None,
        platform_ids: Optional[torch.Tensor] = None,
        organ_ids: Optional[torch.Tensor] = None,
        batch_local_to_global: Optional[Dict] = None,
        centers_global_indices: Optional[List[int]] = None,
        neighbors_local_rows_per_center: Optional[List] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            genes: Gene IDs [B, L]
            input_values: Expression values [B, L]
            padding_attention_mask: Valid position mask [B, L], True = valid
            center_indices: Indices of center spots in the batch
            batch_to_global_map: Neighbor mapping (start, end) tuples
            platform_ids: Platform IDs [B]
            organ_ids: Organ IDs [B]
            batch_local_to_global: Local to global index mapping
            centers_global_indices: Global indices of centers
            neighbors_local_rows_per_center: Neighbor rows per center
            
        Returns:
            Dictionary containing:
                - reconstructed_expr: Reconstructed expression values
                - center_cls: CLS embeddings for center spots
                - center_cls_reconstructed: Bottleneck reconstructed expression
                - context_attention_scores: Attention scores (inference only)
                - contrast_loss: Contrastive loss (training only)
        """
        # Check for finetune mode
        finetune_mode = getattr(self.config, 'finetune_mode', False)
        if finetune_mode:
            return self._forward_finetune_mode(
                genes, input_values, padding_attention_mask, center_indices,
                batch_to_global_map, platform_ids, organ_ids,
                batch_local_to_global, centers_global_indices,
                neighbors_local_rows_per_center
            )

        total_spots, seq_len = input_values.shape
        device = input_values.device

        # 1) Gene embedding
        E_proj = self.gene_embedding_layer(genes)  # [B, L, D]

        # 2) Masking strategy
        if self.inference:
            mask_ratio = 0
            context_mask = padding_attention_mask & (input_values > 0)
            query_mask = padding_attention_mask
            mlm_mask = query_mask.clone()
        else:
            mask_ratio = torch.empty(1).uniform_(0.3, 0.4).item()
            mlm_mask = self._create_batch_mask(input_values, padding_attention_mask, mask_ratio)
            context_mask = (~mlm_mask) & padding_attention_mask & (input_values > 0)
            query_mask = mlm_mask & padding_attention_mask

        # Ensure at least one context token per sample
        context_mask = self._ensure_context(context_mask, padding_attention_mask, mask_ratio, total_spots)

        # 3) FiLM conditioning
        all_cond_gamma_beta, center_cond_gamma_beta = self._get_conditioning(
            platform_ids, organ_ids, center_indices, device
        )

        # 4) Extract and encode context
        context_data = self._extract_context_data(E_proj, input_values, context_mask, genes)
        dtype = context_data[0]['embeddings'].dtype

        extra_spatial_meta = {
            "batch_local_to_global": batch_local_to_global,
            "centers_global_indices": centers_global_indices,
            "neighbors_local_rows_per_center": neighbors_local_rows_per_center
        }

        # 5) Encode context
        if self.inference:
            encoded_context, context_attention_scores = self._encode_context(
                context_data, center_indices, batch_to_global_map,
                all_cond_gamma_beta, center_cond_gamma_beta, extra_spatial_meta
            )
        else:
            encoded_context = self._encode_context(
                context_data, center_indices, batch_to_global_map,
                all_cond_gamma_beta, center_cond_gamma_beta, extra_spatial_meta
            )
            context_attention_scores = {}

        # 6) Extract query data
        query_data = self._extract_query_data(E_proj, genes, query_mask)

        # 7) Get context genes for inference
        center_context_genes_per_center = None
        center_query_genes_per_center = None
        if self.inference and center_indices is not None:
            center_context_genes_per_center, center_query_genes_per_center = \
                self._get_center_genes(genes, context_mask, query_mask, center_indices)

        # 8) Decode
        if self.inference:
            reconstructed_query, decoder_attn_pack = self._decode_query(
                query_data, encoded_context, all_cond_gamma_beta, return_attention=True
            )
        else:
            reconstructed_query = self._decode_query(
                query_data, encoded_context, all_cond_gamma_beta, return_attention=False
            )
            decoder_attn_pack = None

        # 9) Reconstruct full output
        reconstructed_values = self._reconstruct_full_output(
            reconstructed_query, query_data, total_spots, seq_len, device, dtype
        )

        # 10) Compute CLS representation
        all_spots_cls = self._compute_center_representation(encoded_context, context_data, device, dtype)

        # 11) Bottleneck reconstruction
        all_genes_pred = self.cls2genes(all_spots_cls)
        gather_pred = all_genes_pred.gather(dim=1, index=genes)

        # 12) Select center outputs
        if center_indices is not None:
            center_reconstructed = reconstructed_values[center_indices]
            center_mask = mlm_mask[center_indices]
            center_target = input_values[center_indices]
            center_cls = all_spots_cls[center_indices]
            center_gather_pred = gather_pred[center_indices]
            center_context_mask = context_mask[center_indices]
        else:
            center_reconstructed = reconstructed_values
            center_mask = mlm_mask
            center_target = input_values
            center_cls = all_spots_cls
            center_gather_pred = gather_pred
            center_context_mask = context_mask

        # 13) Encoded context for inference
        center_encoded_context = None
        if self.inference and center_indices is not None:
            center_encoded_context = [encoded_context[i]["embeddings"] for i in range(len(center_indices))]

        # 14) Contrastive loss
        contrast_loss, _ = self._compute_contrastive_loss(
            all_spots_cls, center_indices, batch_to_global_map, device
        )

        return {
            "reconstructed_expr": center_reconstructed,
            "mlm_mask": center_mask,
            "center_cls": center_cls,
            "center_cls_reconstructed": center_gather_pred,
            "context_mask": center_context_mask,
            "target_value": center_target,
            "center_encoded_context": center_encoded_context,
            "context_attention_scores": context_attention_scores,
            "context_genes": center_context_genes_per_center,
            "query_genes": center_query_genes_per_center,
            "decoder_attention_pack": decoder_attn_pack,
            "contrast_loss": contrast_loss,
            "input_genes": genes[center_indices] if center_indices is not None else genes,
        }

    def _create_batch_mask(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_ratio: float
    ) -> torch.Tensor:
        """Create batch-wise masking for MLM."""
        batch_size, seq_len = input_values.shape
        device = input_values.device

        if mask_ratio == 0:
            return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        batch_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        valid_nonzero = padding_mask & (input_values > 0)
        valid_zero = padding_mask & (input_values == 0)

        for i in range(batch_size):
            nz = torch.where(valid_nonzero[i])[0]
            z = torch.where(valid_zero[i])[0]

            if len(nz) > 1:
                k = max(1, int(len(nz) * mask_ratio))
                k = min(k, len(nz) - 1)
                perm = torch.randperm(len(nz), device=device)
                batch_mask[i, nz[perm[:k]]] = True

                if len(z) > 0:
                    kz = min(len(z), k)
                    zperm = torch.randperm(len(z), device=device)
                    batch_mask[i, z[zperm[:kz]]] = True
            elif len(z) > 0:
                kz = max(1, int(len(z) * mask_ratio))
                if kz > 0 and len(z) > 1:
                    zperm = torch.randperm(len(z), device=device)
                    batch_mask[i, z[zperm[:min(kz, len(z)-1)]]] = True

        return batch_mask

    def _ensure_context(
        self,
        context_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_ratio: float,
        total_spots: int
    ) -> torch.Tensor:
        """Ensure each sample has at least one context token."""
        has_context = context_mask.any(dim=1)
        if has_context.all():
            return context_mask

        for i in range(total_spots):
            if not has_context[i] and padding_mask[i].any():
                valid_pos = torch.where(padding_mask[i])[0][0]
                context_mask[i, valid_pos] = True
        
        return context_mask

    def _get_conditioning(
        self,
        platform_ids: Optional[torch.Tensor],
        organ_ids: Optional[torch.Tensor],
        center_indices: Optional[List[int]],
        device: torch.device
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Get FiLM conditioning parameters."""
        if platform_ids is not None and organ_ids is not None:
            all_gamma, all_beta = self.cond_head(platform_ids.to(device), organ_ids.to(device))
            all_cond_gamma_beta = (all_gamma, all_beta)
            if center_indices is not None:
                center_gamma = all_gamma[center_indices]
                center_beta = all_beta[center_indices]
                center_cond_gamma_beta = (center_gamma, center_beta)
            else:
                center_cond_gamma_beta = None
        else:
            all_cond_gamma_beta = None
            center_cond_gamma_beta = None
        
        return all_cond_gamma_beta, center_cond_gamma_beta

    def _extract_context_data(
        self,
        E_base: torch.Tensor,
        input_values: torch.Tensor,
        context_mask: torch.Tensor,
        genes: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """Extract and weight context embeddings."""
        context_data = []
        for i in range(E_base.shape[0]):
            valid_idx = torch.where(context_mask[i])[0]
            if len(valid_idx) > 0:
                ctx_emb = E_base[i, valid_idx]
                ctx_vals = input_values[i, valid_idx]
                weights = (ctx_vals + 1e-6).unsqueeze(-1)
                weighted_emb = ctx_emb * weights.to(ctx_emb.dtype)

                item = {
                    'embeddings': weighted_emb,
                    'indices': valid_idx,
                    'batch_idx': i,
                    'weights': ctx_vals
                }
                if genes is not None:
                    item['genes'] = genes[i, valid_idx].detach().clone()
                context_data.append(item)
            else:
                item = {
                    'embeddings': torch.zeros(1, E_base.shape[-1], device=E_base.device, dtype=E_base.dtype),
                    'indices': torch.tensor([0], device=E_base.device),
                    'batch_idx': i,
                    'weights': torch.zeros(1, device=E_base.device, dtype=E_base.dtype)
                }
                if genes is not None:
                    item['genes'] = torch.full((1,), -1, device=genes.device, dtype=genes.dtype)
                context_data.append(item)
        return context_data

    def _extract_query_data(
        self,
        E_base: torch.Tensor,
        genes: torch.Tensor,
        query_mask: torch.Tensor
    ) -> List[Dict]:
        """Extract query position data."""
        query_data = []
        for i in range(E_base.shape[0]):
            q_idx = torch.where(query_mask[i])[0]
            if len(q_idx) > 0:
                query_data.append({
                    'embeddings': E_base[i, q_idx],
                    'genes': genes[i, q_idx],
                    'indices': q_idx,
                    'batch_idx': i
                })
            else:
                query_data.append({
                    'embeddings': torch.zeros(0, E_base.shape[-1], device=E_base.device, dtype=E_base.dtype),
                    'genes': torch.zeros(0, device=genes.device, dtype=genes.dtype),
                    'indices': torch.zeros(0, device=E_base.device, dtype=torch.long),
                    'batch_idx': i
                })
        return query_data

    def _get_center_genes(
        self,
        genes: torch.Tensor,
        context_mask: torch.Tensor,
        query_mask: torch.Tensor,
        center_indices: List[int]
    ) -> Tuple[List, List]:
        """Get gene lists per center for context and query."""
        combined_mask_context = torch.zeros_like(context_mask, dtype=torch.bool)
        combined_mask_context[center_indices] = context_mask[center_indices]
        center_context_genes = genes[combined_mask_context]

        combined_mask_query = torch.zeros_like(query_mask, dtype=torch.bool)
        combined_mask_query[center_indices] = query_mask[center_indices]
        center_query_genes = genes[combined_mask_query]

        ctx_lengths = [int(context_mask[ci].sum().item()) for ci in center_indices]
        qry_lengths = [int(query_mask[ci].sum().item()) for ci in center_indices]

        def split_by_lengths(flat_tensor, lengths):
            pieces = []
            start = 0
            for L in lengths:
                if L > 0:
                    pieces.append(flat_tensor[start:start+L])
                else:
                    pieces.append(flat_tensor.new_empty((0,), dtype=flat_tensor.dtype))
                start += L
            return pieces

        return split_by_lengths(center_context_genes, ctx_lengths), split_by_lengths(center_query_genes, qry_lengths)

    def _encode_context(
        self,
        context_data: List[Dict],
        center_indices: Optional[List[int]],
        batch_to_spots_map: Optional[List],
        all_cond_gamma_beta: Optional[Tuple],
        center_cond_gamma_beta: Optional[Tuple],
        extra_spatial_meta: Optional[Dict] = None
    ):
        """Encode context data through transformer layers."""
        encoded_context = []
        all_attention_scores = {}

        # Pack embeddings
        all_embeddings = [d['embeddings'] for d in context_data]
        batch_lengths = [d['embeddings'].shape[0] for d in context_data]
        packed_embeddings = torch.cat(all_embeddings, dim=0)

        # Batch
        max_len = max(batch_lengths)
        B = len(context_data)
        D = packed_embeddings.shape[-1]
        device = packed_embeddings.device
        dtype = packed_embeddings.dtype

        batched_embeddings = torch.zeros(B, max_len, D, device=device, dtype=dtype)
        attention_mask = torch.ones(B, max_len, device=device, dtype=torch.bool)

        start = 0
        for i, L in enumerate(batch_lengths):
            batched_embeddings[i, :L] = packed_embeddings[start:start + L]
            attention_mask[i, :L] = False
            start += L

        # Build spatial data
        spatial_data = None
        if center_indices is not None:
            spatial_data = {
                'center_indices': center_indices,
                'batch_to_spots_map': batch_to_spots_map,
                'batch_lengths': batch_lengths
            }
            
            if self.use_spatial_attention and batch_to_spots_map is not None:
                if any('genes' in d for d in context_data):
                    genes_ctx = torch.full((B, max_len), -1, device=device, dtype=context_data[0]['genes'].dtype)
                    for i, data in enumerate(context_data):
                        g = data.get('genes')
                        if g is not None and g.numel() > 0:
                            genes_ctx[i, :g.numel()] = g
                    spatial_data['genes_ctx'] = genes_ctx

                if extra_spatial_meta:
                    for key in ['batch_local_to_global', 'centers_global_indices', 'neighbors_local_rows_per_center']:
                        if extra_spatial_meta.get(key) is not None:
                            spatial_data[key] = extra_spatial_meta[key]

        # Encode through layers
        x = batched_embeddings
        for layer_idx, layer in enumerate(self.context_encoder_layers):
            if self.inference and layer_idx in [3, 5]:
                x, layer_attention_scores = layer(
                    src=x,
                    src_key_padding_mask=attention_mask,
                    spatial_data=spatial_data,
                    all_cond_gamma_beta=all_cond_gamma_beta,
                    center_cond_gamma_beta=center_cond_gamma_beta,
                    return_attention=True
                )
                all_attention_scores[f"context_encoder_layer_{layer_idx}"] = layer_attention_scores
            else:
                x = layer(
                    src=x,
                    src_key_padding_mask=attention_mask,
                    spatial_data=spatial_data,
                    all_cond_gamma_beta=all_cond_gamma_beta,
                    center_cond_gamma_beta=center_cond_gamma_beta,
                    return_attention=False
                )

        # Unpack
        for i, data in enumerate(context_data):
            L = batch_lengths[i]
            result = {
                'embeddings': x[i, :L],
                'indices': data['indices'],
                'batch_idx': data['batch_idx']
            }
            if 'weights' in data:
                result['weights'] = data['weights']
            encoded_context.append(result)

        if self.inference:
            return encoded_context, all_attention_scores
        return encoded_context

    def _decode_query(
        self,
        query_data: List[Dict],
        encoded_context: List[Dict],
        all_cond_gamma_beta: Optional[Tuple],
        return_attention: bool = False
    ):
        """Decode query positions using cross-attention."""
        reconstructed_query = []
        B = len(query_data)

        if B == 0:
            return reconstructed_query

        Nq_list = [qd['embeddings'].shape[0] for qd in query_data]
        Nc_list = [ec['embeddings'].shape[0] for ec in encoded_context]

        if sum(Nq_list) == 0:
            device = encoded_context[0]['embeddings'].device
            for qd in query_data:
                reconstructed_query.append({
                    'values': torch.zeros(0, device=device),
                    'indices': qd['indices'],
                    'batch_idx': qd['batch_idx']
                })
            return reconstructed_query

        D = query_data[0]['embeddings'].shape[-1]
        device = query_data[0]['embeddings'].device
        dtype = query_data[0]['embeddings'].dtype

        maxNq = max(Nq_list)
        maxNc = max(Nc_list)

        Q = torch.zeros(B, maxNq, D, device=device, dtype=dtype)
        q_pad = torch.ones(B, maxNq, dtype=torch.bool, device=device)
        KV = torch.zeros(B, maxNc, D, device=device, dtype=dtype)
        kv_pad = torch.ones(B, maxNc, dtype=torch.bool, device=device)

        for i in range(B):
            nq, nc = Nq_list[i], Nc_list[i]
            if nq > 0:
                Q[i, :nq] = query_data[i]['embeddings']
                q_pad[i, :nq] = False
            if nc > 0:
                KV[i, :nc] = encoded_context[i]['embeddings']
                kv_pad[i, :nc] = False

        batched_gamma_beta = None
        if all_cond_gamma_beta is not None:
            gamma, beta = all_cond_gamma_beta
            batched_gamma_beta = (gamma.to(device=device, dtype=Q.dtype),
                                  beta.to(device=device, dtype=Q.dtype))

        X = Q
        decoder_attns = [] if return_attention else None

        for layer in self.cross_attention_decoder:
            if return_attention:
                X, attn_w = layer(
                    query=X, key_value=KV, key_padding_mask=kv_pad,
                    all_cond_gamma_beta=batched_gamma_beta, return_attention=True
                )
                decoder_attns.append(attn_w)
            else:
                X = layer(
                    query=X, key_value=KV, key_padding_mask=kv_pad,
                    all_cond_gamma_beta=batched_gamma_beta, return_attention=False
                )

        X = self.norm_before_head(X)
        preds = self.reconstruction_head(X).squeeze(-1)

        for i in range(B):
            nq = Nq_list[i]
            reconstructed_query.append({
                'values': preds[i, :nq] if nq > 0 else torch.zeros(0, device=device, dtype=preds.dtype),
                'indices': query_data[i]['indices'],
                'batch_idx': query_data[i]['batch_idx']
            })

        if return_attention:
            return reconstructed_query, {
                'attentions': decoder_attns,
                'Nq_list': Nq_list,
                'Nc_list': Nc_list
            }
        return reconstructed_query

    def _reconstruct_full_output(
        self,
        reconstructed_query: List[Dict],
        query_data: List[Dict],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Reassemble reconstructed values to original shape."""
        out = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        for recon in reconstructed_query:
            b = recon['batch_idx']
            idx = recon['indices']
            val = recon['values']
            if len(idx) > 0:
                out[b, idx] = val.to(dtype=dtype)
        return out

    def _compute_center_representation(
        self,
        encoded_context: List[Dict],
        context_data: List[Dict],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute CLS representation via pooling."""
        reps = []
        for context in encoded_context:
            emb = context['embeddings']
            if emb.shape[0] > 0:
                if emb.device != device or emb.dtype != dtype:
                    emb = emb.to(device=device, dtype=dtype)
                
                if self.cls_pooling_mode == 'weighted' and 'weights' in context:
                    weights = context['weights'].to(device=device, dtype=dtype)
                    weights = weights + 1e-6
                    weights_sum = weights.sum()
                    if weights_sum > 0:
                        normalized_weights = weights / weights_sum
                        weighted_avg = torch.einsum('n,nd->d', normalized_weights, emb)
                        reps.append(weighted_avg)
                    else:
                        reps.append(emb.mean(dim=0))
                else:
                    reps.append(emb.mean(dim=0))
            else:
                reps.append(torch.zeros(self.d_model, device=device, dtype=dtype))
        return torch.stack(reps, dim=0)

    def _compute_contrastive_loss(
        self,
        all_spots_cls: torch.Tensor,
        center_indices: Optional[List[int]],
        batch_to_global_map: Optional[List],
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute local-global contrastive loss."""
        try:
            num_spots = all_spots_cls.size(0)
            if center_indices is None:
                center_rows = torch.arange(num_spots, device=device, dtype=torch.long)
            else:
                center_rows = torch.as_tensor(center_indices, device=device, dtype=torch.long) \
                    if not torch.is_tensor(center_indices) else center_indices

            B0 = center_rows.numel()
            if B0 == 0:
                return None, None

            z = all_spots_cls
            h_latent = self.contrast_proj(z)
            emb = self.readout_act(h_latent)
            emb = F.normalize(emb, p=2, dim=1)

            if batch_to_global_map is None or len(batch_to_global_map) != B0:
                batch_to_global_map = [(0, 0) for _ in range(B0)]

            all_rows = torch.arange(num_spots, device=device)
            pos_indices_per_center, neg_pool_per_center, Np_list = [], [], []

            for i in range(B0):
                ci = center_rows[i].item()
                s, e = batch_to_global_map[i]
                s = int(max(0, min(s, num_spots)))
                e = int(max(0, min(e, num_spots)))
                neigh_idx = torch.arange(s, e, device=device, dtype=torch.long)
                
                if (neigh_idx == ci).sum() == 0:
                    neigh_idx = torch.cat([neigh_idx, torch.tensor([ci], device=device, dtype=torch.long)])
                neigh_idx = neigh_idx.unique()
                
                pos_indices_per_center.append(neigh_idx)
                Np_list.append(neigh_idx.numel())
                
                neg_pool = all_rows[~torch.isin(all_rows, neigh_idx)]
                if neg_pool.numel() == 0:
                    neg_pool = torch.tensor([ci], device=device)
                neg_pool_per_center.append(neg_pool)

            Np_max = max(Np_list) if Np_list else 1
            Nn_max = Np_max

            H = emb.size(1)
            emb_pos = torch.zeros(B0, Np_max, H, device=device, dtype=emb.dtype)
            emb_neg = torch.zeros(B0, Nn_max, H, device=device, dtype=emb.dtype)
            mask_pos = torch.zeros(B0, Np_max, device=device, dtype=torch.bool)
            mask_neg = torch.ones(B0, Nn_max, device=device, dtype=torch.bool)

            for i in range(B0):
                pos_idx = pos_indices_per_center[i]
                Np_i = pos_idx.numel()
                emb_pos[i, :Np_i] = emb[pos_idx]
                mask_pos[i, :Np_i] = True

                neg_pool = neg_pool_per_center[i]
                if neg_pool.numel() >= Nn_max:
                    sel = neg_pool[torch.randperm(neg_pool.numel(), device=device)[:Nn_max]]
                else:
                    sel = neg_pool[torch.randint(neg_pool.numel(), (Nn_max,), device=device)]
                emb_neg[i, :Nn_max] = emb[sel]

            mpos = mask_pos.to(dtype=emb_pos.dtype)
            denom = mpos.sum(dim=1, keepdim=True).clamp_min(1.0)
            g = (emb_pos * mpos.unsqueeze(-1)).sum(dim=1) / denom
            g = F.normalize(g, p=2, dim=1)

            logits = self.disc_D(g, emb_pos, emb_neg)

            labels = torch.cat([
                torch.ones(B0, Np_max, device=device, dtype=logits.dtype),
                torch.zeros(B0, Nn_max, device=device, dtype=logits.dtype)
            ], dim=1)

            mask_concat = torch.cat([mask_pos, mask_neg], dim=1)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            contrast_loss = (bce * mask_concat.to(bce.dtype)).sum() / mask_concat.sum().clamp_min(1.0)

            return contrast_loss, logits.detach()

        except Exception as e:
            logger.warning(f"Contrastive loss computation failed: {e}")
            return None, None

    def _forward_finetune_mode(
        self,
        genes: torch.Tensor,
        input_values: torch.Tensor,
        padding_attention_mask: torch.Tensor,
        center_indices: Optional[List[int]],
        batch_to_global_map: Optional[List] = None,
        platform_ids: Optional[torch.Tensor] = None,
        organ_ids: Optional[torch.Tensor] = None,
        batch_local_to_global: Optional[Dict] = None,
        centers_global_indices: Optional[List[int]] = None,
        neighbors_local_rows_per_center: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Finetune mode: Only compute CLS reconstruction, skip decoder and contrastive.
        """
        total_spots, seq_len = input_values.shape
        device = input_values.device

        E_proj = self.gene_embedding_layer(genes)
        context_mask = padding_attention_mask & (input_values > 0)
        context_mask = self._ensure_context(context_mask, padding_attention_mask, 0, total_spots)

        all_cond_gamma_beta, center_cond_gamma_beta = self._get_conditioning(
            platform_ids, organ_ids, center_indices, device
        )

        context_data = self._extract_context_data(E_proj, input_values, context_mask, genes)
        dtype = context_data[0]['embeddings'].dtype

        extra_spatial_meta = {
            "batch_local_to_global": batch_local_to_global,
            "centers_global_indices": centers_global_indices,
            "neighbors_local_rows_per_center": neighbors_local_rows_per_center
        }

        if self.inference:
            encoded_context, context_attention_scores = self._encode_context(
                context_data, center_indices, batch_to_global_map,
                all_cond_gamma_beta, center_cond_gamma_beta, extra_spatial_meta
            )
        else:
            encoded_context = self._encode_context(
                context_data, center_indices, batch_to_global_map,
                all_cond_gamma_beta, center_cond_gamma_beta, extra_spatial_meta
            )
            context_attention_scores = {}

        all_spots_cls = self._compute_center_representation(encoded_context, context_data, device, dtype)
        all_genes_pred = self.cls2genes(all_spots_cls)
        gather_pred = all_genes_pred.gather(dim=1, index=genes)

        if center_indices is not None:
            center_cls_reconstructed = gather_pred[center_indices]
            center_target = input_values[center_indices]
            center_context_mask = context_mask[center_indices]
            center_cls = all_spots_cls[center_indices]
        else:
            center_cls_reconstructed = gather_pred
            center_target = input_values
            center_context_mask = context_mask
            center_cls = all_spots_cls

        center_encoded_context = None
        center_context_genes_per_center = None

        if self.inference and center_indices is not None:
            center_encoded_context = [encoded_context[i]["embeddings"] for i in range(len(center_indices))]
            center_context_genes_per_center, _ = self._get_center_genes(
                genes, context_mask, context_mask, center_indices
            )

        return {
            "center_cls_reconstructed": center_cls_reconstructed,
            "target_value": center_target,
            "context_mask": center_context_mask,
            "reconstructed_expr": torch.zeros_like(center_cls_reconstructed),
            "mlm_mask": torch.zeros_like(center_context_mask),
            "contrast_loss": None,
            "center_cls": center_cls,
            "center_encoded_context": center_encoded_context,
            "context_attention_scores": context_attention_scores,
            "context_genes": center_context_genes_per_center,
            "query_genes": None,
            "decoder_attention_pack": None,
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        genes: torch.Tensor,
        input_values: torch.Tensor,
        padding_attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract CLS embeddings for spots.
        
        Convenience method for inference that only returns embeddings.
        
        Args:
            genes: Gene IDs [B, L]
            input_values: Expression values [B, L]
            padding_attention_mask: Valid position mask [B, L]
            
        Returns:
            Spot embeddings [B, d_model]
        """
        self.eval()
        outputs = self.forward(genes, input_values, padding_attention_mask, **kwargs)
        return outputs['center_cls']


