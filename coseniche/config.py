"""
Configuration class for CoseNiche model.
"""

import json
from typing import Optional, List
import torch


class CoseNicheConfig:
    """
    Configuration for the CoseNiche model.
    
    This class contains all hyperparameters and settings for the model architecture,
    training, and inference.
    
    Attributes:
        d_model (int): Model embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        gene_encoder_layers (int): Number of context encoder layers
        decoder_layers (int): Number of cross-attention decoder layers
        max_seq_len (int): Maximum sequence length (number of genes per spot)
        max_neighbors (int): Maximum number of spatial neighbors
        mask_ratio (float): Masking ratio for training
        inference (bool): Whether in inference mode
    """
    
    def __init__(
        self,
        # Model architecture
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        gene_encoder_layers: int = 6,
        decoder_layers: int = 4,
        
        # Vocabulary paths
        vocab_file: Optional[str] = None,
        platform_vocab_path: Optional[str] = None,
        organ_vocab_path: Optional[str] = None,
        pretrained_gene_embeddings_path: Optional[str] = None,
        
        # Model settings
        inference: bool = True,
        finetune_mode: bool = False,
        
        # Sequence settings
        max_seq_len: int = 1700,
        mask_ratio: float = 0.4,
        include_zero_gene: bool = True,
        
        # Spatial settings
        max_neighbors: int = 6,
        use_spatial_attention: bool = True,
        spatial_attention_layers: Optional[List[int]] = None,
        distance_threshold: float = 1800.0,
        adjacency_type: str = "value",
        neighbor_hop: int = 2,
        
        # Metadata settings
        num_platforms: int = 5,
        num_organs: int = 43,
        num_species: int = 10,
        
        # Data processing
        input_style: str = "log1p",
        custom_mean_norm: Optional[str] = None,
        cell_emb_style: str = "cls",
        
        # Special tokens
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        special_tokens: Optional[List[str]] = None,
        mask_value: int = -1,
        pad_value: int = -2,
        
        # Preprocessing
        filter_gene_by_counts: int = 50,
        filter_cell_by_genes: int = 10,
        subset_hvg: int = 2000,
        data_is_raw: bool = True,
        
        # Training settings (kept for compatibility)
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-5,
        num_epochs: int = 10,
        accumulation_steps: int = 32,
        precision: str = "fp16",
        
        # Validation and checkpointing
        validation_split: float = 0.05,
        validation_interval: int = 50,
        checkpoint_interval: int = 50,
        
        # Loss weights
        recon_loss_weight: float = 1.0,
        contrast_weight: float = 1.0,
        expr_loss_weight: float = 1.0,
        clip_loss_weight: float = 1.0,
        direct_recon_loss_weight: float = 1.0,
        
        # Device settings
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        
        # Misc
        random_seed: int = 42,
        model_name: str = "CoseNiche",
        **kwargs
    ):
        # Model architecture
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.gene_encoder_layers = gene_encoder_layers
        self.decoder_layers = decoder_layers
        
        # Vocabulary paths
        self.vocab_file = vocab_file
        self.platform_vocab_path = platform_vocab_path
        self.organ_vocab_path = organ_vocab_path
        self.pretrained_gene_embeddings_path = pretrained_gene_embeddings_path
        
        # Model settings
        self.inference = inference
        self.finetune_mode = finetune_mode
        
        # Sequence settings
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.include_zero_gene = include_zero_gene
        
        # Spatial settings
        self.max_neighbors = max_neighbors
        self.use_spatial_attention = use_spatial_attention
        self.spatial_attention_layers = spatial_attention_layers or [2, 5]
        self.distance_threshold = distance_threshold
        self.adjacency_type = adjacency_type
        self.neighbor_hop = neighbor_hop
        
        # Metadata settings
        self.num_platforms = num_platforms
        self.num_organs = num_organs
        self.num_species = num_species
        
        # Data processing
        self.input_style = input_style
        self.custom_mean_norm = custom_mean_norm
        self.cell_emb_style = cell_emb_style
        
        # Special tokens
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.special_tokens = special_tokens or ["[PAD]", "[CLS]", "[EOC]"]
        self.mask_value = mask_value
        self.pad_value = pad_value
        
        # Preprocessing
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_genes = filter_cell_by_genes
        self.subset_hvg = subset_hvg
        self.data_is_raw = data_is_raw
        
        # Training settings
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.precision = precision
        
        # Validation and checkpointing
        self.validation_split = validation_split
        self.validation_interval = validation_interval
        self.checkpoint_interval = checkpoint_interval
        
        # Loss weights
        self.recon_loss_weight = recon_loss_weight
        self.contrast_weight = contrast_weight
        self.expr_loss_weight = expr_loss_weight
        self.clip_loss_weight = clip_loss_weight
        self.direct_recon_loss_weight = direct_recon_loss_weight
        
        # Device settings
        self.device = device
        self.dtype = dtype
        
        # Misc
        self.random_seed = random_seed
        self.model_name = model_name
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        def convert(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif hasattr(obj, '__name__') and not isinstance(obj, type):
                return obj.__name__
            else:
                return str(obj)
        return {k: convert(v) for k, v in self.__dict__.items()}
    
    def to_json_string(self) -> str:
        """Return config as JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json_string())
    
    @classmethod
    def from_json(cls, path: str) -> "CoseNicheConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Handle dtype conversion
        if 'dtype' in config_dict:
            dtype_str = config_dict['dtype']
            if 'float16' in dtype_str:
                config_dict['dtype'] = torch.float16
            elif 'float32' in dtype_str:
                config_dict['dtype'] = torch.float32
            elif 'bfloat16' in dtype_str:
                config_dict['dtype'] = torch.bfloat16
        
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return f"CoseNicheConfig({self.to_json_string()})"


