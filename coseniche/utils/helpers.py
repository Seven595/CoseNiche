"""
Utility functions for CoseNiche
"""

import os
import logging
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = 'coseniche',
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_tensor(
    data: Union[np.ndarray, torch.Tensor, list],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Convert data to tensor on specified device.
    
    Args:
        data: Input data (numpy array, tensor, or list)
        device: Target device
        dtype: Target dtype
        
    Returns:
        PyTorch tensor
    """
    if data is None:
        return None
    
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    
    if device is not None:
        tensor = tensor.to(device=device)
    
    return tensor


def to_numpy_pack(x: Any) -> Any:
    """
    Recursively convert tensors to numpy arrays.
    
    Useful for serializing model outputs.
    
    Args:
        x: Input data (tensor, dict, list, etc.)
        
    Returns:
        Data with tensors converted to numpy
    """
    if x is None:
        return None
    
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    
    if isinstance(x, dict):
        return {k: to_numpy_pack(v) for k, v in x.items()}
    
    if isinstance(x, (list, tuple)):
        return [to_numpy_pack(item) for item in x]
    
    return x


def monitor_resources(rank: int = 0) -> Dict[str, float]:
    """
    Monitor system resource usage.
    
    Args:
        rank: Process rank (for distributed training)
        
    Returns:
        Dictionary with resource metrics
    """
    import psutil
    
    metrics = {}
    
    # CPU
    process = psutil.Process()
    mem_info = process.memory_info()
    metrics['cpu_memory_mb'] = mem_info.rss / (1024 * 1024)
    metrics['cpu_percent'] = process.cpu_percent()
    
    # GPU
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        metrics['gpu_utilization'] = torch.cuda.utilization(device) if hasattr(torch.cuda, 'utilization') else -1
    
    return metrics


def log_model_size(model: torch.nn.Module, logger: Optional[logging.Logger] = None):
    """
    Log model size statistics.
    
    Args:
        model: PyTorch model
        logger: Optional logger (uses print if None)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    msg = (f"Model size: {total_params:,} total params, "
           f"{trainable_params:,} trainable ({total_mb:.2f} MB)")
    
    if logger:
        logger.info(msg)
    else:
        print(msg)


def disable_ddp_functions():
    """
    Disable DDP functions for single-GPU inference.
    
    Returns:
        Dictionary of original functions to restore later
    """
    import torch.distributed as dist
    
    original_funcs = {}
    
    if hasattr(dist, 'all_reduce'):
        original_funcs['all_reduce'] = dist.all_reduce
        dist.all_reduce = lambda *args, **kwargs: None
    
    if hasattr(dist, 'barrier'):
        original_funcs['barrier'] = dist.barrier
        dist.barrier = lambda *args, **kwargs: None
    
    return original_funcs


def restore_ddp_functions(original_funcs: Dict):
    """
    Restore DDP functions after inference.
    
    Args:
        original_funcs: Dictionary from disable_ddp_functions
    """
    import torch.distributed as dist
    
    for name, func in original_funcs.items():
        setattr(dist, name, func)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid positions
        
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Correlation
    if len(predictions) > 1:
        corr = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
    else:
        corr = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'correlation': float(corr) if not np.isnan(corr) else 0.0,
    }


class EarlyStopping:
    """
    Early stopping handler.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


