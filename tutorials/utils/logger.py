"""
Logging utilities for CoseNiche tutorials.

Provides consistent logging setup across all tutorial scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logger(name: str = 'coseniche',
                level: Union[str, int] = 'INFO',
                log_file: Optional[Union[str, Path]] = None,
                format_string: Optional[str] = None,
                date_format: str = '%Y-%m-%d %H:%M:%S') -> logging.Logger:
    """
    Set up logger with consistent formatting.
    
    Parameters
    ----------
    name : str, optional
        Logger name (default: 'coseniche')
    level : str or int, optional
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or logging level constant (default: 'INFO')
    log_file : str or Path, optional
        Path to log file. If None, only log to console
    format_string : str, optional
        Custom format string. If None, uses default format
    date_format : str, optional
        Date format string (default: '%Y-%m-%d %H:%M:%S')
        
    Returns
    -------
    logger : logging.Logger
        Configured logger object
        
    Examples
    --------
    >>> logger = setup_logger('deconvolution', level='INFO')
    >>> logger.info("Starting analysis...")
    
    >>> logger = setup_logger('attention', log_file='analysis.log')
    >>> logger.debug("Debug information")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system and environment information.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger object
        
    Examples
    --------
    >>> logger = setup_logger('analysis')
    >>> log_system_info(logger)
    """
    import platform
    import torch
    import numpy as np
    import scanpy as sc
    
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Scanpy: {sc.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA: Not available")
    
    logger.info("=" * 60)


def log_parameters(logger: logging.Logger, params: dict, title: str = "Parameters") -> None:
    """
    Log parameter dictionary in formatted way.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger object
    params : dict
        Parameters to log
    title : str, optional
        Section title (default: "Parameters")
        
    Examples
    --------
    >>> logger = setup_logger('deconvolution')
    >>> params = {'lr': 0.001, 'epochs': 100, 'batch_size': 32}
    >>> log_parameters(logger, params)
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    max_key_len = max(len(str(k)) for k in params.keys()) if params else 0
    
    for key, value in params.items():
        logger.info(f"{str(key):<{max_key_len}} : {value}")
    
    logger.info("=" * 60)


class TqdmToLogger:
    """
    Redirect tqdm output to logger.
    
    Examples
    --------
    >>> logger = setup_logger('analysis')
    >>> tqdm_logger = TqdmToLogger(logger, level=logging.INFO)
    >>> for i in tqdm(range(100), file=tqdm_logger):
    ...     pass
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ''
    
    def write(self, buf: str):
        self.buf = buf.strip('\r\n\t ')
    
    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)


def create_progress_callback(logger: logging.Logger, 
                            total_steps: int,
                            log_interval: int = 10) -> callable:
    """
    Create a progress logging callback.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger object
    total_steps : int
        Total number of steps
    log_interval : int, optional
        Log every N steps (default: 10)
        
    Returns
    -------
    callback : callable
        Progress callback function
        
    Examples
    --------
    >>> logger = setup_logger('training')
    >>> callback = create_progress_callback(logger, total_steps=100)
    >>> for i in range(100):
    ...     # ... training code ...
    ...     callback(i, loss=loss_value)
    """
    def callback(step: int, **metrics):
        if step % log_interval == 0 or step == total_steps - 1:
            progress_pct = (step + 1) / total_steps * 100
            metric_str = ', '.join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"Progress: {progress_pct:.1f}% ({step+1}/{total_steps}) | {metric_str}")
    
    return callback
