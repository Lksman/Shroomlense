import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import torch
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import json
import random
import numpy as np

from src.config import Config
####################################################################################
# Disclaimer to prevent self-plagiarism: I (Lukas) have implemented and used this  #
# logger in other TU Projects before. (NLP, KDDM2, ...)                            #
####################################################################################


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color to log level names
    """
    # Define color mappings for each log level
    COLOR_MAPPINGS = {
        "CRITICAL": "\033[1;36m",
        "ERROR": "\033[1;31m",
        "WARNING": "\033[1;33m",
        "INFO": "\033[1;32m",
        "DEBUG": "\033[1;30m"
    }

    def format(self, record: logging.LogRecord) -> str: 
        # Apply color to log level name
        level_name = record.levelname
        if level_name in self.COLOR_MAPPINGS:
            color_code = self.COLOR_MAPPINGS[level_name]
            level_name = f"{color_code}{level_name:<8}\033[0m"
        
        record.levelname = level_name
        return super().format(record)


class CustomLogger(logging.Logger):
    """
    Custom logger class that adds color-coded log levels and a specific log message format
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the logger with configuration from Config class

        Parameters
        ----------
        name : str
            Logger name (usually __name__)
        """
        super().__init__(name, Config.LOG_LEVEL)
        
        # Create logs directory if it doesn't exist
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Add handlers
        self.add_stream_handler()
        self.add_rotating_file_handler()

    def add_stream_handler(self) -> None:
        """
        Add console handler with colored output
        """
        console_handler = logging.StreamHandler()
        color_formatter = ColoredFormatter('[%(asctime)s] [%(levelname)-8s] --- %(message)s (%(filename)s:%(lineno)d)', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(color_formatter)
        self.addHandler(console_handler)

    def add_rotating_file_handler(self, max_bytes: int = 1_000_000, backup_count: int = 1) -> None:
        """
        Add rotating file handlers for each log level
        
        Parameters
        ----------
        max_bytes : int
            Maximum size of each log file
        backup_count : int
            Number of backup log files to keep
        """
        file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)-8s] --- %(message)s (%(filename)s:%(lineno)d)', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Define log levels and their corresponding file names
        log_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        
        # Create a handler for each log level
        for level_name, level in log_levels.items():
            log_file = Config.LOG_DIR / f"{level_name}.log"
            handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            handler.setFormatter(file_formatter)
            handler.setLevel(level)
            self.addHandler(handler)

def get_logger(name: str) -> CustomLogger:
    """
    Get a logger instance for the given name
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    
    Returns
    -------
    CustomLogger
        Configured logger instance
    """
    return CustomLogger(name)

def save_model(model: torch.nn.Module, model_name: str, metrics: dict, save_dir: Path) -> Path:
    """Save model state dict."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save only the model state dict (weights), this prevents us from unsafe loading of the model.
    model_path = save_dir / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    
    metrics_path = save_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return model_path

def plot_class_distribution(data: dict, title: str, xlabel: str, ylabel: str, save_path: Path) -> None:
    """Plot a bar chart of class distribution using seaborn."""
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(range(len(data))), y=list(data.values()))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def set_all_seeds() -> None:
    """Set all seeds to ensure reproducibility."""    
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed_all(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Test the logger
    logger = get_logger(__name__)
    logger.debug('Test debug message')
    logger.info('Test info message')
    logger.warning('Test warning message')
    logger.error('Test error message')
    logger.critical('Test critical message')