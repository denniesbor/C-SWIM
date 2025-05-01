"""
Configuration module for GIC calculations.

Provides utilities for:
- Data directory management
- Custom logger setup
"""

import os
import sys
import logging
from pathlib import Path

# Application name - used for the logger
APP_NAME = "gic_analysis"

# Path settings
DEFAULT_DATA_DIR = Path("__file__").resolve().parent / "data"


def get_data_dir(subdir=None):
    """
    Get the path to a data directory, creating it if it doesn't exist.

    Parameters
    ----------
    subdir : str or Path, optional
        Subdirectory within the data directory, if None returns the main data directory

    Returns
    -------
    Path
        Path to the requested data directory
    """
    if subdir:
        data_dir = DEFAULT_DATA_DIR / subdir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level=logging.INFO):
    """
    Set up a custom logger with console and optional file output.

    Parameters
    ----------
    name : str, optional
        Logger name, default is APP_NAME
    log_file : str or Path, optional
        Path to log file, if None no file logging is set up
    level : int, optional
        Logging level, default is INFO

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers if present
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if requested
    if log_file:
        # Ensure the log directory exists
        log_path = Path(log_file)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
