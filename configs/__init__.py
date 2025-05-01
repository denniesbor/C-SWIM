"""
Configuration package for GIC calculation.

Provides utilities for data directory management and custom logging.
"""

from .settings import (
    APP_NAME,
    DEFAULT_DATA_DIR,
    get_data_dir,
    setup_logger,
)

__all__ = [
    "APP_NAME",
    "DEFAULT_DATA_DIR",
    "get_data_dir",
    "setup_logger",
]
