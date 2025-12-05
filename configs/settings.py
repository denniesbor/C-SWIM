"""
Configuration module for GIC calculations.
Authors: Dennies and Ed
"""

import os
import sys
import logging
from pathlib import Path

APP_NAME = "gic_analysis"

DEFAULT_DATA_DIR = Path("__file__").resolve().parent / "data"

# Leave out sites for current system model
LEAVE_OUT_SITES = ["GUA", "HON"]  # as questioned by C Balch and advised by R Weigel

# Ration of substations with blocking devices (BD)
P_TRAFO_BD = 0.01

# Ground gic Dir - specify a dir with enough space ~ 300 GB for 200 sim
try:
    GROUND_GIC_DIR = Path("/data/archives/nfs/spw-geophy/data/gic/ground_gic")
    GROUND_GIC_DIR.mkdir(parents=True, exist_ok=True)
except (FileNotFoundError, PermissionError):
    GROUND_GIC_DIR = DEFAULT_DATA_DIR / "gic" / "ground_gic"
    GROUND_GIC_DIR.mkdir(parents=True, exist_ok=True)


def get_data_dir(subdir=None):
    """Get data directory path, creating if needed."""
    if subdir:
        data_dir = DEFAULT_DATA_DIR / subdir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level=logging.INFO):
    """Setup logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers if present
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# EHV cutoff voltage threshold (kV)
cut_off_volt = 160  # All above 161 kV are considered EHV lines
