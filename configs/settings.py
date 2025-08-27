"""
Configuration module for GIC calculations.

Provides utilities for:
- Data directory management
- Custom logger setup
- GIC processing parameters and scenarios
"""

import os
import sys
import logging
from pathlib import Path

# Application name - used for the logger
APP_NAME = "spwio"

# Path settings
DEFAULT_DATA_DIR = Path("__file__").resolve().parent / "data"

# =============================================================================
# GIC Processing Configuration
# =============================================================================

# Scenario configuration
USE_ALPHA_BETA_SCENARIO = False  # Set to True to use LUCY's GIC MAX PRED WITH ALPHA AND BETA FACTORS

# Alpha-beta scenarios (uncertainty-quantified GIC predictions)
ALPHA_BETA_SCENARIOS = [
    'gic_75yr_conf_68_lower', 'gic_75yr_mean_prediction', 'gic_75yr_conf_68_upper',
    'gic_100yr_conf_68_lower', 'gic_100yr_mean_prediction', 'gic_100yr_conf_68_upper',
    'gic_125yr_conf_68_lower', 'gic_125yr_mean_prediction', 'gic_125yr_conf_68_upper',
    'gic_150yr_conf_68_lower', 'gic_150yr_mean_prediction', 'gic_150yr_conf_68_upper',
    'gic_175yr_conf_68_lower', 'gic_175yr_mean_prediction', 'gic_175yr_conf_68_upper',
    'gic_200yr_conf_68_lower', 'gic_200yr_mean_prediction', 'gic_200yr_conf_68_upper',
    'gic_225yr_conf_68_lower', 'gic_225yr_mean_prediction', 'gic_225yr_conf_68_upper',
    'gic_250yr_conf_68_lower', 'gic_250yr_mean_prediction', 'gic_250yr_conf_68_upper'
]

# Regular scenarios (original approach)
REGULAR_SCENARIOS = [
    'e_75-year-hazard A/ph', 'e_gannon-year-hazard A/ph',
    'e_100-year-hazard A/ph', 'e_50-year-hazard A/ph',
    'e_175-year-hazard A/ph', 'e_125-year-hazard A/ph',
    'e_200-year-hazard A/ph', 'e_150-year-hazard A/ph',
    'e_250-year-hazard A/ph', 'e_225-year-hazard A/ph'
]

# Economic sector columns
GDP_COLUMNS = [
    'GDP_AGR', 'GDP_MINING', 'GDP_UTIL_CONST', 'GDP_MANUF',
    'GDP_TRADE_TRANSP', 'GDP_INFO', 'GDP_FIRE',
    'GDP_PROF_OTHER', 'GDP_EDUC_ENT', 'GDP_G'
]

EST_COLUMNS = [
    'EST_AGR', 'EST_MINING', 'EST_UTIL_CONST', 'EST_MANUF',
    'EST_TRADE_TRANSP', 'EST_INFO', 'EST_FIRE',
    'EST_PROF_OTHER', 'EST_EDUC_ENT', 'EST_G'
]

# Data processing configuration
DROP_COLUMNS = ['n_samples', 'n_substations', 'total_pop']
CSV_DTYPES = {'sub_id': 'category'}

# Simulation parameters
DEFAULT_THETA0 = 75.0           # Default fragility parameter
DEFAULT_TOLERANCE = 0.1         # Convergence tolerance for simulations
DEFAULT_MAX_ITERATIONS = 20000  # Maximum simulation iterations
DEFAULT_BATCH_SIZE = 2000       # Batch size for vectorized operations
DEFAULT_SAVE_BATCH_SIZE = 50    # Files per processing batch


DENNIES_DATA_LOC = Path('/home/pve_ubuntu/spw-geophy-io/data')
IPOPT_EXEC = "/home/pve_ubuntu/miniconda3/envs/spw-env/bin/ipopt"

# File paths for alpha-beta scenario
ALPHA_BETA_GIC_FILE = DENNIES_DATA_LOC / "/regression" / "substations_with_gic_uncertainty.geojson"

# Regular GIC file directories
GIC_DIRECTORIES = [
    DENNIES_DATA_LOC / "gic_eff",
    "~/ubuntu/archives/spw-geophy/data/final_gic/gic_eff"
]

# Output file configurations
OUTPUT_FILES = {
    'alpha_beta_uncertainty': 'scenario_summary_alpha_beta_uncertainty.nc',
    'alpha_beta_regular': 'scenario_summary_alpha_beta.nc',
    'regular_all': 'scenario_summary_all_v2.nc',
    'vulnerable_alpha_beta_uncertainty': 'vulnerable_substations_alpha_beta_uncertainty.parquet',
    'vulnerable_alpha_beta_regular': 'vulnerable_substations_alpha_beta.parquet',
    'vulnerable_regular': 'vulnerable_substations'
}

# Figures DIR
FIGURES_DIR = Path("__file__").resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def get_scenarios():
    """Get the appropriate scenario list based on configuration."""
    return ALPHA_BETA_SCENARIOS if USE_ALPHA_BETA_SCENARIO else REGULAR_SCENARIOS


def get_simulation_config():
    """Get simulation parameters as a dictionary."""
    return {
        'theta0': DEFAULT_THETA0,
        'tolerance': DEFAULT_TOLERANCE,
        'max_iterations': DEFAULT_MAX_ITERATIONS,
        'batch_size': DEFAULT_BATCH_SIZE,
        'save_batch_size': DEFAULT_SAVE_BATCH_SIZE
    }


def get_data_dir(subdir=None):
    """
    Get the path to a data directory, creating it if it doesn't exist.
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
