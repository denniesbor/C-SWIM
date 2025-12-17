"""
Configuration module for GIC calculations.

Provides utilities for:
- Data directory management
- Custom logger setup
- GIC processing parameters and scenarios
"""

import os
from pickle import TRUE
import sys
import logging
from pathlib import Path

APP_NAME = "spwio"

DEFAULT_DATA_DIR = Path("__file__").resolve().parent / "data"

USE_ALPHA_BETA_SCENARIO = (
    os.getenv("USE_ALPHA_BETA_SCENARIO", "false").lower() == "true"
)

ALPHA_BETA_SCENARIOS = [
    "gic_75yr_conf_68_lower",
    "gic_75yr_mean_prediction",
    "gic_75yr_conf_68_upper",
    "gic_100yr_conf_68_lower",
    "gic_100yr_mean_prediction",
    "gic_100yr_conf_68_upper",
    "gic_125yr_conf_68_lower",
    "gic_125yr_mean_prediction",
    "gic_125yr_conf_68_upper",
    "gic_150yr_conf_68_lower",
    "gic_150yr_mean_prediction",
    "gic_150yr_conf_68_upper",
    "gic_175yr_conf_68_lower",
    "gic_175yr_mean_prediction",
    "gic_175yr_conf_68_upper",
    "gic_200yr_conf_68_lower",
    "gic_200yr_mean_prediction",
    "gic_200yr_conf_68_upper",
    "gic_225yr_conf_68_lower",
    "gic_225yr_mean_prediction",
    "gic_225yr_conf_68_upper",
    "gic_250yr_conf_68_lower",
    "gic_250yr_mean_prediction",
    "gic_250yr_conf_68_upper",
]

REGULAR_SCENARIOS = [
    "e_75-year-hazard A/ph",
    "e_gannon-year-hazard A/ph",
    "e_100-year-hazard A/ph",
    "e_50-year-hazard A/ph",
    "e_175-year-hazard A/ph",
    "e_125-year-hazard A/ph",
    "e_200-year-hazard A/ph",
    "e_150-year-hazard A/ph",
    "e_250-year-hazard A/ph",
    "e_225-year-hazard A/ph",
]

GDP_COLUMNS = [
    "GDP_AGR",
    "GDP_MINING",
    "GDP_UTIL_CONST",
    "GDP_MANUF",
    "GDP_TRADE_TRANSP",
    "GDP_INFO",
    "GDP_FIRE",
    "GDP_PROF_OTHER",
    "GDP_EDUC_ENT",
    "GDP_G",
]

EST_COLUMNS = [
    "EST_AGR",
    "EST_MINING",
    "EST_UTIL_CONST",
    "EST_MANUF",
    "EST_TRADE_TRANSP",
    "EST_INFO",
    "EST_FIRE",
    "EST_PROF_OTHER",
    "EST_EDUC_ENT",
    "EST_G",
]

DROP_COLUMNS = ["n_samples", "n_substations", "total_pop"]
CSV_DTYPES = {"sub_id": "category"}

DEFAULT_THETA0 = 75.0
DEFAULT_TOLERANCE = 0.1
DEFAULT_MAX_ITERATIONS = 20000
DEFAULT_BATCH_SIZE = 2000
DEFAULT_SAVE_BATCH_SIZE = 50

# Need to set this (this is )
DENNIES_DATA_LOC = Path("/home/pve_ubuntu/spw-geophy-io/data")
IPOPT_EXEC = "/home/pve_ubuntu/miniconda3/envs/spw-env/bin/ipopt"

ALPHA_BETA_GIC_FILE = (
    DENNIES_DATA_LOC / "regression" / "substations_with_gic_uncertainty_scaled.geojson"
)                                                                                                                                                                                                                                                                                                                               

EFF_GIC_DIR = [
    DENNIES_DATA_LOC / "gic_eff",
    "~/ubuntu/archives/spw-geophy/data/final_gic/gic_eff",
]

GND_GIC_DIR = [
    DENNIES_DATA_LOC / "gnd_gic",
    "~/ubuntu/archives/spw-geophy/data/final_gic/gnd_gic",
]

PROCESS_GND_FILES = False

OUTPUT_FILES = {
    "alpha_beta_uncertainty": "scenario_summary_alpha_beta_uncertainty.nc",
    "alpha_beta_regular": "scenario_summary_alpha_beta.nc",
    "regular_all": (
        "scenario_summary_gnd_gic.nc"
        if PROCESS_GND_FILES
        else "scenario_summary_eff_gic.nc"
    ),
    "vulnerable_alpha_beta_uncertainty": "vulnerable_substations_alpha_beta_uncertainty.parquet",
    "vulnerable_alpha_beta_regular": "vulnerable_substations_alpha_beta.parquet",
    "vulnerable_regular": (
        "vulnerable_substations_gnd_gic"
        if PROCESS_GND_FILES
        else "vulnerable_substations_eff_gic"
    ),
}

FIGURES_DIR = Path("__file__").resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_scenarios():
    """Get the appropriate scenario list based on configuration."""
    return ALPHA_BETA_SCENARIOS if USE_ALPHA_BETA_SCENARIO else REGULAR_SCENARIOS


def get_simulation_config():
    """Get simulation parameters as a dictionary."""
    return {
        "theta0": DEFAULT_THETA0,
        "tolerance": DEFAULT_TOLERANCE,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "save_batch_size": DEFAULT_SAVE_BATCH_SIZE,
    }


def get_data_dir(subdir=None):
    """Get the path to a data directory, creating it if it doesn't exist."""
    if subdir:
        data_dir = DEFAULT_DATA_DIR / subdir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level=logging.INFO):
    """Set up a custom logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

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