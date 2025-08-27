# entry file
"""
Configuration package for GIC calculation.

Provides utilities for data directory management, custom logging,
and GIC processing configuration parameters.
"""

from .settings import (
    # Core utilities
    APP_NAME,
    DEFAULT_DATA_DIR,
    get_data_dir,
    setup_logger,
    
    # GIC processing configuration
    USE_ALPHA_BETA_SCENARIO,
    ALPHA_BETA_SCENARIOS,
    REGULAR_SCENARIOS,
    GDP_COLUMNS,
    EST_COLUMNS,
    DROP_COLUMNS,
    CSV_DTYPES,
    
    # Simulation parameters
    DEFAULT_THETA0,
    DEFAULT_TOLERANCE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAVE_BATCH_SIZE,
    
    # File paths and configurations
    ALPHA_BETA_GIC_FILE,
    GIC_DIRECTORIES,
    OUTPUT_FILES,
    DENNIES_DATA_LOC,
    FIGURES_DIR,
    IPOPT_EXEC,

    # Helper functions
    get_scenarios,
    get_simulation_config,
)

__all__ = [
    # Core utilities
    "APP_NAME",
    "DEFAULT_DATA_DIR",
    "get_data_dir",
    "setup_logger",
    
    # GIC processing configuration
    "USE_ALPHA_BETA_SCENARIO",
    "ALPHA_BETA_SCENARIOS", 
    "REGULAR_SCENARIOS",
    "GDP_COLUMNS",
    "EST_COLUMNS",
    "DROP_COLUMNS",
    "CSV_DTYPES",
    
    # Simulation parameters
    "DEFAULT_THETA0",
    "DEFAULT_TOLERANCE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SAVE_BATCH_SIZE",
    
    # File paths and configurations
    "ALPHA_BETA_GIC_FILE",
    "GIC_DIRECTORIES",
    "OUTPUT_FILES",
    "DENNIES_DATA_LOC",
    "FIGURES_DIR",
    "IPOPT_EXEC",

    # Helper functions
    "get_scenarios",
    "get_simulation_config",
]