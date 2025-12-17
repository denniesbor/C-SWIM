# Economic Impact Analysis Module

**Location:** `c-swim/econ/`

Socio-economic impact modeling for geomagnetically induced currents (GIC) on the U.S. power grid.

## Project Structure
```
c-swim/
├── run_econ.py          # ← Economic pipeline orchestrator (run from here)
├── configs/             # Unified configuration
├── data/                # Shared data directory
│   ├── admittance_matrix/   # C-SWIM outputs
│   ├── gic_eff/            # GIC simulation results
│   ├── gnd_gic/            # Ground GIC results
│   └── econ_data/          # ← Economic analysis data
│       ├── raw_econ_data/       # Census, GDP inputs
│       ├── processed_econ/      # Processed economic data
│       ├── land_mask/           # NLCD land cover data
│       ├── 10sector/            # IO tables
│       └── sam/                 # Social Accounting Matrix
├── figures/             # Output visualizations
└── econ/                # ← You are here
    ├── preprocess/      # Data preparation
    ├── scripts/         # Analysis pipeline
    ├── models/          # IO/CGE economic models
    └── viz/             # Visualization tools
```

## Overview

This module extends C-SWIM's physics-engineering GIC simulations with economic impact assessment using Input-Output (IO) and Computable General Equilibrium (CGE) models.

## Quick Start

**⚠️ Important:** All commands must be run from the **c-swim root directory**, not from `econ/`
```bash
# Navigate to c-swim root
cd ~/c-swim

# Run economic analysis pipeline
python run_econ.py preprocess   # Prepare economic data
python run_econ.py analysis     # Run IO/CGE models
python run_econ.py viz          # Generate figures
python run_econ.py all          # Run full pipeline
```

### Prerequisites
- C-SWIM GIC simulation outputs in `data/gic_eff/`
- Economic data (census, NLCD, GDP) in `data/econ_data/raw_econ_data/`
- Python environment: `conda activate spw-env`

### Alpha-Beta Scenario Mode
Use when you have results from Alpha-Beta regression models (See [SWERVE](https://github.com/lucywilkerson/SWERVE))
```bash
python run_econ.py analysis --alpha-beta
```

## Module Structure
```
preprocess/          # Data preparation
├── p_econ_data.py        # Census, GDP data processing
├── downsample_nlcd.py    # Land cover data processing
├── p_areal_intp.py       # Dasymetric mapping
├── p_gic_files.py        # GIC processing & Monte Carlo
├── p_technology.py       # IO coefficient tables
└── p_us_sam.py          # Social Accounting Matrix

scripts/             # Analysis pipeline
├── econ_analysis.py      # Main IO/CGE analysis
├── l_prepr_data.py       # Data loading utilities
└── run_policy_test.py    # Policy scenario testing

models/              # Economic models
├── io_model.py           # Input-Output model
├── cge_model.py          # CGE model core
└── cge_data_model.py     # CGE data management

viz/                 # Visualization
├── viz.py                # Main visualization script
├── plot_utils.py         # Plotting utilities
└── plot_fragility.py     # Reliability plots
```

## Data Flow

1. **C-SWIM outputs** (`../data/gic_eff/`, `../data/admittance_matrix/`) → Economic module inputs
2. **Economic preprocessing** → Spatial economic data at substation level
3. **GIC processing** → Monte Carlo transformer failure scenarios
4. **IO/CGE analysis** → Direct and indirect economic impacts
5. **Visualization** → Maps, charts, confidence intervals

## Key Outputs

Located in `../figures/`:
- `io_model_results.csv` - Sector-level economic impacts
- `confidence_intervals.csv` - Uncertainty quantification
- `hazard_maps.png` - Spatial GIC distribution
- `vulnerable_trafos_*.pdf` - Transformer failure maps

Located in `../data/econ_data/`:
- `scenario_summary_*.nc` - Complete results dataset
- `vulnerable_substations_*.parquet` - Substation vulnerability data
- `gic_processing_batches/` - Intermediate processing results

## Configuration

Settings in [`../configs/settings.py`](../configs/settings.py):
- `USE_ALPHA_BETA_SCENARIO` - Use [Lucy Wilkerson's Alpha-Beta GIC model](https://github.com/lucywilkerson/SWERVE)
- `PROCESS_GND_FILES` - Use ground instead of effective GIC
- `DEFAULT_THETA0` - Transformer fragility parameter
- Economic sector mappings (`GDP_COLUMNS`, `EST_COLUMNS`)

## Related Publication

Oughton, E. J., Bor, D. K., Weigel, R., Gaunt, C. T., Dogan, R., Huang, L., Love, J. J., & Wiltberger, M. (2024). Major Space Weather Risks Identified via Coupled Physics-Engineering-Economic Modeling. [*arXiv:2412.18032*](https://arxiv.org/abs/2412.18032)

## Contact

**Dennies Bor** - Modeling  
**Ed Oughton** - Research lead

## Related Repositories
- [SWERVE](https://github.com/lucywilkerson/SWERVE) - Statistical GIC prediction models