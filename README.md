<p align="center">
  <img src="C-SWIM-Logo.png" alt="C-SWIM Logo" width="400"/>
</p>

# C-SWIM: Coupled Space Weather Impact Model
A Coupled Physics–Engineering–Economic Pipeline for Impact Assessment of Extreme Space Weather

## Summary

This repository provides a reproducible end-to-end pipeline that:

1. Collects and preprocesses geomagnetic and grid datasets
2. Derives extreme geoelectric field scenarios statistically
3. Builds a geospatial/electrical admittance model of the extra-high voltage (EHV) grid
4. Simulates geomagnetically induced currents (GIC) under synthetic and historical storms
5. Performs socio-economic and reliability impact modeling using Input-Output (IO) and Computable General Equilibrium (CGE) models

This work builds upon the geoelectric hazard analysis framework developed by Lucas et al. (2020) for 100-year return period assessments, and then extends this approach to the substation-level for the U.S. high-voltage power grid. C-SWIM then uses vulnerable substation information to undertake a comprehensive socio-economic impact analysis.

Paper citation
-----------------
 
- Oughton, E.J., Bor, D.K., Weigel, R., Gaunt, C.T., Dogan, R., Huang, L., Love, J.J., & Wiltberger, M. (2024). Major Space Weather Risks Identified via Coupled Physics-Engineering-Economic Modeling. *arXiv preprint* [doi.org](https://arxiv.org/abs/2412.18032)

---

## Quick Start

### Option A: Using Zenodo Data (Recommended)

Download the prepared data bundle to bypass lengthy data acquisition:

```bash
# Clone the repository
git clone https://github.com/Space-Weather-Research/c-swim.git
cd c-swim

# Download prepared physics data bundle
wget -O data-tl-emtf-storms.tar.gz "https://zenodo.org/records/16994602/files/data-tl-emtf-storms.tar.gz?download=1"

# Extract to data directory
tar -xzf data-tl-emtf-storms.tar.gz

# Economic data must be obtained separately from listed sources (see Data Requirements section)
```

### Option B: Environment Setup

Create the conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate spw-env
pip install bezpy pysecs
```

---

## Pipeline Architecture

The complete pipeline includes both physics-engineering and economic analysis:

### Physics-Engineering Pipeline (Stages 1-8)

1. **Data acquisition & preprocessing** — geomagnetic observatories, EMTF/MT transfer functions, storm identification, transmission grid & substations
2. **Storm maxima extraction** — space-time windowing of **B**, derived **E**, and induced line voltages **V** at MT sites & along lines  
3. **Statistical extremes** — fit return-period scaling/distributions for **B/E/V**
4. **Grid network & admittance** — substation-line topology; transformer archetype assignment
5. **Scenario synthesis** — Gannon event + 75/100/150/200/250-year extremes
6. **GIC simulation** — Lehtinen-Pirjola via `est_gic.py`; (2/3) thousand Monte-Carlo transformer realizations
7. **Post-processing** — effective/aggregated GIC metrics
8. **Validation** — IEEE benchmark test case (Horton grid)

### Economic Impact Pipeline (Stage 9)

9. **Socio-economic analysis** — IO/CGE modeling of direct and indirect economic impacts

See [`econ/README.md`](econ/README.md) for detailed documentation of the economic analysis module.

---

## Directory Structure
```
c-swim/
├── configs/                     # Unified configuration (Phys-Eng-Econ)
│   ├── __init__.py
│   └── settings.py
├── preprocess/                  # Physics data acquisition & preprocessing
│   ├── p_identify_storm_periods.py
│   ├── dl_intermagnet.py
│   ├── dl_usgs_pre_1990.py
│   ├── dl_nrcan_pre_1990.py
│   ├── p_geomag_data.py
│   ├── dl_power_grid_update.py
│   └── p_power_grid.py
├── scripts/                     # Core GIC analysis pipeline
│   ├── calc_storm_maxes.py
│   ├── stat_analysis.py
│   ├── build_admittance_matrix.py
│   ├── est_gic.py
│   └── horton_grid.py
├── postprocess/                 # GIC results aggregation
│   ├── agg_scenario_gic.py
│   ├── aggregate_gannon_gic.py
│   └── calc_eff_gic.py
├── econ/                        # Economic impact analysis
│   ├── README.md                    # Detailed economic module documentation
│   ├── preprocess/                  # Economic data processing
│   ├── scripts/                     # IO/CGE analysis
│   ├── models/                      # Economic models
│   └── viz/                         # Visualization tools
├── preprocess_wrapper.py        # Physics preprocessing orchestrator
├── run_scenarios.py             # GIC analysis orchestrator
├── run_postprocess.py           # GIC post-processing orchestrator
├── run_econ.py                  # Economic analysis orchestrator
├── data/                        # Data directory
│   ├── admittance_matrix/           # Grid electrical model
│   ├── gic_eff/                     # Effective GIC outputs
│   ├── gnd_gic/                     # Ground GIC outputs
│   └── econ_data/                   # Economic analysis data
│       ├── raw_econ_data/               # Census, GDP inputs
│       ├── processed_econ/              # Processed economic data
│       ├── land_mask/                   # NLCD land cover
│       ├── 10sector/                    # IO coefficient tables
│       └── sam/                         # Social Accounting Matrix
├── figures/                     # All output visualizations
└── logs/                        # Execution logs
```

---

## Data Requirements

### Physics-Engineering Data

**Geomagnetic Data:**
- **Kp/ap indices** — [GFZ Potsdam](https://kp.gfz.de/en/)
- **Dst index** — [WDC for Geomagnetism, Kyoto](https://wdc.kugi.kyoto-u.ac.jp/dstdir/)
- **INTERMAGNET** observatory data — [INTERMAGNET.org](https://intermagnet.org/)
- **USGS** geomagnetism — [USGS Geomagnetism Program](https://www.usgs.gov/programs/geomagnetism/data)
- **NRCan** geomagnetic data — [NRCan Geomagnetic Services](https://geomag.nrcan.gc.ca/data-donnee/sd-en.php)

**Electromagnetic Transfer Functions:**
- **EMTF/MT transfer functions** — [NSF SAGE/IRIS SPUD EMTF Repository](https://ds.iris.edu/spud/emtf/)

**Power Grid Infrastructure:**
- **Transmission lines (U.S.)** — [HIFLD](https://hifld-geoplatform.hub.arcgis.com/) (currently inaccessible)
- **Substations (OSM)** — [Overpass Turbo](https://overpass-turbo.eu/)

### Economic Analysis Data

**Demographic & Economic Data:**
- **Population (2020)** — [U.S. Census Bureau Decennial Census](https://www.census.gov/programs-surveys/decennial-census/data.html)
  - 2020 Population and Housing State Data
- **ZIP Code Business Patterns (ZBP, 2021)** — [U.S. Census Bureau Statistics of U.S. Businesses](https://www.census.gov/programs-surveys/susb.html)
  - Business establishment counts by NAICS sector at ZIP Code level
- **ZIP to ZCTA Crosswalk (2020)** — [HRSA Geo-Crosswalk](https://data.hrsa.gov/DataDownload/GeoCareNavigator/ZIP%20Code%20to%20ZCTA%20Crosswalk.xlsx)
  - Translates ZIP codes to ZCTA geography
- **State GDP by Industry (2023)** — [Bureau of Economic Analysis (BEA)](https://www.bea.gov/data/gdp/gdp-state)
  - Sectoral GDP at state level
- **Supply-Use Tables (2023)** — [BEA Input-Output Accounts](https://www.bea.gov/industry/input-output-accounts-data)
  - Inter-industry production relationships for IO model
- **National Income and Product Accounts (NIPA, 2023)** — [BEA NIPA Tables](https://apps.bea.gov/iTable/)
  - Used to complete Social Accounting Matrix

**Land Cover Data:**
- **National Land Cover Database (NLCD, 2023)** — [USGS MRLC](https://www.mrlc.gov/data)
  - 30-meter resolution land cover classification
  - Used for dasymetric interpolation with built environment classes (21-24)

**Geospatial Boundaries:**
- **ZCTA Shapefiles (2020)** — [U.S. Census Bureau TIGER/Line Files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
  - ZIP Code Tabulation Area boundaries for spatial analysis

---

## Execution

### Complete Pipeline

```bash
# 1. Physics-Engineering Pipeline
python preprocess_wrapper.py          # Data acquisition & preprocessing
python run_scenarios.py all           # GIC simulations
python run_postprocess.py all         # Post-processing

# 2. Economic Analysis Pipeline
python run_econ.py preprocess         # Prepare economic data
python run_econ.py analysis           # Run IO/CGE models
python run_econ.py viz                # Generate figures
```

### Physics-Engineering Pipeline

#### 1. Preprocessing Wrapper: `preprocess_wrapper.py`

Handles storm identification, data downloads, and preprocessing:

```bash
python preprocess_wrapper.py                # full pipeline (parallel)
python preprocess_wrapper.py --sequential   # deterministic serial run  
python preprocess_wrapper.py --download-only
python preprocess_wrapper.py --preprocess-only --skip-storm-id
```

#### 2. Scenario Modeling: `run_scenarios.py`

```bash
python run_scenarios.py storm       # Extract storm maxima
python run_scenarios.py stat        # Statistical analysis
python run_scenarios.py admittance  # Build grid model
python run_scenarios.py gic         # Run GIC simulations
python run_scenarios.py all         # Run complete analysis
```

#### 3. Post-processing: `run_postprocess.py`

```bash
python run_postprocess.py aggregate   # Aggregate results
python run_postprocess.py effective   # Calculate effective GICs
python run_postprocess.py all         # Run all post-processing
```

### Economic Analysis Pipeline

See [`econ/README.md`](econ/README.md) for detailed documentation.

```bash
python run_econ.py preprocess   # Prepare economic data
python run_econ.py analysis     # Run IO/CGE models
python run_econ.py viz          # Generate figures
python run_econ.py all          # Run full economic pipeline

# Alpha-Beta scenario mode (requires SWERVE outputs)
python run_econ.py analysis --alpha-beta
```

### Validation

```bash
python scripts/horton_grid.py   # IEEE benchmark validation
```

---

## Key Outputs

**Physics-Engineering:**
- `data/admittance_matrix/` — Grid electrical model
- `data/gic_eff/` — Effective GIC values per scenario
- `data/gnd_gic/` — Ground GIC values per scenario
- `figures/hazard_maps*.png` — Spatial GIC distributions

**Economic Analysis:**
- `figures/io_model_results.csv` — Sector-level economic impacts
- `figures/confidence_intervals.csv` — Uncertainty quantification
- `figures/vulnerable_trafos*.pdf` — Transformer failure visualizations
- `data/econ_data/scenario_summary*.nc` — Complete results datasets

---

## Configuration

All settings are managed in `configs/settings.py`:

**Physics-Engineering:**
- `GROUND_GIC_DIR` — Storage path for large GIC datasets (~300GB)
- `cut_off_volt` — EHV threshold (default: 160 kV)
- `P_TRAFO_BD` — Ratio of transformers with blocking devices

**Economic Analysis:**
- `USE_ALPHA_BETA_SCENARIO` — Use [SWERVE](https://github.com/lucywilkerson/SWERVE) Alpha-Beta GIC models
- `PROCESS_GND_FILES` — Use ground vs effective GIC
- `DEFAULT_THETA0` — Transformer fragility parameter
- Economic sector mappings (`GDP_COLUMNS`, `EST_COLUMNS`)

---

## Related Repositories

- [**SWERVE**](https://github.com/lucywilkerson/SWERVE) — Statistical GIC prediction models (Lucy Wilkerson)

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{oughton2024physics,
  title={Major Space Weather Risks Identified via Coupled Physics-Engineering-Economic Modeling}, 
  author={Oughton, Edward J. and Bor, Dennies K. and Weigel, Robert and Gaunt, C. Trevor and Dogan, Ramiz and Huang, Lucy and Love, Jeffrey J. and Wiltberger, Michael},
  year={2024},
  eprint={2412.18032},
  archivePrefix={arXiv},
  primaryClass={physics.geo-ph},
  url={https://arxiv.org/abs/2412.18032}
}
```

This work builds upon the foundational methodology from:

```bibtex
@article{lucas2020100year,
  title={100-year Geoelectric Hazard Analysis for the U.S. High-Voltage Power Grid},
  author={Lucas, G. M. and Love, J. J. and Kelbert, A. and Bedrosian, P. A. and Rigler, E. J.},
  journal={Space Weather},
  volume={18},
  year={2020},
  doi={10.1029/2019SW002329}
}
```

For the IEEE benchmark test case:

```bibtex
@article{horton_test_2012,
  title = {A Test Case for the Calculation of Geomagnetically Induced Currents},
  volume = {27},
  number = {4},
  pages = {2368--2373},
  year = {2012},
  issn = {1937-4208},
  doi = {10.1109/TPWRD.2012.2206407},
  journal = {IEEE Transactions on Power Delivery},
  author = {Horton, Randy and Boteler, David and Overbye, Thomas J. and Pirjola, Risto and Dugan, Roger C.}
}
```

For the data products:

```bibtex
@dataset{bor2025geomag,
  author = {Bor, Dennies K.},
  title = {C-SWIM Geomagnetic and Grid Data},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.16994602},
  url = {https://doi.org/10.5281/zenodo.16994602}
}
```

---

## Acknowledgments

This work builds upon data and services provided by: GFZ Potsdam; WDC Kyoto; INTERMAGNET; USGS; NRCan; EarthScope/IRIS EMTF/USArray; HIFLD; OpenStreetMap contributors; U.S. Census Bureau; Bureau of Economic Analysis.

---

## License

[MIT License](LICENSE) — This repository provides research code and derived data products as-is for academic and non-commercial use.
