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
5. Validates the GIC model against real measurements and IEEE benchmark test cases
6. Performs socio-economic and reliability impact modeling using Input-Output (IO) and Computable General Equilibrium (CGE) models

This work builds upon the geoelectric hazard analysis framework developed by Lucas et al. (2020) for 100-year return period assessments, and extends this approach to the substation-level for the U.S. high-voltage power grid. C-SWIM then uses vulnerable substation information to undertake a comprehensive socio-economic impact analysis.

## Paper citation

- Oughton, E.J., Bor, D.K., Weigel, R., Gaunt, C.T., Dogan, R., Huang, L., Love, J.J., & Wiltberger, M. (2024). Major Space Weather Risks Identified via Coupled Physics-Engineering-Economic Modeling. *arXiv preprint* [doi.org](https://arxiv.org/abs/2412.18032)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/denniesbor/C-SWIMs.git
cd c-swim

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate spw-env

# 3. Install the package (no pip dependency resolution — conda owns deps)
pip install -e . --no-deps
```

---

## External Repositories

Two external repositories are required and must be cloned and installed separately:

**SWERVE** — Statistical GIC prediction models:
```bash
git clone https://github.com/lucywilkerson/SWERVE
pip install -e /path/to/SWERVE --no-deps
```

**utilrsw** — Utility functions required by SWERVE:
```bash
pip install git+https://github.com/rweigel/utilrsw.git
```

Update `SWERVE_DIR` in `configs/settings.py` to point to your local SWERVE clone.

---

## Data Requirements

### Option A: Zenodo Bundle (Recommended)

Download the prepared data bundle to bypass lengthy data acquisition:

```bash
wget -O data-tl-emtf-storms.tar.gz "https://zenodo.org/records/16994602/files/data-tl-emtf-storms.tar.gz?download=1"
tar -xzf data-tl-emtf-storms.tar.gz
```

Economic data must be obtained separately from the sources listed below.

### Option B: Manual Data Acquisition

**Geomagnetic Data:**
- Kp/ap indices — [GFZ Potsdam](https://kp.gfz.de/en/)
- Dst index — [WDC for Geomagnetism, Kyoto](https://wdc.kugi.kyoto-u.ac.jp/dstdir/)
- INTERMAGNET observatory data — [INTERMAGNET.org](https://intermagnet.org/)
- USGS geomagnetism — [USGS Geomagnetism Program](https://www.usgs.gov/programs/geomagnetism/data)
- NRCan geomagnetic data — [NRCan Geomagnetic Services](https://geomag.nrcan.gc.ca/data-donnee/sd-en.php)

**Electromagnetic Transfer Functions:**
- EMTF/MT transfer functions — [NSF SAGE/IRIS SPUD EMTF Repository](https://ds.iris.edu/spud/emtf/)

**Power Grid Infrastructure:**
- Transmission lines (U.S.) — [HIFLD](https://hifld-geoplatform.hub.arcgis.com/)
- Substations (OSM) — [Overpass Turbo](https://overpass-turbo.eu/)

**Economic Data:**
- Population (2020) — [U.S. Census Bureau](https://www.census.gov/programs-surveys/decennial-census/data.html)
- ZIP Code Business Patterns (ZBP, 2021) — [U.S. Census Bureau SUSB](https://www.census.gov/programs-surveys/susb.html)
- State GDP by Industry (2023) — [BEA](https://www.bea.gov/data/gdp/gdp-state)
- Supply-Use Tables (2023) — [BEA Input-Output Accounts](https://www.bea.gov/industry/input-output-accounts-data)
- NLCD Land Cover (2023) — [USGS MRLC](https://www.mrlc.gov/data)
- ZCTA Shapefiles (2020) — [U.S. Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

---

## Pipeline Architecture

### Physics-Engineering Pipeline

```
preprocess/     — Storm identification, geomagnetic and grid data acquisition
scripts/        — Storm maxima, statistical extremes, admittance matrix, GIC simulation
postprocess/    — GIC aggregation and effective GIC computation
validation/     — Model validation against measurements and IEEE benchmark
```

### Economic Pipeline

```
econ/preprocess/   — Economic data processing and spatial interpolation
econ/scripts/      — IO/CGE economic impact analysis
econ/models/       — IO and CGE model implementations
viz/               — Economic and fragility visualizations
```

---

## Execution

### Physics-Engineering Pipeline

```bash
# Preprocessing — storm identification, downloads, grid processing
python run_preprocess.py

# GIC analysis
python run_scenarios.py storm       # Extract storm maxima
python run_scenarios.py stat        # Statistical return period analysis
python run_scenarios.py admittance  # Build admittance matrix
python run_scenarios.py gic         # Run GIC simulations
python run_scenarios.py all         # Run complete analysis

# Post-processing
python run_postprocess.py

# Validation
python run_val.py
```

### Economic Pipeline

```bash
python run_econ.py preprocess    # Prepare economic data
python run_econ.py analysis      # Run IO/CGE models
python run_econ.py all           # Run full economic pipeline

# Alpha-Beta scenario mode (requires SWERVE outputs)
python run_econ.py analysis --alpha-beta
```

### Visualization

```bash
python run_viz.py                # Generate all figures
python run_viz.py --steps 01     # Economic impact figures only
python run_viz.py --steps 02     # Fragility curves only
```

---

## Validation

The `validation/` module validates the GIC model against two independent sources:

**TVA Measurements (Gannon Storm, May 2024)**
Compares simulated GIC values against real transformer neutral current measurements from the Tennessee Valley Authority (TVA) during the May 2024 Gannon geomagnetic storm. Includes frequency analysis, time-series comparison, magnetometer field comparison, and alpha-beta regression fitting.

**IEEE Horton Benchmark Test Case**
Validates the Lehtinen-Pirjola GIC solver against the published IEEE PES test case (Horton et al., 2012) using a synthetic 8-substation EHV network with known analytical solutions. Monte Carlo variants of transformer configurations are compared against the deterministic baseline.

```bash
python run_val.py                  # Run all validation steps
python run_val.py --steps 08 09    # IEEE Horton benchmark only
python run_val.py --steps 01 02    # TVA GIC comparison only
```

---

## Directory Structure

```
c-swim/
├── configs/              # Unified configuration
├── preprocess/           # Data acquisition and preprocessing (numbered)
├── scripts/              # Core GIC analysis scripts
├── postprocess/          # GIC post-processing (numbered)
├── validation/           # Model validation (numbered)
├── econ/                 # Economic impact analysis
│   ├── preprocess/
│   ├── scripts/
│   └── models/
├── viz/                  # Visualization scripts
├── run_preprocess.py     # Preprocessing runner
├── run_scenarios.py      # GIC scenario runner
├── run_postprocess.py    # Post-processing runner
├── run_val.py            # Validation runner
├── run_econ.py           # Economic analysis runner
├── run_viz.py            # Visualization runner
├── environment.yml       # Conda environment
└── pyproject.toml        # Package installation
```

---

## Configuration

All settings are in `configs/settings.py`:

- `SWERVE_DIR` — Path to local SWERVE repository clone
- `LUCY_DATA_LOC` — Path to external TVA/NERC measurement data
- `IPOPT_EXEC` — Path to IPOPT solver binary
- `cut_off_volt` — EHV voltage threshold (default: 160 kV)
- `P_TRAFO_BD` — Fraction of transformers with GIC blocking devices
- `USE_ALPHA_BETA_SCENARIO` — Toggle alpha-beta GIC scenario mode

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

[MIT License](https://opensource.org/licenses/MIT) — Research code provided as-is for academic and non-commercial use.