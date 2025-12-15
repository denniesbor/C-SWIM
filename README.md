<p align="center">
  <img src="C-SWIM-Logo.png" alt="C-SWIM Logo" width="400"/>
</p>

# C-SWIM: Coupled Space Weather Impact Modeling
A Coupled Physics–Engineering–Economic Pipeline for Impact Assessment of Extreme Space Weather

## Summary

This repository provides a reproducible pipeline that (1) collects and preprocesses geomagnetic and grid datasets, (2) derives extreme geoelectric field scenarios statistically, (3) builds a geospatial/electrical admittance model of the extra-high voltage (EHV) grid, (4) simulates geomagnetically induced currents (GIC) under synthetic and historical storms, and (5) exports outputs for downstream socio-economic and reliability impact modeling (handled in the separate [`spwio`](https://github.com/denniesbor/spwio/) repository).

This work builds upon the foundational geoelectric hazard analysis framework developed by Lucas et al. (2020) for 100-year return period assessments of the U.S. high-voltage power grid.

**Related Publication:** Oughton, E. J., et al. (2024). A physics-engineering-economic model coupling approach for estimating the socio-economic impacts of space weather scenarios. *arXiv preprint* arXiv:2412.18032.

---

## Quick Start

### Option A: Using Zenodo Data (Recommended)

Download the prepared data bundle to bypass lengthy data acquisition. Once you clone this repository, extract the data at the root level (same level as preprocess, scripts, etc.):
```bash
# Download prepared data bundle
wget -O data-tl-emtf-storms.tar.gz "https://zenodo.org/records/16994602/files/data-tl-emtf-storms.tar.gz?download=1"

# Extract data to root directory (same level as preprocess/, scripts/, etc.)
tar -xzf data-tl-emtf-storms.tar.gz
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

The pipeline follows a nine-stage process:

1. **Data acquisition & preprocessing** — geomagnetic observatories, EMTF/MT transfer functions, storm identification, transmission grid & substations
2. **Storm maxima extraction** — space-time windowing of **B**, derived **E**, and induced line voltages **V** at MT sites & along lines  
3. **Statistical extremes** — fit return-period scaling/distributions for **B/E/V**
4. **Grid network & admittance** — substation-line topology; transformer archetype assignment
5. **Scenario synthesis** — Gannon event + 75/100/150/200/250-year extremes
6. **GIC simulation** — Lehtinen-Pirjola via `est_gic.py`; (2/3) thousand Monte-Carlo transformer realizations
7. **Post-processing** — effective/aggregated GIC metrics
8. **Validation** — IEEE benchmark test case (Horton grid)
9. **Export** — artifacts for economic/reliability modeling (not in this repo; see [`spwio`](https://github.com/denniesbor/spwio/))

---

## Directory Structure
```
├── configs/
│   ├── __init__.py
│   └── settings.py              # Configuration settings
├── preprocess/                  # Data acquisition and preprocessing
│   ├── p_identify_storm_periods.py
│   ├── dl_intermagnet.py
│   ├── dl_usgs_pre_1990.py
│   ├── dl_nrcan_pre_1990.py
│   ├── p_geomag_data.py
│   ├── dl_power_grid_update.py
│   └── p_power_grid.py
├── scripts/                     # Core analysis pipeline
│   ├── calc_storm_maxes.py
│   ├── stat_analysis.py
│   ├── build_admittance_matrix.py
│   ├── est_gic.py
│   └── horton_grid.py
├── postprocess/                 # Results aggregation
│   ├── agg_scenario_gic.py
│   ├── aggregate_gannon_gic.py
│   └── calc_eff_gic.py
├── preprocess_wrapper.py        # Preprocessing orchestrator
├── run_scenarios.py             # Analysis orchestrator
├── run_postprocess.py           # Post-processing orchestrator
├── data/                        # Data directory (created by pipeline)
└── logs/                        # Execution logs
```

### Script Functions

**Preprocessing:**
- `p_identify_storm_periods.py` → writes `kp_ap_indices/storm_periods.csv` (Kp/Dst merged & filtered)
- `dl_intermagnet.py` → downloads post-1990 observatory magnetic data (INTERMAGNET)
- `dl_usgs_pre_1990.py`, `dl_nrcan_pre_1990.py` → pre-1990 USGS/NRCan magnetic data
- `p_geomag_data.py` → harmonizes observatory + EMTF outputs → `processed_geomag_data.nc`
- `dl_power_grid_update.py` → pulls OSM substations
- `p_power_grid.py` → processes substations + transmission lines (filters ≥ `cut_off_volt`), geometry normalization

**Analysis:**
- `calc_storm_maxes.py` → extracts B/E/V maxima for identified storm windows
- `stat_analysis.py` → fits statistical/extreme-value models  
- `build_admittance_matrix.py` → builds electrical admittance matrices & bus/substation mapping → `data/admittance_matrix/`
- `est_gic.py` → Lehtinen-Pirjola GIC over Monte-Carlo transformer configs & scenarios
- `horton_grid.py` → IEEE benchmark test case validation (Horton et al. 2012) with Monte Carlo sensitivity analysis

**Post-processing:**
- `agg_scenario_gic.py` → Aggregate GIC results across scenarios
- `calc_eff_gic.py` → Calculate effective GICs for downstream impact modeling
- `aggregate_gannon_gic.py` → Aggregates Gannon storm ground GIC metrics

---

## Configuration

### Storage Paths for Large Datasets

Ground GIC simulations generate ~300 GB of data for 200+ Monte Carlo runs. Specify where to store this data in `configs/settings.py`. This is where data is stored in `est_gic.py` and accessed for downstream post processing.

The configuration automatically falls back to the local `data/` directory if the specified path is unavailable.

---

## Execution

### Recommended: Pipeline Orchestrators

#### 1. Preprocessing Wrapper: `preprocess_wrapper.py`

Handles storm identification, data downloads, and preprocessing in three phases:

1. Storm identification → `p_identify_storm_periods.py`
2. Downloads → `dl_intermagnet.py`, `dl_nrcan_pre_1990.py`, `dl_usgs_pre_1990.py`, `dl_power_grid_update.py`  
3. Preprocessing → `p_geomag_data.py`, `p_power_grid.py` (only if required downloads succeed)

**CLI Options:**
```bash
--max-retries <INT>   # retry count for pre-1990 fetches (default 10)
--sequential          # run tasks serially  
--download-only       # only storm-ID (unless skipped) + downloads
--preprocess-only     # assume downloads present; run preprocessing
--skip-storm-id       # skip storm identification (storms already computed)
```

**Usage:**
```bash
python preprocess_wrapper.py                # full pipeline (parallel)
python preprocess_wrapper.py --sequential   # deterministic serial run  
python preprocess_wrapper.py --download-only
python preprocess_wrapper.py --preprocess-only --skip-storm-id
```

#### 2. Scenario Modeling Wrapper: `run_scenarios.py`

Targets `scripts/` (does not redo preprocessing).

**Sub-commands:**
```bash  
storm       -> calc_storm_maxes.py
stat        -> stat_analysis.py
admittance  -> build_admittance_matrix.py
gic         -> est_gic.py
all         -> storm -> stat -> gic
```

**Optional Flag:**
```bash
--gannon-only   # passed through to est_gic.py
```

**Examples:**
```bash
python run_scenarios.py storm
python run_scenarios.py stat  
python run_scenarios.py admittance
python run_scenarios.py gic --gannon-only
python run_scenarios.py all
```

#### 3. Post-processing Wrapper: `run_postprocess.py`

Handles result aggregation and effective GIC calculations.

**Sub-commands:**
```bash
aggregate  -> aggregate_gannon_gic.py
effective  -> calc_eff_gic.py
all        -> aggregate + effective
```

**CLI Options:**
```bash
--sequential    # run scripts serially
--max-retries   # maximum retries per script (default: 1)
```

**Usage:**
```bash
python run_postprocess.py aggregate
python run_postprocess.py effective
python run_postprocess.py all
```

### Validation

Run IEEE benchmark test case (Horton et al. 2012) to validate GIC calculation modules:
```bash
python scripts/horton_grid.py
```

This script:
- Tests all GIC calculation modules against IEEE benchmark data
- Computes line GICs, transformer winding GICs, and substation ground GICs
- Performs Monte Carlo sensitivity analysis (5000 scenarios) with random grid configuration variants
- Saves validation results to `data/horton_grid/mc_results.pkl`

### Manual Sequential Execution

For step-by-step control, run scripts in order:
```bash
# 1. Storm identification
python preprocess/p_identify_storm_periods.py

# 2. Data downloads  
python preprocess/dl_usgs_pre_1990.py
python preprocess/dl_nrcan_pre_1990.py
python preprocess/dl_intermagnet.py
python preprocess/dl_power_grid_update.py

# 3. Data preprocessing
python preprocess/p_geomag_data.py
python preprocess/p_power_grid.py

# 4. Analysis pipeline
python scripts/calc_storm_maxes.py
python scripts/stat_analysis.py
python scripts/build_admittance_matrix.py
python scripts/est_gic.py

# 5. Post-processing
python postprocess/agg_scenario_gic.py
python postprocess/calc_eff_gic.py
python postprocess/aggregate_gannon_gic.py

# 6. Validation (optional)
python scripts/horton_grid.py
```

---

## Data Requirements

### Geomagnetic Data Sources

- **Kp/ap indices** — [GFZ Potsdam](https://kp.gfz.de/en/): [Geomagnetic Kp index](https://www.gfz.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index)
- **Dst index** — [WDC for Geomagnetism, Kyoto](https://wdc.kugi.kyoto-u.ac.jp/dstdir/) (real-time: [DST realtime](https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/))
- **INTERMAGNET** observatory data: [INTERMAGNET.org](https://intermagnet.org/) ([download portal](https://intermagnet.org/data_download.html))
- **USGS** geomagnetism: [USGS Geomagnetism Program](https://www.usgs.gov/programs/geomagnetism/data)  
- **NRCan** geomagnetic data: [NRCan Geomagnetic Services](https://geomag.nrcan.gc.ca/data-donnee/sd-en.php)

### Electromagnetic Transfer Functions

- **EMTF/MT transfer functions** — [NSF SAGE/IRIS SPUD](https://ds.iris.edu/ds/products/emtf/): [SPUD EMTF Repository](https://ds.iris.edu/spud/emtf/)

### Power Grid Infrastructure

- **Transmission lines (U.S.)** — [HIFLD](https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines-1/about) (as of updating this README file, inaccessible)
- **Substations (OSM)** — [Overpass Turbo](https://overpass-turbo.eu/)


## Related Repositories

- **[`spwio`](https://github.com/denniesbor/spwio/)** — Socio-economic and reliability impact modeling using outputs from this pipeline

---

## Citation

If you use this pipeline in your research, please cite:
```bibtex
@misc{oughton2024physics,
  title={A physics-engineering-economic model coupling approach for estimating the socio-economic impacts of space weather scenarios}, 
  author={Edward Oughton and others},
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
  doi={10.1029/2019SW002329},
  note={Code Author: Greg Lucas (greg.lucas@lasp.colorado.edu)}
}
```

For the IEEE benchmark test case, please cite:
```bibtex
@article{horton_test_2012,
  title = {A {Test} {Case} for the {Calculation} of {Geomagnetically} {Induced} {Currents}},
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

For the data products, please also cite:
```bibtex
@dataset{bor2025geomag,
  author = {Bor, D.},
  title = {Geomag data},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.16994602},
  url = {https://doi.org/10.5281/zenodo.16994602}
}
```

---

## Acknowledgments

This work builds upon data and services provided by: GFZ Potsdam; WDC Kyoto; INTERMAGNET; USGS; NRCan; EarthScope/IRIS EMTF/USArray; HIFLD; OpenStreetMap contributors.

---

## License

[MIT License](LICENSE) — This repository provides research code and derived data products as-is. Downstream socio-economic and reliability modeling is handled in the separate [`spwio`](https://github.com/denniesbor/spwio/) repository.