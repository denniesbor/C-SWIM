"""
Configuration for reproducible_mapping GIC validation.
Author: Dennies
"""
from pathlib import Path
from configs import DATA_DIR, FIGURES_DIR, LUCY_DATA_LOC, setup_logger, setup_matplotlib

# Create a new figures dir for reproducible mapping outputs
FIGURES_DIR = FIGURES_DIR / "rep_mapping"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible mapping data paths
REP_DATA_DIR = DATA_DIR / "rep_data"
TVA_DIR      = REP_DATA_DIR / "tva"
UIUC_DIR     = REP_DATA_DIR / "uiuc150"
TVA_BOUNDARY = REP_DATA_DIR / "tva_boundaries" / "TVA_Power_Service_Area_clean.geojson"
UIUC_XLSX    = UIUC_DIR / "uiuc-150bus[1].xlsx"

# Depends on Lucy Wilkerson's data path
GANNON_DS    = DATA_DIR / "storm_maxes" / "ds_gannon.nc"
DEVICES_NC   = LUCY_DATA_LOC / "tva_gic.nc"
HIFLD_PATH   = DATA_DIR / "Electric__Power_Transmission_Lines" / "Electric__Power_Transmission_Lines.shp"
OSM_SUB_PATH = DATA_DIR / "substation_locations" / "us_substations_full.geojson"
GRID_MAPPING = DATA_DIR / "grid_mapping.csv"

# GIC solver constants
LINE_RESISTANCE = {
    765: 0.010, 500: 0.0141, 345: 0.0283,
    230: 0.0500, 161: 0.0800, 138: 0.0900,
}
TRAFO_WINDING_R = {
    "GY-GY":   {"pri": 0.04,  "sec": 0.06},
    "Auto":    {"pri": 0.04,  "sec": 0.06},
    "GY-GY-D": {"pri": 0.20,  "sec": 0.10},
    "GY-D":    {"pri": 0.05,  "sec": float("inf")},
    "GSU":     {"pri": 0.15,  "sec": float("inf")},
}
RG_REAL      = 0.2
RG_SYNTHETIC = 10.0
POOL_GEN     = ["GSU", "GY-D"]
POOL_TRANS   = ["GY-GY", "Auto", "GY-GY-D"]

VOLTAGE_THRESHOLD    = 229
SNAP_DISTANCE        = 1500
KEEP_SYNTHETIC_ABOVE = 230
CRS_PROJ             = "EPSG:3857"
CRS_GEO              = "EPSG:4326"

for d in (TVA_DIR, UIUC_DIR, REP_DATA_DIR / "tva_boundaries"):
    d.mkdir(parents=True, exist_ok=True)