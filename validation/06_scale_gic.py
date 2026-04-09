"""
Author: Dennies Bor, Ed Oughton
Role: Scale alpha-beta GIC uncertainty predictions to different return periods.
      Uses E-field scaling ratios derived from the statistical return period analysis.
Inputs:
    - data/regression/substations_with_gic_uncertainty.geojson
    - data/statistical_analysis/geomagnetic_data_return_periods.h5
Outputs:
    - data/regression/substations_with_gic_uncertainty_scaled.geojson
"""

import numpy as np
import geopandas as gpd
import h5py
from scipy.spatial import cKDTree

from configs import DATA_DIR, setup_logger

logger = setup_logger(log_file="logs/val_scale_gic.log")

INPUT_FILE = DATA_DIR / "regression" / "substations_with_gic_uncertainty.geojson"
OUTPUT_FILE = (
    DATA_DIR / "regression" / "substations_with_gic_uncertainty_scaled.geojson"
)
H5_PATH = DATA_DIR / "statistical_analysis" / "geomagnetic_data_return_periods.h5"

UNCERTAINTY_COLS = [
    "mean_prediction",
    "std_uncertainty",
    "conf_68_lower",
    "conf_68_upper",
]
RETURN_PERIODS = [75, 100, 125, 150, 175, 200, 225, 250]


def load_return_period_data(h5_path):
    """Load MT coordinates, Gannon E-field, and return period E-fields from HDF5."""
    logger.info(f"Loading return period data from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        mt_coords = f["sites/mt_sites/coordinates"][:]
        gannon_e = f["events/gannon/E"][:] / 1000
        e_fields = {
            period: f[f"predictions/E/{period}_year"][:] for period in RETURN_PERIODS
        }
    logger.info(
        f"Loaded {len(mt_coords)} MT coordinates and {len(RETURN_PERIODS)} return periods"
    )
    return mt_coords, gannon_e, e_fields


def calculate_e_magnitude(e_data):
    """Calculate E-field magnitude from one or two components."""
    return (
        e_data if e_data.ndim == 1 else np.sqrt(e_data[:, 0] ** 2 + e_data[:, 1] ** 2)
    )


def interpolate_to_substations(trafo_gdf, mt_coords, e_values):
    """Nearest-neighbour interpolation of E-field values to substation locations."""
    substation_coords = np.array([[geom.x, geom.y] for geom in trafo_gdf.geometry])
    tree = cKDTree(mt_coords)
    distances, indices = tree.query(substation_coords)
    trafo_gdf = trafo_gdf.copy()
    trafo_gdf["nearest_mt_distance"] = distances
    return trafo_gdf, e_values[indices]


def add_e_field_columns(trafo_gdf, mt_coords, gannon_e, e_fields):
    """Add interpolated E-field magnitude columns for Gannon and all return periods."""
    logger.info("Interpolating E-field columns to substations")
    gannon_mag = calculate_e_magnitude(gannon_e)
    trafo_gdf, gannon_interp = interpolate_to_substations(
        trafo_gdf, mt_coords, gannon_mag
    )
    trafo_gdf["e_gannon_mag"] = gannon_interp

    for period, e_data in e_fields.items():
        e_mag = calculate_e_magnitude(e_data)
        _, e_interp = interpolate_to_substations(trafo_gdf, mt_coords, e_mag)
        trafo_gdf[f"e_{period}yr_mag"] = e_interp

    logger.info(f"Added E-field columns for {len(e_fields)} return periods")
    return trafo_gdf


def scale_gic_uncertainty_to_return_periods(
    trafo_gdf, uncertainty_cols, return_periods
):
    """Scale GIC uncertainty predictions by E-field ratio for each return period."""
    logger.info(
        f"Scaling {len(uncertainty_cols)} uncertainty columns to {len(return_periods)} return periods"
    )
    for col in uncertainty_cols:
        for rp in return_periods:
            scaling_factor = (
                trafo_gdf[f"e_{rp}yr_mag"] / trafo_gdf["e_gannon_mag"]
            ).fillna(1.0)
            trafo_gdf[f"gic_{rp}yr_{col}"] = trafo_gdf[col] * scaling_factor
    logger.info("GIC uncertainty scaling completed")
    return trafo_gdf


def main():
    """Run GIC scaling pipeline from Gannon predictions to return period estimates."""
    logger.info("Starting GIC scaling analysis")

    logger.info(f"Loading substation data from {INPUT_FILE}")
    trafo_gdf = gpd.read_file(INPUT_FILE)
    logger.info(f"Loaded {len(trafo_gdf)} substations")

    mt_coords, gannon_e, e_fields = load_return_period_data(H5_PATH)
    trafo_gdf = add_e_field_columns(trafo_gdf, mt_coords, gannon_e, e_fields)
    trafo_gdf = scale_gic_uncertainty_to_return_periods(
        trafo_gdf, UNCERTAINTY_COLS, RETURN_PERIODS
    )

    logger.info(f"Saving scaled results to {OUTPUT_FILE}")
    trafo_gdf.to_file(OUTPUT_FILE, driver="GeoJSON")
    logger.info("GIC scaling analysis complete")
    return trafo_gdf


if __name__ == "__main__":
    main()
