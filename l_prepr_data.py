"""Module to load preprocessed data from p_gic_files.py and p_areal_int.py
Author: Dennies Bor & Ed Oughton
"""

import warnings
import pickle

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm

from configs import (
    setup_logger,
    get_data_dir,
    USE_ALPHA_BETA_SCENARIO,
    DENNIES_DATA_LOC,
    ALPHA_BETA_SCENARIOS,
    REGULAR_SCENARIOS,
)

warnings.filterwarnings("ignore")
DATA_LOC = get_data_dir()
logger = setup_logger("Load preprocessed data")


# Return periods for GIC calculations
return_periods = np.arange(50, 251, 25)


def load_and_aggregate_tiles():
    """Load all tile files and aggregate economic data by substation"""
    land_mask_dir = DATA_LOC / "land_mask"
    tile_dir = land_mask_dir / "tiles"

    parts = []
    for fp in tile_dir.glob("tile_*.gpkg"):
        gdf = gpd.read_file(fp)
        parts.append(gdf)

    if not parts:
        raise RuntimeError("No tile files found")

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)

    econ_cols = [
        c
        for c in merged.columns
        if c.startswith("GDP_") or c.startswith("EST_") or c == "POP20"
    ]

    aggregated = (
        merged.sort_values("sub_id")
        .groupby("sub_id", as_index=False)
        .agg({**{c: "mean" for c in econ_cols}, "geometry": "first"})
    )

    aggregate_gdf = gpd.GeoDataFrame(
        aggregated, geometry=aggregated.geometry, crs=merged.crs
    )

    return aggregate_gdf


def load_gic_results():
    """Load GIC analysis results based on scenario type"""
    batch_dir = DATA_LOC / "gic_processing_batches"

    if USE_ALPHA_BETA_SCENARIO:
        combined_ds = xr.open_dataset(
            DATA_LOC / "scenario_summary_alpha_beta_uncertainty.nc"
        )
        combined_vuln = pd.read_parquet(
            DATA_LOC / "vulnerable_substations_alpha_beta_uncertainty.parquet"
        )

        logger.info(f"Alpha-beta uncertainty results loaded:")
        logger.info(f"Summary dataset shape: {combined_ds.dims}")
        logger.info(f"Vulnerability data shape: {combined_vuln.shape}")
        logger.info(f"Scenarios: {list(combined_ds.scenario.values)}")
        logger.info(f"GIC realizations: {list(combined_ds.gic_realization.values)}")

    else:
        summary_files = sorted(batch_dir.glob("summary_batch_*.nc"))
        datasets = [xr.open_dataset(f) for f in summary_files]
        combined_ds = xr.concat(datasets, dim="gic_realization")

        vuln_files = sorted(batch_dir.glob("vuln_batch_*.parquet"))
        vuln_dfs = [pd.read_parquet(f) for f in vuln_files]
        combined_vuln = pd.concat(vuln_dfs, ignore_index=True)

        logger.info(f"Multi-file results loaded:")
        logger.info(f"Summary dataset shape: {combined_ds.dims}")
        logger.info(f"Vulnerability data shape: {combined_vuln.shape}")
        logger.info(f"Scenarios: {list(combined_ds.scenario.values)}")
        logger.info(f"GIC realizations: {len(combined_ds.gic_realization)}")

    vuln_table = pa.Table.from_pandas(combined_vuln)

    return combined_ds, combined_vuln, vuln_table


def process_vulnerability_chunks(combined_vuln, chunk_size=50, max_realizations=2000):
    """Process vulnerability data in chunks to compute mean failure probabilities"""
    sum_failure = {}
    count_failure = {}
    coordinates = {}

    for scenario_name in combined_vuln["scenario"].unique():
        logger.info(f"Processing {scenario_name}...")

        scenario_data = combined_vuln[combined_vuln["scenario"] == scenario_name].copy()
        scenario_data["gic_realization"] = pd.to_numeric(
            scenario_data["gic_realization"]
        )

        all_realizations = sorted(scenario_data["gic_realization"].unique())
        max_real = min(len(all_realizations), max_realizations)

        for start_idx in tqdm(
            range(0, max_real, chunk_size), desc=f"Chunks for {scenario_name}"
        ):
            end_idx = min(start_idx + chunk_size, max_real)
            chunk_realizations = all_realizations[start_idx:end_idx]

            chunk_data = scenario_data[
                scenario_data["gic_realization"].isin(chunk_realizations)
            ].copy()

            if scenario_name not in coordinates:
                coords = chunk_data.groupby("sub_id")[["latitude", "longitude"]].first()
                coordinates[scenario_name] = coords

            chunk_grouped = chunk_data.groupby("sub_id")["failure_prob"].agg(
                ["sum", "count"]
            )

            for sub_id, row in chunk_grouped.iterrows():
                key = (sub_id, scenario_name)

                if key not in sum_failure:
                    sum_failure[key] = 0
                    count_failure[key] = 0

                sum_failure[key] += row["sum"]
                count_failure[key] += row["count"]

    results = []
    for (sub_id, scenario), total_sum in sum_failure.items():
        total_count = count_failure[(sub_id, scenario)]
        mean_prob = total_sum / total_count if total_count > 0 else 0

        lat = coordinates[scenario].loc[sub_id, "latitude"]
        lon = coordinates[scenario].loc[sub_id, "longitude"]

        results.append(
            {
                "sub_id": sub_id,
                "scenario": scenario,
                "latitude": lat,
                "longitude": lon,
                "mean_failure_prob": mean_prob,
                "n_realizations": total_count,
            }
        )

    return pd.DataFrame(results)


def read_pickle(file_path):
    """Read data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_and_process_gic_data(df_lines):
    """Load and process geomagnetically induced current (GIC) data from HDF5 files."""

    results_path = (
        DENNIES_DATA_LOC / "statistical_analysis" / "geomagnetic_data_return_periods.h5"
    )

    logger.info("Loading and processing GIC data...")

    with h5py.File(DATA_LOC / results_path, "r") as f:

        logger.info("Reading geomagnetic data from geomagnetic_data.h5")
        mt_names = f["sites/mt_sites/names"][:]
        mt_coords = f["sites/mt_sites/coordinates"][:]

        line_ids = f["sites/transmission_lines/line_ids"][:]
        line_ids_str = [
            id.decode("utf-8") if isinstance(id, bytes) else str(id) for id in line_ids
        ]

        halloween_e = f["events/halloween/E"][:] / 1000
        halloween_b = f["events/halloween/B"][:]
        halloween_v = f["events/halloween/V"][:]

        st_patricks_e = f["events/st_patricks/E"][:] / 1000
        st_patricks_b = f["events/st_patricks/B"][:]
        st_patricks_v = f["events/st_patricks/V"][:]

        gannon_e = f["events/gannon/E"][:] / 1000
        gannon_b = f["events/gannon/B"][:]
        gannon_v = f["events/gannon/V"][:]

        e_fields, b_fields, v_fields = {}, {}, {}

        for period in return_periods:
            e_fields[period] = f[f"predictions/E/{period}_year"][:]
            b_fields[period] = f[f"predictions/B/{period}_year"][:]
            v_fields[period] = f[f"predictions/V/{period}_year"][:]

    v_cols = ["V_halloween", "V_st_patricks", "V_gannon"] + [
        f"V_{period}" for period in return_periods
    ]

    id_to_index = {id: i for i, id in enumerate(line_ids_str)}
    indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])
    mask = indices != -1

    logger.info(f"Mask coverage: {mask.sum()}/{len(mask)}")

    df_lines.loc[mask, "V_halloween"] = halloween_v[indices[mask]]
    df_lines.loc[mask, "V_st_patricks"] = st_patricks_v[indices[mask]]
    df_lines.loc[mask, "V_gannon"] = gannon_v[indices[mask]]

    for period in return_periods:
        df_lines.loc[mask, f"V_{period}"] = v_fields[period][indices[mask]]

    df_lines[v_cols] = df_lines[v_cols].fillna(0)

    logger.info("GIC data loaded and processed successfully.")

    return (
        df_lines,
        mt_coords,
        mt_names,
        e_fields,
        b_fields,
        v_fields,
        gannon_e,
        v_cols,
    )


def load_network_data():
    """Load transmission network data"""
    tl_path = DENNIES_DATA_LOC / "admittance_matrix" / "transmission_lines.csv"
    df_lines = pd.read_csv(tl_path)
    df_lines["geometry"] = df_lines["geometry"].apply(wkt.loads)
    df_lines = gpd.GeoDataFrame(df_lines, geometry="geometry", crs="EPSG:4326")

    substation_path = DENNIES_DATA_LOC / "admittance_matrix" / "substation_info.csv"
    df_substations = pd.read_csv(substation_path)

    # pickle subs
    GRID_DATA = DENNIES_DATA_LOC / "grid_processed"
    SUB_DF_PATH = GRID_DATA / "ss_df.pkl"

    ss_gdf_pkl = read_pickle(SUB_DF_PATH)

    return df_lines, df_substations, ss_gdf_pkl


def find_vulnerable_substations(
    mean_vuln_data, years_of_interest=[100, 150, 200, 250], threshold=0.5
):
    """Find substations vulnerable across multiple scenarios"""
    if USE_ALPHA_BETA_SCENARIO:
        target_scenarios = [
            sc
            for sc in ALPHA_BETA_SCENARIOS
            if any(str(y) in sc for y in years_of_interest)
        ]
    else:
        target_scenarios = [
            sc
            for sc in REGULAR_SCENARIOS
            if any(str(y) in sc for y in years_of_interest)
        ]

    filtered_data = mean_vuln_data[
        (mean_vuln_data["scenario"].isin(target_scenarios))
        & (mean_vuln_data["mean_failure_prob"] > threshold)
    ]

    scenario_count = filtered_data.groupby("sub_id")["scenario"].nunique()
    common_vulnerable = scenario_count[scenario_count == len(target_scenarios)].index

    vulnerability_matrix = mean_vuln_data[
        (mean_vuln_data["sub_id"].isin(common_vulnerable))
        & (mean_vuln_data["scenario"].isin(target_scenarios))
    ].pivot_table(index="sub_id", columns="scenario", values="mean_failure_prob")

    return common_vulnerable, vulnerability_matrix, target_scenarios


if __name__ == "__main__":
    aggregate_gdf = load_and_aggregate_tiles()
    combined_ds, combined_vuln, vuln_table = load_gic_results()
    mean_vuln_all = process_vulnerability_chunks(
        combined_vuln, chunk_size=50, max_realizations=2000
    )
    df_lines, df_substations = load_network_data()
    common_vulnerable, vulnerability_matrix, target_scenarios = (
        find_vulnerable_substations(mean_vuln_all)
    )
