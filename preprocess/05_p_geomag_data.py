"""
Process time-series geomagnetic data from multiple sources.

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025
"""

import os
import warnings
from pathlib import Path
from multiprocessing import Pool
from itertools import chain

import xarray as xr
import numpy as np
import pandas as pd
from scipy import signal
import bezpy.mag

from configs import setup_logger, get_data_dir

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/p_geomag.log")


def process_magnetic_files(file_path, baseline=None, is_processed=False):
    """Process magnetic field data from file.
    
    If baseline dict {X, Y, Z} is provided, subtract it.
    Otherwise fall back to per-file median (original behaviour).
    """
    if is_processed:
        data = pd.read_csv(file_path, index_col=0)
        data.index = pd.to_datetime(data.index)
        Latitude = data["Latitude"].iloc[0]
        Longitude = data["Longitude"].iloc[0]
        iaga_code = Path(file_path).parent.name
    else:
        data, headers = bezpy.mag.read_iaga(file_path, return_header=True)
        data.index.name = "Timestamp"
        Latitude = float(headers["geodetic latitude"])
        Longitude = float(headers["geodetic longitude"]) - 360
        iaga_code = headers["iaga code"]

    for component in ["X", "Y", "Z"]:
        s = data[component].interpolate(method="nearest").bfill().ffill()
        if baseline is not None:
            data[component] = s - baseline[component]
        else:
            data[component] = s - np.nanmedian(s.to_numpy())

    data["Latitude"] = Latitude
    data["Longitude"] = Longitude

    ds = xr.Dataset.from_dataframe(data)
    ds.attrs.update({"Latitude": Latitude, "Longitude": Longitude, "Name": iaga_code})

    return ds, data


def _load_day_file(dir_path, obs_name, date):
    """
    Load a single day file for an observatory.
    Uses glob to handle filename variations (min.min, vmin.min, dmin.min etc.)
    Returns DataFrame or None.
    """
    date_str = date.strftime("%Y%m%d")
    dir_path = Path(dir_path)

    # Search for any matching .min or .csv file for this date
    candidates = list(dir_path.glob(f"{obs_name}{date_str}*.min"))
    candidates += list(dir_path.glob(f"{obs_name}{date_str}*.csv"))

    for fpath in candidates:
        try:
            is_csv = str(fpath).endswith(".csv")
            if is_csv:
                df = pd.read_csv(str(fpath), index_col=0)
                df.index = pd.to_datetime(df.index)
            else:
                df, _ = bezpy.mag.read_iaga(str(fpath), return_header=True)
            for comp in ["X", "Y", "Z"]:
                df[comp] = df[comp].interpolate(method="nearest").bfill().ffill()
            return df, is_csv
        except Exception as e:
            logger.error(f"Error loading {fpath}: {e}")

    return None, None


def compute_storm_baseline(dir_path, storm_start, storm_end, pad_hours=12):
    """
    Load all files in the extended storm window and compute per-component median.
    Extended window = storm_start - pad_hours to storm_end + pad_hours.
    Uses glob to handle filename variations.
    Returns dict {X, Y, Z} or None if insufficient data.
    """
    extended_start = storm_start - pd.Timedelta(hours=pad_hours)
    extended_end   = storm_end   + pd.Timedelta(hours=pad_hours)

    dates = pd.date_range(
        extended_start.normalize(),
        extended_end.normalize(),
        freq="D"
    )

    obs_name = os.path.basename(dir_path).lower()
    all_frames = []

    for date in dates:
        df, _ = _load_day_file(dir_path, obs_name, date)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        logger.warning(f"[{obs_name}] No files found for baseline computation")
        return None

    combined = pd.concat(all_frames)
    combined = combined[
        (combined.index >= extended_start) &
        (combined.index <= extended_end)
    ]

    if len(combined) < 30:
        logger.warning(f"[{obs_name}] Less than 30 min of data in storm window — skipping baseline")
        return None

    logger.info(
        f"[{obs_name}] Baseline computed from {len(combined)} samples "
        f"({extended_start} to {extended_end}): "
        f"X={np.nanmedian(combined['X'].to_numpy()):.2f}, "
        f"Y={np.nanmedian(combined['Y'].to_numpy()):.2f}, "
        f"Z={np.nanmedian(combined['Z'].to_numpy()):.2f}"
    )

    return {
        "X": np.nanmedian(combined["X"].to_numpy()),
        "Y": np.nanmedian(combined["Y"].to_numpy()),
        "Z": np.nanmedian(combined["Z"].to_numpy()),
    }


def process_directory(dir_path, storm_df=None):
    """Process all magnetic field files in a directory.
    
    If storm_df is provided, files belonging to a storm window use
    a single extended-window baseline. Otherwise falls back to
    original per-file median (original behaviour preserved).
    """
    logger.info(f"Processing {dir_path}")
    obs_name = os.path.basename(dir_path).lower()

    # Pre-compute baselines for all storms — avoids recomputing per file
    storm_baselines = {}
    if storm_df is not None:
        for idx, row in storm_df.iterrows():
            baseline = compute_storm_baseline(
                dir_path, row["Start"], row["End"], pad_hours=12
            )
            storm_baselines[idx] = baseline

    def process_file(filename):
        if not filename.endswith((".min", ".csv")):
            return None, None
        is_processed = filename.endswith(".csv")
        file_path = os.path.join(dir_path, filename)

        # Determine baseline for this file
        baseline = None
        if storm_df is not None:
            try:
                # Extract date from filename using obs_name length as offset
                # e.g. ott20240511vmin.min -> date part starts after obs code
                name_no_ext = Path(filename).stem  # e.g. ott20240511vmin
                # find 8-digit date string
                import re
                match = re.search(r"(\d{8})", name_no_ext)
                if match:
                    file_date = pd.Timestamp(match.group(1))
                    storm_match = storm_df[
                        (storm_df["Start"] - pd.Timedelta(hours=12) <= file_date) &
                        (storm_df["End"]   + pd.Timedelta(hours=12) >= file_date)
                    ]
                    if not storm_match.empty:
                        idx = storm_match.index[0]
                        baseline = storm_baselines.get(idx)
                        if baseline is None:
                            logger.warning(f"No baseline for {filename} storm idx={idx}")
            except Exception as e:
                logger.warning(f"Could not determine baseline for {filename}: {e}")
                baseline = None

        try:
            ds, result = process_magnetic_files(
                file_path, baseline=baseline, is_processed=is_processed
            )
            result["site_id"] = obs_name
            return ds, result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None, None

    results = list(
        filter(
            lambda x: x[0] is not None,
            map(process_file, sorted(os.listdir(dir_path)))
        )
    )

    datasets, file_results = zip(*results) if results else ([], [])
    if datasets:
        return (
            obs_name,
            xr.concat(datasets, dim="Timestamp"),
            list(file_results),
        )

    logger.warning(f"No valid datasets found in {dir_path}")
    return obs_name, None, None


def process_all_directories(geomag_folder, usgs_obs, nrcan_obs, storm_df=None):
    """Process all geomagnetic observation directories."""
    all_dirs = [
        os.path.join(root, d)
        for root, dirs, _ in os.walk(geomag_folder)
        for d in dirs
        if d.upper() in usgs_obs + nrcan_obs
    ]

    from functools import partial
    process_dir_fn = partial(process_directory, storm_df=storm_df)

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_dir_fn, all_dirs)

    obsv_xarrays = {
        dir: dataset
        for dir, dataset, _ in results
        if dataset is not None and not np.isnan(dataset.X.min().values)
    }

    return results, obsv_xarrays


def combine_results(results, obsv_xarrays):
    """Combine processed results into a single list."""
    return list(
        chain.from_iterable(
            result_list
            for _, _, result_list in results
            if result_list and result_list[0].site_id.iloc[0] in obsv_xarrays
        )
    )

def get_mode(series):
    """Get the mode of a series."""
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return np.nan
    value_counts = series_clean.value_counts()
    if len(value_counts) > 0:
        return value_counts.index[0]
    return np.nan


def prepare_dataset(combined_df):
    """Prepare dataset for analysis."""
    combined_df = combined_df.sort_values(by=["site_id", "Timestamp"])
    unique_lat_lon = (
        combined_df.groupby("site_id")
        .agg({"Longitude": get_mode, "Latitude": get_mode})
        .reset_index()
    )
    time_steps = sorted(set(pd.to_datetime(combined_df.index.unique()).to_list()))
    dB = np.stack(
        [
            combined_df.pivot_table(
                index=combined_df.index, columns="site_id", values=col, aggfunc="first"
            )
            .reindex(time_steps)
            .values
            for col in ["X", "Y", "Z"]
        ],
        axis=-1,
    )
    ds = xr.Dataset(
        coords={
            "longitude": ("site", unique_lat_lon["Longitude"].values),
            "latitude":  ("site", unique_lat_lon["Latitude"].values),
            "site":      unique_lat_lon["site_id"].values,
            "component": ["X", "Y", "Z"],
            "time":      time_steps,
        },
        data_vars={"B": (("time", "site", "component"), dB)},
    )
    return ds


def run_function(geomag_folder, usgs_obs, nrcan_obs, storm_df=None):
    """Process all geomagnetic data and save results."""
    path = geomag_folder / "combined_geomag_df.csv"

    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Removed existing file: {path}")

    results, obsv_xarrays = process_all_directories(
        geomag_folder, usgs_obs, nrcan_obs, storm_df=storm_df
    )
    logger.info(f"Processed {len(obsv_xarrays)} valid observatories")

    combined_df = pd.concat(combine_results(results, obsv_xarrays))
    combined_df.to_csv(path)
    logger.info(f"Saved combined data to: {path}")

    ds = prepare_dataset(combined_df)
    netcdf_path = geomag_folder / "processed_geomag_data.nc"
    ds.to_netcdf(netcdf_path, format="NETCDF4", engine="netcdf4")
    logger.info(f"Saved NetCDF data to: {netcdf_path}")


if __name__ == "__main__":

    geomag_folder = DATA_LOC / "geomag_data"

    usgs_obs = [
        obs.upper()
        for obs in [
            "bou", "brw", "bsl", "cmo", "ded", "frd", "frn",
            "gua", "hon", "new", "shu", "sit", "sjg", "tuc",
        ]
    ]
    nrcan_obs = [
        "ALE", "BLC", "BRD", "CBB", "FCC", "IQA",
        "MEA", "OTT", "RES", "STJ", "VIC", "YKC",
    ]

    storm_df = pd.read_csv(DATA_LOC / "kp_ap_indices" / "storm_periods.csv")
    storm_df["Start"] = pd.to_datetime(storm_df["Start"])
    storm_df["End"]   = pd.to_datetime(storm_df["End"])

    run_function(geomag_folder, usgs_obs, nrcan_obs, storm_df=storm_df)