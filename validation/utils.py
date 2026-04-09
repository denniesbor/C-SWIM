"""
Author: Dennies Bor
Role: Utility functions for GIC and magnetic field validation analysis.
      Provides data loading, preprocessing, matching, and metric computation.
Inputs:
    - NERC/TVA GIC measurement files
    - NERC/TVA magnetometer data
    - Simulated GIC cache and transformer GIC results
    - SWERVE site info
Outputs:
    - Cached NetCDF datasets for NERC/TVA GIC and magnetometer data
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch, coherence
from scipy import integrate
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

from configs import DATA_DIR as DENNIES_DATA_LOC

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def haversine_dist(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km between two points."""
    R = 6371
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def get_matches_df(gdf_mean_sim, gdf_monitors, threshold=0.5, op="nerc"):
    """Match simulated substations to GIC monitors within a distance threshold."""
    gdf_mean_sim_ = gdf_mean_sim.drop_duplicates(subset="sub_id").reset_index()
    sim_lats = gdf_mean_sim_["latitude"].to_numpy()
    sim_lons = gdf_mean_sim_["longitude"].to_numpy()
    mon_lats = gdf_monitors["latitude"].to_numpy()
    mon_lons = gdf_monitors["longitude"].to_numpy()

    dists = haversine_dist(
        sim_lats[:, None], sim_lons[:, None], mon_lats[None, :], mon_lons[None, :]
    )
    sim_idx, mon_idx = np.nonzero(dists <= threshold)

    df = pd.DataFrame(
        {
            "substation": gdf_mean_sim_.iloc[sim_idx]["sub_id"].to_numpy(),
            "sub_lat": gdf_mean_sim_.iloc[sim_idx]["latitude"].to_numpy(),
            "sub_lon": gdf_mean_sim_.iloc[sim_idx]["longitude"].to_numpy(),
            "monitor_name": gdf_monitors.iloc[mon_idx]["device"].to_numpy(),
            "monitor_lat": gdf_monitors.iloc[mon_idx]["latitude"].to_numpy(),
            "monitor_lon": gdf_monitors.iloc[mon_idx]["longitude"].to_numpy(),
        }
    )
    return df.drop_duplicates(["substation", "monitor_name"])


def preprocess_magnetic_field(ds):
    """Fill missing values and subtract storm-window median baseline per component."""
    for comp in ["Bx", "By", "Bz"]:
        if comp in ds:
            ds[comp] = ds[comp].bfill(dim="time").interpolate_na(dim="time")
            ds[comp] = ds[comp].where(np.isfinite(ds[comp]), other=np.nan)
            ds[comp] = (
                ds[comp].ffill(dim="time").interpolate_na(dim="time", method="linear")
            )
            baseline = ds[comp].median(dim="time", skipna=True)
            ds[comp] = ds[comp] - baseline
    return ds


def get_close_mag_sites(ds_operator, ds_sim, threshold=100):
    """For each magnetometer station find the closest MT site within threshold km."""
    ds_op_lat = ds_operator.latitude.values
    ds_op_lon = ds_operator.longitude.values
    ds_sim_lat = ds_sim.latitude.values
    ds_sim_lon = ds_sim.longitude.values

    dists = haversine_dist(
        ds_op_lat[:, None], ds_op_lon[:, None], ds_sim_lat[None, :], ds_sim_lon[None, :]
    )

    closest_mt_sites, closest_distances = [], []
    for row in dists:
        valid = np.where(row <= threshold)[0]
        if valid.size:
            idx = valid[np.argmin(row[valid])]
            closest_mt_sites.append(idx)
            closest_distances.append(row[idx])
        else:
            closest_mt_sites.append(np.nan)
            closest_distances.append(np.nan)

    ds_operator = ds_operator.assign_coords(
        nearest_mt_site=("device", closest_mt_sites),
        nearest_distance=("device", closest_distances),
    )
    return ds_operator.sortby("nearest_distance")


def read_gnd_gic(cache_file):
    """Load simulated GIC cache from npz file."""
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["data_array"],
            data["peak_times"],
            data["median_values"],
            data["mean_values"],
            data["uncertainty_arr"],
            data["substation_names"],
        )


def is_nan_or_empty(entry):
    """Check if an entry is NaN or an empty array/list."""
    return entry is np.nan or (
        isinstance(entry, (list, np.ndarray)) and len(entry) == 0
    )


def find_close_matches(
    trafo_gic_gdf, ds_gic_meas, site_ids=None, threshold=2, nerc=True
):
    """Match simulated transformer substations to measured GIC devices within threshold km."""
    trafo_unique = trafo_gic_gdf.drop_duplicates(subset="sub_id").reset_index(drop=True)
    trafos = trafo_unique.sub_id.values
    ds_other = ds_gic_meas.copy()

    if nerc:
        site_ids_clean = [s for s in site_ids if s in ds_other.device.values]
        ds_other = ds_other.sel(device=site_ids_clean)

    dists = haversine_dist(
        trafo_unique.latitude.values[:, None],
        trafo_unique.longitude.values[:, None],
        ds_other.latitude.values[None, :],
        ds_other.longitude.values[None, :],
    )

    match_ids, match_dists = [], []
    for row in dists:
        valid = np.where(row <= threshold)[0]
        if valid.size:
            match_ids.append(ds_other.device.values[valid])
            match_dists.append(row[valid])
        else:
            match_ids.append(np.nan)
            match_dists.append(np.nan)

    valid_ids = np.array([not is_nan_or_empty(s) for s in match_ids])
    valid_match_ids = np.array(match_ids, dtype=object)[valid_ids]
    valid_dists = np.array(match_dists, dtype=object)[valid_ids]
    valid_substations = trafos[valid_ids]

    return valid_match_ids, valid_dists, valid_substations, valid_dists, trafo_unique


def get_pred_metrics(observed, predicted):
    """Return prediction efficiency and Pearson correlation."""
    mse = np.mean((predicted - observed) ** 2)
    var_obs = np.var(observed)
    pe = 1 - (mse / var_obs)
    pr_corr = np.corrcoef(observed, predicted)[0, 1]
    return pe, pr_corr


def compute_coherence_and_psd(
    gic_meas, gic_sim, fs=1 / 60, nperseg=256, noverlap=128, window="hann"
):
    """Compute coherence and Welch PSD between measured and simulated GIC signals."""
    f_coh, Cxy = coherence(
        gic_meas, gic_sim, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    f_welch, S_meas = welch(
        gic_meas, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    _, S_sim = welch(gic_sim, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    return f_coh, Cxy, f_welch, S_meas, S_sim


def load_trafo_gic_data(dennies_data_loc):
    """Load transformer GIC simulation results as a GeoDataFrame."""
    trafo_gic_df = pd.read_csv(dennies_data_loc / "gic" / "winding_gic_rand_0.csv")
    geometry = [
        Point(xy) for xy in zip(trafo_gic_df["longitude"], trafo_gic_df["latitude"])
    ]
    return gpd.GeoDataFrame(trafo_gic_df, geometry=geometry, crs="EPSG:4326")


def load_nerc_gic_monitors(nerc_gic_path):
    """Load NERC GIC monitor metadata as a GeoDataFrame."""
    nerc_gic_csv_files = sorted(list(nerc_gic_path.glob("*.csv")))
    nerc_gic_monitors_df = pd.read_csv(nerc_gic_csv_files[-1])
    nerc_gic_monitors_df.rename(
        columns={
            " Latitude": "latitude",
            " Longitude": "longitude",
            "Device ID": "device",
        },
        inplace=True,
    )
    nerc_gic_monitors_df["longitude"] = nerc_gic_monitors_df["longitude"].apply(
        lambda x: -x if x > 0 and x > 60 else x
    )
    geometry = [
        Point(xy)
        for xy in zip(
            nerc_gic_monitors_df["longitude"], nerc_gic_monitors_df["latitude"]
        )
    ]
    return gpd.GeoDataFrame(nerc_gic_monitors_df, geometry=geometry, crs="EPSG:4326")


def load_tva_gic_metadata(tva_gic_meas_path):
    """Load TVA GIC monitor metadata."""
    df = pd.read_csv(tva_gic_meas_path / "GIC_monitors.dat")
    df.rename(
        columns={
            "Node Name": "device",
            "Type": "type",
            "Latitude": "latitude",
            "Longitude": "longitude",
        },
        inplace=True,
    )
    return df


def load_or_create_nerc_gic_dataset(nerc_gic_path, lucy_data_loc, gdf_monitors_nerc):
    """Load cached NERC GIC dataset or build it from CSV files."""
    cache_path = lucy_data_loc / "nerc_gic.nc"
    if os.path.exists(cache_path):
        return xr.open_dataset(cache_path)

    nerc_gic_csv_files = sorted(list(nerc_gic_path.glob("*.csv")))
    device_data, timestamps_set = {}, set()

    for file in nerc_gic_csv_files:
        try:
            df = pd.read_csv(file)
            df["SampleDateTime"] = pd.to_datetime(
                df["SampleDateTime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
            )
            for device_id, group in df.groupby("GICDeviceID"):
                if device_id not in device_data:
                    device_data[device_id] = {}
                timestamps_set.update(group["SampleDateTime"])
                device_data[device_id].update(
                    dict(zip(group["SampleDateTime"], group["GICMeasured"]))
                )
        except Exception as e:
            print(f"Error processing {file}: {e}")

    timestamps = sorted(timestamps_set)
    device_ids = sorted(device_data.keys())
    data_array = np.full((len(timestamps), len(device_ids)), np.nan)

    for j, device_id in enumerate(device_ids):
        for i, timestamp in enumerate(timestamps):
            if timestamp in device_data[device_id]:
                data_array[i, j] = device_data[device_id][timestamp]

    ds_gic_nerc = xr.Dataset(
        data_vars={"gic": (["time", "device"], data_array)},
        coords={"time": timestamps, "device": device_ids},
    )

    gic_monitors_dict = gdf_monitors_nerc.set_index("device").to_dict()
    for col in [
        "latitude",
        "longitude",
        " Installation Type",
        " Connection",
        " Minimum Value in Measurement Range",
    ]:
        ds_gic_nerc = ds_gic_nerc.assign_coords(
            {
                col.lower().replace(" ", "_"): (
                    "device",
                    [gic_monitors_dict[col].get(d, np.nan) for d in device_ids],
                )
            }
        )

    ds_gic_nerc.to_netcdf(cache_path)
    return ds_gic_nerc


def load_or_create_tva_gic_dataset(
    tva_gic_meas_path, lucy_data_loc, tva_name_map, tva_gic_meas_metadat
):
    """Load cached TVA GIC dataset or build it from CSV files."""
    cache_path = lucy_data_loc / "tva_gic.nc"
    if os.path.exists(cache_path):
        return xr.open_dataset(cache_path)

    device_data, timestamps_set = {}, set()

    for fpath in tva_gic_meas_path.glob("*.csv"):
        file_stem = fpath.stem.split("_")[0].split("-")[1]
        name = tva_name_map.get(file_stem)
        if name is None:
            continue
        df = pd.read_csv(fpath, names=["SampleDateTime", "GICMeasured"])
        df["GICDeviceID"] = name
        df["SampleDateTime"] = pd.to_datetime(df["SampleDateTime"], errors="coerce")
        for device_id, group in df.groupby("GICDeviceID"):
            if device_id not in device_data:
                device_data[device_id] = {}
            timestamps_set.update(group["SampleDateTime"])
            device_data[device_id].update(
                dict(zip(group["SampleDateTime"], group["GICMeasured"]))
            )

    timestamps = sorted(timestamps_set)
    device_ids = sorted(device_data.keys())
    data_array = np.full((len(timestamps), len(device_ids)), np.nan)

    for j, device_id in enumerate(device_ids):
        for i, timestamp in enumerate(timestamps):
            if timestamp in device_data[device_id]:
                data_array[i, j] = device_data[device_id][timestamp]

    ds_gic_tva = xr.Dataset(
        data_vars={"gic": (["time", "device"], data_array)},
        coords={"time": timestamps, "device": device_ids},
    )

    gic_monitors_dict = tva_gic_meas_metadat.set_index("device").to_dict()
    for col in ["latitude", "longitude", "type"]:
        ds_gic_tva = ds_gic_tva.assign_coords(
            {
                col: (
                    "device",
                    [gic_monitors_dict[col].get(d, np.nan) for d in device_ids],
                )
            }
        )

    ds_gic_tva.to_netcdf(cache_path)
    return ds_gic_tva


def get_filtered_site_ids(swerve_dir, default_site_ids):
    """Load valid site IDs from SWERVE info, falling back to defaults."""
    try:
        info_df = pd.read_csv(swerve_dir / "info" / "info.csv")
        site_ids = info_df[info_df.error.isna()].site_id.unique()
        return [int(s) for s in site_ids]
    except Exception:
        return default_site_ids


def prepare_time_window(peak_times, start_time, end_time):
    """Return duration mask, indices, and trimmed peak times for the given window."""
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    duration_mask = (peak_times >= start_time) & (peak_times <= end_time)
    duration = np.where(duration_mask)[0]
    return duration_mask, duration, peak_times[duration_mask]


def filter_and_validate_data(
    valid_substations,
    valid_match_ids,
    substation_names,
    mean_values,
    median_values,
    uncertainty_arr,
    ds,
    duration,
    peak_times_trimmed,
    savgol_window=5,
    savgol_polyorder=3,
):
    """Extract raw measured and simulated GIC for each matched substation.

    No smoothing applied — raw signals used to preserve spectral content.
    savgol parameters retained in signature for backward compatibility.
    """
    sub_to_idx = {
        sub: np.where(substation_names == sub)[0][0]
        for sub in substation_names
        if sub in set(valid_substations)
    }

    valid_selected_indices = []
    filtered_measured_data = {}
    filtered_simulated_data = {}

    for idx in range(len(valid_substations)):
        sub = valid_substations[idx]
        if sub not in sub_to_idx:
            continue

        sim_idx = sub_to_idx[sub]
        close_site_list = valid_match_ids[idx]
        median_sim = median_values[sim_idx, 1:][duration]

        site_data = {}
        for close_site in close_site_list:
            site_data[close_site] = ds.gic.sel(
                device=close_site, time=peak_times_trimmed
            ).values

        valid_selected_indices.append(idx)
        filtered_measured_data[idx] = site_data
        filtered_simulated_data[idx] = median_sim

    return valid_selected_indices, filtered_measured_data, filtered_simulated_data


def prepare_validation_data(
    trafo_gic_gdf,
    ds_operator,
    site_ids,
    ground_truth_data,
    start_time="2024-05-10T12:00:00",
    end_time="2024-05-12T12:00:00",
    threshold=2.0,
    savgol_window=5,
    savgol_polyorder=3,
    nerc=True,
):
    """Match, time-window, and prepare all data needed for validation plotting."""
    (
        data_array,
        peak_times,
        median_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    ) = ground_truth_data

    start_dt = np.datetime64(start_time)
    end_dt = np.datetime64(end_time)
    duration_mask = (peak_times >= start_dt) & (peak_times <= end_dt)
    duration = np.where(duration_mask)[0]
    peak_times_trimmed = peak_times[duration_mask]

    valid_match_ids, valid_dists, valid_substations, _, trafo_unique = (
        find_close_matches(
            trafo_gic_gdf,
            ds_operator,
            site_ids if nerc else None,
            threshold=threshold,
            nerc=nerc,
        )
    )

    valid_selected_indices, filtered_measured_data, filtered_simulated_data = (
        filter_and_validate_data(
            valid_substations,
            valid_match_ids,
            substation_names,
            mean_values,
            median_values,
            uncertainty_arr,
            ds_operator,
            duration,
            peak_times_trimmed,
            savgol_window,
            savgol_polyorder,
        )
    )

    return {
        "selected_indices": valid_selected_indices,
        "valid_match_ids": valid_match_ids,
        "valid_substations": valid_substations,
        "trafo_unique": trafo_unique,
        "filtered_measured_data": filtered_measured_data,
        "filtered_simulated_data": filtered_simulated_data,
        "ds_operator": ds_operator,
        "peak_times_trimmed": peak_times_trimmed,
    }


def load_nerc_magnetometer_data():
    """Load and preprocess NERC magnetometer data, caching to NetCDF."""
    from configs import LUCY_DATA_LOC, nerc_mag_folder

    cache_file = LUCY_DATA_LOC / "nerc_magnetometer_data.nc"
    if os.path.exists(cache_file):
        return xr.open_dataset(cache_file)

    nerc_mag_files = list(nerc_mag_folder.glob("2024*.csv"))
    nerc_mag_locs_df = pd.read_csv(nerc_mag_folder / "magnetometers.csv")
    nerc_mag_locs_df.rename(
        columns={
            "Device ID": "device_id",
            " Latitude": "latitude",
            " Longitude": "longitude",
            " Orientation": "orientation",
        },
        inplace=True,
    )
    nerc_mag_locs_df = nerc_mag_locs_df[
        nerc_mag_locs_df["orientation"] == "1 - Geographic"
    ].copy()
    nerc_mag_locs_df["longitude"] = nerc_mag_locs_df["longitude"].apply(
        lambda x: -x if x > 0 and x > 60 else x
    )

    nerc_mag_df = pd.concat([pd.read_csv(f) for f in nerc_mag_files], ignore_index=True)
    nerc_mag_df.rename(
        columns={
            "MagnetometerDeviceID": "device_id",
            "SampleDateTime": "time",
            "GeoBfieldMeasureNorth": "Bx",
            "GeoBfieldMeasureEast": "By",
            "GeoBfieldMeasureVertical": "Bz",
        },
        inplace=True,
    )
    nerc_mag_df["time"] = pd.to_datetime(
        nerc_mag_df["time"], format="%m/%d/%Y %I:%M:%S %p"
    )
    nerc_mag_df = nerc_mag_df.merge(nerc_mag_locs_df, on="device_id", how="inner")

    pivot_kwargs = {"index": "time", "columns": "device_id"}
    ds_mag_nerc = xr.Dataset(
        {
            "Bx": (
                ["time", "device"],
                nerc_mag_df.pivot(**pivot_kwargs, values="Bx").values,
            ),
            "By": (
                ["time", "device"],
                nerc_mag_df.pivot(**pivot_kwargs, values="By").values,
            ),
            "Bz": (
                ["time", "device"],
                nerc_mag_df.pivot(**pivot_kwargs, values="Bz").values,
            ),
        },
        coords={
            "time": nerc_mag_df["time"].unique(),
            "device": nerc_mag_df["device_id"].unique(),
            "latitude": (
                ["device"],
                nerc_mag_df.drop_duplicates("device_id")
                .set_index("device_id")["latitude"]
                .values,
            ),
            "longitude": (
                ["device"],
                nerc_mag_df.drop_duplicates("device_id")
                .set_index("device_id")["longitude"]
                .values,
            ),
            "orientation": (
                ["device"],
                nerc_mag_df.drop_duplicates("device_id")
                .set_index("device_id")["orientation"]
                .values,
            ),
        },
    )

    ds_mag_nerc = ds_mag_nerc.sortby("time")
    ds_mag_nerc = preprocess_magnetic_field(ds_mag_nerc)
    ds_mag_nerc.to_netcdf(cache_file)
    return ds_mag_nerc


def load_tva_magnetometer_data():
    """Load and preprocess TVA magnetometer data, caching to NetCDF."""
    from configs import LUCY_DATA_LOC, tva_mag

    cache_file = LUCY_DATA_LOC / "tva_magnetometer_data.nc"
    if os.path.exists(cache_file):
        return xr.open_dataset(cache_file)

    tva_mag_files = list(tva_mag.glob("*.csv"))
    tvamag_meta_dat = pd.read_csv(
        tva_mag / "TVAmagmetadata.dat", names=["Device", "latitude", "longitude"]
    )

    device_map = {
        "ackerman": "Ackerman",
        "bullrun": "Bull Run",
        "lagooncreek": "Lagoon Creek",
        "paradise": "Paradise",
        "raccoonmountain": "Raccoon Mountain",
        "union": "Union",
        "wattsbar": "Watts Bar",
    }

    dfs = []
    for f in tva_mag_files:
        df = pd.read_csv(f, parse_dates=["datetime"])
        df = df.rename(columns={"x": "Bx", "y": "By", "z": "Bz"})
        df["device"] = device_map.get(
            f.stem.split("_")[0].lower(), f.stem.split("_")[0].lower()
        )
        dfs.append(df)

    tva_mag_df = pd.concat(dfs, ignore_index=True)
    tva_mag_df = tva_mag_df.merge(
        tvamag_meta_dat, left_on="device", right_on="Device", how="left"
    )

    pivot_kwargs = {"index": "datetime", "columns": "device"}
    ds_tva_mag = xr.Dataset(
        {
            "Bx": (
                ["time", "device"],
                tva_mag_df.pivot(**pivot_kwargs, values="Bx").values,
            ),
            "By": (
                ["time", "device"],
                tva_mag_df.pivot(**pivot_kwargs, values="By").values,
            ),
            "Bz": (
                ["time", "device"],
                tva_mag_df.pivot(**pivot_kwargs, values="Bz").values,
            ),
        },
        coords={
            "time": tva_mag_df["datetime"].unique(),
            "device": tva_mag_df["device"].unique(),
            "latitude": (
                ["device"],
                tva_mag_df.drop_duplicates("device")
                .set_index("device")["latitude"]
                .values,
            ),
            "longitude": (
                ["device"],
                tva_mag_df.drop_duplicates("device")
                .set_index("device")["longitude"]
                .values,
            ),
        },
    )

    ds_tva_mag = preprocess_magnetic_field(ds_tva_mag)
    ds_tva_mag.to_netcdf(cache_file)
    return ds_tva_mag


def load_simulated_data():
    """Load SECS-derived B and E field predictions for the Gannon storm."""
    simulated_ds = xr.open_dataset(DENNIES_DATA_LOC / "storm_maxes" / "ds_gannon.nc")
    return simulated_ds.rename({"site_x": "latitude", "site_y": "longitude"})


def save_validation_csvs(
    validation_data, ground_truth_data, operator_name, output_dir, old_subs=None
):
    """Save per-substation validation CSVs with simulated and measured GIC time series."""
    save_dir = Path(output_dir) / "gic-comparison" / operator_name.lower()
    save_dir.mkdir(parents=True, exist_ok=True)

    (
        data_array,
        peak_times,
        median_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    ) = ground_truth_data

    selected_indices = validation_data["selected_indices"]
    valid_match_ids = validation_data["valid_match_ids"]
    valid_substations = validation_data["valid_substations"]
    trafo_unique = validation_data["trafo_unique"]
    ds_operator = validation_data["ds_operator"]
    peak_times_trimmed = validation_data["peak_times_trimmed"]

    if old_subs is not None:
        old_subs_set = set(old_subs)
        selected_indices = [
            idx for idx in selected_indices if valid_substations[idx] in old_subs_set
        ]
        matched_subs = set(valid_substations[idx] for idx in selected_indices)
        missing = old_subs_set - matched_subs
        if missing:
            print(
                f"{operator_name}: {len(missing)} subs not matched: {sorted(missing)}"
            )
        print(
            f"{operator_name}: Saving {len(selected_indices)}/{len(old_subs)} substations"
        )

    start_time = peak_times_trimmed[0]
    end_time = peak_times_trimmed[-1]
    duration_mask = (peak_times >= start_time) & (peak_times <= end_time)
    duration = np.where(duration_mask)[0]

    substation_indices_map = {
        sub: np.where(substation_names == sub)[0][0]
        for sub in substation_names
        if sub in valid_substations
    }

    for idx in tqdm(selected_indices, desc=f"Saving {operator_name} CSVs"):
        substation_id = valid_substations[idx]
        close_site_list = valid_match_ids[idx]

        sub_data = trafo_unique[trafo_unique.sub_id == substation_id]
        if sub_data.empty or substation_id not in substation_indices_map:
            continue

        sub_lat = sub_data["latitude"].values[0]
        sub_lon = sub_data["longitude"].values[0]
        sub_idx = substation_indices_map[substation_id]

        sim_median = median_values[sub_idx, 1:][duration]
        sim_lower = uncertainty_arr[0, sub_idx, 1:][duration]
        sim_upper = uncertainty_arr[1, sub_idx, 1:][duration]

        rows = []
        for t_idx, timestamp in enumerate(peak_times_trimmed):
            row = {
                "timestamp": pd.Timestamp(timestamp),
                "substation": substation_id,
                "Sim GIC (Median)": sim_median[t_idx],
                "Sim GIC (2.5)": sim_lower[t_idx],
                "Sim GIC (97.5)": sim_upper[t_idx],
                "sub lat": sub_lat,
                "sub lon": sub_lon,
            }
            for site_num, device_id in enumerate(close_site_list, start=1):
                measured_gic = ds_operator.gic.sel(
                    device=device_id, time=timestamp
                ).values.item()
                device_lat = ds_operator.latitude.sel(device=device_id).values.item()
                device_lon = ds_operator.longitude.sel(device=device_id).values.item()
                distance = haversine_dist(sub_lat, sub_lon, device_lat, device_lon)
                row[f"site_{site_num}_device"] = device_id
                row[f"site_{site_num}_gic"] = measured_gic
                row[f"site_{site_num}_Latitude"] = device_lat
                row[f"site_{site_num}_Longitude"] = device_lon
                row[f"site_{site_num}_Distance_km"] = distance
            rows.append(row)

        pd.DataFrame(rows).to_csv(save_dir / f"site_{substation_id}.csv", index=False)

    print(f"Saved {len(selected_indices)} CSV files to {save_dir}")
