"""
Calculate maximum magnetic fields, electric fields, and voltages at MT sites during geomagnetic storms.
Uses SECS (Spherical Elementary Current Systems) model to fit observatory data and calculate fields.
Adapted from Greg Lucas's 2018 Hazard Paper.
Authors: Dennies and Ed
"""

# %%
import os
import time
import psutil
import pickle
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import bezpy
import powerlaw
import numpy as np
import pandas as pd
import xarray as xr
from pysecs import SECS
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from configs import setup_logger, get_data_dir

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/storm_maxes.log")


def process_xml_file(full_path):
    """Read and parse EMTF XML files."""
    try:
        site_name = os.path.basename(full_path).split(".")[1]
        site = bezpy.mt.read_xml(full_path)

        if site.rating < 3:
            logger.info(f"Skipped: {full_path} (Outside region or low rating)")
            return None

        logger.info(f"Processed: {full_path}")
        return site
    except Exception as e:
        logger.error(f"Error processing {full_path}: {e}")
        return None


def process_sites(directory):
    """Process all XML files in directory and subdirectories."""
    MT_sites = []
    completed = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".xml"):
                    full_path = os.path.join(root, file)
                    futures.append(executor.submit(process_xml_file, full_path))

        for future in as_completed(futures):
            result = future.result()
            if result:
                completed += 1
                MT_sites.append(result)

                if completed % 100 == 0:
                    logger.info(f"Completed {completed} files")

    logger.info(f"Completed processing {completed} files")

    return MT_sites


def load_mt_sites(mt_sites_pickle, emtf_path):
    """Load MT sites from pickle file or process EMTF files."""
    if os.path.exists(mt_sites_pickle):
        with open(mt_sites_pickle, "rb") as pkl:
            return pickle.load(pkl)
    else:
        MT_sites = process_sites(emtf_path)
        with open(mt_sites_pickle, "wb") as pkl:
            pickle.dump(MT_sites, pkl)
        return MT_sites


def read_transmission_lines(df_lines_EHV, trans_lines_pickle, site_xys):
    """Read transmission lines and setup Delaunay triangulation weights."""

    if os.path.exists(trans_lines_pickle):
        with open(trans_lines_pickle, "rb") as pkl:
            df = pickle.load(pkl)

        return df

    else:

        t1 = time.time()
        with open(df_lines_EHV, "rb") as f:
            trans_lines_gdf = pickle.load(f)

        trans_lines_gdf.rename({"LINE_ID": "line_id"}, inplace=True, axis=1)

        trans_lines_gdf = trans_lines_gdf.to_crs(epsg=4326)

        trans_lines_gdf["obj"] = trans_lines_gdf.apply(
            bezpy.tl.TransmissionLine, axis=1
        )
        trans_lines_gdf["length"] = trans_lines_gdf.obj.apply(lambda x: x.length)

        trans_lines_gdf.obj.apply(lambda x: x.set_delaunay_weights(site_xys))
        logger.info(f"Done filling interpolation weights: {time.time() - t1} s")

        # Remove lines with bad integration
        E_test = np.ones((1, len(site_xys), 2))
        arr_delaunay = np.zeros(shape=(1, len(trans_lines_gdf)))
        for i, tLine in enumerate(trans_lines_gdf.obj):
            arr_delaunay[:, i] = tLine.calc_voltages(E_test, how="delaunay")

        trans_lines_gdf_not_na = trans_lines_gdf[~np.isnan(arr_delaunay[0, :])]

        with open(trans_lines_pickle, "wb") as pkl:
            pickle.dump(trans_lines_gdf_not_na, pkl)

        return trans_lines_gdf_not_na


def load_data(start_date=None, end_date=None):
    """Load geomagnetic data, MT sites, and transmission line data."""
    emtf_path = DATA_LOC / "EMTF"
    mt_sites_pickle = emtf_path / "mt_pickle.pkl"
    geomag_path = DATA_LOC / "geomag_data"
    mag_data_path = geomag_path / "processed_geomag_data.nc"
    translines_path = DATA_LOC / "grid_processed"
    trans_lines_pickle = translines_path / "trans_lines_pickle.pkl"
    df_lines_EHV = translines_path / "df_lines_EHV.pkl"

    storm_data_loc = DATA_LOC / "kp_ap_indices" / "storm_periods.csv"
    storm_df = pd.read_csv(storm_data_loc)
    storm_df["Start"] = pd.to_datetime(storm_df["Start"])
    storm_df["End"] = pd.to_datetime(storm_df["End"])

    start_date = None
    end_date = None
    if start_date and end_date:
        storm_df = storm_df[
            (storm_df["Start"] >= start_date) & (storm_df["End"] <= end_date)
        ]

    magnetic_data = xr.open_dataset(mag_data_path)
    MT_sites = load_mt_sites(mt_sites_pickle, emtf_path)

    site_xys = [(site.latitude, site.longitude) for site in MT_sites]

    df = read_transmission_lines(df_lines_EHV, trans_lines_pickle, site_xys)

    obs_dict = {
        site.lower(): magnetic_data.sel(site=site) for site in magnetic_data.site.values
    }

    return magnetic_data, MT_sites, df, storm_df, obs_dict, site_xys


# %%
R_earth = 6371e3


def calculate_SECS(B, obs_xy, pred_xy):
    """Calculate SECS output magnetic field.

    B shape: (ntimes, nobs, 3 (xyz))
    obs_xy shape: (nobs, 2 (lat, lon))
    pred_xy shape: (npred, 2 (lat, lon))
    """
    if obs_xy.shape[0] != B.shape[1]:
        raise ValueError("Number of observation points doesn't match B input")

    obs_lat_lon_r = np.zeros((len(obs_xy), 3))
    obs_lat_lon_r[:, 0] = obs_xy[:, 0]
    obs_lat_lon_r[:, 1] = obs_xy[:, 1]
    obs_lat_lon_r[:, 2] = R_earth

    B_std = np.ones(B.shape)
    B_std[..., 2] = np.inf

    # SECS grid specification
    lat, lon, r = np.meshgrid(
        np.linspace(15, 85, 36),
        np.linspace(-175, -25, 76),
        R_earth + 110000,
        indexing="ij",
    )
    secs_lat_lon_r = np.hstack(
        (lat.reshape(-1, 1), lon.reshape(-1, 1), r.reshape(-1, 1))
    )

    secs = SECS(sec_df_loc=secs_lat_lon_r)

    secs.fit(obs_loc=obs_lat_lon_r, obs_B=B, obs_std=B_std, epsilon=0.05)

    pred_lat_lon_r = np.zeros((len(pred_xy), 3))
    pred_lat_lon_r[:, 0] = pred_xy[:, 0]
    pred_lat_lon_r[:, 1] = pred_xy[:, 1]
    pred_lat_lon_r[:, 2] = R_earth

    B_pred = secs.predict_B(pred_lat_lon_r)

    return B_pred


def pick_peak_times(
    E_pred,
    smooth_minutes=3,
    min_separation_min=10,
    top_k=3,
    agg="median",
    prominence_frac=0.2,
):
    """
    E_pred: [time, site, 2]
    Returns sorted list of peak indices to evaluate TL voltages at.
    """
    # |E| per site
    site_mag = np.sqrt(np.sum(E_pred**2, axis=2))  # [T, S]

    # robust network metric S(t)
    if agg == "median":
        S = np.nanmedian(site_mag, axis=1)
    elif agg == "sum":
        S = np.nansum(site_mag, axis=1)
    else:
        raise ValueError("agg must be 'median' or 'sum'")

    # 3-min moving average
    m = max(1, int(round(smooth_minutes)))
    S_sm = uniform_filter1d(S, size=m, mode="nearest")

    # peak picking
    dist = max(1, int(round(min_separation_min)))  # minutes since 1 sample/min
    prom = max(1e-12, prominence_frac * np.nanmax(S_sm))
    peaks, props = signal.find_peaks(S_sm, distance=dist, prominence=prom)

    if peaks.size == 0:
        return [int(np.nanargmax(S_sm))]

    # rank by smoothed height (or props['prominences']); take top_k
    order = np.argsort(S_sm[peaks])[::-1]
    sel = peaks[order[:top_k]]
    return sorted(map(int, sel))


def site_series(ds, t0, t1):
    """Return (times, B[nt,3]) for one site without adding extra dims."""
    win = ds.sel(time=slice(t0, t1))
    if win.sizes.get("time", 0) == 0:
        return None, None
    if "B" not in win:
        raise ValueError("Dataset missing variable 'B' with dims (time, component).")
    B = win["B"].values  # (nt, 3)
    if B.ndim != 2 or B.shape[1] < 3 or np.isnan(B[:, :3]).any():
        return None, None
    t = pd.to_datetime(win["time"].values)
    return t, B[:, :3]


def build_common_stack_from_obsdict(obs_dict, t0, t1):
    """
    Build B_obs aligned on exact common timestamps (NO resample/interp).
    Returns: B_obs (nt,nobs,3), obs_xy (nobs,2), times (DatetimeIndex), kept_names (list).
    """
    series, metas, names = [], [], []
    for name, ds in obs_dict.items():
        t, B = site_series(ds, t0, t1)
        if t is None:
            continue
        lat = (
            float(ds["latitude"].values)
            if "latitude" in ds
            else float(getattr(ds, "Latitude"))
        )
        lon = (
            float(ds["longitude"].values)
            if "longitude" in ds
            else float(getattr(ds, "Longitude"))
        )
        series.append((t, B))
        metas.append((lat, lon))
        names.append(str(name))

    if not series:
        raise RuntimeError("No usable observatory series in the window.")

    lens = np.array([len(t) for t, _ in series])

    common = set(series[0][0])
    for t, _ in series[1:]:
        common &= set(t)
    common = pd.DatetimeIndex(sorted(common))

    if len(common) < 10:
        ref = series[0][0]
        inter = sorted(
            ((nm, len(set(t) & set(ref)), len(t)) for nm, (t, _) in zip(names, series)),
            key=lambda x: x[1],
        )

    aligned, kept_xy, kept_names = [], [], []
    for (t, B), meta, nm in zip(series, metas, names):
        df = pd.DataFrame(B, index=t, columns=["Bx", "By", "Bz"]).reindex(common)
        if df.isna().any().any():
            continue
        aligned.append(df.to_numpy())
        kept_xy.append(meta)
        kept_names.append(nm)

    if not aligned:
        raise RuntimeError("No site matches the common timestamps exactly (no fill).")

    nt, nobs = len(common), len(aligned)
    B_obs = np.empty((nt, nobs, 3), dtype=float)
    for j, arr in enumerate(aligned):
        B_obs[:, j, :] = arr
    obs_xy = np.asarray(kept_xy, dtype=float)

    return B_obs, obs_xy, common, kept_names


def calculate_maxes(
    start_time, end_time, calcV=False, if_gannon=True, use_peaks=False, top_k=3
):
    """Calculate maximum values of magnetic and electric fields."""

    t0 = time.time()

    B_obs, obs_xy, times, kept_names = build_common_stack_from_obsdict(
        obs_dict, start_time, end_time
    )
    site_xys = np.array([(s.latitude, s.longitude) for s in MT_sites], dtype=float)

    B_pred = calculate_SECS(B_obs, obs_xy, site_xys)
    logger.info(f"Done calculating magnetic fields: {time.time() - t0}")

    site_maxB = np.max(np.sqrt(B_pred[:, :, 0] ** 2 + B_pred[:, :, 1] ** 2), axis=0)

    # E_pred (trim only for FFT edge effects)
    T = B_pred.shape[0]
    min_per_day = 1440
    trim = int(round(1.2 * min_per_day))
    if 2 * trim >= T:
        trim = max(0, (T - 1) // 2)

    Te = T - 2 * trim
    E_pred = np.zeros((Te, len(site_xys), 2), dtype=float)
    for i, site in enumerate(MT_sites):
        Ex, Ey = site.convolve_fft(B_pred[:, i, 0], B_pred[:, i, 1], dt=60)
        E_pred[:, i, 0] = Ex[trim : T - trim]
        E_pred[:, i, 1] = Ey[trim : T - trim]

    np.nan_to_num(E_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if E_pred.shape[0] == 0:
        logger.warning(
            "Empty E_pred after trim; setting site_maxE=0 and skipping peaks."
        )
        site_maxE = np.zeros(len(site_xys), dtype=float)
    else:
        if use_peaks:
            peak_idxs = pick_peak_times(
                E_pred,
                smooth_minutes=None,
                min_separation_min=10,
                top_k=top_k,
                agg="median",
                prominence_frac=0.2,
            )
            peak_time = int(peak_idxs[0])
            site_maxE = np.sqrt(np.sum(E_pred[peak_time] ** 2, axis=1))
        else:
            mag = np.sqrt(np.sum(E_pred**2, axis=2))  # [Te, S]
            site_maxE = mag.max(axis=0)  # zeros already for NaNs/Infs

    logger.info(f"Done calculating electric fields: {time.time() - t0}")

    # Voltages
    if calcV:
        if if_gannon or not use_peaks:
            # Ground truth: per-minute, unsmoothed
            logger.info("Calculating voltages (full per-minute series)...")
            arr_delaunay = np.zeros((Te, n_trans_lines))
            for i, tLine in enumerate(df.obj):
                arr_delaunay[:, i] = tLine.calc_voltages(E_pred, how="delaunay")
            line_maxV = (
                np.ma.masked_invalid(np.abs(arr_delaunay)).max(axis=0).filled(np.nan)
            )
            logger.info(f"Done calculating voltages: {time.time() - t0}")

            # return full series only for Gannon
            if if_gannon:
                return (site_maxB, site_maxE, arr_delaunay, B_pred, E_pred)
        else:
            # Speed mode: evaluate at selected peaks (still unsmoothed selection)
            peak_idxs = pick_peak_times(
                E_pred,
                smooth_minutes=None,
                min_separation_min=10,
                top_k=top_k,
                agg="median",
                prominence_frac=0.2,
            )
            logger.info("Calculating voltages at peak snapshots...")
            arr_delaunay = np.zeros((len(peak_idxs), n_trans_lines))
            for j, t_idx in enumerate(peak_idxs):
                e_snap = E_pred[t_idx][None, ...]
                for i, tLine in enumerate(df.obj):
                    arr_delaunay[j, i] = tLine.calc_voltages(e_snap, how="delaunay")
            line_maxV = (
                np.ma.masked_invalid(np.abs(arr_delaunay)).max(axis=0).filled(np.nan)
            )
            logger.info(f"Done calculating voltages at peaks: {time.time() - t0}")
    else:
        line_maxV = np.zeros(n_trans_lines)

    logger.info(f"B_pred shape: {B_pred.shape}")
    return (site_maxB, site_maxE, line_maxV, B_pred, E_pred)


def process_storm(args):
    """Process a single storm event."""
    i, row, calcV, gannon_storm = args
    try:
        storm_times = (row["Start"], row["End"])
        logger.info(f"Working on storm: {i + 1}")
        maxB, maxE, maxV, B_pred, E_pred = calculate_maxes(
            storm_times[0], storm_times[1], calcV, if_gannon=gannon_storm
        )
        return i, maxB, maxE, maxV, B_pred, E_pred
    except Exception as e:
        logger.error(f"Error processing storm {i + 1}: {e}")
        return i, None, None, None, None, None


def process_storm(args):
    """Process a single storm event."""
    i, row, calcV, gannon_storm = args
    try:
        storm_times = (row["Start"], row["End"])
        logger.info(f"Working on storm: {i + 1}")
        i, maxB, maxE, maxV, B_pred, E_pred = i, *calculate_maxes(
            storm_times[0], storm_times[1], calcV, if_gannon=gannon_storm
        )

        mar0 = pd.Timestamp("1989-03-01 00:00:00")
        mar1 = pd.Timestamp("1989-03-31 23:59:59")
        if (storm_times[0] <= mar1) and (storm_times[1] >= mar0):
            if maxV is None:
                logger.warning("[1989-03] maxV is None")
            else:
                m = np.ma.masked_invalid(np.abs(maxV))
                if m.count() == 0:
                    logger.warning("[1989-03] maxV is all-NaN")
                else:
                    logger.info(
                        f"[1989-03] global nanmax(|V|) = {float(m.max()):.1f} V"
                    )

        return i, maxB, maxE, maxV, B_pred, E_pred
    except Exception as e:
        logger.error(f"Error processing storm {i + 1}: {e}")
        return i, None, None, None, None, None


def main():
    file_path = DATA_LOC / "storm_maxes"
    os.makedirs(file_path, exist_ok=True)

    maxB_file = file_path / "maxB_arr_testing_2.npy"
    maxE_file = file_path / "maxE_arr_testing_2.npy"
    maxV_file = file_path / "maxV_arr_testing_2.npy"
    gannon_delaunay = file_path / "delaunay_v_gannon.npy"

    site_xys = np.array([(site.latitude, site.longitude) for site in MT_sites])
    mt_site_names = [site.name for site in MT_sites]

    logger.info(f"Done loading data, Obs in obs_dict: {obs_dict.keys()}")

    CALCULATE_VALUES = True
    if CALCULATE_VALUES:
        t0 = time.time()
        logger.info(f"Starting to calculate storm maxes...")

        n_storms = len(storm_df)
        n_sites = len(site_xys)
        calcV = True

        if os.path.exists(maxB_file) and os.path.exists(maxE_file):
            maxB_arr = np.load(maxB_file)
            maxE_arr = np.load(maxE_file)
            logger.info(f"Loaded existing maxB and maxE arrays")
        else:
            maxB_arr = np.zeros((n_sites, n_storms))
            maxE_arr = np.zeros((n_sites, n_storms))

        if calcV and os.path.exists(maxV_file):
            maxV_arr = np.load(maxV_file)
        else:
            maxV_arr = np.zeros((n_trans_lines, n_storms))

        args = []
        for i, row in storm_df.iterrows():
            if np.all(maxB_arr[:, i] == 0):  # Only process unprocessed storms
                # For storm 91 (Gannon), calculate full time series data
                is_gannon = i == 91
                args.append((i, row, calcV, is_gannon))

        logger.info(f"Processing {len(args)} remaining storms")

        with multiprocessing.Pool(12) as pool:
            for result in pool.imap_unordered(process_storm, args):
                i, maxB, maxE, maxV, B_pred, E_pred = result

                if result[1] is None:  # Skip if error occurred
                    continue

                maxB_arr[:, i] = maxB
                maxE_arr[:, i] = maxE

                if calcV:
                    if i == 91:  # Special handling for Gannon storm
                        np.save(gannon_delaunay, maxV)  # Full time series
                        maxV_arr[:, i] = np.nanmax(np.abs(maxV), axis=0)
                    else:
                        maxV_arr[:, i] = maxV

                    np.save(maxV_file, maxV_arr)

                np.save(maxB_file, maxB_arr)
                np.save(maxE_file, maxE_arr)

                if i == 91:
                    logger.info("Processing Gannon storm data...")

                    gannon_start = storm_df.loc[91, "Start"]
                    gannon_end = storm_df.loc[91, "End"]

                    time_b = pd.date_range(
                        start=gannon_start, periods=B_pred.shape[0], freq="60s"
                    )
                    T, Te = B_pred.shape[0], E_pred.shape[0]
                    trim = (T - Te) // 2

                    time_e = time_b[trim : trim + Te]
                    B_pred_xy = B_pred[trim : trim + Te, :, :2]

                    ds_gannon = xr.Dataset(
                        data_vars=dict(
                            B_pred=(("time", "site", "bcomp"), B_pred_xy),
                            E_pred=(("time", "site", "ecomp"), E_pred),
                        ),
                        coords=dict(
                            time=time_e,
                            site=np.arange(B_pred.shape[1]),
                            name=(["site"], mt_site_names),
                            site_x=(["site"], site_xys[:, 0]),
                            site_y=(["site"], site_xys[:, 1]),
                            bcomp=["Bx", "By"],
                            ecomp=["Ex", "Ey"],
                        ),
                    )

                    ds_gannon.to_netcdf(file_path / "ds_gannon.nc")
                    logger.info("Gannon storm dataset saved to ds_gannon.nc")

                logger.info(f"Processed and saved storm: {i + 1}")

        logger.info(f"Done calculating storm maxes: {time.time() - t0}")
        logger.info(f"Saved results to {file_path}")


if __name__ == "__main__":
    magnetic_data, MT_sites, df, storm_df, obs_dict, site_xys = load_data()
    n_trans_lines = df.shape[0]
    main()

# %%
