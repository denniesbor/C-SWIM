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
from scipy.ndimage import uniform_filter1d

from configs import setup_logger, get_data_dir, LEAVE_OUT_SITES

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/storm_maxes.log")


def read_usgs_accepted_sites():
    file_path = DATA_LOC / "EMTF" / "USMTArray_unique_grid_points_1616_sites.txt"
    """Read USGS accepted sites from a text file."""
    try:
        df = pd.read_csv(
            file_path, header=None, names=["Station", "Latitude", "Longitude"]
        )
        logger.info(f"Loaded {len(df)} USGS accepted sites from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading USGS accepted sites: {e}")
        return pd.DataFrame()


usgs_accepted_sites = read_usgs_accepted_sites()
usgs_accepted_set = set(usgs_accepted_sites["Station"].str.upper().tolist())
LOAD_NEW = (
    True  # toggle to force reloading and processing emtf and transmission lines data
)


BAD_SITES = {"KSR32", "CAY09", "RER23"}

# Adjustables
MAX_ABS_Z = 100.0
MIN_GOOD_PERIODS = 8


def sanitize_site_variances(
    site, max_abs_Z: float = MAX_ABS_Z, min_good_periods: int = MIN_GOOD_PERIODS
):
    """
    Remove bad frequency bands (zero/neg/nonfinite variance, nonfinite Z, or |Z| too large).
    Returns the site with arrays trimmed in-place, or None if too few good bands remain.
    """
    Z = np.asarray(site.Z)
    if Z.ndim != 2:
        return None

    if not hasattr(site, "Z_var"):
        return site
    Z_var = np.asarray(site.Z_var)
    if Z_var.shape != Z.shape:
        return None

    bad_var = ~np.isfinite(Z_var) | (Z_var <= 0.0)
    bad_mag = ~np.isfinite(Z) | (np.abs(Z) > max_abs_Z)

    bad_by_band = np.any(bad_var, axis=0) | np.any(bad_mag, axis=0)
    keep = ~bad_by_band
    n_keep = int(keep.sum())
    n_total = keep.size

    if n_keep < min_good_periods:
        return None

    def _mask_attr(obj, name):
        if hasattr(obj, name):
            A = np.asarray(getattr(obj, name))
            if A.shape and A.shape[-1] == n_total:
                setattr(obj, name, A[..., keep])

    site.Z = Z[:, keep]
    site.Z_var = Z_var[:, keep]
    _mask_attr(site, "periods")
    _mask_attr(site, "freqs")
    for name in (
        "Z_invsigcov",
        "Z_residcov",
        "T",
        "T_var",
        "T_invsigcov",
        "T_residcov",
        "coherence",
        "multiple_coherence",
    ):
        _mask_attr(site, name)

    dropped = np.flatnonzero(~keep)
    if dropped.size:
        logger.info(
            f"[{site.name}] dropped bands idx={dropped.tolist()} "
            f"(kept {n_keep}/{n_total})"
        )

    return site


def process_xml_file(full_path):
    """Read and parse EMTF XML files."""
    try:
        site_name = os.path.basename(full_path).split(".")[1]
        site = bezpy.mt.read_xml(full_path)

        # If usgs accepted sites are available, filter based on them
        if not usgs_accepted_sites.empty and site.name.upper() not in usgs_accepted_set:
            logger.info(f"Skipped: {full_path} (Not in USGS accepted sites)")
            return None

        # If USGS sites are not available, use quality filters / site ratings
        if usgs_accepted_sites.empty:
            if site.rating < 3:
                logger.info(f"Skipped: {full_path} (Outside region or low rating)")
                return None

        if site.name.upper() in BAD_SITES:
            cleaned = sanitize_site_variances(site)
            if cleaned is None:
                logger.info(f"Skipped: {full_path} (Too many bad bands after sanitize)")
                return None
            site = cleaned

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

    if LOAD_NEW:
        if os.path.exists(mt_sites_pickle):
            os.remove(mt_sites_pickle)
            logger.info(f"Removed existing pickle file: {mt_sites_pickle}")

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

    if LOAD_NEW:
        # Remove existing pickle to force reprocessing
        if os.path.exists(trans_lines_pickle):
            os.remove(trans_lines_pickle)
            logger.info(f"Removed existing pickle file: {trans_lines_pickle}")

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
        site.lower(): magnetic_data.sel(site=site)
        for site in magnetic_data.site.values
        if site.lower() not in [s.lower() for s in LEAVE_OUT_SITES]
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

    #  network metric S(t)
    if agg == "median":
        S = np.nanmedian(site_mag, axis=1)
    elif agg == "sum":
        S = np.nansum(site_mag, axis=1)
    else:
        raise ValueError("agg must be 'median' or 'sum'")

    # 3-min moving average
    m = max(1, int(round(smooth_minutes)))
    S_sm = uniform_filter1d(S, size=m, mode="nearest")

    # Peak picking
    dist = max(1, int(round(min_separation_min)))  # minutes since 1 sample/min
    prom = max(1e-12, prominence_frac * np.nanmax(S_sm))
    peaks, props = signal.find_peaks(S_sm, distance=dist, prominence=prom)

    if peaks.size == 0:
        return [int(np.nanargmax(S_sm))]

    # rank by smoothed height
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
    start_time, end_time, calcV=False, is_special=True, use_peaks=False, top_k=3
):
    """Calculate maximum values of magnetic and electric fields."""

    t0 = time.time()

    B_obs, obs_xy, times, kept_names = build_common_stack_from_obsdict(
        obs_dict, start_time, end_time
    )
    site_xys = np.array([(s.latitude, s.longitude) for s in MT_sites], dtype=float)

    B_pred = calculate_SECS(B_obs, obs_xy, site_xys)
    logger.info(f"Done calculating magnetic fields: {time.time() - t0}")

    bad_b_sites = []
    for i, (lat, lon) in enumerate(site_xys):
        b_xy = B_pred[:, i, :2]
        reason = []
        if np.isnan(b_xy).any():
            reason.append("NaN in B_pred")
        if np.isinf(b_xy).any():
            reason.append("Inf in B_pred")
        if np.allclose(b_xy, 0.0):
            reason.append("all-zero B_pred")
        if reason:
            bad_b_sites.append((i, MT_sites[i].name, lat, lon, "; ".join(reason)))
            logger.warning(
                f"[SECS] {MT_sites[i].name} (idx {i}, lat={lat:.3f}, lon={lon:.3f}) -> {'; '.join(reason)}"
            )

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
        if is_special or not use_peaks:
            logger.info("Calculating voltages (full per-minute series)...")
            arr_delaunay = np.zeros((Te, n_trans_lines))
            for i, tLine in enumerate(df.obj):
                arr_delaunay[:, i] = tLine.calc_voltages(E_pred, how="delaunay")
            line_maxV = (
                np.ma.masked_invalid(np.abs(arr_delaunay)).max(axis=0).filled(np.nan)
            )
            logger.info(f"Done calculating voltages: {time.time() - t0}")

            # return full series only for Gannon
            if is_special:
                return (site_maxB, site_maxE, arr_delaunay, B_pred, E_pred)
        else:
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
    i, row, calcV, special_storm = args
    try:
        storm_times = (row["Start"], row["End"])
        logger.info(f"Working on storm: {i + 1}")
        i, maxB, maxE, maxV, B_pred, E_pred = i, *calculate_maxes(
            storm_times[0], storm_times[1], calcV, is_special=special_storm
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
    """Main processing function for storm maximum calculations."""
    file_path = DATA_LOC / "storm_maxes"
    os.makedirs(file_path, exist_ok=True)

    maxB_file = file_path / "maxB_arr_testing_2.npy"
    maxE_file = file_path / "maxE_arr_testing_2.npy"
    maxV_file = file_path / "maxV_arr_testing_2.npy"

    def _is_special_storm(row):
        """Check if storm is one of the special target storms."""
        s, e = row["Start"], row["End"]
        mar0 = pd.Timestamp("1989-03-01 00:00:00")
        mar1 = pd.Timestamp("1989-03-31 23:59:59")
        hal0 = pd.Timestamp("2003-10-27 00:00:00")
        hal1 = pd.Timestamp("2003-11-01 23:59:59")
        may24_0 = pd.Timestamp("2024-05-01 00:00:00")
        may24_1 = pd.Timestamp("2024-05-31 23:59:59")
        return (
            (s <= mar1 and e >= mar0)
            or (s <= hal1 and e >= hal0)
            or (s <= may24_1 and e >= may24_0)
        )

    site_xys = np.array([(site.latitude, site.longitude) for site in MT_sites])
    mt_site_names = [site.name for site in MT_sites]

    logger.info(f"Done loading data, Obs in obs_dict: {obs_dict.keys()}")

    CALCULATE_VALUES = True
    if CALCULATE_VALUES:
        t0 = time.time()
        logger.info("Starting to calculate storm maxes...")

        n_storms = len(storm_df)
        n_sites = len(site_xys)
        calcV = True

        if os.path.exists(maxB_file) and os.path.exists(maxE_file):
            maxB_arr = np.load(maxB_file)
            maxE_arr = np.load(maxE_file)
            logger.info("Loaded existing maxB and maxE arrays")
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
                is_special = _is_special_storm(row)
                args.append((i, row, calcV, is_special))

        logger.info(f"Processing {len(args)} remaining storms")

        with multiprocessing.Pool(12) as pool:
            for result in pool.imap_unordered(process_storm, args):
                i, maxB, maxE, maxV, B_pred, E_pred = result

                if result[1] is None:  # Skip if error occurred
                    continue

                maxB_arr[:, i] = maxB
                maxE_arr[:, i] = maxE

                if calcV:
                    if _is_special_storm(storm_df.loc[i]):  # Special storms (89, 03, 24, etc)
                        # Save individual storm data
                        if i == 91:
                            storm_filename = file_path / "delaunay_v_gannon.npy"
                        elif (
                            storm_df.loc[i]["Start"].year == 1989
                            and storm_df.loc[i]["Start"].month == 3
                        ):
                            storm_filename = file_path / "delaunay_v_march89.npy"
                        elif (
                            storm_df.loc[i]["Start"].year == 2003
                            and storm_df.loc[i]["Start"].month == 10
                        ):
                            storm_filename = file_path / "delaunay_v_halloween2003.npy"
                        else:
                            storm_filename = file_path / f"delaunay_v_storm_{i}.npy"

                        np.save(storm_filename, maxV)
                        maxV_arr[:, i] = np.nanmax(np.abs(maxV), axis=0)
                    else:
                        maxV_arr[:, i] = maxV

                    np.save(maxV_file, maxV_arr)

                np.save(maxB_file, maxB_arr)
                np.save(maxE_file, maxE_arr)

                if _is_special_storm(storm_df.loc[i]):
                    storm_name = "special storm"
                    if i == 91:
                        storm_name = "Gannon"
                    elif (
                        storm_df.loc[i]["Start"].year == 1989
                        and storm_df.loc[i]["Start"].month == 3
                    ):
                        storm_name = "March 1989"
                    elif (
                        storm_df.loc[i]["Start"].year == 2003
                        and storm_df.loc[i]["Start"].month == 10
                    ):
                        storm_name = "Halloween 2003"

                    logger.info(f"Processing {storm_name} storm data...")

                    storm_start = storm_df.loc[i, "Start"]
                    storm_end = storm_df.loc[i, "End"]

                    time_b = pd.date_range(
                        start=storm_start, periods=B_pred.shape[0], freq="60s"
                    )
                    T, Te = B_pred.shape[0], E_pred.shape[0]
                    trim = (T - Te) // 2

                    time_e = time_b[trim : trim + Te]
                    B_pred_xy = B_pred[trim : trim + Te, :, :2]

                    ds_special = xr.Dataset(
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

                    filename = f"ds_{storm_name.lower().replace(' ', '_')}.nc"
                    ds_special.to_netcdf(file_path / filename)
                    logger.info(f"{storm_name} storm dataset saved to {filename}")

                logger.info(f"Processed and saved storm: {i + 1}")

        logger.info(f"Done calculating storm maxes: {time.time() - t0}")
        logger.info(f"Saved results to {file_path}")

        try:
            # A site is "usable" for bootstrap if it has at least one finite, nonzero max
            # over all storms. We'll check both B and E, but E is the usual driver for TL.
            okB = np.isfinite(maxB_arr) & (maxB_arr != 0)
            okE = np.isfinite(maxE_arr) & (maxE_arr != 0)

            validB_counts = okB.sum(axis=1)  # per-site counts
            validE_counts = okE.sum(axis=1)

            all_zero_or_nan_B = validB_counts == 0
            all_zero_or_nan_E = validE_counts == 0

            # Problematic for bootstrap if either B or E has no usable samples
            problem_mask = all_zero_or_nan_B | all_zero_or_nan_E
            problem_indices = np.where(problem_mask)[0]

            # Log a compact summary and write a CSV for forensic checks
            rows = []
            for idx in problem_indices:
                name = mt_site_names[idx] if idx < len(mt_site_names) else f"site_{idx}"
                lat, lon = (float(site_xys[idx][0]), float(site_xys[idx][1]))
                row = {
                    "site_idx": int(idx),
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "validB_count": int(validB_counts[idx]),
                    "validE_count": int(validE_counts[idx]),
                    "all_zero_or_nan_B": bool(all_zero_or_nan_B[idx]),
                    "all_zero_or_nan_E": bool(all_zero_or_nan_E[idx]),
                    "any_nan_B": bool(np.isnan(maxB_arr[idx]).any()),
                    "any_nan_E": bool(np.isnan(maxE_arr[idx]).any()),
                    "all_zero_B": bool(np.all(maxB_arr[idx] == 0)),
                    "all_zero_E": bool(np.all(maxE_arr[idx] == 0)),
                }
                rows.append(row)
                logger.warning(
                    "[SITE DIAG] idx=%d name=%s lat=%.4f lon=%.4f "
                    "validB=%d validE=%d allZeroOrNaN(B/E)=%s/%s anyNaN(B/E)=%s/%s allZero(B/E)=%s/%s",
                    row["site_idx"],
                    row["name"],
                    row["lat"],
                    row["lon"],
                    row["validB_count"],
                    row["validE_count"],
                    row["all_zero_or_nan_B"],
                    row["all_zero_or_nan_E"],
                    row["any_nan_B"],
                    row["any_nan_E"],
                    row["all_zero_B"],
                    row["all_zero_E"],
                )

            # Emit the exact list to logs (handy if it's a small set like 3)
            if problem_indices.size > 0:
                names_joined = ", ".join(
                    f"{int(i)}:{mt_site_names[int(i)]}" for i in problem_indices
                )
                logger.error(
                    "[SITE DIAG] Problematic sites (count=%d): %s",
                    int(problem_indices.size),
                    names_joined,
                )
            else:
                logger.info(
                    "[SITE DIAG] No problematic sites found (all have usable samples)."
                )

            # CSV artifact for later inspection
            try:
                diag_df = pd.DataFrame(rows)
                diag_path = file_path / "problem_sites_summary.csv"
                diag_df.to_csv(diag_path, index=False)
                logger.info(
                    "[SITE DIAG] Wrote CSV: %s (rows=%d)", str(diag_path), len(rows)
                )
            except Exception as _csv_exc:
                logger.error(
                    "[SITE DIAG] Failed to write problem_sites_summary.csv: %s",
                    _csv_exc,
                )

            # Also, surface the TOP offenders by how empty they are (E first, then B)
            # This helps if you want the exact three right in logs:
            if problem_indices.size > 0:
                # sites with 0 valid E, then sort by validB to break ties
                order = np.lexsort((validB_counts, validE_counts))  # ascending
                top = [int(i) for i in order[: min(3, len(order))]]
                logger.error(
                    "[SITE DIAG] Top-3 worst sites by (validE, then validB): %s",
                    ", ".join(
                        f"{i}:{mt_site_names[i]}(validE={int(validE_counts[i])},validB={int(validB_counts[i])})"
                        for i in top
                    ),
                )
        except Exception as _diag_exc:
            logger.error("[SITE DIAG] diagnostics block failed: %s", _diag_exc)


if __name__ == "__main__":
    magnetic_data, MT_sites, df, storm_df, obs_dict, site_xys = load_data()
    n_trans_lines = df.shape[0]
    main()

# %%
