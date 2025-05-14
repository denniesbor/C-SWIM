# --------------------------------------------------------------------------
# This script calculates the maximum magnetic fields, electric fields, and voltages at MT sites during geomagnetic storms.
# It uses the SECS (Spherical Elementary Current Systems) model to fit observatory data and calculate the fields.
# The script processes EMTF data, geomagnetic data, and transmission lines data to calculate the fields.
# Some of the code is adapted from Greg Lucas's 2018 Hazard Paper.
# Author: Dennies Bor
# --------------------------------------------------------------------------
""" """
import os
import time
import sys
import psutil
import pickle
import datetime
from pathlib import Path
import multiprocessing
import powerlaw
import xarray as xr
import pandas as gpd
import bezpy
from pysecs import SECS
from scipy import signal
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from pysecs import SECS
from shapely.geometry import LineString


from configs import setup_logger, get_data_dir

# Get data data log and configure logger
DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/storm_maxes.log")


# Routines and modules that are used to extract max Es, Bs, and Vs during a storm
# -----------------------------------------------------------------------
# Looad and Prepare EMTF data
# -----------------------------------------------------------------------
def process_xml_file(full_path):
    """
    Read and parse EMTF XML files.

    Parameters
    ----------
    full_path : str
        The full path to the XML file.

    Returns
    -------
    bezpy.mt.Site or None
        The processed site object if successful, None otherwise.

    Notes
    -----
    This function reads an EMTF XML file, checks its rating, and returns
    the processed site object if the rating is 3 or higher.
    """
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
    """
    Process all XML files in the given directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The root directory to search for XML files.

    Returns
    -------
    list
        A list of processed MT site objects.

    Notes
    -----
    This function uses a ThreadPoolExecutor to process XML files concurrently.
    It prints progress every 100 completed files.
    """
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
    """
    Load MT sites from pickle file or process EMTF files.
    """
    if os.path.exists(mt_sites_pickle):
        with open(mt_sites_pickle, "rb") as pkl:
            return pickle.load(pkl)
    else:
        MT_sites = process_sites(emtf_path)
        with open(mt_sites_pickle, "wb") as pkl:
            pickle.dump(MT_sites, pkl)
        return MT_sites


# --------------------------------------------------------------------------
# Load and Prepare transmission lines data
# ----------------------------------------------------------------
def read_transmission_lines(translines_shp, trans_lines_pickle, site_xys):
    """
    Read transmission lines shapefile and pickle the data.

    Parameters
    ----------
    translines_shp : str
        The path to the transmission lines
    translines_pickle : str
        The path to the processed and pickled transmission lines

    Returns
    -------
    geopandas.GeoDataFrame
        The processed transmission lines data
    """

    # Check if transmission lines is pickled
    if os.path.exists(trans_lines_pickle):
        with open(trans_lines_pickle, "rb") as pkl:
            df = pickle.load(pkl)

        return df

    else:

        # Use Delaunay triangulation to set weights
        t1 = time.time()
        # US Transmission lines
        trans_lines_gdf = gpd.read_file(translines_shp)

        # Rename the ID column
        trans_lines_gdf.rename({"ID": "line_id"}, inplace=True, axis=1)

        # Apply crs
        trans_lines_gdf = trans_lines_gdf.to_crs(epsg=4326)

        # Translate MultiLineString to LineString geometries, taking only the first LineString
        trans_lines_gdf.loc[
            trans_lines_gdf["geometry"].apply(lambda x: x.geom_type)
            == "MultiLineString",
            "geometry",
        ] = trans_lines_gdf.loc[
            trans_lines_gdf["geometry"].apply(lambda x: x.geom_type)
            == "MultiLineString",
            "geometry",
        ].apply(
            lambda x: list(x.geoms)[0]
        )

        # Let's focus on high voltage transmissions - > 200 kV
        trans_lines_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] >= 200)]

        # Use bezpy to get the tranmsission line lenght
        trans_lines_gdf["obj"] = trans_lines_gdf.apply(
            bezpy.tl.TransmissionLine, axis=1
        )
        trans_lines_gdf["length"] = trans_lines_gdf.obj.apply(lambda x: x.length)

        trans_lines_gdf.obj.apply(lambda x: x.set_delaunay_weights(site_xys))
        print("Done filling interpolation weights: {0} s".format(time.time() - t1))

        # Remove lines with bad integration
        E_test = np.ones((1, len(site_xys), 2))
        arr_delaunay = np.zeros(shape=(1, len(trans_lines_gdf)))
        for i, tLine in enumerate(trans_lines_gdf.obj):
            arr_delaunay[:, i] = tLine.calc_voltages(E_test, how="delaunay")

        # Filter the trans_lines_gdf
        trans_lines_gdf_not_na = trans_lines_gdf[~np.isnan(arr_delaunay[0, :])]

        # Pickle the trans_lines_gdf
        with open(trans_lines_pickle, "wb") as pkl:
            pickle.dump(trans_lines_gdf_not_na, pkl)

        return trans_lines_gdf_not_na


# %%
# --------------------------------------------------------------------------
# Prepare the paths and load the datasets
# --------------------------------------------------------------------------


def load_data(start_date=None, end_date=None):
    """
    Load geomagnetic data, MT sites, and transmission line data.

    Parameters
    ----------
    start_date : str, optional
        Start date for storm filtering (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for storm filtering (format: 'YYYY-MM-DD')

    Returns
    -------
    tuple
        magnetic_data : magnetic field data
        MT_sites : MT site data
        df : transmission line data
        storm_df : storm period dataframe
        obs_dict : observation dictionary for SECS
    """
    # Set up paths
    emtf_path = DATA_LOC / "EMTF"
    mt_sites_pickle = emtf_path / "mt_pickle.pkl"
    geomag_path = DATA_LOC / "geomag_data"
    mag_data_path = geomag_path / "processed_geomag_data.nc"
    translines_path = DATA_LOC / "Electric__Power_Transmission_Lines"
    trans_lines_pickle = translines_path / "trans_lines_pickle.pkl"
    translines_shp = translines_path / "Electric__Power_Transmission_Lines.shp"

    # Load storm periods
    storm_data_loc = DATA_LOC / "kp_ap_indices" / "storm_periods.csv"
    storm_df = pd.read_csv(storm_data_loc)
    storm_df["Start"] = pd.to_datetime(storm_df["Start"])
    storm_df["End"] = pd.to_datetime(storm_df["End"])

    # Filter storm data if dates provided
    if start_date and end_date:
        storm_df = storm_df[
            (storm_df["Start"] >= start_date) & (storm_df["End"] <= end_date)
        ]

    # Load datasets
    magnetic_data = xr.open_dataset(mag_data_path)  # Load geomagnetic data
    MT_sites = load_mt_sites(mt_sites_pickle, emtf_path)

    # Get site coordinates
    site_xys = [(site.latitude, site.longitude) for site in MT_sites]

    # Load transmission lines
    df = read_transmission_lines(translines_shp, trans_lines_pickle, site_xys)

    # Prepare SECS observation dictionary
    obs_dict = {
        site.lower(): magnetic_data.sel(site=site) for site in magnetic_data.site.values
    }

    return magnetic_data, MT_sites, df, storm_df, obs_dict, site_xys


# Data
magnetic_data, MT_sites, df, storm_df, obs_dict, site_xys = load_data()
n_trans_lines = df.shape[0]

# %%
# --------------------------------------------------------------------------
# Fit the observatory data to ionospheric model -secs
# Compute the electric field and magnetic field at the MT sites
# Adapted from pysecs by Greg Lucas
# --------------------------------------------------------------------------

R_earth = 6371e3


def calculate_SECS(B, obs_xy, pred_xy):
    """Calculate SECS output magnetic field

    B shape: (ntimes, nobs, 3 (xyz))

    obs_xy shape: (nobs, 2 (lat, lon))

    pred_xy shape: (npred, 2 (lat, lon))"""
    if obs_xy.shape[0] != B.shape[1]:
        raise ValueError("Number of observation points doesn't match B input")

    obs_lat_lon_r = np.zeros((len(obs_xy), 3))
    obs_lat_lon_r[:, 0] = obs_xy[:, 0]
    obs_lat_lon_r[:, 1] = obs_xy[:, 1]
    obs_lat_lon_r[:, 2] = R_earth

    # B = np.dstack((B, np.full(B.shape[:2], 1)))

    B_std = np.ones(B.shape)
    B_std[..., 2] = np.inf

    # specify the SECS grid
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

    # Create prediction points
    pred_lat_lon_r = np.zeros((len(pred_xy), 3))
    pred_lat_lon_r[:, 0] = pred_xy[:, 0]
    pred_lat_lon_r[:, 1] = pred_xy[:, 1]
    pred_lat_lon_r[:, 2] = R_earth

    B_pred = secs.predict_B(pred_lat_lon_r)

    return B_pred


# Find peaks
def find_storm_maximum(E_pred, window_hours=(20 / 60)):
    """
    Find storm maximum using windowed analysis of 1-minute resolution data.

    Parameters:
    -----------
    E_pred : numpy.ndarray
        Array of shape (time, sites, components) containing E-field values at 1-min resolution
    window_hours : float
        Size of the analysis window in hours

    Returns:
    --------
    dict containing:
        - optimal_time : int
            Index of the determined storm maximum
        - site_magnitudes : numpy.ndarray
            Magnitude at each site at the optimal time
    """
    # Convert window hours to number of samples (1 sample per minute)
    window_samples = int(window_hours * 60)

    # Calculate magnitude at each site and time
    site_maxE_mags = np.sqrt(np.sum(E_pred**2, axis=2))

    # Calculate the total magnitude across all sites
    total_magnitude = np.nansum(site_maxE_mags, axis=1)

    # Create a centered moving average
    window = np.ones(window_samples) / window_samples
    # smoothed_magnitude = np.convolve(total_magnitude, window, mode='same')
    smoothed_magnitude = gaussian_filter1d(total_magnitude, sigma=window_samples // 2)

    # Find peaks in the smoothed data
    peaks, _ = signal.find_peaks(
        smoothed_magnitude,
        distance=window_samples // 2,  # Minimum distance between peaks
        prominence=0.2 * np.max(smoothed_magnitude),  # Minimum prominence
    )

    if len(peaks) == 0:
        # If no peaks found, use the maximum point
        optimal_time = np.argmax(smoothed_magnitude)
    else:
        # Among the peaks, find the one with highest average in surrounding window
        peak_scores = []
        for peak in peaks:
            start_idx = max(0, peak - window_samples // 2)
            end_idx = min(len(smoothed_magnitude), peak + window_samples // 2)
            window_mean = np.mean(smoothed_magnitude[start_idx:end_idx])
            peak_scores.append(window_mean)

        optimal_time = peaks[np.argmax(peak_scores)]

    # Get data at the optimal time
    return {
        "optimal_time": optimal_time,
        "site_magnitudes": E_pred[optimal_time],
        "total_magnitude": total_magnitude[optimal_time],
        "smoothed_magnitude": smoothed_magnitude[optimal_time],
    }


# --------------------------------------------------------------------------
# Calculate the maxes for all storms
# --------------------------------------------------------------------------
def calculate_maxes(start_time, end_time, calcV=False, if_gannon=True):
    """
    Calculate maximum magnetic fields, electric fields, and voltages for given time range and observation data.

    Parameters
    ----------
    start_time : datetime
        The start time of the calculation period.
    end_time : datetime
        The end time of the calculation period.
    calcV : bool, optional
        Whether to calculate voltages, by default True.
    obs_dict : dict, optional
        Dictionary of observation datasets, by default None.

    Returns
    -------
    tuple
        A tuple containing:
        - site_maxB : ndarray
            Maximum magnetic field magnitude for each site.
        - site_maxE : ndarray
            Maximum electric field magnitude for each site.
        - line_maxV : ndarray
            Maximum voltage magnitude for each transmission line (if calcV is True).

    Notes
    -----
    This function processes observation data, calculates SECS (Spherical Elementary Current Systems),
    and determines maximum values for magnetic fields, electric fields, and optionally, voltages at the MT sites.
    It uses interpolation and filtering techniques on the input data.
    """
    t0 = time.time()

    obs_xy = []
    B_obs = []
    site_xys = np.array([(site.latitude, site.longitude) for site in MT_sites])

    for name, dataset in obs_dict.items():
        data = dataset.loc[{"time": slice(start_time, end_time)}].interpolate_na("Time")
        if len(data["time"]) == 0:
            continue

        data = np.array(data.loc[{"time": slice(start_time, end_time)}].to_array().T)
        if np.any(np.isnan(data)):
            continue

        obs_xy.append((dataset.latitude, dataset.longitude))
        B_obs.append(data)

    obs_xy = np.squeeze(np.array(obs_xy))
    B_obs = np.squeeze(np.array(B_obs)).transpose(2, 0, 1)

    B_pred = calculate_SECS(B_obs, obs_xy, site_xys)
    logger.info(f"Done calculating magnetic fields: {time.time() - t0}")

    site_maxB = np.max(np.sqrt(B_pred[:, :, 0] ** 2 + B_pred[:, :, 1] ** 2), axis=0)

    E_pred = np.zeros((len(B_obs), len(site_xys), 2))
    for i, site in enumerate(MT_sites):
        Ex, Ey = site.convolve_fft(B_pred[:, i, 0], B_pred[:, i, 1], dt=60)
        E_pred[:, i, 0] = Ex
        E_pred[:, i, 1] = Ey

    # Storm peaks
    e_pred_peak_data = find_storm_maximum(
        E_pred, window_hours=(20 / 60)
    )  # Every 20 minutes for thermal heating
    peak_time = e_pred_peak_data["optimal_time"]
    e_pred_peak = E_pred[peak_time, :, :]

    e_pred_magnitude = np.sqrt(np.sum(e_pred_peak**2, axis=1))

    logger.info(f"Done calculating electric fields: {time.time() - t0}")
    # site_maxE = np.max(np.sqrt(E_pred[:, :, 0] ** 2 + E_pred[:, :, 1] ** 2), axis=0)
    site_maxE = e_pred_magnitude

    if calcV:
        if if_gannon:
            logger.info("Calculating voltages, Gannon only...")
            arr_delaunay = np.zeros(shape=(E_pred.shape[0], n_trans_lines))
            for i, tLine in enumerate(df.obj):
                arr_delaunay[:, i] = tLine.calc_voltages(E_pred, how="delaunay")
            line_maxV = np.nanmax(np.abs(arr_delaunay), axis=0)
            logger.info(f"Done calculating voltages, Gannon: {time.time() - t0}")

            return (site_maxB, site_maxE, arr_delaunay, B_pred, E_pred)

        else:
            e_pred = e_pred_peak.reshape(1, e_pred_peak.shape[0], e_pred_peak.shape[1])
            logger.info("Calculating voltages...")
            arr_delaunay = np.zeros(shape=(e_pred.shape[0], n_trans_lines))
            for i, tLine in enumerate(df.obj):
                arr_delaunay[:, i] = tLine.calc_voltages(e_pred, how="delaunay")
            line_maxV = np.squeeze(arr_delaunay)
            logger.info(f"Done calculating voltages: {time.time() - t0}")

    else:
        logger.info("Skipping voltage calculation")
        line_maxV = np.zeros(n_trans_lines)

    # Apply Gaussinan smoothing to the electric field
    print("B_pred shape", B_pred.shape)

    return (site_maxB, site_maxE, line_maxV, B_pred, E_pred)


# --------------------------------------------------------------------------
# Calculating storm maxes
# --------------------------------------------------------------------------
def process_storm(args):
    """
    Process a single storm event.

    Parameters
    ----------
    args : tuple
        Contains (i, row, calcV, MT_sites, obs_dict)

    Returns
    -------
    tuple
        Contains (i, maxB, maxE, maxV)
    """
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


# %%
def main():

    # File paths
    maxB_file = DATA_LOC / "maxB_arr_testing_2.npy"
    maxE_file = DATA_LOC / "maxE_arr_testing_2.npy"
    maxV_file = DATA_LOC / "maxV_arr_testing_2.npy"
    gannon_delaunay = DATA_LOC / "delaunay_v_gannon.npy"

    site_xys = np.array([(site.latitude, site.longitude) for site in MT_sites])
    mt_site_names = [site.name for site in MT_sites]

    # Log done loading data, and print unique obs in obs_dict
    logger.info(f"Done loading data, Obs in obs_dict: {obs_dict.keys()}")

    CALCULATE_VALUES = True
    if CALCULATE_VALUES:
        t0 = time.time()
        logger.info(f"Starting to calculate storm maxes...")

    if CALCULATE_VALUES:
        n_storms = len(storm_df)
        n_sites = len(site_xys)
        calcV = True
        gannon_storm = True

        # Check if saved results exist, else initialize arrays
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

        # Prepare the args list for only unprocessed storms OR if i == 91 (force processing)
        args = []
        for i, row in storm_df.iterrows():
            if np.all(maxB_arr[:, i] == 0) or i == 91:  # Always process i == 91
                args.append((i, row, calcV, gannon_storm))

        logger.info(f"Processing {len(args)} remaining storms")

        # Use multiprocessing with a pool of workers
        with multiprocessing.Pool(12) as pool:
            for result in pool.imap_unordered(process_storm, args):
                i, maxB, maxE, maxV, B_pred, E_pred = result

                # Update the arrays with results
                maxB_arr[:, i] = maxB
                maxE_arr[:, i] = maxE
                if calcV:
                    if gannon_storm:
                        np.save(gannon_delaunay, maxV)
                    else:
                        maxV_arr[:, i] = maxV
                        np.save(maxV_file, maxV_arr)

                # Save intermediate results after each storm
                np.save(maxB_file, maxB_arr)
                np.save(maxE_file, maxE_arr)

                logger.info(f"Processed and saved storm: {i + 1}")

        logger.info(f"Done calculating storm maxes: {time.time() - t0}")
        logger.info(f"Saved results to {DATA_LOC}")

        # If Gannon storm, save additional data
        if gannon_storm and i == 91:
            logger.info("Processing Gannon storm data...")
            gannon_start = storm_df.loc[91, "Start"]
            gannon_end = storm_df.loc[91, "End"]

            # Generate time stamps with 60s intervals
            time_stamps = pd.date_range(start=gannon_start, end=gannon_end, freq="60S")
            print("B_pred shape and time stamps leng", B_pred.shape, len(time_stamps))

            # Ensure B_pred aligns with time dimension
            if B_pred.shape[0] != len(time_stamps):
                logger.warning(
                    f"Mismatch in time length: B_pred {B_pred.shape[0]} vs time {len(time_stamps)}"
                )
                time_stamps = time_stamps[: B_pred.shape[0]]  # Trim if necessary

            B_pred = B_pred[:, :, :2]  # Only use Bx and By

            ds_gannon = xr.Dataset(
                {
                    "B_pred": (["time", "site", "component"], B_pred),
                    "E_pred": (["time", "site", "component"], E_pred),
                },
                coords={
                    "time": time_stamps,
                    "name": mt_site_names,
                    "site_x": site_xys[:, 0],
                    "site_y": site_xys[:, 1],
                    "component": ["Bx", "By"],
                },
            )
            ds_gannon.to_netcdf(DATA_LOC / "ds_gannon.nc")
            logger.info(f"Gannon storm dataset saved to ds_gannon.nc")


if __name__ == "__main__":

    magnetic_data, MT_sites, df, storm_df, obs_dict, site_xys = load_data()
    n_trans_lines = df.shape[0]

    main()
