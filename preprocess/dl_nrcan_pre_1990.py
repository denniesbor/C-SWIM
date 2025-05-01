# --------------------------------------------------------------------------------
# This script downloads geomagnetic data from the Canadian National Geomagnetic
# Network (C2) for a given storm period. The data is downloaded for a set of
# observatories and saved to a CSV file.
# Some of the requests may fail due to network issues or other reasons. So rerun this
# script if some of the data is missing. Please validate the data before using it.
# Author: Dennies Bor - GMU
# --------------------------------------------------------------------------------

# %%
import time
import os
from pathlib import Path
import tempfile
import obspy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from configs import setup_logger, get_data_dir

# Get data data log and configure logger
DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/d_nrcan_mag.log")

# Load the storm data
data_dir = DATA_LOC
storm_data_loc = data_dir / "kp_ap_indices" / "storm_periods.csv"

storm_df = pd.read_csv(storm_data_loc)
storm_df["Start"] = pd.to_datetime(storm_df["Start"])
storm_df["End"] = pd.to_datetime(storm_df["End"])
geomag_folder = data_dir / "geomag_data"

nrcan_obs = ["blc", "cbb", "fcc", "mea", "ott", "res", "stj", "vic", "ykc"]

nrcan_loc = {
    "ALE": {"latitude": 82.497, "longitude": 297.647},
    "BLC": {"latitude": 64.318, "longitude": 263.988},
    "BRD": {"latitude": 49.870, "longitude": 260.026},
    "CBB": {"latitude": 69.123, "longitude": 254.969},
    "EUA": {"latitude": 80.000, "longitude": 274.100},
    "FCC": {"latitude": 58.759, "longitude": 265.912},
    "GLN": {"latitude": 49.645, "longitude": 262.880},
    "GWC": {"latitude": 55.300, "longitude": 282.250},
    "IQA": {"latitude": 63.753, "longitude": 291.482},
    "MBC": {"latitude": 76.315, "longitude": 240.638},
    "MEA": {"latitude": 54.616, "longitude": 246.653},
    "OTT": {"latitude": 45.403, "longitude": 284.448},
    "PBQ": {"latitude": 55.277, "longitude": 282.255},
    "RES": {"latitude": 74.690, "longitude": 265.105},
    "STJ": {"latitude": 47.595, "longitude": 307.323},
    "SNK": {"latitude": 56.500, "longitude": 280.800},
    "VIC": {"latitude": 48.520, "longitude": 236.580},
    "WHS": {"latitude": 49.80, "longitude": 264.750},
    "YKC": {"latitude": 62.480, "longitude": 245.518},
}

# %%
def download_nrcan_geomag_data(network, station, location, channel, starttime, 
    endtime, max_retries=3):
    """
    Downloads geomagnetic data from the NRCan (Natural Resources Canada) website.
    Parameters:
    - network (str): The network code.
    - station (str): The station code.
    - location (str): The location code.
    - channel (str): The channel code.
    - starttime (str): The start time of the data in UTC format.
    - endtime (str): The end time of the data in UTC format.
    - max_retries (int): The maximum number of retries in case of download failure. Default is 3.
    Returns:
    - df (pandas.DataFrame): The downloaded data as a pandas DataFrame with the following columns:
        - Timestamp: The timestamp of each data point.
        - Channel: The data values for the specified channel.
    Note:
    - This function requires the `Client` class from the `obspy.clients.earthquakescanada` module.
    - The data is downloaded using the `get_waveforms` method of the `Client` class.
    - The downloaded data is processed and returned as a pandas DataFrame.
    - If the download fails, the function will retry up to `max_retries` times with an exponential backoff.
    - If the download still fails after `max_retries` attempts, None is returned.

    """
    client = Client("https://www.earthquakescanada.nrcan.gc.ca")
    start = UTCDateTime(starttime)
    end = UTCDateTime(endtime)

    for attempt in range(max_retries):
        try:
            st = client.get_waveforms(network, station, location, channel, start, end)

            if st and len(st) > 0:
                tr = st[0]
                df = pd.DataFrame(
                    {
                        "Timestamp": tr.times("timestamp"),
                        channel[-1]: tr.detrend("linear").data,
                    }
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
                df.set_index("Timestamp", inplace=True)
                logger.info(
                    f"Successfully downloaded data for {network}.{station}.{location}.{channel}"
                )
                return df
            else:
                logger.warning(
                    f"No data available for {network}.{station}.{location}.{channel} (Attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

        except Exception as e:
            logger.error(
                f"Error downloading data for {network}.{station}.{location}.{channel} (Attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue

    logger.error(
        f"Failed to download data for {network}.{station}.{location}.{channel} after {max_retries} attempts"
    )
    return None


# Download and process data for a single station
def download_and_process_nrcan_station(network, station, location, channels, 
    start_time, end_time, nrcan_loc):
    station_data = []
    for channel in channels:
        data = download_nrcan_geomag_data(
            network, station, location, channel, start_time, end_time
        )
        if data is not None:
            logger.info(
                f"Successfully downloaded data for {network}.{station}.{location}.{channel}"
            )
            station_data.append(data)
        elif channel in ["UFX", "UFY", "UFZ"]:
            logger.info(
                f"No data available for {network}.{station}.{location}.{channel}"
            )
            return None

    if station_data:
        combined_data = pd.concat(station_data, axis=1)
        latitude = nrcan_loc[station]["latitude"]
        longitude = nrcan_loc[station]["longitude"] - 360  # Convert to -180 to 180
        combined_data["Latitude"] = latitude
        combined_data["Longitude"] = longitude
        return station, combined_data
    return None


def process_nrcan_stations(stations, start_time, end_time, nrcan_loc):
    network = "C2"  # Canadian National Geomagnetic Network
    location = "R0"
    channels = ["UFX", "UFY", "UFZ", "UFF"]

    data_dict = {}

    for station in stations:
        try:
            result = download_and_process_nrcan_station(
                network, station, location, channels, start_time, end_time, nrcan_loc
            )
            if result:
                station, data = result
                data_dict[station] = data
        except Exception as e:
            logger.error(f"Error processing station {station}: {str(e)}")

    return data_dict


# %%
def process_and_save_data(observatory_name, data, start_time, base_dir=geomag_folder):
    start_date = pd.to_datetime(start_time)
    year = start_date.year
    month = start_date.month
    day = start_date.day

    dir_path = os.path.join(base_dir, str(year), observatory_name.upper())
    os.makedirs(dir_path, exist_ok=True)

    filename = f"{observatory_name.lower()}{year:04d}{month:02d}{day:02d}processed.csv"
    file_path = os.path.join(dir_path, filename)

    if not os.path.exists(file_path):
        data.to_csv(file_path)
    logger.info(f"Saved processed data for {observatory_name} to {file_path}")


def process_storm_period(row, nrcan_obs, nrcan_loc):
    start_time = row["Start"].strftime("%Y-%m-%dT%H:%M:%S")
    end_time = row["End"].strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"Processing storm period: {start_time} to {end_time}")

    storm_data = {}

    # Process NRCAN data
    nrcan_stations = [station.upper() for station in nrcan_obs]
    nrcan_storm_data = process_nrcan_stations(
        nrcan_stations, start_time, end_time, nrcan_loc
    )

    for station, data in nrcan_storm_data.items():
        storm_data[station] = data
        process_and_save_data(station, data, start_time)

    return storm_data


# %%
def process_all_storms(storm_df, nrcan_obs, nrcan_loc):
    all_storm_data = {}

    # Index from 1985 to 1991 start
    storm_df_filtered = storm_df[
        (storm_df["Start"] > pd.to_datetime("1985-01-01"))
        & (storm_df["Start"] < pd.to_datetime("1991-01-01"))
    ]

    for index, row in storm_df_filtered.iterrows():
        try:
            result = process_storm_period(row, nrcan_obs, nrcan_loc)
            all_storm_data[index] = result
            logger.info(f"Processed storm {index}")
        except Exception as e:
            logger.error(f"Error processing storm {index}: {str(e)}")

    return all_storm_data


# %%
if __name__ == "__main__":

    try:
        all_storm_data = process_all_storms(storm_df, nrcan_obs, nrcan_loc)
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")

# %%
