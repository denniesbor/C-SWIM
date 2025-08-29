"""
USGS Geomag file transfer service downloader for storm periods between 1985-1991.
Run repeatedly since some requests can return errors.
Authors: Dennies and Ed
"""

import os
import requests
import multiprocessing
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import xarray as xr
from bezpy import mag
from scipy import signal

from configs import setup_logger, get_data_dir

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/dl_usgs.log")

data_dir = DATA_LOC

storm_df_path = data_dir / "kp_ap_indices" / "storm_periods.csv"
geomag_folder = data_dir / "geomag_data"

os.makedirs(geomag_folder, exist_ok=True)

storm_df = pd.read_csv(storm_df_path)
storm_df["Start"] = pd.to_datetime(storm_df["Start"])
storm_df["End"] = pd.to_datetime(storm_df["End"])

# USGS observatory codes
usgs_obs = list(
    set(
        [
            "bou",
            "brw",
            "bsl",
            "cmo",
            "ded",
            "frd",
            "frn",
            "gua",
            "hon",
            "new",
            "shu",
            "sit",
            "sjg",
            "tuc",
            "bou",
            "brw",
            "bsl",
            "cmo",
            "ded",
            "dlr",
            "frd",
            "frn",
            "gua",
            "hon",
            "new",
            "shu",
            "sit",
            "sjg",
            "tuc",
        ]
    )
)


def usgs_mag_requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def process_usgs_magnetic_files(file_path):
    """Process downloaded IAGA file with interpolation and detrending."""
    data, headers = mag.read_iaga(file_path, return_header=True)
    data.index.name = "Timestamp"

    # Fill NaNs at the start and end, then interpolate remaining NaNs
    for component in ["X", "Y", "Z", "F"]:

        data[component] = (
            data[component]
            .interpolate(method="nearest")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

        # Detrend data linearly
        data[component] = signal.detrend(data[component])

    ds = xr.Dataset.from_dataframe(data)
    Latitude = float(headers["geodetic latitude"])
    Longitude = float(headers["geodetic longitude"]) - 360

    ds.attrs["Latitude"] = Latitude
    ds.attrs["Longitude"] = Longitude

    data["Latitude"] = Latitude
    data["Longitude"] = Longitude

    ds.attrs["Name"] = headers["iaga code"]

    ds.attrs.update(headers)

    # Remove original keys / Eliminate duplication
    del ds.attrs["geodetic latitude"]
    del ds.attrs["geodetic longitude"]
    del ds.attrs["iaga code"]

    return ds, data


def fetch_and_process_usgs_data(
    obsv_name, start_date, end_date, base_dir=geomag_folder
):
    """Fetch data from USGS API and process IAGA file."""
    logger.info(f"Processing {obsv_name}")
    obsv_name = obsv_name.upper()

    api_url = f"http://geomag.usgs.gov/ws/data/?id={obsv_name}&type=definitive&starttime={start_date}&endtime={end_date}"

    try:
        session = usgs_mag_requests_retry_session()
        response = session.get(api_url, timeout=30)
        response.raise_for_status()

        start_date_obj = pd.to_datetime(start_date)
        year = start_date_obj.year

        dir_path = os.path.join(base_dir, "interim", str(year), obsv_name)
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{obsv_name.lower()}{start_date_obj.strftime('%Y%m%d')}dmin.min"
        file_path = os.path.join(dir_path, filename)

        # if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(response.text)

        ds, data_df = process_usgs_magnetic_files(file_path)
        return obsv_name, data_df

    except Exception as e:
        logger.error(f"Error processing data for {obsv_name}: {e}")
        return obsv_name, None


def process_and_save_data(observatory_name, data, start_time, base_dir=geomag_folder):
    """Save processed data to CSV with year/observatory directory structure."""
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


def process_storm_period(row, usgs_obs):
    """Process all USGS observatories for a single storm period using ThreadPool."""
    start_time = row["Start"].strftime("%Y-%m-%dT%H:%M:%S")
    end_time = row["End"].strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"Processing storm period: {start_time} to {end_time}")

    storm_data = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_usgs = {
            executor.submit(
                fetch_and_process_usgs_data, obsv_name, start_time, end_time
            ): obsv_name
            for obsv_name in usgs_obs
        }
        for future in as_completed(future_to_usgs):
            obsv_name, data = future.result()
            if data is not None:
                storm_data[obsv_name] = data
                process_and_save_data(obsv_name, data, start_time)

    return storm_data


def process_storm_wrapper(args):
    row, usgs_obs = args
    return process_storm_period(row, usgs_obs)


def process_all_storms(storm_df, usgs_obs):
    """Process all storm periods between 1985-1991 using multiprocessing."""
    all_storm_data = {}

    # Index from 1985 to 1991 start
    storm_df_filtered = storm_df[
        (storm_df["Start"] > pd.to_datetime("1985-01-01"))
        & (storm_df["Start"] < pd.to_datetime("1991-01-01"))
    ]

    storm_args = [(row, usgs_obs) for _, row in storm_df_filtered.iterrows()]  # [:1]

    with multiprocessing.Pool() as pool:
        results = pool.map(process_storm_wrapper, storm_args)

    for index, result in enumerate(results):
        all_storm_data[index] = result

    return all_storm_data


if __name__ == "__main__":
    all_storm_data = process_all_storms(storm_df, usgs_obs)
    logger.info("Finished processing all storms")
