"""
Script to download substation data from the OpenStreetMap database.

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025
"""

import os
import pickle
import requests
from pathlib import Path
import warnings
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd

from configs import setup_logger, get_data_dir

warnings.filterwarnings("ignore")

# Get data data log and configure logger
DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/dl_power_grid.log")

# Define the data path
data_path = DATA_LOC / "substation_locations"


def download_data(url, save_path):
    """This function downloads the data from the given URL
    and saves it to the specified path.
    params:
    ---------
    url: str
      URL to download the data from
    save_path: str
      Path to save the downloaded data
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"{save_path.name} downloaded successfully.")
    else:
        logger.error(
            f"Failed to download {save_path.name}. Status code: {response.status_code}"
        )


def read_text_file(file_path):
    """This custom function reads large text/csv
    files, especially those with encoding errors due
    to compression
    params:
    ---------
    file: str
      data path of the input file
    returns:
    ---------
    dataframe: pd.dataframe
      pandas dataframe
    """

    with open(file_path, encoding="utf-8", errors="ignore") as file:
        content = file.read()

    content_io = StringIO(content)
    return pd.read_csv(content_io, low_memory=False)


def get_pwr_stations(state: str, max_retries=5, retry_delay=10):
    """
    Fetches power substations data for a given state with retry mechanism.
    Parameters:
        state (str): The state for which to fetch the data.
        max_retries (int): Number of times to retry fetching data before failing.
        retry_delay (int): Delay in seconds before retrying.
    Returns:
        substations (GeoDataFrame or None): A GeoDataFrame containing the power substations data, or None if failed.
    """
    area_of_interest = f"{state}, USA"
    tags = {"power": "substation"}

    for attempt in range(max_retries):
        try:
            substations = ox.features_from_place(area_of_interest, tags)
            if not substations.empty:
                return substations
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {state}: {e}")

        print(
            f"Retrying {state} in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})"
        )
        time.sleep(retry_delay)

    print(
        f"Failed to fetch power substations for {state} after {max_retries} attempts."
    )
    return None


def dl_per_state(state):
    """
    Download OSM data on a per state basis.

    """
    out_path = data_path / f"{state}.pkl"

    if os.path.exists(out_path):
        return

    results = get_pwr_stations(state)

    with open(out_path, "wb") as file:
        pickle.dump(results, file)

    return


def export_all_data(us_states_df):
    """
    Load all state-level pickle files, combine them, and export the result.

    Parameters:
        us_states_df (pd.DataFrame): DataFrame containing state names under the column 'stname'.
        data_path (str): Directory where the pickle files are stored.
    """
    export_path = data_path / "substations.geojson"

    df_list = []

    for _, row in tqdm(
        us_states_df.iterrows(), total=len(us_states_df), desc="Combining States"
    ):
        path_in = os.path.join(data_path, f"{row['stname']}.pkl")

        if os.path.exists(path_in):
            try:
                df = pd.read_pickle(path_in)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_list.append(df)
            except Exception as e:
                print(f"Error loading {path_in}: {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_file(export_path, driver="GeoJSON")
        print(f"Exported combined data to {export_path}")
    else:
        print("No valid data found to combine.")


if __name__ == "__main__":

    # Create the data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    # print(data_path)
    # Define the URLs
    states_url = "https://gist.githubusercontent.com/dantonnoriega/bf1acd2290e15b91e6710b6fd3be0a53/raw/11d15233327c8080c9646c7e1f23052659db251d/us-state-ansi-fips.csv"
    counties_url = "https://raw.githubusercontent.com/kjhealy/us-county/master/data/census/fips-by-state.csv"

    # Save paths
    states_data_path = data_path / "us-state-ansi-fips.csv"
    counties_data_path = data_path / "fips-by-state.csv"

    download_states_counties = True

    if download_states_counties:
        # Download and save the states data
        download_data(states_url, states_data_path)

        # Download and save the counties data
        download_data(counties_url, counties_data_path)

    us_states_df = read_text_file(states_data_path)

    for idx, state in tqdm(
        us_states_df.iterrows(), total=len(us_states_df), desc="Processing States"
    ):

        # if not state['stname'] == 'Rhode Island':
        #     continue

        print(f"--Working on {state['stname']}")
        dl_per_state(state["stname"])

    export_all_data(us_states_df)
