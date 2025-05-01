# ----------------------------------------------------------------------
# Script to download geomagnetic data from the INTERMAGNET GIN from
# identified storm periods between 1991 and 2024.
#  To get the old data use download_nrcan_old.py, and download_usgs_old.py scripts
# Author: Dennies Bor-GMU
# ----------------------------------------------------------------------
import os
import sys
from pathlib import Path
import shutil
import re
from urllib import request
from urllib.error import URLError
from contextlib import ExitStack
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing
from functools import partial

from configs import setup_logger, get_data_dir

# Get data data log and configure logger
DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/dl_intermag.log")

data_path = DATA_LOC
data_path.mkdir(exist_ok=True)

storm_df_path = data_path / "kp_ap_indices" / "storm_periods.csv"

# Load storm periods
storm_df = pd.read_csv(storm_df_path)
storm_df["Start"] = pd.to_datetime(storm_df["Start"])
storm_df["End"] = pd.to_datetime(storm_df["End"])
storm_df["Duration"] = storm_df["End"] - storm_df["Start"]

# Filter from > "1991-01-01"
storm_df = storm_df[storm_df["Start"] > datetime(1991, 1, 1)]

usgs_obs = list(
    set(
        [
            obs.upper()
            for obs in [
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
        ]
    )
)
nrcan_obs = [
    "ALE",
    "BLC",
    "BRD",
    "CBB",
    "FCC",
    "IQA",
    "MEA",
    "OTT",
    "RES",
    "STJ",
    "VIC",
    "YKC",
]

# All obs
all_obs = usgs_obs + nrcan_obs


# Configurable parameters - these variables control use of proxy
# servers at the users site - there's also an option to use
# authentication - change them as required (though the
# defaults should work OK)
gin_username = ""
gin_password = ""
proxy_address = ""
n_retries = 4
loc = data_path / "geomag_data"


# ----------------------------------------------------------------------
# upd_op_co:         Increment and serialize the operation count
# Input parameters:  op_count the current operation number
# Returns:           op_count is incremented and returned (and written to disk)
def upd_op_co(op_count):
    op_count = op_count + 1
    with open("counter.dat", "w") as f:
        f.write("%d" % op_count)
    return op_count


# ----------------------------------------------------------------------
# safemd:            Safely create a folder (no error if it already exists)
# Input parameters:  folder the directory to create
#                    op_number the operation number for this call
#                    op_count the current operation number
# Returns:           op_count is incremented and returned (and written to disk)
def safemd(folder, op_number, op_count):
    if op_number >= op_count:
        if op_number == 0:
            logger.info("Creating directories...")
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            logger.info("Error: unable to create directory: " + str(folder))
            sys.exit(1)
        op_count = upd_op_co(op_count)
    return op_count


# ----------------------------------------------------------------------
# getfile:           Download a file from a web server
# Input parameters:  url URL to download from
#                    local_file local file to download to
#                    n_retries number of retries to do
#                    op_number the operation number for this call
#                    gin_username the username of the GIN (or empty string)
#                    gin_password the username of the GIN (or empty string)
#                    proxy_address address of proxy server (or empty string)
#                    n_folders the number of folders to create
#                    n_downloads the number of files to download
#                    op_count the current operation number
# Returns:           op_count is incremented and returned (and written to disk)
def getfile(
    url,
    local_file,
    n_retries,
    op_number,
    gin_username,
    gin_password,
    proxy_address,
    n_folders,
    n_downloads,
    op_count,
):
    if op_number >= op_count:
        # tell the user what's going on
        percent = ((op_number - n_folders) * 100) / n_downloads
        logger.info("%d%% - downloading file: %s" % (percent, local_file))

        # Check if the file already exists, pass
        if os.path.exists(local_file):
            logger.info("File already exists: " + local_file)  # File already exists
            op_count = upd_op_co(op_count)
            return op_count

        # remove any existing file
        try:
            os.remove(local_file)
        except FileNotFoundError:
            pass
        except OSError:
            logger.info("Error: unable to remove file: " + str(local_file))
            sys.exit(1)

        # handle authentication and proxy server
        proxy = auth = None
        if len(proxy_address) > 0:
            proxy = request.ProxyHandler(
                {"http": proxy_address, "https": proxy_address}
            )
        if len(gin_username) > 0:
            pwd_mgr = request.HTTPPasswordMgrWithPriorAuth()
            pwd_mgr.add_password(
                None,
                "https://imag-data.bgs.ac.uk/GIN_V1",
                gin_username,
                gin_password,
                is_authenticated=True,
            )
            auth = request.HTTPBasicAuthHandler(pwd_mgr)
        if url.startswith("https"):
            default_handler = request.HTTPSHandler
        else:
            default_handler = request.HTTPHandler
        if auth and proxy:
            opener = request.build_opener(proxy, auth, default_handler)
        elif auth:
            opener = request.build_opener(auth, default_handler)
        elif proxy:
            opener = request.build_opener(proxy, default_handler)
        else:
            opener = request.build_opener(default_handler)

        # download the file
        success = False
        while (not success) and (n_retries > 0):
            try:
                with opener.open(url) as f_in:
                    with open(local_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out, 4096)
                success = True
            except (URLError, IOError, OSError):
                n_retries -= 1
        if not success:
            logger.info("Error: cannot download " + local_file)
            sys.exit(1)

        # rename IAGA-2002 files
        dt = None
        try:
            with open(local_file, "r") as f:
                for line in f.readlines():
                    if re.search("^ Data Type", line):
                        dt = line[24:25].lower()
        except (IOError, OSError):
            pass
        if dt:
            if not dt.isalpha():
                dt = None
        if dt:
            new_local_file = (
                local_file[: len(local_file) - 7]
                + dt
                + local_file[len(local_file) - 7 :]
            )
            try:
                os.remove(new_local_file)
            except (FileNotFoundError, OSError):
                pass
            try:
                os.rename(local_file, new_local_file)
            except (IOError, OSError):
                logger.info(
                    "Warning: unable to rename " + local_file + " to " + new_local_file
                )
        else:
            logger.info(
                "Warning: unable to determine data type for renaming of " + local_file
            )

        op_count = upd_op_co(op_count)
    return op_count


# Download storm data
def download_observatory_data(observatory, storm_df, n_folders, n_downloads):
    op_count = 0
    for index, storm in storm_df.iterrows():
        start_date = storm["Start"].date()
        end_date = storm["End"].date()
        current_date = start_date

        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day

            folder = os.path.join(loc, str(year), observatory)
            op_count = safemd(folder, op_count, op_count)

            local_file = os.path.join(
                folder, f"{observatory.lower()}{year}{month:02d}{day:02d}min.min"
            )
            url = (
                f"https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=IAGA2002"
                f"&testObsys=0&observatoryIagaCode={observatory}&samplesPerDay=1440&orientation=Native"
                f"&publicationState=adj-or-rep&recordTermination=UNIX"
                f"&dataStartDate={year}-{month:02d}-{day:02d}&dataDuration=1"
            )

            op_count = getfile(
                url,
                local_file,
                n_retries,
                op_count,
                gin_username,
                gin_password,
                proxy_address,
                n_folders,
                n_downloads,
                op_count,
            )

            current_date += timedelta(days=1)

    logger.info(f"100% - data download complete for {observatory}")


def download_storm_data(storm_df, observatories):
    n_folders = storm_df["Start"].dt.year.nunique()
    n_downloads = storm_df["Duration"].dt.days.sum() * len(observatories)

    # Create a partial function with fixed arguments
    download_func = partial(
        download_observatory_data,
        storm_df=storm_df,
        n_folders=n_folders,
        n_downloads=n_downloads,
    )

    # Use multiprocessing to download data for each observatory in parallel
    with multiprocessing.Pool(
        processes=min(len(observatories), multiprocessing.cpu_count())
    ) as pool:
        pool.map(download_func, observatories)

    # Tidy up
    try:
        os.remove("counter.dat")
    except FileNotFoundError:
        pass


if __name__ == "__main__":

    # Download data for identified storm periods
    download_storm_data(storm_df, all_obs)
