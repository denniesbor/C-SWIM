"""
Identify and analyze geomagnetic storm periods using Kp and DST indices.
Authors: Dennies and Ed
"""

import os
import pandas as pd
import requests

from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from configs import setup_logger, get_data_dir

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/identify_storms.log")

data_loc = DATA_LOC
kp_dst_path = data_loc / "kp_ap_indices"
kp_dst_path.mkdir(parents=True, exist_ok=True)

logger.info(f"Working directory: {data_loc}")


def download_file(url, file_path):
    """Download file from URL with error handling."""
    response = requests.get(url)
    response.raise_for_status()
    with file_path.open("wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded and saved to {file_path}")


def parse_combined_kp_ap_file(file_path):
    """Parse combined Kp and Ap index data file into DataFrame."""

    columns = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "FracHour",
        "DecimalDate1",
        "DecimalDate2",
        "Kp",
        "Ap",
        "Flag",
    ]
    kp_df = pd.read_csv(file_path, sep="\s+", header=None, names=columns)
    kp_df["Kp_0to9"] = (kp_df["Kp"] * 3).round().astype(int)
    kp_df["Datetime"] = pd.to_datetime(kp_df[["Year", "Month", "Day", "Hour"]])

    return kp_df


def analyze_kp_ap_data(df):
    logger.info(f"Data range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    logger.info("\nBasic statistics:")
    logger.info(df[["Kp", "Kp_0to9", "Ap"]].describe())
    logger.info("\nDates with highest Ap index:")
    logger.info(df.nlargest(10, "Ap")[["Datetime", "Ap", "Kp"]])
    logger.info("\nDistribution of Kp values (0-9 scale):")
    logger.info(df["Kp_0to9"].value_counts().sort_index())


def parse_dst_file(file_path):
    """Parse DST data file into DataFrame."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if "|" in line:
            continue
        parts = line.split()
        if len(parts) == 4:
            datetime = parts[0] + " " + parts[1]
            doy, dst = parts[2], parts[3]
            data.append([datetime, doy, dst])

    return pd.DataFrame(data, columns=["Datetime", "DOY", "DST"])


def process_dst_data(df):
    """Clean and prepare DST DataFrame with datetime index."""
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["DOY"] = df["DOY"].astype(int)
    df["DST"] = df["DST"].astype(float)
    df = df[df["DST"] != 99999.99]
    df.set_index("Datetime", inplace=True)
    return df


def identify_storms(
    dst_df, df_kp, dst_threshold=-150, kp_threshold=8, time_delta_days=1
):
    """Identify storm periods by combining DST and Kp thresholds."""

    dst_df = dst_df.set_index(pd.to_datetime(dst_df.index))
    df_kp = df_kp.set_index(pd.to_datetime(df_kp.index))

    dst_storms = dst_df[dst_df["DST"] <= dst_threshold].index
    kp_storms = df_kp[df_kp["Kp"] >= kp_threshold].index

    all_storm_times = sorted(set(dst_storms) | set(kp_storms))

    storm_periods = []
    current_storm_start = None
    current_storm_end = None

    for time in all_storm_times:
        if current_storm_start is None:
            current_storm_start = time
            current_storm_end = time
        elif (time - current_storm_end) <= timedelta(days=time_delta_days):
            current_storm_end = time
        else:
            storm_periods.append((current_storm_start, current_storm_end))
            current_storm_start = time
            current_storm_end = time

    if current_storm_start is not None:
        storm_periods.append((current_storm_start, current_storm_end))

    # Merge overlapping or close storm periods
    merged_storm_periods = []
    for start, end in storm_periods:
        if not merged_storm_periods or start - merged_storm_periods[-1][1] > timedelta(
            days=time_delta_days * 2
        ):
            merged_storm_periods.append([start, end])
        else:
            merged_storm_periods[-1][1] = max(merged_storm_periods[-1][1], end)

    # Extend storm periods by time_delta
    extended_storm_periods = []
    for start, end in merged_storm_periods:
        extended_start = start - timedelta(days=time_delta_days)
        extended_end = end + timedelta(days=time_delta_days)
        extended_storm_periods.append((extended_start, extended_end))

    storm_df = pd.DataFrame(extended_storm_periods, columns=["Start", "End"])
    storm_df["Duration"] = storm_df["End"] - storm_df["Start"]

    return storm_df


def visualize_storm_periods(dst_df, kp_df):
    """Create visualization of DST and Kp indices with storm thresholds."""

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    # DST plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dst_df.index, dst_df["DST"], color="#1f77b4", label="DST")
    ax1.set_ylabel("DST Index (nT)")
    ax1.set_title(
        "Geomagnetic Activity: DST and Kp Indices", fontsize=16, fontweight="bold"
    )
    ax1.legend(loc="upper right")

    # Kp plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(kp_df.index, kp_df["Kp"], color="#ff7f0e", label="Kp")
    ax2.set_ylabel("Kp Index")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Add gridlines
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Adjust y-axis for Kp plot to show 0-9 range
    ax2.set_ylim(0, 9)
    ax2.set_yticks(range(0, 10))

    # Add horizontal lines for storm thresholds
    ax1.axhline(y=-50, color="r", linestyle="--", alpha=0.7)
    ax1.text(
        dst_df.index.max(),
        -50,
        "Moderate Storm",
        va="bottom",
        ha="right",
        color="r",
        alpha=0.7,
    )

    ax2.axhline(y=5, color="r", linestyle="--", alpha=0.7)
    ax2.text(
        kp_df.index.max(), 5, "G1 Storm", va="bottom", ha="right", color="r", alpha=0.7
    )

    plt.tight_layout()
    fig.savefig(kp_dst_path / "storm_periods.png", bbox_inches="tight", dpi=150)


def main():
    kp_url = "https://kp.gfz-potsdam.de/kpdata?startdate=1980-01-01&enddate=2024-09-16&format=kp2#kpdatadownload-143"
    kp_file_path = kp_dst_path / "kpdata.txt"
    download_file(kp_url, kp_file_path)

    kp_df = parse_combined_kp_ap_file(kp_file_path)
    analyze_kp_ap_data(kp_df)
    logger.info("\nKP Analysis complete. Check the current directory for output plots.")

    # DST data manually downloaded from Kyoto World Data Center for Geomagnetism
    # Server restricts to 25 years per request, so files were merged manually
    dst_file_path = kp_dst_path / "dst_data.txt"
    dst_df = parse_dst_file(dst_file_path)
    dst_df = process_dst_data(dst_df)

    kp_df.set_index("Datetime", inplace=True)

    logger.info("\nData processing complete.")
    logger.info(f"KP data shape: {kp_df.shape}")
    logger.info(f"DST data shape: {dst_df.shape}")

    storm_df = identify_storms(dst_df, kp_df)

    # Save storms from 1985 onwards
    storm_df = storm_df[storm_df["Start"] > pd.to_datetime("1985-01-01")]
    storm_df.to_csv((kp_dst_path / "storm_periods.csv"), index=False)

    # visualize_storm_periods(dst_df, kp_df)

    # Greg Lucas method for storm identification
    kp_url = "https://kp.gfz-potsdam.de/kpdata?startdate=1980-01-01&enddate=2024-09-16&format=kp2#kpdatadownload-143"
    kp_file_path = kp_dst_path / "kpdata.txt"
    if not os.path.exists(kp_file_path):
        download_file(kp_url, kp_file_path)

    kp_df = parse_combined_kp_ap_file(kp_file_path)
    analyze_kp_ap_data(kp_df)
    logger.info("\nKP Analysis complete. Check the current directory for output plots.")

    dst_file_path = kp_dst_path / "dst_data.txt"
    dst_df = parse_dst_file(dst_file_path)
    dst_df = process_dst_data(dst_df)

    kp_df.set_index("Datetime", inplace=True)

    import datetime

    dst_df["Kp"] = kp_df["1957":"2025"]["Kp"].resample("1H").ffill()
    storm_time_df = dst_df["1985":"2025"].copy()
    storm_time_df["storm"] = False

    delta_t = datetime.timedelta(days=1.5)

    list_of_times = []

    # Identify DST storms (DST < -140)
    curr_dst = -1000
    while curr_dst < -140:
        dsts = storm_time_df[~storm_time_df["storm"]]["DST"]
        dst_min = dsts.idxmin()
        storm_time_df.loc[dst_min - delta_t : dst_min + delta_t, "storm"] = True
        curr_dst = storm_time_df.loc[dst_min, "DST"]
        list_of_times.append(dst_min)

    # Identify Kp storms (Kp >= 8)
    curr_kp = 10
    while curr_kp >= 8:
        kp = storm_time_df[~storm_time_df["storm"]]["Kp"]
        kp_max = kp.idxmax()
        storm_time_df.loc[kp_max - delta_t : kp_max + delta_t, "storm"] = True
        curr_kp = storm_time_df.loc[kp_max, "Kp"]
        list_of_times.append(kp_max)

    logger.info(f"Initial number of storms: {len(list_of_times)}")
    temp = storm_time_df["storm"]
    # first row is a True preceded by a False
    fst = temp.index[temp & ~temp.shift(1).fillna(False)]

    # last row is a True followed by a False
    lst = temp.index[temp & ~temp.shift(-1).fillna(False)]

    storm_times = []
    for i in range(len(fst)):
        delta_t = lst[i] - fst[i]
        # Storm width in hours
        storm_hours = delta_t.days * 24 + delta_t.seconds / 3600
        if storm_hours < 3:
            continue

        storm_times.append((fst[i], lst[i]))

    logger.info(
        f"Number of storms after combining overlapping times: {len(storm_times)}"
    )

    nDst = 0
    nKp = 0
    nBoth = 0
    for x in storm_times:
        maxKp = storm_time_df.loc[x[0] : x[1]]["Kp"].max()
        minDst = storm_time_df.loc[x[0] : x[1]]["DST"].min()
        if maxKp >= 8 and minDst <= -140:
            nBoth += 1
        elif maxKp >= 8:
            nKp += 1
        elif minDst <= -140:
            nDst += 1
        else:
            logger.error(f"Error, shouldn't get here! maxKp: {maxKp}, minDst: {minDst}")

    logger.info(f"Number of events with both selections satisfied: {nBoth}")
    logger.info(f"Number of events with only Kp satisfied: {nKp}")
    logger.info(f"Number of events with only Dst satisfied: {nDst}")

    storm_df = pd.DataFrame(storm_times, columns=["Start", "End"])
    storm_df.to_csv((kp_dst_path / "storm_periods.csv"), index=False)


if __name__ == "__main__":

    main()
