"""Aggregate and cache ground GIC simulations for Gannon peak times.
Authors: Dennies and Ed. Oughton
"""

import gc
import os

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures

from configs import setup_logger, get_data_dir, GROUND_GIC_DIR

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/aggregate_gannon_gic.log")

data_path = Path(
    "/data/archives/nfs/spw-geophy/data"
)  # wll need to set up your data path here
ground_gic_folder = Path(GROUND_GIC_DIR)
peak_times_path = DATA_LOC / "peak_times_1.npy"


def read_ground_gic_simulations(
    ground_gic_folder, peak_times_path, cache_file=data_path / "gic_data.npz"
):
    """Load ground GIC batches, compute running stats, and cache results."""
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

    gnd_files = list(ground_gic_folder.glob("ground_gic_gannon*.csv"))
    logger.info(f"Found {len(gnd_files)} ground GIC files in {ground_gic_folder}")
    peak_times = np.load(peak_times_path)

    batch_size = 100

    def load_csv(file):
        try:
            df = pd.read_csv(file, engine="c")
            data = df.iloc[:, 1:].astype(np.float16).values
            return data, data.shape
        except (ValueError, pd.errors.ParserError) as e:
            logger.warning(f"Skipping bad file {file}: {e}")
            return None, None

    running_sum = None
    running_sum_sq = None
    total_count = 0
    expected_shape = None
    save_points = [200, 500, 1000, 1500, 2000]
    last_saved = 0

    sample_df = pd.read_csv(gnd_files[0], engine="c")
    substation_names = sample_df["Substation"].values

    for i in range(0, len(gnd_files), batch_size):
        batch_files = gnd_files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(gnd_files)-1)//batch_size + 1}"
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(load_csv, batch_files),
                    total=len(batch_files),
                    desc=f"Loading batch {i//batch_size + 1}",
                )
            )

        batch_data = []
        for data, shape in results:
            if data is not None:
                if expected_shape is None:
                    expected_shape = shape
                if shape == expected_shape:
                    batch_data.append(data)
                else:
                    logger.warning(
                        f"Shape mismatch: expected {expected_shape}, got {shape}"
                    )

        if not batch_data:
            logger.warning(f"No valid files in batch {i//batch_size + 1}")
            continue

        batch_array = np.stack(batch_data, axis=0)

        # Welford's online algorithm for running mean and variance
        if running_sum is None:
            running_sum = np.sum(batch_array, axis=0, dtype=np.float64)
            running_sum_sq = np.sum(batch_array**2, axis=0, dtype=np.float64)
        else:
            running_sum += np.sum(batch_array, axis=0, dtype=np.float64)
            running_sum_sq += np.sum(batch_array**2, axis=0, dtype=np.float64)

        total_count += len(batch_data)

        # Save intermediate results at checkpoints
        if total_count >= min(
            [sp for sp in save_points if sp > last_saved], default=float("inf")
        ):
            temp_mean = running_sum / total_count
            temp_variance = running_sum_sq / total_count - temp_mean**2
            temp_std = np.sqrt(temp_variance)
            temp_cache = cache_file.parent / f"gic_data_partial_{total_count}.npz"
            np.savez(
                temp_cache,
                data_array=np.array([]),
                peak_times=peak_times,
                median_values=temp_mean,
                mean_values=temp_mean,
                uncertainty_arr=np.array(
                    [temp_mean - 2 * temp_std, temp_mean + 2 * temp_std]
                ),
                substation_names=substation_names,
            )
            logger.info(
                f"Saved partial results to {temp_cache} with {total_count} files"
            )
            last_saved = total_count

        del batch_data, batch_array
        gc.collect()

    mean_values = running_sum / total_count
    variance = running_sum_sq / total_count - mean_values**2
    std_values = np.sqrt(variance)
    uncertainty_arr = np.array(
        [mean_values - 2 * std_values, mean_values + 2 * std_values]
    )
    median_values = mean_values

    np.savez(
        cache_file,
        data_array=np.array([]),
        peak_times=peak_times,
        median_values=median_values,
        mean_values=mean_values,
        uncertainty_arr=uncertainty_arr,
        substation_names=substation_names,
    )

    return (
        np.array([]),
        peak_times,
        median_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    )


if __name__ == "__main__":
    logger.info("Starting to read ground GIC simulations...")
    (
        data_array,
        peak_times,
        median_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    ) = read_ground_gic_simulations(ground_gic_folder, peak_times_path)

    logger.info(f"Peak times shape: {peak_times.shape}")
    logger.info(f"Median values shape: {median_values.shape}")
    logger.info(f"Mean values shape: {mean_values.shape}")
    logger.info(f"Uncertainty array shape: {uncertainty_arr.shape}")
    logger.info(f"Substation names: {substation_names[:5]}...")
