"""
Aggregate ground GIC simulations from compact batches and/or CSVs into summary statistics.
Supports incremental partial saves per machine and final merge across distributed runs.
Authors: Dennies Bor, Ed Oughton
"""

import argparse
import gc
import json
import socket
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from configs import setup_logger, get_data_dir, GANNON_GND_GIC_DIR

DATA_LOC = Path(get_data_dir())
COMPACT_DIR = DATA_LOC / "gannon_gic_compact"
CSV_DIR = Path(GANNON_GND_GIC_DIR)
OUT_DIR = DATA_LOC / "gannon_gic_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger(log_file=str(OUT_DIR / "aggregate_gannon.log"))


def extract_iteration_idx(filename):
    """Extract the iteration index from a GIC output filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return None


def load_csv(file):
    """Load a single GIC CSV and return float16 array and its shape."""
    try:
        df = pd.read_csv(file, engine="c")
        data = df.iloc[:, 1:].astype(np.float16).values
        return data, data.shape
    except Exception as e:
        logger.warning(f"Skipping bad file {file}: {e}")
        return None, None


def get_machine_id():
    """Get unique machine identifier."""
    return socket.gethostname()


def load_partials(out_dir):
    """Load all partial files and return merged running stats."""
    partial_files = sorted(Path(out_dir).glob("partial_*.npz"))
    if not partial_files:
        return None, None, 0, set(), None

    logger.info(f"Found {len(partial_files)} partial files")

    running_sum = None
    running_sum_sq = None
    total_count = 0
    processed_iters = set()
    substation_names = None

    for pf in partial_files:
        data = np.load(pf, allow_pickle=True)

        if substation_names is None:
            substation_names = data["substation_names"]

        if running_sum is None:
            running_sum = data["_running_sum"]
            running_sum_sq = data["_running_sum_sq"]
        else:
            running_sum += data["_running_sum"]
            running_sum_sq += data["_running_sum_sq"]

        total_count += int(data["n_iterations"])
        processed_iters.update(data["completed_iterations"].tolist())

        logger.info(f"Loaded {pf.name}: {data['n_iterations']} iterations")

    return running_sum, running_sum_sq, total_count, processed_iters, substation_names


def save_partial(
    out_dir,
    running_sum,
    running_sum_sq,
    total_count,
    processed_iters,
    substation_names,
    peak_times,
):
    """Save partial results for this machine."""
    machine_id = get_machine_id()
    partial_file = Path(out_dir) / f"partial_{machine_id}.npz"

    mean_values = running_sum / total_count
    variance = running_sum_sq / total_count - mean_values**2
    std_values = np.sqrt(np.maximum(variance, 0))
    uncertainty_arr = np.array(
        [mean_values - 2 * std_values, mean_values + 2 * std_values]
    )

    np.savez(
        partial_file,
        # Final stats format (for compatibility)
        data_array=np.array([]),
        peak_times=peak_times,
        median_values=mean_values,
        mean_values=mean_values,
        uncertainty_arr=uncertainty_arr,
        substation_names=substation_names,
        # For merging
        _running_sum=running_sum,
        _running_sum_sq=running_sum_sq,
        n_iterations=total_count,
        completed_iterations=np.array(sorted(processed_iters), dtype=np.int32),
        machine_id=machine_id,
    )

    logger.info(f"Saved partial to {partial_file} ({total_count} iterations)")
    return partial_file


def save_final(
    out_dir, running_sum, running_sum_sq, total_count, substation_names, peak_times
):
    """Save final aggregated results."""
    cache_file = Path(out_dir) / "gic_data.npz"

    mean_values = running_sum / total_count
    variance = running_sum_sq / total_count - mean_values**2
    std_values = np.sqrt(np.maximum(variance, 0))
    uncertainty_arr = np.array(
        [mean_values - 2 * std_values, mean_values + 2 * std_values]
    )

    np.savez(
        cache_file,
        data_array=np.array([]),
        peak_times=peak_times,
        median_values=mean_values,
        mean_values=mean_values,
        uncertainty_arr=uncertainty_arr,
        substation_names=substation_names,
    )

    logger.info(f"Saved final stats to {cache_file} ({total_count} iterations)")
    return cache_file


def aggregate(
    source="all",
    compact_dir=COMPACT_DIR,
    csv_dir=CSV_DIR,
    peak_times_path=DATA_LOC / "peak_times_1.npy",
    out_dir=OUT_DIR,
    batch_size=100,
    finalize=False,
):
    """
    Aggregate GIC data from compact batches and/or CSVs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    peak_times = np.load(peak_times_path)

    # Load existing partials
    running_sum, running_sum_sq, total_count, processed_iters, substation_names = (
        load_partials(out_dir)
    )
    expected_shape = None

    if running_sum is not None:
        expected_shape = running_sum.shape
        logger.info(
            f"Loaded {total_count} iterations from partials, {len(processed_iters)} unique"
        )

    new_count = 0

    # Process compact batches
    if source in ("compact", "all"):
        batch_files = sorted(Path(compact_dir).glob("batch_*.npz"))
        if batch_files:
            logger.info(f"Found {len(batch_files)} compact batch files")

            ref_file = Path(compact_dir) / "reference.json"
            if ref_file.exists() and substation_names is None:
                with open(ref_file) as f:
                    ref = json.load(f)
                substation_names = np.array(ref["substations"])
                expected_shape = (ref["n_substations"], ref["n_scenarios"])

            for bf in tqdm(batch_files, desc="Processing compact batches"):
                data = np.load(bf)
                cube = data["cube"]
                iterations = data["iterations"]

                if expected_shape is None:
                    expected_shape = cube.shape[1:]

                for i, arr in enumerate(cube):
                    iter_idx = int(iterations[i])

                    if iter_idx in processed_iters:
                        continue

                    if arr.shape != expected_shape:
                        logger.warning(
                            f"Shape mismatch: {arr.shape} vs {expected_shape}"
                        )
                        continue

                    if running_sum is None:
                        running_sum = arr.astype(np.float64)
                        running_sum_sq = (arr**2).astype(np.float64)
                    else:
                        running_sum += arr.astype(np.float64)
                        running_sum_sq += (arr**2).astype(np.float64)

                    total_count += 1
                    new_count += 1
                    processed_iters.add(iter_idx)

                del cube, data
                gc.collect()

            logger.info(f"Added {new_count} new iterations from compact batches")

    # Process CSVs
    if source in ("csv", "all"):
        csv_files = sorted(Path(csv_dir).glob("ground_gic_gannon*.csv"))
        if csv_files:
            # Filter already processed
            csv_files = [
                f
                for f in csv_files
                if extract_iteration_idx(f.name) not in processed_iters
            ]

            if csv_files:
                logger.info(f"Found {len(csv_files)} new CSV files to process")

                if substation_names is None:
                    sample_df = pd.read_csv(csv_files[0], engine="c")
                    substation_names = sample_df["Substation"].values

                csv_new_count = 0
                for i in range(0, len(csv_files), batch_size):
                    batch_files = csv_files[i : i + batch_size]
                    logger.info(
                        f"Processing CSV batch {i // batch_size + 1}/{(len(csv_files) - 1) // batch_size + 1}"
                    )

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = list(
                            tqdm(
                                executor.map(load_csv, batch_files),
                                total=len(batch_files),
                                desc=f"Loading CSV batch {i // batch_size + 1}",
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
                        logger.warning(
                            f"No valid files in CSV batch {i // batch_size + 1}"
                        )
                        continue

                    batch_array = np.stack(batch_data, axis=0)

                    if running_sum is None:
                        running_sum = np.sum(batch_array, axis=0, dtype=np.float64)
                        running_sum_sq = np.sum(
                            batch_array**2, axis=0, dtype=np.float64
                        )
                    else:
                        running_sum += np.sum(batch_array, axis=0, dtype=np.float64)
                        running_sum_sq += np.sum(
                            batch_array**2, axis=0, dtype=np.float64
                        )

                    csv_new_count += len(batch_data)
                    total_count += len(batch_data)

                    # Track processed iterations
                    for bf in batch_files[: len(batch_data)]:
                        idx = extract_iteration_idx(bf.name)
                        if idx is not None:
                            processed_iters.add(idx)

                    del batch_data, batch_array
                    gc.collect()

                new_count += csv_new_count
                logger.info(f"Added {csv_new_count} new iterations from CSVs")

    if total_count == 0:
        logger.error("No data aggregated")
        return None

    logger.info(f"Total iterations: {total_count}")

    # Check if there's more data to process
    has_more_compact = bool(list(Path(compact_dir).glob("batch_*.npz")))
    has_more_csv = bool(
        [
            f
            for f in Path(csv_dir).glob("ground_gic_gannon*.csv")
            if extract_iteration_idx(f.name) not in processed_iters
        ]
    )

    if finalize or (not has_more_compact and not has_more_csv and new_count == 0):
        # No more data, save final
        save_final(
            out_dir,
            running_sum,
            running_sum_sq,
            total_count,
            substation_names,
            peak_times,
        )
    else:
        # Save partial for this machine
        save_partial(
            out_dir,
            running_sum,
            running_sum_sq,
            total_count,
            processed_iters,
            substation_names,
            peak_times,
        )

    mean_values = running_sum / total_count
    variance = running_sum_sq / total_count - mean_values**2
    std_values = np.sqrt(np.maximum(variance, 0))
    uncertainty_arr = np.array(
        [mean_values - 2 * std_values, mean_values + 2 * std_values]
    )

    return (
        np.array([]),
        peak_times,
        mean_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    )


def main():
    """Parse arguments and run the aggregation pipeline."""
    parser = argparse.ArgumentParser(description="Aggregate Gannon GIC simulations")
    parser.add_argument(
        "--source",
        choices=["compact", "csv", "all"],
        default="all",
        help="Data source: compact batches, CSVs, or all (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of CSVs to load at once (default: 100)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: gannon_gic_processed/)",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Force save final gic_data.npz even if more data might exist",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    logger.info(f"Starting aggregation from source: {args.source}")
    logger.info(f"Machine ID: {get_machine_id()}")

    results = aggregate(
        source=args.source,
        out_dir=out_dir,
        batch_size=args.batch_size,
        finalize=args.finalize,
    )

    if results:
        _, peak_times, median_values, mean_values, uncertainty_arr, substation_names = (
            results
        )
        logger.info(f"Peak times shape: {peak_times.shape}")
        logger.info(f"Mean values shape: {mean_values.shape}")
        logger.info(f"Uncertainty array shape: {uncertainty_arr.shape}")
        logger.info(f"Substation names: {substation_names[:5]}...")


if __name__ == "__main__":
    main()
