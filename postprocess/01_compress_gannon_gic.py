"""Compact Gannon GIC simulations with explicit index tracking."""

import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from configs import setup_logger, get_data_dir, GANNON_GND_GIC_DIR

DATA_LOC = Path(get_data_dir())
SRC_DIR = Path(GANNON_GND_GIC_DIR)
OUT_DIR = DATA_LOC / "gannon_gic_compact"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger(log_file=str(OUT_DIR / "compact_gannon.log"))


def extract_iteration_idx(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return None


def load_csv(file):
    try:
        idx = extract_iteration_idx(file.name)
        if idx is None:
            return None, None, None, None
        df = pd.read_csv(file, engine="c")
        subs = df["Substation"].astype(str).values
        data = df.iloc[:, 1:].astype(np.float32).values
        cols = list(df.columns[1:])
        return idx, subs, cols, data
    except Exception as e:
        logger.warning(f"Skipping bad file {file}: {e}")
        return None, None, None, None


def compact_and_track(chunk_size=5, batch_size=100):
    """Process CSVs into compact batches. Loads chunk_size files at a time, saves every batch_size iterations."""
    csv_files = sorted(SRC_DIR.glob("ground_gic_gannon*.csv"))
    if not csv_files:
        logger.info("No Gannon CSVs to process")
        return

    file_idx_pairs = []
    for f in csv_files:
        idx = extract_iteration_idx(f.name)
        if idx is not None:
            file_idx_pairs.append((idx, f))

    file_idx_pairs.sort(key=lambda x: x[0])
    logger.info(f"Found {len(file_idx_pairs)} valid Gannon CSV files")

    if not file_idx_pairs:
        return

    sample_df = pd.read_csv(file_idx_pairs[0][1], engine="c")
    substation_names = sample_df["Substation"].astype(str).values
    scenario_cols = list(sample_df.columns[1:])
    n_subs = len(substation_names)
    n_scen = len(scenario_cols)

    logger.info(f"Substations: {n_subs}, Scenarios: {n_scen}")

    ref_file = OUT_DIR / "reference.json"
    with open(ref_file, "w") as f:
        json.dump(
            {
                "n_substations": n_subs,
                "n_scenarios": n_scen,
                "substations": substation_names.tolist(),
                "scenarios": scenario_cols,
            },
            f,
            indent=2,
        )

    batch_num = 0
    all_batch_info = []

    # Accumulate until batch_size
    batch_iterations = []
    batch_data = []

    for i in tqdm(range(0, len(file_idx_pairs), chunk_size), desc="Processing"):
        chunk_pairs = file_idx_pairs[i : i + chunk_size]
        chunk_files = [f for _, f in chunk_pairs]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_csv, chunk_files))

        for idx, subs, cols, data in results:
            if idx is not None and data is not None:
                if data.shape == (n_subs, n_scen):
                    batch_iterations.append(idx)
                    batch_data.append(data)
                else:
                    logger.warning(f"Shape mismatch for iter {idx}: {data.shape}")

        # Save batch when we hit batch_size
        if len(batch_data) >= batch_size:
            _save_batch(
                OUT_DIR, batch_num, batch_data, batch_iterations, all_batch_info
            )
            batch_num += 1
            batch_iterations = []
            batch_data = []
            gc.collect()

    # Save remaining
    if batch_data:
        _save_batch(OUT_DIR, batch_num, batch_data, batch_iterations, all_batch_info)
        batch_num += 1

    # Final metadata
    all_iters = []
    for b in all_batch_info:
        all_iters.extend(b["iterations"])

    meta = {
        "completed_iterations": sorted(all_iters),
        "n_completed": len(all_iters),
        "batches": all_batch_info,
    }
    with open(OUT_DIR / "completed.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Completed {len(all_iters)} iterations in {batch_num} batches")


def _save_batch(out_dir, batch_num, batch_data, batch_iterations, all_batch_info):
    """Save a single batch to disk."""
    sort_order = np.argsort(batch_iterations)
    batch_cube = np.stack(batch_data, axis=0).astype(np.float32)[sort_order]
    batch_iters = np.array(sorted(batch_iterations), dtype=np.int32)

    batch_file = out_dir / f"batch_{batch_num:04d}.npz"
    np.savez_compressed(batch_file, cube=batch_cube, iterations=batch_iters)

    all_batch_info.append(
        {
            "batch_file": batch_file.name,
            "iterations": batch_iters.tolist(),
            "n_iterations": len(batch_iters),
            "iter_min": int(batch_iters.min()),
            "iter_max": int(batch_iters.max()),
        }
    )

    logger.info(
        f"Saved {batch_file.name}: {len(batch_iters)} iterations [{batch_iters.min()}-{batch_iters.max()}]"
    )


def get_missing(total_target=2000):
    """Return iterations not yet processed."""
    meta_file = OUT_DIR / "completed.json"
    if not meta_file.exists():
        logger.warning("No completed.json found")
        return list(range(total_target))

    with open(meta_file) as f:
        meta = json.load(f)

    done = set(meta["completed_iterations"])
    missing = sorted([i for i in range(total_target) if i not in done])
    logger.info(f"Completed: {len(done)}, Missing: {len(missing)}")
    return missing


def merge_batches(output_path=None):
    """Merge all batches into single file."""
    if output_path is None:
        output_path = OUT_DIR / "gannon_gic_merged.npz"

    with open(OUT_DIR / "reference.json") as f:
        ref = json.load(f)

    batch_files = sorted(OUT_DIR.glob("batch_*.npz"))
    all_cubes = []
    all_iters = []

    for bf in tqdm(batch_files, desc="Merging"):
        data = np.load(bf)
        all_cubes.append(data["cube"])
        all_iters.extend(data["iterations"].tolist())

    merged_cube = np.concatenate(all_cubes, axis=0)
    merged_iters = np.array(all_iters, dtype=np.int32)

    sort_idx = np.argsort(merged_iters)
    merged_cube = merged_cube[sort_idx]
    merged_iters = merged_iters[sort_idx]

    np.savez_compressed(
        output_path,
        cube=merged_cube,
        substations=np.array(ref["substations"]),
        scenarios=np.array(ref["scenarios"]),
        iterations=merged_iters,
    )

    logger.info(f"Merged {len(merged_iters)} iterations -> {output_path}")
    logger.info(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    compact_and_track(chunk_size=5, batch_size=100)
