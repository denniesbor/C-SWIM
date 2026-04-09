#!/usr/bin/env python3
# Author: Dennies Bor
# Role: Master runner for preprocessing pipeline
# Usage: python run_preprocess.py [--steps 01 02] [--dry-run]

import argparse
import subprocess
import sys
from pathlib import Path

from configs import setup_logger

logger = setup_logger(log_file="logs/run_preprocess.log")

PREPROCESS_DIR = Path(__file__).parent / "preprocess"

STEPS = [
    ("01", "01_identify_storm_periods.py",       "Identify geomagnetic storm periods"),
    ("02", "02_dl_nrcan_pre_1990.py",             "Download NRCAN pre-1990 magnetic data"),
    ("03", "03_dl_usgs_pre_1990.py",              "Download USGS pre-1990 magnetic data"),
    ("04", "04_dl_intermagnet.py",                "Download Intermagnet data 1991-present"),
    ("05", "05_p_geomag_data.py",                 "Process geomagnetic data"),
    ("06", "06_dl_power_grid_update.py",          "Download OSM substations"),
    ("07", "07_dl_substations_relations_only.py", "Download OSM substation relations"),
    ("08", "08_p_power_grid.py",                  "Process power grid data"),
]

STEP_MAP = {n: (s, l) for n, s, l in STEPS}


def run_step(script: str, label: str, dry_run: bool = False) -> bool:
    path = PREPROCESS_DIR / script
    if not path.exists():
        logger.error(f"Script not found: {path}")
        return False
    logger.info(f"Running: {label}")
    if dry_run:
        logger.info(f"  [dry-run] would run: python {path}")
        return True
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        logger.error(f"Failed: {script} (exit code {result.returncode})")
        return False
    logger.info(f"Done: {label}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--steps",     nargs="+", default=None,
                        help="Run specific steps e.g. --steps 02 03")
    parser.add_argument("--from-step", type=str,  default=None,
                        help="Start from this step e.g. --from-step 05")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--list",      action="store_true")
    args = parser.parse_args()

    if args.list:
        for n, s, l in STEPS:
            logger.info(f"  {n}  {l}")
        return

    all_ids = [n for n, _, _ in STEPS]
    steps   = STEPS

    if args.steps:
        invalid = [s for s in args.steps if s not in STEP_MAP]
        if invalid:
            logger.error(f"Unknown steps: {invalid}")
            sys.exit(1)
        steps = [(n, *STEP_MAP[n]) for n in args.steps]
    elif args.from_step:
        if args.from_step not in all_ids:
            logger.error(f"Step {args.from_step} not found")
            sys.exit(1)
        steps = STEPS[all_ids.index(args.from_step):]

    logger.info(f"Running {len(steps)} preprocess steps")
    for n, script, label in steps:
        logger.info(f"  Step {n}: {label}")

    for n, script, label in steps:
        if not run_step(script, label, dry_run=args.dry_run):
            logger.error(f"Pipeline stopped at step {n}")
            sys.exit(1)

    logger.info("Preprocessing pipeline complete")


if __name__ == "__main__":
    main()