#!/usr/bin/env python3
# Author: Dennies Bor
# Role: Master runner for post-processing pipeline
# Usage: python run_postprocess.py [--steps 01 02] [--dry-run]

import argparse
import subprocess
import sys
from pathlib import Path

from configs import setup_logger

logger = setup_logger(log_file="logs/run_postprocess.log")

POSTPROCESS_DIR = Path(__file__).parent / "postprocess"

STEPS = [
    ("01", "01_compress_gannon_gic.py", "Compress Gannon GIC outputs"),
    ("02", "02_agg_gannon_gic.py",      "Aggregate Gannon ground GIC"),
    ("03", "03_calc_eff_gic.py",         "Compute effective GIC"),
    ("04", "04_agg_scenario_gic.py",     "Aggregate scenario GIC results"),
]

STEP_MAP = {n: (s, l) for n, s, l in STEPS}


def run_step(script: str, label: str, dry_run: bool = False) -> bool:
    path = POSTPROCESS_DIR / script
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
    parser = argparse.ArgumentParser(description="Run post-processing pipeline")
    parser.add_argument("--steps",     nargs="+", default=None,
                        help="Run specific steps e.g. --steps 02 03")
    parser.add_argument("--from-step", type=str,  default=None,
                        help="Start from this step e.g. --from-step 03")
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

    logger.info(f"Running {len(steps)} post-processing steps")
    for n, script, label in steps:
        logger.info(f"  Step {n}: {label}")

    for n, script, label in steps:
        if not run_step(script, label, dry_run=args.dry_run):
            logger.error(f"Pipeline stopped at step {n}")
            sys.exit(1)

    logger.info("Post-processing pipeline complete")


if __name__ == "__main__":
    main()