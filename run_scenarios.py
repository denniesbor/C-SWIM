#!/usr/bin/env python3
"""
Run GIC analysis scenarios.
1. Calculate storm maxima
2. Fit power law
3. Estimate GIC (includes admittance matrix)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from configs import setup_logger
from preprocess_wrapper import ProgressTracker

logger = setup_logger(log_file="logs/run_scenarios.log")


def run_script(script_name, extra_args=None):
    """Run a script in the scripts folder with timing via ProgressTracker."""
    script_path = Path(__file__).parent / "scripts" / script_name
    env = os.environ.copy()
    parent_dir = str(Path(__file__).parent)
    env["PYTHONPATH"] = f"{parent_dir}:{env.get('PYTHONPATH', '')}"
    extra_args = extra_args or []

    logger.info(f"Running {script_name}")
    progress = ProgressTracker(script_name)
    progress.start()
    try:
        subprocess.run(
            [sys.executable, str(script_path), *extra_args], check=True, env=env
        )
        progress.stop()
        logger.info(f"✓ {script_name} completed")
        return True
    except subprocess.CalledProcessError:
        progress.stop()
        logger.error(f"✗ {script_name} failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run GIC analysis scripts")
    parser.add_argument(
        "script",
        choices=["storm", "stat", "gic", "admittance", "all"],
        help="Which script to run",
    )
    parser.add_argument(
        "--gannon-only",
        action="store_true",
        help="Pass --gannon-only to est_gic.py",
    )
    args = parser.parse_args()

    script_map = {
        "storm": ["calc_storm_maxes.py"],
        "stat": ["stat_analysis.py"],
        "gic": ["est_gic.py"],
        "admittance": ["build_admittance_matrix.py"],
        "all": ["calc_storm_maxes.py", "stat_analysis.py", "est_gic.py"],
    }

    for script in script_map[args.script]:
        extra = (
            ["--gannon-only"] if (args.gannon_only and script == "est_gic.py") else []
        )
        if not run_script(script, extra):
            logger.error("Pipeline stopped due to error")
            sys.exit(1)

    logger.info("Completed successfully")


if __name__ == "__main__":
    main()
