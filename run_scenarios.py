#!/usr/bin/env python3
# Author: Dennies Bor
# Role: Master runner for GIC analysis scenarios
# Usage: python run_scenarios.py [storm|stat|gic|admittance|all] [--gannon-only]

import os
import sys
import argparse
import subprocess
import threading
import time
from pathlib import Path

from configs import setup_logger

logger = setup_logger(log_file="logs/run_scenarios.log")


class ProgressTracker:
    def __init__(self, script_name):
        self.script_name = script_name
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._show_progress, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        elapsed = time.time() - self.start_time
        logger.info(f"Completed {self.script_name} in {elapsed:.1f}s")

    def _show_progress(self):
        while self.running:
            elapsed = time.time() - self.start_time
            logger.info(f"Running {self.script_name}... {elapsed:.0f}s elapsed")
            time.sleep(30)


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
        logger.info(f"Completed {script_name}")
        return True
    except subprocess.CalledProcessError:
        progress.stop()
        logger.error(f"Failed {script_name}")
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
        "storm":      ["calc_storm_maxes.py"],
        "stat":       ["stat_analysis.py"],
        "gic":        ["est_gic.py"],
        "admittance": ["build_admittance_matrix.py"],
        "all":        ["calc_storm_maxes.py", "stat_analysis.py", "est_gic.py"],
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