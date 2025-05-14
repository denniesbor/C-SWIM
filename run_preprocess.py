#!/usr/bin/env python3
"""
Automated preprocessing pipeline for GIC analysis.
Manages downloads and preprocessing with proper dependency handling.

Author: Dennies Bor
Date: February 2025
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import setup_logger

logger = setup_logger(log_file="logs/run_preprocess.log")


class PreprocessingPipeline:
    def __init__(self, max_retries: int = 10, parallel: bool = True):
        self.preprocess_dir = Path(__file__).parent / "preprocess"
        self.max_retries = max_retries
        self.parallel = parallel
        self.completed_downloads = set()
        self.completed_preprocessors = set()

    def run_script(self, script_name: str, description: str, retries: int = 1) -> bool:
        """Run a single script with retry logic."""
        script_path = self.preprocess_dir / script_name

        # Set PYTHONPATH to include the parent directory for configs module
        env = os.environ.copy()
        parent_dir = str(Path(__file__).parent)
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{parent_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = parent_dir

        for attempt in range(retries):
            logger.info(
                f"Running {script_name} - {description} (Attempt {attempt + 1}/{retries})"
            )

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,  # Pass the modified environment
                )
                logger.info(f"Successfully completed {script_name}")
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed {script_name} (Attempt {attempt + 1}/{retries})")
                logger.error(f"Error: {e.stderr}")

                if attempt < retries - 1:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error(f"Maximum retries reached for {script_name}")
                    return False

        return False

    def run_download_phase(self):
        """Run all download scripts with appropriate retry logic."""
        logger.info("=" * 50)
        logger.info("PHASE 1: DATA DOWNLOAD")
        logger.info("=" * 50)

        download_tasks = [
            ("dl_intermagnet.py", "Download Intermagnet data (1990-present)", 5),
            (
                "dl_nrcan_pre_1990.py",
                "Download NRCAN pre-1990 magnetic data",
                self.max_retries,
            ),
            ("dl_power_grid_update.py", "Download OSM substations", 3),
            ("dl_usgs_pre_1990.py", "Download USGS pre-1990 magnetic data", 5),
        ]

        if self.parallel:
            # Run downloads in parallel
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {}

                for script, desc, retries in download_tasks:
                    future = executor.submit(self.run_script, script, desc, retries)
                    futures[future] = script

                for future in as_completed(futures):
                    script = futures[future]
                    if future.result():
                        self.completed_downloads.add(script)
                    else:
                        logger.warning(
                            f"Download script {script} failed after all retries"
                        )
        else:
            # Run downloads sequentially
            for script, desc, retries in download_tasks:
                if self.run_script(script, desc, retries):
                    self.completed_downloads.add(script)
                else:
                    logger.warning(f"Download script {script} failed after all retries")

        logger.info(
            f"Download phase complete. Successful: {len(self.completed_downloads)}/4"
        )

    def run_preprocessing_phase(self):
        """Run preprocessing scripts based on completed downloads."""
        logger.info("=" * 50)
        logger.info("PHASE 2: DATA PREPROCESSING")
        logger.info("=" * 50)

        preprocessing_tasks = []

        # Check dependencies and add tasks
        if all(
            script in self.completed_downloads
            for script in [
                "dl_intermagnet.py",
                "dl_nrcan_pre_1990.py",
                "dl_usgs_pre_1990.py",
            ]
        ):
            preprocessing_tasks.append(
                ("p_geomag_data.py", "Process geomagnetic data", 1)
            )
        else:
            logger.warning("Skipping p_geomag_data.py - missing required downloads")

        # Storm periods can run independently
        preprocessing_tasks.append(
            ("p_identify_storm_periods.py", "Identify storm periods", 1)
        )

        # Power grid needs OSM data
        if "dl_power_grid_update.py" in self.completed_downloads:
            preprocessing_tasks.append(
                ("p_power_grid.py", "Process power grid data", 1)
            )
        else:
            logger.warning("Skipping p_power_grid.py - missing OSM substation data")

        if self.parallel and len(preprocessing_tasks) > 1:
            # Run preprocessing in parallel where possible
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = {}

                for script, desc, retries in preprocessing_tasks:
                    future = executor.submit(self.run_script, script, desc, retries)
                    futures[future] = script

                for future in as_completed(futures):
                    script = futures[future]
                    if future.result():
                        self.completed_preprocessors.add(script)
                    else:
                        logger.error(f"Preprocessing script {script} failed")
        else:
            # Run sequentially
            for script, desc, retries in preprocessing_tasks:
                if self.run_script(script, desc, retries):
                    self.completed_preprocessors.add(script)
                else:
                    logger.error(f"Preprocessing script {script} failed")

        logger.info(
            f"Preprocessing phase complete. Successful: {len(self.completed_preprocessors)}/{len(preprocessing_tasks)}"
        )

    def run_pipeline(self):
        """Run the complete preprocessing pipeline."""
        start_time = time.time()

        logger.info("Starting GIC preprocessing pipeline")
        logger.info(f"Max retries: {self.max_retries}")
        logger.info(f"Parallel processing: {self.parallel}")

        # Phase 1: Downloads
        self.run_download_phase()

        # Phase 2: Preprocessing
        self.run_preprocessing_phase()

        # Summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Downloads completed: {len(self.completed_downloads)}/4")
        logger.info(f"Preprocessing completed: {len(self.completed_preprocessors)}")

        # List any failures
        all_downloads = {
            "dl_intermagnet.py",
            "dl_nrcan_pre_1990.py",
            "dl_power_grid_update.py",
            "dl_usgs_pre_1990.py",
        }
        failed_downloads = all_downloads - self.completed_downloads

        if failed_downloads:
            logger.warning(f"Failed downloads: {failed_downloads}")

        if len(self.completed_preprocessors) < 3:
            logger.warning("Some preprocessing scripts did not complete successfully")

        return len(failed_downloads) == 0 and len(self.completed_preprocessors) >= 2


def main():
    parser = argparse.ArgumentParser(description="Run GIC preprocessing pipeline")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum retries for unreliable downloads (default: 10)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run scripts sequentially instead of in parallel",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Run only download phase"
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run only preprocessing phase (assumes downloads complete)",
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        max_retries=args.max_retries, parallel=not args.sequential
    )

    try:
        if args.download_only:
            pipeline.run_download_phase()
        elif args.preprocess_only:
            # Assume all downloads completed for preprocessing only
            pipeline.completed_downloads = {
                "dl_intermagnet.py",
                "dl_nrcan_pre_1990.py",
                "dl_power_grid_update.py",
                "dl_usgs_pre_1990.py",
            }
            pipeline.run_preprocessing_phase()
        else:
            success = pipeline.run_pipeline()
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
