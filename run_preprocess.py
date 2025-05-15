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

# Add parent directory to path so configs can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import setup_logger

logger = setup_logger(log_file="logs/run_preprocess.log")


class PreprocessingPipeline:
    def __init__(self, max_retries: int = 10, parallel: bool = True):
        self.root_dir = Path(__file__).parent
        self.preprocess_dir = self.root_dir / "preprocess"
        self.max_retries = max_retries
        self.parallel = parallel
        self.completed_steps = set()
    
    def run_script(self, script_name: str, description: str, retries: int = 1) -> bool:
        """Run a single script with retry logic."""
        script_path = self.preprocess_dir / script_name
        
        for attempt in range(retries):
            logger.info(f"Running {script_name} - {description} (Attempt {attempt + 1}/{retries})")
            
            try:
                # Run with the parent directory as the working directory
                # This ensures configs can be imported
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=self.root_dir
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
    

    
    def run_storm_identification(self):
        """Run storm identification first as it's needed by downloads."""
        logger.info("="*50)
        logger.info("PHASE 1: STORM IDENTIFICATION")
        logger.info("="*50)
        
        if self.run_script("p_identify_storm_periods.py", "Identify storm periods", 1):
            self.completed_steps.add("p_identify_storm_periods.py")
            return True
        else:
            logger.error("Failed to identify storm periods - this is required for downloads")
            return False
    
    def run_download_phase(self):
        """Run all download scripts with appropriate retry logic."""
        logger.info("="*50)
        logger.info("PHASE 2: DATA DOWNLOAD")
        logger.info("="*50)
        
        # Note: Pre-1990 scripts need many attempts because different magnetic stations
        # may be temporarily unavailable. The scripts skip already-downloaded data.
        
        download_tasks = [
            ("dl_intermagnet.py", "Download Intermagnet data (1990-present)", 3),
            ("dl_nrcan_pre_1990.py", "Download NRCAN pre-1990 magnetic data", self.max_retries),
            ("dl_power_grid_update.py", "Download OSM substations", 3),
            ("dl_usgs_pre_1990.py", "Download USGS pre-1990 magnetic data", self.max_retries),
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
                        self.completed_steps.add(script)
                    else:
                        # For pre-1990 scripts, this isn't necessarily a failure
                        if "pre_1990" in script:
                            logger.info(f"{script} completed attempts - some stations may still be missing data")
                            self.completed_steps.add(script)  # Mark as complete anyway
                        else:
                            logger.warning(f"Download script {script} failed after all retries")
        else:
            # Run downloads sequentially
            for script, desc, retries in download_tasks:
                if self.run_script(script, desc, retries):
                    self.completed_steps.add(script)
                else:
                    # For pre-1990 scripts, this isn't necessarily a failure
                    if "pre_1990" in script:
                        logger.info(f"{script} completed attempts - some stations may still be missing data")
                        self.completed_steps.add(script)  # Mark as complete anyway
                    else:
                        logger.warning(f"Download script {script} failed after all retries")
        
        download_count = len([s for s in self.completed_steps if s.startswith("dl_")])
        logger.info(f"Download phase complete. Successful: {download_count}/4")
        logger.info("Note: Pre-1990 scripts may need additional runs to get all available data")
    
    def run_preprocessing_phase(self):
        """Run preprocessing scripts based on completed downloads."""
        logger.info("="*50)
        logger.info("PHASE 3: DATA PREPROCESSING")
        logger.info("="*50)
        
        preprocessing_tasks = []
        
        # Check dependencies and add tasks
        mag_downloads = ["dl_intermagnet.py", "dl_nrcan_pre_1990.py", "dl_usgs_pre_1990.py"]
        mag_downloads_complete = all(script in self.completed_steps for script in mag_downloads)
        
        if mag_downloads_complete:
            preprocessing_tasks.append(("p_geomag_data.py", "Process geomagnetic data", 1))
        else:
            logger.warning("Skipping p_geomag_data.py - missing required downloads")
        
        # Power grid needs OSM data
        if "dl_power_grid_update.py" in self.completed_steps:
            preprocessing_tasks.append(("p_power_grid.py", "Process power grid data", 1))
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
                        self.completed_steps.add(script)
                    else:
                        logger.error(f"Preprocessing script {script} failed")
        else:
            # Run sequentially
            for script, desc, retries in preprocessing_tasks:
                if self.run_script(script, desc, retries):
                    self.completed_steps.add(script)
                else:
                    logger.error(f"Preprocessing script {script} failed")
        
        preprocess_count = len([s for s in self.completed_steps if s.startswith("p_") and s != "p_identify_storm_periods.py"])
        logger.info(f"Preprocessing phase complete. Successful: {preprocess_count}/{len(preprocessing_tasks)}")
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline."""
        start_time = time.time()
        
        logger.info("Starting GIC preprocessing pipeline")
        logger.info(f"Max retries: {self.max_retries}")
        logger.info(f"Parallel processing: {self.parallel}")
        
        # Phase 1: Storm identification (required for downloads)
        if not self.run_storm_identification():
            logger.error("Cannot proceed without storm identification")
            return False
        
        # Phase 2: Downloads
        self.run_download_phase()
        
        # Phase 3: Preprocessing
        self.run_preprocessing_phase()
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info("="*50)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*50)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Steps completed: {len(self.completed_steps)}")
        
        # List completed steps
        logger.info("Completed steps:")
        for step in sorted(self.completed_steps):
            logger.info(f"  ✓ {step}")
        
        # List any expected steps that didn't complete
        all_expected = {
            "p_identify_storm_periods.py",
            "dl_intermagnet.py", "dl_nrcan_pre_1990.py", 
            "dl_power_grid_update.py", "dl_usgs_pre_1990.py",
            "p_geomag_data.py", "p_power_grid.py"
        }
        failed_steps = all_expected - self.completed_steps
        
        if failed_steps:
            logger.warning("Failed or skipped steps:")
            for step in sorted(failed_steps):
                logger.warning(f"  ✗ {step}")
        
        return len(failed_steps) <= 2  # Allow up to 2 failures


def main():
    parser = argparse.ArgumentParser(description="Run GIC preprocessing pipeline")
    parser.add_argument("--max-retries", type=int, default=10,
                       help="Maximum retries for unreliable downloads (default: 10)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run scripts sequentially instead of in parallel")
    parser.add_argument("--download-only", action="store_true",
                       help="Run only download phase")
    parser.add_argument("--preprocess-only", action="store_true",
                       help="Run only preprocessing phase (assumes downloads complete)")
    parser.add_argument("--skip-storm-id", action="store_true",
                       help="Skip storm identification (if already done)")
    
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline(
        max_retries=args.max_retries,
        parallel=not args.sequential
    )
    
    try:
        if args.download_only:
            if not args.skip_storm_id:
                pipeline.run_storm_identification()
            pipeline.run_download_phase()
        elif args.preprocess_only:
            # Assume all required steps completed
            if not args.skip_storm_id:
                pipeline.completed_steps.add("p_identify_storm_periods.py")
            pipeline.completed_steps.update({
                "dl_intermagnet.py", "dl_nrcan_pre_1990.py",
                "dl_power_grid_update.py", "dl_usgs_pre_1990.py"
            })
            pipeline.run_preprocessing_phase()
        else:
            success = pipeline.run_pipeline()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()