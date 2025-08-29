#!/usr/bin/env python3
"""Automated post-processing pipeline for GIC analysis."""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure repo root on path for configs
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from configs import setup_logger  # noqa: E402
from run_scenarios import ProgressTracker  # noqa: E402

logger = setup_logger(log_file="logs/run_postprocess.log")


class PostprocessPipeline:
    def __init__(self, max_retries: int = 1, parallel: bool = True):
        self.root_dir = repo_root
        self.scripts_dir = self.root_dir / "postprocess"
        self.max_retries = max_retries
        self.parallel = parallel
        self.completed_steps = set()

    def run_script(self, script_name: str, description: str, retries: int = 1) -> bool:
        """Execute a script with retries and timing."""
        script_path = self.scripts_dir / script_name
        for attempt in range(retries):
            logger.info(
                f"Starting {script_name} - {description} (Attempt {attempt + 1}/{retries})"
            )
            progress = ProgressTracker(script_name)
            progress.start()
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
                subprocess.run(
                    [sys.executable, str(script_path)],
                    check=True,
                    cwd=self.root_dir,
                    env=env,
                )
                progress.stop()
                return True
            except subprocess.CalledProcessError as e:
                progress.stop()
                elapsed = time.time() - progress.start_time
                logger.error(
                    f"Failed {script_name} after {elapsed:.1f}s (Attempt {attempt + 1}/{retries})"
                )
                if attempt < retries - 1:
                    logger.info("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error(f"Maximum retries reached for {script_name}: {e}")
                    return False
        return False

    def run_postprocess(self, tasks: list[tuple[str, str, int]]):
        """Run selected post-processing tasks."""
        if self.parallel and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=min(2, len(tasks))) as executor:
                futures = {
                    executor.submit(self.run_script, s, d, r): s for s, d, r in tasks
                }
                for fut in as_completed(futures):
                    script = futures[fut]
                    if fut.result():
                        self.completed_steps.add(script)
                    else:
                        logger.error(f"Post-process script {script} failed")
        else:
            for script, desc, retries in tasks:
                if self.run_script(script, desc, retries):
                    self.completed_steps.add(script)
                else:
                    logger.error(f"Post-process script {script} failed")

    def run_pipeline(self, which: str):
        """Execute the post-processing pipeline."""
        start_time = time.time()
        logger.info("Starting GIC post-processing pipeline")
        logger.info(f"Parallel processing: {self.parallel}")

        task_map = {
            "aggregate": [
                (
                    "aggregate_gannon_gic.py",
                    "Aggregate Gannon ground GIC",
                    self.max_retries,
                )
            ],
            "effective": [
                ("calc_eff_gic.py", "Compute effective GIC", self.max_retries)
            ],
            "all": [
                (
                    "aggregate_gannon_gic.py",
                    "Aggregate Gannon ground GIC",
                    self.max_retries,
                ),
                ("calc_eff_gic.py", "Compute effective GIC", self.max_retries),
            ],
        }
        self.run_postprocess(task_map[which])

        elapsed_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("POST-PROCESS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Steps completed: {len(self.completed_steps)}")
        for step in sorted(self.completed_steps):
            logger.info(f"  ✓ {step}")

        expected = {t[0] for t in task_map[which]}
        failed = expected - self.completed_steps
        if failed:
            logger.warning("Failed steps:")
            for step in sorted(failed):
                logger.warning(f"  ✗ {step}")

        return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Run GIC post-processing pipeline")
    parser.add_argument(
        "task",
        choices=["aggregate", "effective", "all"],
        help="Which post-process task to run",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run scripts sequentially instead of in parallel",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retries per script (default: 1)",
    )
    args = parser.parse_args()

    pipeline = PostprocessPipeline(
        max_retries=args.max_retries, parallel=not args.sequential
    )
    try:
        success = pipeline.run_pipeline(which=args.task)
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
