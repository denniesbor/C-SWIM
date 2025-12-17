#!/usr/bin/env python3
"""
Space Weather Economic Impact Analysis - main orchestrator script
Author: Dennies Bor, Aug 2025
"""
import argparse
import subprocess
import threading
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm

import configs

from configs import (
    setup_logger,
    get_data_dir,
    DENNIES_DATA_LOC,
    FIGURES_DIR,
    ALPHA_BETA_GIC_FILE,
    EFF_GIC_DIR,
    PROCESS_GND_FILES,
)

logger = setup_logger("SPWIO Pipeline")
DATA_LOC = get_data_dir()
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

USE_ALPHA_BETA_SCENARIO = configs.USE_ALPHA_BETA_SCENARIO


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
        logger.info(f"✅ Completed {self.script_name} in {elapsed:.1f}s")

    def _show_progress(self):
        while self.running:
            elapsed = time.time() - self.start_time
            logger.info(f"⏳ Running {self.script_name}... {elapsed:.0f}s elapsed")
            time.sleep(30)


def essential_p_econ_data(data_loc: Path = DATA_LOC / "raw_econ_data") -> bool:
    files = [data_loc / "pop_2020_zcta.csv", data_loc / "zbp21detail.txt"]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing p_econ_data essentials:\n" + "\n".join([f"  {f}" for f in missing])
        )
    return True


def essential_downsample_nlcd(data_loc: Path = DATA_LOC / "land_mask") -> bool:
    files = [data_loc / "Annual_NLCD_LndCov_2023_CU_C1V0.tif"]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing downsample_nlcd essentials:\n"
            + "\n".join([f"  {f}" for f in missing])
        )
    return True


def essential_raster_interp(data_loc: Path = DATA_LOC) -> bool:
    econ_dir = data_loc / "processed_econ"
    land_dir = data_loc / "land_mask"
    den_dir = DENNIES_DATA_LOC
    files = [
        econ_dir / "socioeconomic_data.pkl",
        land_dir / "coarse" / "nlcd_coarse_mask.tif",
        den_dir / "admittance_matrix" / "substation_info.csv",
    ]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raster_interpolation essentials. "
            "Run p_econ_data and downsample_nlcd first, and ensure model coupling with GIC files is available:\n"
            + "\n".join([f"  {f}" for f in missing])
        )
    return True


def essential_build_tech_sam(data_loc: Path = DATA_LOC) -> bool:
    tables_dir = data_loc / "supply_use_tables"
    g_output_dir = data_loc / "gross_output"
    files = [
        tables_dir / "use_tables.csv",
        tables_dir / "supply_tables.csv",
        g_output_dir / "gross_output.csv",
    ]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Production Technology / US SAM essentials:\n"
            + "\n".join([f"  {f}" for f in missing])
        )
    return True


def essential_gic_eff(data_loc: Path = DATA_LOC) -> bool:
    if USE_ALPHA_BETA_SCENARIO:
        if not ALPHA_BETA_GIC_FILE.exists():
            raise FileNotFoundError(
                f"Missing alpha–beta GIC file:\n  {ALPHA_BETA_GIC_FILE}"
            )
        return True
    for gic_dir_str in EFF_GIC_DIR:
        gic_dir = Path(gic_dir_str).expanduser()
        files = sorted(gic_dir.glob("effective_gic_rand_*.csv"))
        if files:
            return True
    raise FileNotFoundError(
        "Missing effective GIC realization files under model coupling outputs."
    )


def essential_econ_analysis(data_loc: Path = DATA_LOC) -> bool:
    from configs import OUTPUT_FILES

    reg_out = data_loc / OUTPUT_FILES["regular_all"]
    ab_out = data_loc / OUTPUT_FILES["alpha_beta_uncertainty"]
    tech_dir = data_loc / "10sector"
    tech_files = [
        tech_dir / "direct_requirements.csv",
        tech_dir / "gross_output.csv",
        tech_dir / "value_added.csv",
        tech_dir / "final_demand.csv",
    ]
    sam_file = data_loc / "sam" / "us_balanced_sam.csv"

    missing = []
    logger.info(f"Checking if use alpha beta scenario {USE_ALPHA_BETA_SCENARIO}...")
    if USE_ALPHA_BETA_SCENARIO and not ab_out.exists():
        missing.append(ab_out)
    if not USE_ALPHA_BETA_SCENARIO and not reg_out.exists():
        missing.append(reg_out)
    for f in tech_files:
        if not f.exists():
            missing.append(f)
    if not sam_file.exists():
        missing.append(sam_file)

    if missing:
        raise FileNotFoundError(
            "Missing econ_analysis essentials. Run preprocess first:\n"
            + "\n".join([f"  {m}" for m in missing])
        )
    return True


def essential_viz(figures_dir=FIGURES_DIR) -> bool:
    files = []

    if USE_ALPHA_BETA_SCENARIO:
        files = [
            figures_dir / "io_model_results_alpha_beta.csv",
            figures_dir / "confidence_intervals_alpha_beta.csv",
        ]
    elif PROCESS_GND_FILES:
        files = [
            figures_dir / "io_model_results_gnd_gic.csv",
            figures_dir / "confidence_intervals_gnd_gic.csv",
        ]
    else:
        files = [
            figures_dir / "io_model_results.csv",
            figures_dir / "confidence_intervals.csv",
        ]

    missing = [f for f in files if not f.exists()]
    if missing:
        scenario_type = (
            "alpha_beta"
            if USE_ALPHA_BETA_SCENARIO
            else ("gnd_gic" if PROCESS_GND_FILES else "gic_sim")
        )
        raise FileNotFoundError(
            f"Missing visualization essentials for {scenario_type} scenario. "
            "Run `scripts/econ_analysis.py` first to generate required files.\n"
            + "\n".join([f"  {f}" for f in missing])
        )
    return True


def run_preprocess():
    tracker = ProgressTracker("Preprocessing")
    tracker.start()
    try:
        logger.info("Starting preprocessing steps...")
        steps = [
            "preprocess.p_econ_data",
            "preprocess.downsample_nlcd",
            "preprocess.p_areal_intp",
            "preprocess.p_gic_files",
            "preprocess.p_technology",
            "preprocess.p_us_sam",
        ]
        env = os.environ.copy()
        for step in steps:
            logger.info(f"Running {step} ...")
            subprocess.run([sys.executable, "-m", step], check=True, env=env)
    finally:
        tracker.stop()


def run_analysis():
    tracker = ProgressTracker("Economic Analysis")
    tracker.start()
    try:
        logger.info("Running economic analysis...")
        env = os.environ.copy()
        subprocess.run(
            [sys.executable, "-m", "scripts.econ_analysis"], check=True, env=env
        )
    finally:
        tracker.stop()


def run_viz():
    tracker = ProgressTracker("Visualization")
    tracker.start()
    try:
        logger.info("Running visualization...")
        env = os.environ.copy()

        cmd = [sys.executable, "-m", "viz.viz"]
        res = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning(
                "`python -m viz.viz` failed; falling back to file. stderr:\n"
                + res.stderr.strip()
            )
            subprocess.run([sys.executable, "viz/viz.py"], check=True, env=env)
    finally:
        tracker.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPWIO Pipeline Entry Point")
    parser.add_argument(
        "step", choices=["preprocess", "analysis", "viz", "all"], help="Step to run"
    )
    parser.add_argument(
        "--alpha-beta",
        action="store_true",
        help="Use alpha–beta scenario mode for GIC processing",
    )
    args = parser.parse_args()

    os.environ["USE_ALPHA_BETA_SCENARIO"] = "true" if args.alpha_beta else "false"
    logger.info(
        f"USE_ALPHA_BETA_SCENARIO (env) = {os.environ['USE_ALPHA_BETA_SCENARIO']}"
    )

    USE_ALPHA_BETA_SCENARIO = args.alpha_beta

    import configs

    import importlib

    importlib.reload(configs)
    logger.info(
        f"USE_ALPHA_BETA_SCENARIO (env) = {os.environ['USE_ALPHA_BETA_SCENARIO']}"
    )

    try:
        if args.step == "preprocess":
            essential_p_econ_data()
            essential_downsample_nlcd()
            run_preprocess()

        elif args.step == "analysis":
            essential_p_econ_data()
            essential_downsample_nlcd()
            essential_raster_interp()
            essential_gic_eff()
            essential_build_tech_sam()
            essential_econ_analysis()
            run_analysis()

        elif args.step == "viz":
            essential_p_econ_data()
            essential_downsample_nlcd()
            essential_raster_interp()
            essential_gic_eff()
            essential_build_tech_sam()
            essential_econ_analysis()
            essential_viz()
            run_viz()

        elif args.step == "all":
            logger.info("Running full pipeline: preprocess → analysis → viz")
            essential_p_econ_data()
            essential_downsample_nlcd()
            run_preprocess()

            essential_raster_interp()
            essential_gic_eff()
            essential_build_tech_sam()
            essential_econ_analysis()
            run_analysis()

            essential_viz()
            run_viz()

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)