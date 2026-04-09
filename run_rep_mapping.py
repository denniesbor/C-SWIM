#!/usr/bin/env python3
"""
Master runner for reproducible mapping GIC validation pipeline.
Author: Dennies Bor
Usage:
    python run_rep_mapping.py preprocess        # build grids and compute voltages
    python run_rep_mapping.py validate          # run GIC solver and validation
    python run_rep_mapping.py viz               # generate all figures
    python run_rep_mapping.py all               # full pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path

from configs import setup_logger
from rep_mapping.rep_config import (
    TVA_DIR, UIUC_DIR, DEVICES_NC, GANNON_DS, HIFLD_PATH, OSM_SUB_PATH
)

logger = setup_logger("rep_mapping")


def check_essentials():
    missing = []
    for f in [DEVICES_NC, GANNON_DS, HIFLD_PATH, OSM_SUB_PATH]:
        if not f.exists():
            missing.append(f)
    if missing:
        raise FileNotFoundError(
            "Missing NFS dependencies:\n" +
            "\n".join([f"  {f}" for f in missing])
        )


def check_grids():
    missing = []
    for f in [
        TVA_DIR / "G_backbone.pkl",
        TVA_DIR / "substations_df.parquet",
        UIUC_DIR / "df_lines.pkl",
        UIUC_DIR / "bus_coords.pkl",
    ]:
        if not f.exists():
            missing.append(f)
    if missing:
        raise FileNotFoundError(
            "Grid files missing. Run preprocess first:\n" +
            "\n".join([f"  {f}" for f in missing])
        )


def check_voltages():
    missing = []
    for f in [
        TVA_DIR / "df_lines_with_voltages.parquet",
        UIUC_DIR / "df_lines_with_voltages.pkl",
    ]:
        if not f.exists():
            missing.append(f)
    if missing:
        raise FileNotFoundError(
            "Voltage files missing. Run preprocess first:\n" +
            "\n".join([f"  {f}" for f in missing])
        )


def run_step(module, args=None):
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    logger.info(f"Running {module} {args or ''}")
    subprocess.run(cmd, check=True)


def run_preprocess():
    logger.info("Building grids ...")
    run_step("rep_mapping.preprocess.build_tva_grid")
    run_step("rep_mapping.preprocess.build_uiuc_grid")

    logger.info("Computing Gannon voltages ...")
    run_step("rep_mapping.scripts.compute_v_tva")
    run_step("rep_mapping.scripts.compute_v_uiuc")


def run_validate():
    check_grids()
    check_voltages()

    logger.info("Running GIC solver ...")

    # TVA Gannon storm
    run_step("rep_mapping.scripts.run_gic", ["--grid", "tva"])

    # UIUC150 uniform 1 V/km benchmark validation
    run_step("rep_mapping.scripts.run_gic", ["--grid", "uiuc"])

    # UIUC150 Gannon storm
    run_step("rep_mapping.scripts.run_gic", ["--grid", "uiuc", "--gannon"])


def run_viz():
    logger.info("Generating figures ...")
    run_step("rep_mapping.viz.uiuc_map")
    run_step("rep_mapping.viz.tva_map")
    run_step("rep_mapping.viz.tva_uiuc_map")
    run_step("rep_mapping.viz.compare_uiuc")
    run_step("rep_mapping.viz.compare_tva")
    run_step("rep_mapping.viz.comp_gannon_tva_u")
    run_step("rep_mapping.viz.compare_uiuc_tva")


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible mapping GIC validation pipeline")
    parser.add_argument(
        "step",
        choices=["preprocess", "validate", "viz", "all"],
        help="Pipeline step to run"
    )
    args = parser.parse_args()

    try:
        check_essentials()

        if args.step == "preprocess":
            run_preprocess()

        elif args.step == "validate":
            run_validate()

        elif args.step == "viz":
            run_viz()

        elif args.step == "all":
            run_preprocess()
            run_validate()
            run_viz()

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()