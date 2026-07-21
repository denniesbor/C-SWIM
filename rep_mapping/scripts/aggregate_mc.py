"""
Aggregate TVA Monte Carlo GIC ensemble to per-substation statistics.
Role: post-processing step between run_tva_mc and figure scripts.
Description: Reads ground_gic_mc.parquet (1000 runs x 84 substations) and
writes ground_gic_mc_per_sub.parquet (84 substations x {median, mean, p5, p95}).
With --deterministic, falls back to the single-run ground_gic_ts.parquet so
downstream figures remain usable without the full ensemble.
Author: Dennies
"""

import argparse
import numpy as np
import pandas as pd
from rep_mapping.rep_config import TVA_DIR, setup_logger

logger = setup_logger(log_file="logs/aggregate_mc.log")


def aggregate_mc(parquet_path, out_path):
    """Compute per-substation median/mean/p5/p95 across MC runs."""
    mc = pd.read_parquet(parquet_path)
    logger.info("Loaded %s: %d runs x %d substations", parquet_path.name, *mc.shape)
    peak = mc.to_numpy(dtype=float)
    per_sub = pd.DataFrame(
        {
            "median": np.nanmedian(peak, axis=0),
            "mean": np.nanmean(peak, axis=0),
            "p5": np.nanpercentile(peak, 5, axis=0),
            "p95": np.nanpercentile(peak, 95, axis=0),
        },
        index=mc.columns,
    )
    per_sub.index.name = "sub_id"
    per_sub.to_parquet(out_path)
    logger.info(
        "Saved %s — median range [%.2f, %.2f] A  p95 range [%.2f, %.2f] A",
        out_path.name,
        per_sub["median"].min(),
        per_sub["median"].max(),
        per_sub["p95"].min(),
        per_sub["p95"].max(),
    )
    return per_sub


def aggregate_deterministic(ts_path, out_path):
    """Compute per-substation peak |GIC| from a single-run time series."""
    ts = pd.read_parquet(ts_path)
    peak = ts.abs().max(axis=1)
    per_sub = pd.DataFrame(
        {
            "median": peak.values,
            "mean": peak.values,
            "p5": peak.values,
            "p95": peak.values,
        },
        index=ts.index,
    )
    per_sub.index.name = "sub_id"
    per_sub.to_parquet(out_path)
    logger.info("Saved deterministic per-substation stats to %s", out_path.name)
    return per_sub


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate TVA MC GIC ensemble to per-substation stats."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic single-run ground_gic_ts.parquet instead of MC ensemble",
    )
    args = parser.parse_args()

    out_path = TVA_DIR / "ground_gic_mc_per_sub.parquet"

    if args.deterministic:
        aggregate_deterministic(TVA_DIR / "ground_gic_ts.parquet", out_path)
    else:
        aggregate_mc(TVA_DIR / "ground_gic_mc.parquet", out_path)

        summary_path = TVA_DIR / "ground_gic_mc_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            logger.info("Summary CSV confirmed (%d rows):", len(summary))
            for _, row in summary.iterrows():
                logger.info(
                    "  %s: median=%.2f [p5=%.2f, p95=%.2f] std=%.2f",
                    row["metric"],
                    row["median"],
                    row["p5"],
                    row["p95"],
                    row["std"],
                )
        else:
            logger.warning("ground_gic_mc_summary.csv not found at %s", summary_path)


if __name__ == "__main__":
    main()
