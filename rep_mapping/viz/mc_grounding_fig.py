"""
Role: Illustrate Monte Carlo grounding resistance sampling for the manuscript.
Description: Two-panel figure. Panel (a) shows the Uniform prior distributions
for grounding resistance: Uniform(0.1-1.0) Ohm for the 60 OSM substations and
Uniform(2.0-20.0) Ohm for the 24 HIFLD synthetic endpoints, with the fixed
deterministic baseline values marked. Panel (b) shows the resulting distribution
of the headline 95th-percentile peak ground GIC across 1000 MC runs compared
to the fixed-Rg deterministic result.
Author: Bor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rep_mapping.rep_config import TVA_DIR, FIGURES_DIR, setup_logger, setup_matplotlib

logger = setup_logger(log_file="logs/mc_grounding_fig.log")

setup_matplotlib()

SEED_BASE = 42
N_RUNS = 1000
N_REAL = 60
N_SYNTH = 24

RG_REAL_LO, RG_REAL_HI = 0.1, 1.0
RG_SYNTH_LO, RG_SYNTH_HI = 2.0, 20.0
RG_BASELINE_REAL = 0.2
RG_BASELINE_SYNTH = 10.0

C_REAL = "#0072B2"
C_SYNTH = "#E69F00"
C_BASE = "#CC79A7"


def _headline_p95_per_run(mc: pd.DataFrame) -> np.ndarray:
    return np.percentile(mc.to_numpy(dtype=float), 95, axis=1)


def main() -> None:
    mc = pd.read_parquet(TVA_DIR / "ground_gic_mc.parquet")
    summary = pd.read_csv(TVA_DIR / "ground_gic_mc_summary.csv").set_index("metric")
    p95_per_run = _headline_p95_per_run(mc)
    baseline = float(
        summary.loc["headline_p95_ground_gic_A", "deterministic_baseline"]
    )
    ens_median = float(summary.loc["headline_p95_ground_gic_A", "median"])
    ens_p5 = float(summary.loc["headline_p95_ground_gic_A", "p5"])
    ens_p95 = float(summary.loc["headline_p95_ground_gic_A", "p95"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # (a) Prior distributions on log scale
    ax_a.set_xscale("log")

    real_height = 1.0 / (RG_REAL_HI - RG_REAL_LO)
    synth_height = 1.0 / (RG_SYNTH_HI - RG_SYNTH_LO)
    scale = max(real_height, synth_height)

    ax_a.fill_betweenx(
        [0, real_height / scale],
        RG_REAL_LO, RG_REAL_HI,
        color=C_REAL, alpha=0.35, linewidth=0,
    )
    ax_a.plot(
        [RG_REAL_LO, RG_REAL_LO, RG_REAL_HI, RG_REAL_HI],
        [0, real_height / scale, real_height / scale, 0],
        color=C_REAL, linewidth=1.2,
        label=f"OSM substations (n={N_REAL})",
    )

    ax_a.fill_betweenx(
        [0, synth_height / scale],
        RG_SYNTH_LO, RG_SYNTH_HI,
        color=C_SYNTH, alpha=0.35, linewidth=0,
    )
    ax_a.plot(
        [RG_SYNTH_LO, RG_SYNTH_LO, RG_SYNTH_HI, RG_SYNTH_HI],
        [0, synth_height / scale, synth_height / scale, 0],
        color=C_SYNTH, linewidth=1.2,
        label=f"HIFLD endpoints (n={N_SYNTH})",
    )

    ax_a.axvline(
        RG_BASELINE_REAL, color=C_REAL, linewidth=1.0, linestyle="--", alpha=0.8
    )
    ax_a.axvline(
        RG_BASELINE_SYNTH, color=C_SYNTH, linewidth=1.0, linestyle="--", alpha=0.8
    )

    ax_a.set_xlabel(r"Grounding resistance $R_g$ ($\Omega$)", fontsize=10)
    ax_a.set_ylabel("Relative density", fontsize=10)
    ax_a.set_xlim(0.05, 30)
    ax_a.set_ylim(bottom=0)
    ax_a.tick_params(direction="in", labelsize=9)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(
        frameon=False, fontsize=8.5, loc="upper right",
        handles=[
            mpatches.Patch(color=C_REAL, alpha=0.6, label=f"OSM substations (n={N_REAL})"),
            mpatches.Patch(color=C_SYNTH, alpha=0.6, label=f"HIFLD endpoints (n={N_SYNTH})"),
            plt.Line2D([0], [0], color="#555", linestyle="--", linewidth=1.0,
                       label="Fixed baseline"),
        ],
    )
    ax_a.text(0.0, 1.04, "(a) Grounding resistance priors",
              transform=ax_a.transAxes, fontsize=11, va="bottom")

    # (b) Ensemble headline p95 distribution
    ax_b.hist(
        p95_per_run, bins=40, color=C_REAL, alpha=0.7, edgecolor="none",
        label="MC ensemble (n=1 000 runs)",
    )
    ax_b.axvline(
        baseline, color=C_BASE, linewidth=1.4, linestyle="--",
        label=f"Fixed $R_g$ baseline = {baseline:.1f} A",
    )
    ax_b.axvline(
        ens_median, color=C_REAL, linewidth=1.2, linestyle=":",
        label=f"Ensemble median = {ens_median:.1f} A",
    )
    ax_b.axvspan(ens_p5, ens_p95, color=C_REAL, alpha=0.12,
                 label=f"90% CI [{ens_p5:.1f}, {ens_p95:.1f}] A")

    ax_b.set_xlabel("Headline p95 peak ground GIC (A)", fontsize=10)
    ax_b.set_ylabel("Count (runs)", fontsize=10)
    ax_b.tick_params(direction="in", labelsize=9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax_b.text(0.0, 1.04, "(b) GIC sensitivity to sampled $R_g$",
              transform=ax_b.transAxes, fontsize=11, va="bottom")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "mc_grounding_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mc_grounding_sensitivity.pdf", bbox_inches="tight")
    logger.info("Saved mc_grounding_sensitivity to %s", FIGURES_DIR)
    plt.close()


if __name__ == "__main__":
    main()
