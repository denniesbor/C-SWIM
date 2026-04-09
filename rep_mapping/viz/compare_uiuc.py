"""
Comparison of modelled vs benchmark GIC for UIUC150.
Author: Dennies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rep_mapping.rep_config import UIUC_DIR, FIGURES_DIR, setup_logger, setup_matplotlib

logger = setup_logger(log_file="logs/plot_uiuc_comp.log")

setup_matplotlib()

PANEL_FONTSIZE = 11

# Wong colorblind-safe palette
C_MAIN = "#0072B2"  # blue — modelled / benchmark
C_SECOND = "#E69F00"  # orange — modelled CDF line
C_NEUTRAL = "#aaa"  # gray — near-zero GIC
C_OUTLIER = "#D55E00"  # vermillion — worst outlier


def load_data():
    return pd.read_parquet(UIUC_DIR / "gic_comparison.parquet")


def compute_stats(comp):
    r = np.corrcoef(comp.I_eff, comp.GIC_eff_A)[0, 1]
    r2 = r**2
    rmse = np.sqrt(((comp.I_eff - comp.GIC_eff_A) ** 2).mean())
    mae = (comp.I_eff - comp.GIC_eff_A).abs().mean()
    return r, r2, rmse, mae


def main():
    comp = load_data()
    r, r2, rmse, mae = compute_stats(comp)

    near_zero = comp.GIC_eff_A < 0.5
    outlier = comp.err_pct.idxmax()
    residuals = comp.I_eff - comp.GIC_eff_A

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.1)

    # (a) Scatter — square, top left
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal", adjustable="box")
    lim = max(comp.I_eff.max(), comp.GIC_eff_A.max()) * 1.08

    ax.scatter(
        comp.GIC_eff_A[~near_zero],
        comp.I_eff[~near_zero],
        s=18,
        color=C_MAIN,
        alpha=0.8,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter(
        comp.GIC_eff_A[near_zero],
        comp.I_eff[near_zero],
        s=18,
        color=C_NEUTRAL,
        alpha=0.6,
        edgecolors="none",
        zorder=3,
        label="GIC < 0.5 A",
    )
    ax.scatter(
        comp.GIC_eff_A[outlier],
        comp.I_eff[outlier],
        s=30,
        color=C_OUTLIER,
        alpha=1.0,
        edgecolors="none",
        zorder=5,
    )
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, zorder=2)

    stats_str = (
        f"$r$ = {r:.3f}\n"
        f"$R^2$ = {r2:.3f}\n"
        f"RMSE = {rmse:.2f} A\n"
        f"MAE = {mae:.2f} A"
    )
    ax.text(
        0.97,
        0.05,
        stats_str,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        linespacing=1.6,
    )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Benchmark GIC (A/phase)")
    ax.set_ylabel("Modeled GIC (A/phase)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", labelsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.text(
        0.0,
        1.05,
        "(a) Modeled vs benchmark",
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        va="bottom",
        ha="left",
    )

    # (b) Residuals — top right
    ax = fig.add_subplot(gs[0, 1])

    ax.scatter(
        comp.GIC_eff_A[~near_zero],
        residuals[~near_zero],
        s=18,
        color=C_MAIN,
        alpha=0.8,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter(
        comp.GIC_eff_A[near_zero],
        residuals[near_zero],
        s=18,
        color=C_NEUTRAL,
        alpha=0.6,
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        comp.GIC_eff_A[outlier],
        residuals[outlier],
        s=30,
        color=C_OUTLIER,
        alpha=1.0,
        edgecolors="none",
        zorder=5,
    )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_xlim(left=0)
    ax.set_xlabel("Benchmark GIC (A/phase)")
    ax.set_ylabel("Residual (A/phase)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", labelsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.text(
        0.0,
        1.05,
        "(b) Residuals",
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        va="bottom",
        ha="left",
    )

    # (c) CDF — bottom spanning full width
    ax = fig.add_subplot(gs[1, :])

    bench_sorted = np.sort(comp.GIC_eff_A)
    model_sorted = np.sort(comp.I_eff)
    cdf = np.linspace(0, 1, len(bench_sorted))

    ax.plot(bench_sorted, cdf, color=C_MAIN, linewidth=1.2, label="Benchmark")
    ax.plot(
        model_sorted,
        cdf,
        color=C_SECOND,
        linewidth=1.2,
        linestyle="--",
        label="Modeled",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("GIC (A/phase)")
    ax.set_ylabel("CDF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", labelsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.text(
        0.0,
        1.05,
        "(c) Cumulative distribution",
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        va="bottom",
        ha="left",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / "uiuc150_gic_comparison.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        FIGURES_DIR / "uiuc150_gic_comparison.pdf", dpi=300, bbox_inches="tight"
    )
    logger.info(f"Saved to {FIGURES_DIR}")
    plt.close()


if __name__ == "__main__":
    main()
