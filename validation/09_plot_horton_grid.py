"""
Author: Dennies Bor
Role: Visualisation of IEEE Horton benchmark GIC results.
      Plots ground GIC distributions from Monte Carlo variants
      against the deterministic baseline for northward and eastward fields.
Inputs:
    - data/horton_grid/mc_results.pkl
Outputs:
    - figures/horton_ground_gic_violin.png/.pdf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D

from configs import (
    DATA_DIR,
    HORTON_GRID_DIR,
    VAL_FIGURES_DIR as figures_dir,
    setup_logger,
    setup_matplotlib,
)

setup_matplotlib()

logger = setup_logger(log_file="logs/val_horton_viz.log")

SUBS_ORDER = [f"Substation {i}" for i in range(1, 9)]
COMP_MAP = {
    "GIC_northward": "Northward field (1 V/km)",
    "GIC_eastward": "Eastward field (1 V/km)",
}


def load_mc_results():
    """Load Monte Carlo GIC results from pickle cache."""
    mc_results_path = HORTON_GRID_DIR / "mc_results.pkl"
    logger.info(f"Loading MC results from {mc_results_path}")
    with open(mc_results_path, "rb") as f:
        return pickle.load(f)


def prepare_data(mc_results):
    """Reshape base and MC ground GIC results into long format for plotting."""
    cols = ["GIC_northward", "GIC_eastward"]
    base_grounds = mc_results["base_grounds_gic"]
    mc_grounds = mc_results["mc_grounds_gic"]
    mc_grounds = mc_grounds[mc_grounds["config"] == "var"].copy()

    base_agg = (
        base_grounds.groupby("Substation", as_index=True)[cols]
        .mean()
        .reindex(SUBS_ORDER, fill_value=0.0)
        .reset_index()
    )

    mc_long = mc_grounds.melt(
        id_vars=["Substation", "scenario"],
        value_vars=cols,
        var_name="component",
        value_name="GIC",
    )
    base_long = base_agg.melt(
        id_vars=["Substation"],
        value_vars=cols,
        var_name="component",
        value_name="GIC",
    )

    mc_long["component"] = mc_long["component"].map(COMP_MAP)
    base_long["component"] = base_long["component"].map(COMP_MAP)

    for df in [mc_long, base_long]:
        df["Substation"] = pd.Categorical(
            df["Substation"], categories=SUBS_ORDER, ordered=True
        )

    return mc_long, base_long


def _nice_step(vmax, n=6):
    """Compute a round y-axis tick step for a given max value."""
    raw = (2 * vmax) / n
    exp = np.floor(np.log10(raw))
    base = raw / (10**exp)
    nice = 1 if base <= 1 else 2 if base <= 2 else 5 if base <= 5 else 10
    return nice * (10**exp)


def plot_violin(mc_long, base_long, figures_dir):
    """Plot violin + boxplot of MC GIC distributions with baseline markers."""
    y_all = np.concatenate([mc_long["GIC"].values, base_long["GIC"].values])
    y_abs = np.nanmax(np.abs(y_all))
    y_pad = 1.05
    ylim = (-y_pad * y_abs, y_pad * y_abs)
    ystep = _nice_step(y_pad * y_abs, n=6)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.6, 6.4), sharex=True)

    for i, (ax, comp) in enumerate(zip(axes, COMP_MAP.values())):
        d_mc = mc_long[mc_long["component"] == comp]
        d_b = base_long[base_long["component"] == comp]

        sns.violinplot(
            data=d_mc,
            x="Substation",
            y="GIC",
            order=SUBS_ORDER,
            cut=0,
            inner=None,
            bw="scott",
            ax=ax,
            linewidth=0,
            color="#5fa8d3",
            saturation=1.0,
            width=0.9,
            scale="width",
        )
        for pc in ax.collections:
            try:
                pc.set_alpha(0.65)
            except Exception:
                pass

        sns.boxplot(
            data=d_mc,
            x="Substation",
            y="GIC",
            order=SUBS_ORDER,
            ax=ax,
            width=0.28,
            showcaps=False,
            boxprops=dict(facecolor="none", edgecolor="navy", linewidth=1.0),
            medianprops=dict(color="#d00000", linewidth=1.4),
            whiskerprops=dict(color="navy", linewidth=0.9, alpha=0.6),
            flierprops=dict(
                marker=".",
                markersize=2,
                markeredgewidth=0,
                alpha=0.25,
                markerfacecolor="navy",
            ),
            zorder=4,
        )

        base_vals = d_b.set_index("Substation").loc[SUBS_ORDER, "GIC"].values
        ax.scatter(
            x=np.arange(len(SUBS_ORDER)),
            y=base_vals,
            marker="o",
            s=34,
            linewidths=1.3,
            facecolors="white",
            edgecolors="#d00000",
            zorder=6,
            label="Baseline",
        )

        ax.axhline(0.0, color="0.6", linewidth=0.7, zorder=1)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ystep))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylabel("Ground GIC (A)")
        ax.set_title(f"({chr(97 + i)}) {comp}", fontsize=10, pad=2, loc="left")

    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticks(np.arange(len(SUBS_ORDER)))
    axes[1].set_xticklabels([f"Sub {i}" for i in range(1, 9)], rotation=0, ha="center")
    axes[1].set_xlabel("Substation")

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="#d00000", linewidth=1.4, label="Simulation median")
    )
    labels.append("Simulation median")
    axes[0].legend(handles, labels, frameon=False, loc="upper right", fontsize=8)

    plt.tight_layout(h_pad=0.5)
    for ext in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"horton_ground_gic_violin.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    logger.info(f"Saved horton violin figure to {figures_dir}")


def main():
    """Run Horton benchmark GIC visualisation."""
    mc_results = load_mc_results()
    mc_long, base_long = prepare_data(mc_results)
    plot_violin(mc_long, base_long, figures_dir)


if __name__ == "__main__":
    main()
