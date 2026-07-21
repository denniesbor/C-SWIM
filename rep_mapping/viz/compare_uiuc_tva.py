"""
Plot UIUC150 vs TVA OSM modelled ground GIC for matched substations.
Author: Dennies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rep_mapping.rep_config import (
    UIUC_DIR,
    TVA_DIR,
    FIGURES_DIR,
    setup_logger,
    setup_matplotlib,
)

logger = setup_logger(log_file="logs/plot_modelled_vs_modelled.log")

setup_matplotlib()

FONTSIZE_MAIN = 11
FONTSIZE_INFO = 10

# Wong colorblind-safe palette
C_UIUC = "#0072B2"  # blue
C_TVA = "#E69F00"  # orange

PAIRS = [
    (26, "164024535.0", 1.52),
    (93, "106782533.0", 1.15),
    (95, "422884276.0", 1.03),
    (97, "80436792.0", 0.57),
    (70, "1172767928.0", 2.78),
    (85, "106806181.0", 4.88),
]

uiuc_gic = pd.read_parquet(UIUC_DIR / "ground_gic_ts_gannon.parquet")
tva_gic = pd.read_parquet(TVA_DIR / "ground_gic_ts.parquet")
time_axis = np.load(TVA_DIR / "time_axis.npy", allow_pickle=True)

# Trim to start from 2024-05-10 00:00
time_pd = pd.DatetimeIndex(time_axis)
mask = time_pd >= "2024-05-10 00:00"
time_axis = time_axis[mask]

n = len(PAIRS)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7.8, 9), sharex=True)

for j, (sub_u, sub_t, dist_km) in enumerate(PAIRS):
    ax = axes[j]

    uiuc_vals = uiuc_gic.loc[sub_u].values[mask].copy()
    tva_vals = tva_gic.loc[sub_t].values[mask].copy()

    finite = np.isfinite(uiuc_vals) & np.isfinite(tva_vals)
    r = (
        np.corrcoef(uiuc_vals[finite], tva_vals[finite])[0, 1]
        if finite.sum() > 5
        else np.nan
    )

    polarity_flipped = False
    if not np.isnan(r) and r < 0:
        uiuc_vals = -uiuc_vals
        polarity_flipped = True
        r = -r

    ax.plot(
        time_axis,
        uiuc_vals,
        color=C_UIUC,
        linestyle="--",
        linewidth=0.8,
        label="UIUC150 synthetic",
    )
    ax.plot(
        time_axis,
        tva_vals,
        color=C_TVA,
        linestyle="-",
        linewidth=0.8,
        label="TVA OSM+HIFLD",
    )
    ax.axhline(0, color="gray", linewidth=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", labelsize=10)

    text_x = 1.02
    text_y = 0.95
    dy = 0.15
    ax.text(
        text_x, text_y, f"({chr(97+j)})", transform=ax.transAxes, fontsize=FONTSIZE_INFO
    )
    ax.text(
        text_x,
        text_y - dy,
        f"UIUC {sub_u}",
        transform=ax.transAxes,
        fontsize=FONTSIZE_INFO,
    )
    ax.text(
        text_x,
        text_y - dy * 2,
        f"Sep: {dist_km:.2f} km",
        transform=ax.transAxes,
        fontsize=FONTSIZE_INFO,
    )
    ax.text(
        text_x,
        text_y - dy * 3,
        f"$r$ = {r:.2f}" if not np.isnan(r) else "$r$ = N/A",
        transform=ax.transAxes,
        fontsize=FONTSIZE_INFO,
    )

    if j == 0:
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color=C_UIUC,
                    label="UIUC150 synthetic",
                    linewidth=0.8,
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    color=C_TVA,
                    label="TVA OSM+HIFLD",
                    linewidth=0.8,
                ),
            ],
            loc="upper left",
            frameon=False,
            fontsize=FONTSIZE_MAIN,
            bbox_to_anchor=(0, 1.0),
        )

    if j == n - 1:
        ax.set_xlim(time_axis[0], time_axis[-1])
        tick_pos = [
            time_axis[0],
            time_axis[len(time_axis) // 4],
            time_axis[len(time_axis) // 2],
            time_axis[3 * len(time_axis) // 4],
            time_axis[-1],
        ]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            [pd.Timestamp(t).strftime("%H:%M\n%m/%d") for t in tick_pos], fontsize=10
        )
        ax.set_xlabel("Time (UTC)", fontsize=FONTSIZE_MAIN)

fig.supylabel("Ground GIC (A)", fontsize=FONTSIZE_MAIN, x=0.035)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(
    FIGURES_DIR / "modelled_vs_modelled_gannon.png", dpi=300, bbox_inches="tight"
)
fig.savefig(FIGURES_DIR / "modelled_vs_modelled_gannon.pdf", bbox_inches="tight")
logger.info("Saved to %s", FIGURES_DIR)
