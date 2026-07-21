"""
Role: Vulnerable transformer map, return-period panels with a Northeast inset.
Description: Renders the four-panel substation failure-probability map
(100, 150, 200, 250 year storms). The return-period signal is the growing
count of at-risk substations: nationally the count above 30 percent failure
probability runs 62, 87, 100, 114 across the four panels, and the Northeast
corridor (latitude 39 to 47, longitude -80 to -67) holds 45, 61, 73, 83 of
them. In a national map those substations overlap into one blob, so each
panel carries a zoomed inset of the corridor where the dots separate and the
count growth reads by eye. Marker size is continuous and power-scaled over
probability so size reinforces color. Logic is copied from
viz.plots.plot_vuln_trafos to keep that module untouched while the encoding
is tuned.
Author: Bor
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.collections import LineCollection

from configs import (
    USE_ALPHA_BETA_SCENARIO,
    PROCESS_GND_FILES,
    FIGURES_DIR,
    setup_logger,
    setup_matplotlib,
)
from econ.scripts.l_prepr_data import (
    load_gic_results,
    process_vulnerability_chunks,
    load_network_data,
)

logger = setup_logger("vuln_trafo")

setup_matplotlib()

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Substations below this failure probability (percent) are pooled into the
# floor category and drawn as a faint underlay so the eye tracks only the
# high-risk tail that grows across return periods.
SAFE_CUTOFF_PCT = 30.0

# Colors match create_tl_sub_visualization in plots.py so the two figures
# share a visual language. Navy lines and sky-blue floor sit outside the
# cividis risk scale (blue-to-yellow), so nothing bleeds.
LINE_COLOR = "#00204D"
STATE_COLOR = "#999999"
FLOOR_COLOR = "#b0a0c0"
LINE_WIDTH = 0.35

EXTENT = [-120, -75, 23, 50]

# Mid-Atlantic to southern New England corridor, trimmed to exclude Canada.
NE_EXTENT = [-81.5, -70.0, 35.3, 43.5]
NE_FRAME_COLOR = "#222222"
INSET_GAP_IN = 1

# Everything below this zorder (basemap, lines, dots) is flattened into one
# raster image per axes when saving vector formats. This collapses the cartopy
# node count and bakes marker transparency into pixels, so the EPS/PDF import
# cleanly in Illustrator with only text left as vector. Data dots top out near
# zorder 9, so the cut sits above them and below the box/connector frame.
RASTER_ZORDER = 12

SCENARIOS_ALPHA_BETA = [
    "gic_100yr_mean_prediction",
    "gic_150yr_mean_prediction",
    "gic_200yr_mean_prediction",
    "gic_250yr_mean_prediction",
]
SCENARIOS_REGULAR = [
    "e_100-year-hazard A/ph",
    "e_150-year-hazard A/ph",
    "e_200-year-hazard A/ph",
    "e_250-year-hazard A/ph",
]
SCENARIO_DISPLAY = {
    "gic_100yr_mean_prediction": "100-year",
    "gic_150yr_mean_prediction": "150-year",
    "gic_200yr_mean_prediction": "200-year",
    "gic_250yr_mean_prediction": "250-year",
    "e_100-year-hazard A/ph": "100-year",
    "e_150-year-hazard A/ph": "150-year",
    "e_200-year-hazard A/ph": "200-year",
    "e_250-year-hazard A/ph": "250-year",
}


def _basemap(ax, extent):
    """Local basemap so transmission lines and state boundaries read as
    distinct colors. Mirrors viz.plot_utils.setup_map but recolors the state
    edges to a warm brown that contrasts with the teal transmission lines.
    """
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#777777")
    ax.add_feature(cfeature.STATES, linewidth=0.45, edgecolor=STATE_COLOR)
    ax.add_feature(cfeature.LAKES, alpha=0.5, linewidth=0.5, edgecolor="grey")
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
    return ax


def _tail_labels(edges):
    return [f"{int(edges[i])}–{int(edges[i + 1])}%" for i in range(len(edges) - 1)]


def _marker_size(p_pct, size_min, size_max, gamma):
    """Continuous marker area, power-scaled over the at-risk range.

    gamma > 1 keeps low-probability markers small and lets the severe tail
    balloon, so the growth of high-risk substations across return periods is
    visible by size alone, independent of color.
    """
    t = np.clip(
        (np.asarray(p_pct, dtype=float) - SAFE_CUTOFF_PCT) / (100.0 - SAFE_CUTOFF_PCT),
        0.0,
        1.0,
    )
    return size_min + (t**gamma) * (size_max - size_min)


def _draw_dots(
    ax,
    coords,
    tail_edges,
    bin_colors,
    n_bins,
    size_min,
    size_max,
    gamma,
    size_scale=1.0,
):
    """Draw the faint safe underlay and the binned at-risk dots on one axis.

    High bins draw first so the largest markers sit underneath; lower bins
    overlay on top and stay visible. size_scale shrinks the markers in the
    zoomed inset where the corridor is dense.
    """
    safe = coords[coords["p_pct"] < SAFE_CUTOFF_PCT]
    ax.scatter(
        safe["longitude"],
        safe["latitude"],
        s=3,
        facecolors=FLOOR_COLOR,
        edgecolors="none",
        alpha=0.30,
        transform=ccrs.PlateCarree(),
        zorder=2,
        rasterized=True,
    )

    risk = coords[coords["p_pct"] >= SAFE_CUTOFF_PCT].copy()
    risk["bin"] = np.digitize(risk["p_pct"], tail_edges[1:-1])
    for b in range(n_bins - 1, -1, -1):
        sel = risk[risk["bin"] == b]
        if sel.empty:
            continue
        ax.scatter(
            sel["longitude"],
            sel["latitude"],
            s=_marker_size(sel["p_pct"], size_min, size_max, gamma) * size_scale,
            facecolors=[bin_colors[b]],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.92,
            transform=ccrs.PlateCarree(),
            zorder=3 + (n_bins - 1 - b),
            rasterized=True,
        )


def _add_lines(ax, line_coords, width):
    lc = LineCollection(
        line_coords,
        linewidths=width,
        alpha=0.5,
        colors=LINE_COLOR,
        transform=ccrs.PlateCarree(),
        zorder=1,
        rasterized=True,
    )
    ax.add_collection(lc)


def plot_vuln_trafos(
    vuln_data,
    df_lines,
    *,
    file_suffix,
    tail_edges,
    size_min,
    size_max,
    size_gamma=2.0,
    inset_size_scale=0.62,
    cmap_name="turbo",
    out_dir=FIGURES_DIR,
    formats=("png",),
    dpi=600,
):
    """Render four stacked return-period panels, each with a Northeast inset.

    The four national maps stack in a single column (A4 portrait); each one
    carries a zoomed inset of the Northeast corridor in a dedicated column to
    the right, so the inset never overlays the map. The at-risk substations
    separate in the inset, so the growth in their count across return periods
    reads by eye. tail_edges defines the discrete color bins above
    SAFE_CUTOFF_PCT. Marker size is continuous and power-scaled (size_gamma)
    over probability.
    """
    scenarios = SCENARIOS_ALPHA_BETA if USE_ALPHA_BETA_SCENARIO else SCENARIOS_REGULAR

    tail_labels = _tail_labels(tail_edges)
    n_bins = len(tail_labels)
    cmap = plt.cm.get_cmap(cmap_name, n_bins)
    bin_colors = [cmap(i) for i in range(n_bins)]
    bin_mids = [(tail_edges[b] + tail_edges[b + 1]) / 2 for b in range(n_bins)]

    projection = ccrs.LambertConformal(central_longitude=-97, central_latitude=38)
    inset_proj = ccrs.LambertConformal(central_longitude=-75.5, central_latitude=39.5)

    _fig_w = 7.6
    _avail_w = _fig_w * (0.98 - 0.02)
    _avg_col_w = (_avail_w - INSET_GAP_IN) / 2
    _wspace = INSET_GAP_IN / _avg_col_w

    fig = plt.figure(figsize=(_fig_w, 10.7))
    fig.patch.set_facecolor("#F0F0F0")
    gs = fig.add_gridspec(
        len(scenarios), 2,
        width_ratios=[1.2, 0.8],
        hspace=0.08, wspace=_wspace,
        left=0.02, right=0.98, top=0.97, bottom=0.10,
    )

    line_coords = [list(geom.coords) for geom in df_lines["geometry"]]
    all_coords = vuln_data.groupby("sub_id")[["latitude", "longitude"]].first()

    for i, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[i, 0], projection=projection)
        ax.set_aspect("auto")
        ax.set_rasterization_zorder(RASTER_ZORDER)
        _basemap(ax, EXTENT)
        _add_lines(ax, line_coords, LINE_WIDTH)

        sc_df = vuln_data[vuln_data["scenario"] == scenario]
        p_by_sub = sc_df.groupby("sub_id")["mean_failure_prob"].mean() * 100.0
        coords = all_coords.loc[all_coords.index.intersection(p_by_sub.index)].copy()
        coords["p_pct"] = p_by_sub.reindex(coords.index)

        _draw_dots(
            ax, coords, tail_edges, bin_colors, n_bins, size_min, size_max, size_gamma
        )
        ax.set_title(
            f"({chr(97 + i)}) {SCENARIO_DISPLAY[scenario]} Storm",
            fontsize=10,
            loc="left",
        )
        ax.spines["geo"].set_visible(False)
        ax.set_facecolor("#F0F0F0")

        axi = fig.add_subplot(gs[i, 1], projection=inset_proj)
        axi.set_aspect("auto")
        axi.set_rasterization_zorder(RASTER_ZORDER)
        _basemap(axi, NE_EXTENT)
        _add_lines(axi, line_coords, LINE_WIDTH * 1.6)
        _draw_dots(
            axi,
            coords,
            tail_edges,
            bin_colors,
            n_bins,
            size_min,
            size_max,
            size_gamma,
            size_scale=inset_size_scale,
        )
        axi.spines["geo"].set_visible(True)
        axi.spines["geo"].set_edgecolor(NE_FRAME_COLOR)
        axi.spines["geo"].set_linewidth(1.1)
        axi.set_facecolor("#F0F0F0")

        _connect_inset(fig, ax, axi, projection)

    _build_legends(
        fig, bin_colors, tail_labels, bin_mids, n_bins, size_min, size_max, size_gamma
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        save_kw = {}
        if ext in ("tif", "tiff"):
            save_kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(
            out_dir / f"vuln_trafo_{file_suffix}.{ext}", dpi=dpi, **save_kw
        )
    plt.close(fig)



def _connect_inset(fig, ax_main, axi, projection):
    """Draw the corridor box on the main map and tie its left corners to the
    inset left corners. The inset sits flush to the right of the map, so the
    two dashed connectors read as a magnifier callout."""
    lon0, lon1, lat0, lat1 = NE_EXTENT
    ax_main.add_patch(
        mpatches.Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            fill=False,
            edgecolor=NE_FRAME_COLOR,
            linewidth=0.8,
            linestyle=(0, (4, 2)),
            transform=ccrs.PlateCarree(),
            zorder=20,
            rasterized=True,
        )
    )

    corner_pairs = [
        ((lon0, lat1), (0.0, 1.0)),  # box top-left -> inset top-left
        ((lon0, lat0), (0.0, 0.0)),  # box bottom-left -> inset bottom-left
    ]
    for (lon, lat), axes_xy in corner_pairs:
        px, py = projection.transform_point(lon, lat, ccrs.PlateCarree())
        con = ConnectionPatch(
            xyA=axes_xy,
            coordsA=axi.transAxes,
            xyB=(px, py),
            coordsB=ax_main.transData,
            color=NE_FRAME_COLOR,
            linewidth=0.8,
            linestyle=(0, (4, 2)),
            zorder=21,
            rasterized=True,
        )
        fig.add_artist(con)


def _build_legends(
    fig, bin_colors, tail_labels, bin_mids, n_bins, size_min, size_max, gamma
):
    """Two-row probability legend plus a transmission-line legend."""
    prob_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=FLOOR_COLOR,
            markeredgecolor="none",
            markersize=4,
            label=f"<{int(SAFE_CUTOFF_PCT)}%",
        )
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=bin_colors[b],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=min(
                float(np.sqrt(_marker_size(bin_mids[b], size_min, size_max, gamma))),
                16.0,
            ),
            label=tail_labels[b],
        )
        for b in range(n_bins)
    ]
    # Transmission lines sit to the left of the probability legend on the same
    # row, both centered on a shared baseline near the bottom margin.
    line_handle = [
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.4, label="Transmission lines")
    ]
    leg_tl = fig.legend(
        handles=line_handle,
        loc="center",
        bbox_to_anchor=(0.24, 0.045),
        frameon=False,
        fontsize=9,
    )
    fig.add_artist(leg_tl)

    ncol = int(np.ceil(len(prob_handles) / 2))
    leg_prob = fig.legend(
        handles=prob_handles,
        loc="center",
        bbox_to_anchor=(0.62, 0.045),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        title="Probability of failure",
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.7,
    )
    fig.add_artist(leg_prob)


_EDGES_10 = [30, 40, 50, 60, 70, 80, 90, 100]

# Base name follows the GIC source, matching the original figure naming.
BASE_SUFFIX = "gnd_gic" if PROCESS_GND_FILES else "eff_gic"

# A single delivered figure, turbo colormap.
FIGURE_CASES = [
    {
        "file_suffix": BASE_SUFFIX,
        "tail_edges": _EDGES_10,
        "size_min": 12,
        "size_max": 300,
        "size_gamma": 1.7,
        "inset_size_scale": 0.62,
        "cmap_name": "YlOrRd",
    },
]


def _load_mean_vuln():
    """Return mean_vuln_all, using a cached parquet when one is provided.

    VULN_VIZ_CACHE lets the caller skip the multi-hundred-million-row
    aggregation while tuning the figure. Without it the full pipeline runs.
    """
    cache = os.getenv("VULN_VIZ_CACHE")
    if cache:
        logger.info("Loading cached mean_vuln from %s", cache)
        return pd.read_parquet(cache)

    _, combined_vuln, _ = load_gic_results()
    return process_vulnerability_chunks(
        combined_vuln, chunk_size=50, max_realizations=2000
    )


def main():
    mean_vuln_all = _load_mean_vuln()
    df_lines, _, _ = load_network_data()

    formats = ("tiff", "png", "pdf", "eps")
    for cfg in FIGURE_CASES:
        logger.info("Rendering vuln_trafo_%s", cfg["file_suffix"])
        plot_vuln_trafos(
            mean_vuln_all,
            df_lines,
            out_dir=FIGURES_DIR,
            formats=formats,
            dpi=600,
            **cfg,
        )
    logger.info("Saved %d figures to %s", len(FIGURE_CASES), FIGURES_DIR)


if __name__ == "__main__":
    main()
