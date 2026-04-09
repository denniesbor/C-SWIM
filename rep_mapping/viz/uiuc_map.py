"""
Visualize the UIUC150 synthetic transmission network.
Author: Dennies
"""

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from rep_mapping.rep_config import FIGURES_DIR, UIUC_XLSX, setup_logger, setup_matplotlib

logger = setup_logger(log_file="logs/uiuc_map.log")

setup_matplotlib()

SCHEME = {
    "500kv":   "#FF6B35",
    "230kv":   "#06D6A0",
    "unknown": "#ccc",
    "node":    "#023E8A",
}

EXTENT = [-91.5, -80.5, 34.5, 37.2]


def parse_uiuc150():
    xl = pd.ExcelFile(UIUC_XLSX)

    subs = xl.parse("Substations")
    subs.columns = ["sub_num", "name", "lat", "lon", "kv_max", "Rg"]

    buses = xl.parse("Buses")
    buses.columns = ["bus_num", "bus_name", "sub_num", "kv"]
    buses = buses.merge(subs[["sub_num", "lat", "lon"]], on="sub_num", how="left")

    lines = xl.parse("Lines")
    lines.columns = ["from_bus", "to_bus", "circuit", "R_pu", "X_pu", "B_pu", "mva_limit"]
    kv_map     = dict(zip(buses.bus_num, buses.kv))
    lines["kv"] = lines["from_bus"].map(kv_map)

    return buses, lines, subs


def voltage_class(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    elif v >= 500: return "500kv"
    elif v >= 230: return "230kv"
    else:          return "unknown"


def setup_ax(figsize=(8, 6)):
    proj_data = ccrs.PlateCarree()
    fig, ax   = plt.subplots(figsize=figsize,
                              subplot_kw={"projection": proj_data})
    ax.set_extent(EXTENT, crs=proj_data)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   linewidth=0.6, edgecolor="#555", facecolor="none", zorder=1)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"),
                   linewidth=0.4, edgecolor="#c8dff0", zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="#ccc", alpha=0.5, crs=proj_data)
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = True
    gl.bottom_labels = True
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 7}
    gl.ylabel_style  = {"size": 7}

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax, proj_data


def build_legend(ax):
    handles = [
        mlines.Line2D([], [], color=SCHEME["500kv"], linewidth=1.2, label="500 kV"),
        mlines.Line2D([], [], color=SCHEME["230kv"], linewidth=1.2, label="230 kV"),
        mlines.Line2D([], [], marker="s", color="w",
                      markerfacecolor=SCHEME["node"],
                      markersize=5, label="Substation"),
    ]
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.0, -0.25),
              bbox_transform=ax.transAxes,
              ncol=3, fontsize=7.5, frameon=False,
              borderaxespad=0)


def main():
    logger.info("Loading UIUC150 data")

    buses, lines, subs = parse_uiuc150()

    bus_coords = {int(r.bus_num): (float(r.lon), float(r.lat))
                  for _, r in buses.iterrows()}

    fig, ax, proj_data = setup_ax()

    # Transmission lines coloured by voltage
    vc_segs = {}
    for _, row in lines.iterrows():
        c1 = bus_coords.get(int(row.from_bus))
        c2 = bus_coords.get(int(row.to_bus))
        if c1 is None or c2 is None:
            continue
        vc = voltage_class(row["kv"])
        vc_segs.setdefault(vc, []).append([c1, c2])

    for vc, segs in vc_segs.items():
        if not segs:
            continue
        ax.add_collection(mpl.collections.LineCollection(
            segs, transform=proj_data, color=SCHEME[vc],
            linewidth=0.9, alpha=0.85, zorder=4))

    # Substations — one marker per substation
    sub_xy = [(float(r.lon), float(r.lat)) for _, r in subs.iterrows()]
    if sub_xy:
        sx, sy = zip(*sub_xy)
        sc = ax.scatter(sx, sy, s=8, c=SCHEME["node"], marker="s",
                        transform=proj_data, zorder=7, alpha=0.9)
        sc.set_path_effects([pe.withStroke(linewidth=0.5, foreground="#1a1a1a")])

    build_legend(ax)

    ax.set_title("UIUC 150-Bus Synthetic Transmission Network",
                 fontsize=11, pad=8, loc="left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "uiuc150_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "uiuc150_map.pdf", dpi=300, bbox_inches="tight")
    logger.info(f"Saved to {FIGURES_DIR}")
    plt.close()


if __name__ == "__main__":
    main()