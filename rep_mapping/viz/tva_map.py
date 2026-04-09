"""
Visualize the TVA transmission backbone with monitored devices.
Author: Dennies
"""

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from rep_mapping.rep_config import FIGURES_DIR, TVA_DIR, DATA_DIR,TVA_BOUNDARY, CRS_GEO, setup_logger, setup_matplotlib

logger = setup_logger(log_file="logs/tva_map.log")

setup_matplotlib()

SCHEME = {
    "boundary":   "#E63946",
    "500kv":      "#FF6B35",
    "345kv":      "#8338EC",
    "230kv":      "#06D6A0",
    "161kv":      "#FFB703",
    "unknown":    "#ccc",
    "osm_node":   "#023E8A",
    "synth_node": "#999",
    "monitored":  "#FFD700",
}

EXTENT = [-91.5, -80.5, 32, 37.8]


def voltage_class(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    elif v >= 500: return "500kv"
    elif v >= 345: return "345kv"
    elif v >= 230: return "230kv"
    elif v >= 161: return "161kv"
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


def build_legend(fig, ax):
    col1 = [
        mlines.Line2D([], [], color=SCHEME["500kv"],  linewidth=1.2, label="500 kV"),
        # mlines.Line2D([], [], color=SCHEME["345kv"],  linewidth=1.2, label="345 kV"),
        mlines.Line2D([], [], color=SCHEME["230kv"],  linewidth=1.2, label="230 kV"),
        # mlines.Line2D([], [], color=SCHEME["161kv"],  linewidth=1.2, label="161 kV"),
        mlines.Line2D([], [], color=SCHEME["boundary"], linewidth=1.2,
                      linestyle="--", label="TVA boundary"),
    ]
    col2 = [
        mlines.Line2D([], [], marker="s", color="w",
                    markerfacecolor=SCHEME["osm_node"],
                    markersize=5, label="OSM substation"),
        mlines.Line2D([], [], marker="o", color="w",
                    markerfacecolor=SCHEME["synth_node"],
                    markersize=5, label="Synthetic endpoint"),
        mlines.Line2D([], [], marker="*", color="w",
                    markerfacecolor=SCHEME["monitored"],
                    markersize=9, markeredgecolor="k",
                    markeredgewidth=0.5, label="GIC monitor"),
    ]

    leg1 = ax.legend(handles=col1, loc="lower left",
                     bbox_to_anchor=(0.2, -0.25),
                     bbox_transform=ax.transAxes,
                     ncol=1, fontsize=7.5, frameon=False,
                     borderaxespad=0)
    ax.add_artist(leg1)
    ax.legend(handles=col2, loc="lower left",
              bbox_to_anchor=(0.6, -0.25),
              bbox_transform=ax.transAxes,
              ncol=1, fontsize=7.5, frameon=False,
              borderaxespad=0)


def main():
    logger.info("Loading TVA grid data")

    tva_boundary    = gpd.read_file(TVA_BOUNDARY)
    tva_lines_hifld = gpd.read_parquet(TVA_DIR / "tva_lines_hifld.parquet") \
        if (TVA_DIR / "tva_lines_hifld.parquet").exists() else None

    with open(TVA_DIR / "G_backbone.pkl", "rb") as f:
        G_backbone = pickle.load(f)

    substations_df   = pd.read_parquet(TVA_DIR / "substations_df.parquet")
    matched_devices  = pd.read_parquet(TVA_DIR / "matched_devices.parquet")
    monitored_osmids = set(matched_devices["osmid"].astype(float).unique())

    sub_lookup = substations_df.set_index("name")
    node_pos   = {}
    for n in G_backbone.nodes:
        key = str(n)
        if key in sub_lookup.index:
            row = sub_lookup.loc[key]
            node_pos[n] = (float(row["longitude"]), float(row["latitude"]))

    logger.info(f"Nodes with positions: {len(node_pos)} / {G_backbone.number_of_nodes()}")

    fig, ax, proj_data = setup_ax()

    tva_boundary.to_crs(CRS_GEO).plot(
        ax=ax, facecolor="none", edgecolor=SCHEME["boundary"],
        linewidth=1.4, linestyle="--", zorder=3, transform=proj_data)

    if tva_lines_hifld is not None:
        vc_labels = {
            "500kv": "500+ kV", "345kv": "345 kV",
            "230kv": "230 kV",  "161kv": "161 kV",
        }
        backbone_line_idxs = {d["line_idx"]
                               for _, _, d in G_backbone.edges(data=True)
                               if "line_idx" in d}
        backbone_lines = tva_lines_hifld[
            tva_lines_hifld.index.isin(backbone_line_idxs)].to_crs(CRS_GEO)
        backbone_lines["vc"] = backbone_lines["VOLTAGE"].apply(voltage_class)

        for vc, label in vc_labels.items():
            sub = backbone_lines[backbone_lines["vc"] == vc]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=SCHEME[vc], linewidth=0.9,
                     alpha=0.85, zorder=4, transform=proj_data)
    else:
        vc_segs = {}
        for u, v, d in G_backbone.edges(data=True):
            if u not in node_pos or v not in node_pos:
                continue
            vc = voltage_class(d.get("voltage"))
            vc_segs.setdefault(vc, []).append([node_pos[u], node_pos[v]])
        for vc, segs in vc_segs.items():
            if not segs:
                continue
            ax.add_collection(mpl.collections.LineCollection(
                segs, transform=proj_data, color=SCHEME[vc],
                linewidth=0.9, alpha=0.85, zorder=4))

    osm_xy = [node_pos[n] for n in G_backbone.nodes
               if not (isinstance(n, str) and "_" in str(n))
               and n in node_pos
               and float(n) not in monitored_osmids]
    if osm_xy:
        ox, oy = zip(*osm_xy)
        sc1 = ax.scatter(ox, oy, s=6, c=SCHEME["osm_node"], marker="s",
                         transform=proj_data, zorder=7, alpha=0.85)
        sc1.set_path_effects([pe.withStroke(linewidth=0.5, foreground="#1a1a1a")])

    mon_in_bb = monitored_osmids & set(G_backbone.nodes)
    mon_xy    = [node_pos[n] for n in mon_in_bb if n in node_pos]
    if mon_xy:
        mx, my = zip(*mon_xy)
        ax.scatter(mx, my, s=60, c=SCHEME["monitored"], marker="*",
                   transform=proj_data, zorder=9,
                   edgecolors="black", linewidths=0.8)
        for n in mon_in_bb:
            if n not in node_pos:
                continue
            device = matched_devices[
                matched_devices["osmid"] == n]["device"].values[0]
            ax.annotate(
                device, xy=node_pos[n], fontsize=6, fontweight="bold",
                xytext=(5, 5), textcoords="offset points",
                transform=proj_data, zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, lw=0))
            
    synth_xy = [node_pos[n] for n in G_backbone.nodes
            if isinstance(n, str) and "_" in str(n) and n in node_pos]
    if synth_xy:
        sx, sy = zip(*synth_xy)
        ax.scatter(sx, sy, s=4, c=SCHEME["synth_node"], marker="o",
                transform=proj_data, zorder=6, alpha=0.6)

    build_legend(fig, ax)

    ax.set_title("TVA Transmission Network Derived from OSM and HIFLD",
                 fontsize=11, pad=8, loc="left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "tva_backbone.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "tva_backbone.pdf", dpi=300, bbox_inches="tight")
    logger.info(f"Saved figures to {FIGURES_DIR}")
    plt.close()


if __name__ == "__main__":
    main()