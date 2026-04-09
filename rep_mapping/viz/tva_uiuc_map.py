"""
TVA OSM+HIFLD and UIUC150 synthetic networks — two panel comparison.
Author: Dennies
"""

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr

from rep_mapping.rep_config import (
    UIUC_DIR,
    UIUC_XLSX,
    TVA_DIR,
    DATA_DIR,
    DEVICES_NC,
    FIGURES_DIR,
    TVA_BOUNDARY,
    CRS_GEO,
    setup_logger,
    setup_matplotlib,
)

setup_matplotlib()

SCHEME = {
    "500kv": "#D55E00",  # vermillion — colorblind safe
    "230kv": "#0072B2",  # blue — colorblind safe
    "unknown": "#ccc",
    "boundary": "#CC79A7",  # pink — distinct from both lines
    "osm_node": "#023E8A",  # dark blue — fine, uses square marker
    "synth": "#999",
    "monitor": "#FFD700",  # yellow star — distinct enough
}

EXTENT = [-91.5, -80.5, 32, 37.8]
PANEL_FONTSIZE = 11


def voltage_class(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    elif v >= 500:
        return "500kv"
    elif v >= 230:
        return "230kv"
    elif v >= 161:
        return "161kv"
    else:
        return "unknown"


def setup_ax(ax, proj_data):
    ax.set_extent(EXTENT, crs=proj_data)
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="#555",
        facecolor="none",
        zorder=1,
    )
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f5f0", zorder=0)
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"), linewidth=0.3, edgecolor="#c8dff0", zorder=2
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="#ccc", alpha=0.5, crs=proj_data
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def parse_uiuc150():
    xl = pd.ExcelFile(UIUC_XLSX)
    subs = xl.parse("Substations")
    subs.columns = ["sub_num", "name", "lat", "lon", "kv_max", "Rg"]
    buses = xl.parse("Buses")
    buses.columns = ["bus_num", "bus_name", "sub_num", "kv"]
    buses = buses.merge(subs[["sub_num", "lat", "lon"]], on="sub_num", how="left")
    lines = xl.parse("Lines")
    lines.columns = [
        "from_bus",
        "to_bus",
        "circuit",
        "R_pu",
        "X_pu",
        "B_pu",
        "mva_limit",
    ]
    kv_map = dict(zip(buses.bus_num, buses.kv))
    lines["kv"] = lines["from_bus"].map(kv_map)
    return buses, lines, subs


# Load all data
buses, lines, subs = parse_uiuc150()
bus_coords = {int(r.bus_num): (float(r.lon), float(r.lat)) for _, r in buses.iterrows()}

tva_boundary = gpd.read_file(TVA_BOUNDARY)

with open(TVA_DIR / "G_backbone.pkl", "rb") as f:
    G_backbone = pickle.load(f)

tva_subs = pd.read_parquet(TVA_DIR / "substations_df.parquet")
matched_devices = pd.read_parquet(TVA_DIR / "matched_devices.parquet")
monitored_osmids = set(matched_devices["osmid"].astype(float).unique())
tva_lines_hifld = (
    gpd.read_parquet(TVA_DIR / "tva_lines_hifld.parquet")
    if (TVA_DIR / "tva_lines_hifld.parquet").exists()
    else None
)

sub_lookup = tva_subs.set_index("name")
node_pos = {}
for n in G_backbone.nodes:
    key = str(n)
    if key in sub_lookup.index:
        row = sub_lookup.loc[key]
        node_pos[n] = (float(row["longitude"]), float(row["latitude"]))

ds = xr.open_dataset(DEVICES_NC)

proj_data = ccrs.PlateCarree()
fig, axes = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={"projection": proj_data})

for ax in axes:
    setup_ax(ax, proj_data)
    tva_boundary.to_crs("EPSG:4326").plot(
        ax=ax,
        facecolor="none",
        edgecolor=SCHEME["boundary"],
        linewidth=1.2,
        linestyle="--",
        transform=proj_data,
        zorder=3,
    )
    # Measured devices on both panels
    for i in range(len(ds.device)):
        dev = str(ds.device.values[i])
        lon = float(ds.longitude.values[i])
        lat = float(ds.latitude.values[i])
        ax.scatter(
            lon,
            lat,
            s=70,
            c=SCHEME["monitor"],
            marker="*",
            edgecolors="k",
            linewidths=0.5,
            transform=proj_data,
            zorder=10,
        )
        ax.annotate(
            dev,
            xy=(lon, lat),
            fontsize=5.5,
            xytext=(3, 3),
            textcoords="offset points",
            transform=proj_data,
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0),
        )

# (a) TVA OSM+HIFLD
ax = axes[0]

if tva_lines_hifld is not None:
    backbone_line_idxs = {
        d["line_idx"] for _, _, d in G_backbone.edges(data=True) if "line_idx" in d
    }
    backbone_lines = tva_lines_hifld[
        tva_lines_hifld.index.isin(backbone_line_idxs)
    ].to_crs("EPSG:4326")
    backbone_lines["vc"] = backbone_lines["VOLTAGE"].apply(voltage_class)
    for vc in ["500kv", "230kv", "161kv"]:
        sub = backbone_lines[backbone_lines["vc"] == vc]
        if sub.empty:
            continue
        sub.plot(
            ax=ax,
            color=SCHEME[vc],
            linewidth=0.9,
            alpha=0.85,
            zorder=4,
            transform=proj_data,
        )
else:
    vc_segs = {}
    for u, v, d in G_backbone.edges(data=True):
        if u not in node_pos or v not in node_pos:
            continue
        vc = voltage_class(d.get("voltage"))
        vc_segs.setdefault(vc, []).append([node_pos[u], node_pos[v]])
    for vc, segs in vc_segs.items():
        if segs:
            ax.add_collection(
                mpl.collections.LineCollection(
                    segs,
                    transform=proj_data,
                    color=SCHEME[vc],
                    linewidth=0.9,
                    alpha=0.85,
                    zorder=4,
                )
            )

osm_xy = [
    node_pos[n]
    for n in G_backbone.nodes
    if not (isinstance(n, str) and "_" in str(n))
    and n in node_pos
    and float(n) not in monitored_osmids
]
if osm_xy:
    ox, oy = zip(*osm_xy)
    sc = ax.scatter(
        ox,
        oy,
        s=6,
        c=SCHEME["osm_node"],
        marker="s",
        transform=proj_data,
        zorder=7,
        alpha=0.85,
    )
    sc.set_path_effects([pe.withStroke(linewidth=0.5, foreground="#1a1a1a")])

synth_xy = [
    node_pos[n]
    for n in G_backbone.nodes
    if isinstance(n, str) and "_" in str(n) and n in node_pos
]
if synth_xy:
    sx, sy = zip(*synth_xy)
    ax.scatter(
        sx,
        sy,
        s=4,
        c=SCHEME["synth"],
        marker="o",
        transform=proj_data,
        zorder=6,
        alpha=0.5,
    )

ax.text(
    0.0,
    1.04,
    "(a) TVA OSM+HIFLD transmission network",
    transform=ax.transAxes,
    fontsize=PANEL_FONTSIZE,
    va="bottom",
)

# (b) UIUC150 synthetic
ax = axes[1]

vc_segs = {}
for _, row in lines.iterrows():
    c1 = bus_coords.get(int(row.from_bus))
    c2 = bus_coords.get(int(row.to_bus))
    if c1 is None or c2 is None:
        continue
    vc = voltage_class(row["kv"])
    vc_segs.setdefault(vc, []).append([c1, c2])

for vc, segs in vc_segs.items():
    if segs:
        ax.add_collection(
            mpl.collections.LineCollection(
                segs,
                transform=proj_data,
                color=SCHEME[vc],
                linewidth=0.9,
                alpha=0.85,
                zorder=4,
            )
        )

sub_xy = [(float(r.lon), float(r.lat)) for _, r in subs.iterrows()]
if sub_xy:
    sx, sy = zip(*sub_xy)
    sc = ax.scatter(
        sx,
        sy,
        s=10,
        c=SCHEME["osm_node"],
        marker="s",
        transform=proj_data,
        zorder=7,
        alpha=0.9,
    )
    sc.set_path_effects([pe.withStroke(linewidth=0.5, foreground="#1a1a1a")])

ax.text(
    0.0,
    1.04,
    "(b) UIUC150 synthetic network",
    transform=ax.transAxes,
    fontsize=PANEL_FONTSIZE,
    va="bottom",
)

# Legend below panel (b)
col1 = [
    mlines.Line2D([], [], color=SCHEME["500kv"], linewidth=1.2, label="500 kV"),
    mlines.Line2D([], [], color=SCHEME["230kv"], linewidth=1.2, label="230 kV"),
    # mlines.Line2D([], [], color=SCHEME["161kv"], linewidth=1.2, label="161 kV"),
    mlines.Line2D(
        [],
        [],
        color=SCHEME["boundary"],
        linewidth=1.2,
        linestyle="--",
        label="TVA boundary",
    ),
]
col2 = [
    mlines.Line2D(
        [],
        [],
        marker="s",
        color="w",
        markerfacecolor=SCHEME["osm_node"],
        markersize=5,
        label="Substation",
    ),
    mlines.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor=SCHEME["synth"],
        markersize=5,
        alpha=0.6,
        label="HIFLD synthetic endpoint",
    ),
    mlines.Line2D(
        [],
        [],
        marker="*",
        color="w",
        markerfacecolor=SCHEME["monitor"],
        markersize=9,
        markeredgecolor="k",
        markeredgewidth=0.5,
        label="GIC monitor",
    ),
]

leg1 = axes[1].legend(
    handles=col1,
    loc="lower left",
    bbox_to_anchor=(0.0, -0.18),
    bbox_transform=axes[1].transAxes,
    ncol=1,
    fontsize=8,
    frameon=False,
)
axes[1].add_artist(leg1)
axes[1].legend(
    handles=col2,
    loc="lower left",
    bbox_to_anchor=(0.35, -0.18),
    bbox_transform=axes[1].transAxes,
    ncol=1,
    fontsize=8,
    frameon=False,
)

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES_DIR / "tva_uiuc_backbone.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "tva_uiuc_backbone.pdf", dpi=300, bbox_inches="tight")
plt.close()
