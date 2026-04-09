"""
Author: Dennies Bor
Role: Site mapping and visualization for power grid validation.
      Creates spatial maps showing GIC monitoring stations, magnetometer
      sites, MT stations, and transmission lines in the TVA region.
Inputs:
    - Transformer GIC simulation results (winding_gic_rand_0.csv)
    - TVA GIC measurement data
    - TVA magnetometer data
    - Simulated B/E field data (ds_gannon.nc)
    - EHV transmission line geometries (df_lines_EHV.pkl)
    - Simulated GIC cache (partial_pveubuntu.npz)
Outputs:
    - figures/sites_map.png/.pdf
"""

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.collections
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import box
from pyproj import Geod
from matplotlib_map_utils.core.north_arrow import north_arrow, NorthArrow

from configs import (
    DATA_DIR as DENNIES_DATA_LOC,
    LUCY_DATA_LOC,
    SWERVE_DIR,
    nerc_gic,
    tva_gic,
    cache_file,
    VAL_FIGURES_DIR as figures_dir,
    DEFAULT_SITE_IDS,
    TVA_NAME_MAP,
    DEFAULT_START_TIME,
    DEFAULT_END_TIME,
    setup_logger,
    setup_matplotlib,
)
from utils import (
    haversine_dist,
    read_gnd_gic,
    find_close_matches,
    load_trafo_gic_data,
    load_tva_gic_metadata,
    load_or_create_tva_gic_dataset,
    get_filtered_site_ids,
    prepare_validation_data,
    load_tva_magnetometer_data,
    load_simulated_data,
)

setup_matplotlib()

logger = setup_logger(log_file="logs/val_sites_map.log")


def load_data():
    """Load all datasets required for site mapping."""
    tl_path = DENNIES_DATA_LOC / "grid_processed" / "df_lines_EHV.pkl"
    with open(tl_path, "rb") as f:
        df_lines_EHV = pickle.load(f)

    trafo_gic_gdf        = load_trafo_gic_data(DENNIES_DATA_LOC)
    tva_gic_meas_path    = tva_gic / "GIC-measured"
    tva_gic_meas_metadat = load_tva_gic_metadata(tva_gic_meas_path)
    ds_gic_tva           = load_or_create_tva_gic_dataset(tva_gic_meas_path, LUCY_DATA_LOC, TVA_NAME_MAP, tva_gic_meas_metadat)

    (
        data_array, peak_times, median_values,
        mean_values, uncertainty_arr, substation_names,
    ) = read_gnd_gic(cache_file)

    site_ids = get_filtered_site_ids(SWERVE_DIR, DEFAULT_SITE_IDS)

    logger.info("Processing TVA validation...")
    tva_data = prepare_validation_data(
        trafo_gic_gdf, ds_gic_tva, site_ids,
        (data_array, peak_times, median_values, mean_values, uncertainty_arr, substation_names),
        start_time=DEFAULT_START_TIME, end_time=DEFAULT_END_TIME,
        threshold=0.5, savgol_window=10, nerc=False,
    )

    logger.info("Loading TVA magnetometer data...")
    ds_tva_mag = load_tva_magnetometer_data().resample(time="min").first()

    logger.info("Loading simulated data...")
    simulated_ds = load_simulated_data()

    return df_lines_EHV, tva_data, ds_tva_mag, simulated_ds


def compute_mag_mt_pairs(ds_tva_mag, simulated_ds):
    """Find closest MT site for each TVA magnetometer station."""
    mag_lats    = ds_tva_mag.latitude.values
    mag_lons    = ds_tva_mag.longitude.values
    mag_devices = ds_tva_mag.device.values
    mt_lats     = simulated_ds.latitude.values
    mt_lons     = simulated_ds.longitude.values

    try:
        mt_names = simulated_ds.name.values
    except AttributeError:
        mt_names = simulated_ds.device.values

    pairs = []
    for i, (mag_lat, mag_lon) in enumerate(zip(mag_lats, mag_lons)):
        distances = [haversine_dist(mag_lat, mag_lon, mt_lat, mt_lon) for mt_lat, mt_lon in zip(mt_lats, mt_lons)]
        mt_idx    = int(np.argmin(distances))
        pairs.append({
            "mag":      mag_devices[i],
            "mt":       mt_names[mt_idx],
            "mt_idx":   mt_idx,
            "distance": distances[mt_idx],
        })

    return pairs, mt_names, mt_lats, mt_lons


def compute_bounding_box(filtered_subs, trafo_unique, ds_tva_mag, pairs, mt_lats, mt_lons, mt_names):
    """Compute spatial bounding box covering all validation sites."""
    all_lats, all_lons = [], []

    for sub in filtered_subs:
        lat, lon = trafo_unique[trafo_unique.sub_id == sub][["latitude", "longitude"]].values[0]
        all_lats.append(lat)
        all_lons.append(lon)

    all_lats.extend(ds_tva_mag.latitude.values)
    all_lons.extend(ds_tva_mag.longitude.values)

    for pair in pairs:
        mt_idx = np.where(mt_names == pair["mt"])[0][0]
        all_lats.append(mt_lats[mt_idx])
        all_lons.append(mt_lons[mt_idx])

    return min(all_lats), max(all_lats), min(all_lons), max(all_lons)


def add_geodetic_scale_bar(ax, length_km=None, location=(0.70, 0.05), lw=2):
    """Add a geodetic scale bar to a cartopy axes."""
    geod = Geod(ellps="WGS84")
    x0, x1, y0, y1 = ax.get_extent(ccrs.PlateCarree())
    centre_lat  = 0.5 * (y0 + y1)
    _, _, width_m = geod.inv(x0, centre_lat, x1, centre_lat)

    if length_km is None:
        target = (width_m / 1000) / 4
        for cand in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
            if cand >= target:
                length_km = cand
                break

    lon0 = x0 + (x1 - x0) * location[0]
    lat0 = y0 + (y1 - y0) * location[1]
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000)

    for lon, lat in [(lon0, lat0), (lon1, lat1)]:
        ax.plot([lon, lon], [lat, lat], marker="|", ms=8, transform=ccrs.Geodetic(), color="k", zorder=4)
    ax.plot([lon0, lon1], [lat0, lat1], transform=ccrs.Geodetic(), lw=lw, color="k", zorder=4)
    ax.text((lon0 + lon1) / 2, lat0, f"{length_km} km", transform=ccrs.Geodetic(), ha="center", va="bottom", fontsize=9, weight="bold")


def create_sites_map(df_lines_EHV, tva_data, ds_tva_mag, simulated_ds):
    """Create overview and detail maps of TVA validation sites."""
    filtered_subs = [tva_data["valid_substations"][i] for i in tva_data["selected_indices"]]
    gic_monitors  = [tva_data["valid_match_ids"][i]   for i in tva_data["selected_indices"]]
    trafo_unique  = tva_data["trafo_unique"]

    pairs, mt_names, mt_lats, mt_lons = compute_mag_mt_pairs(ds_tva_mag, simulated_ds)

    for sub in filtered_subs:
        logger.info(f"Substation {sub}: GIC {gic_monitors[filtered_subs.index(sub)]}")
    for pair in pairs:
        logger.info(f"MAG {pair['mag']}: MT {pair['mt']}, Distance {pair['distance']:.2f} km")

    min_lat, max_lat, min_lon, max_lon = compute_bounding_box(filtered_subs, trafo_unique, ds_tva_mag, pairs, mt_lats, mt_lons, mt_names)
    logger.info(f"Bounding box: {min_lat:.3f}, {min_lon:.3f} to {max_lat:.3f}, {max_lon:.3f}")

    lines_gdf        = gpd.GeoDataFrame(df_lines_EHV, geometry="geometry")
    bbox             = box(min_lon, min_lat, max_lon, max_lat)
    lines_within_bbox = lines_gdf[lines_gdf.geometry.intersects(bbox)]
    logger.info(f"Lines within bbox: {len(lines_within_bbox)} of {len(lines_gdf)}")

    if lines_within_bbox.crs not in ("EPSG:4326", 4326):
        lines_within_bbox = lines_within_bbox.to_crs("EPSG:4326")

    pad    = 0.5
    x0, x1 = min_lon - pad, max_lon + pad
    y0, y1 = min_lat - pad, max_lat + pad
    geod   = Geod(ellps="WGS84")

    voltage_values = sorted([v for v in lines_within_bbox["VOLTAGE"].unique() if pd.notna(v)])
    palette        = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#F0E442", "#999999"]
    voltage_colors = {v: palette[i % len(palette)] for i, v in enumerate(voltage_values)}

    proj = ccrs.PlateCarree()
    fig  = plt.figure(figsize=(10, 10))
    gs   = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[9, 1], height_ratios=[2, 1], hspace=0.06, wspace=0.08)

    ax_us  = fig.add_subplot(gs[0, :], projection=proj)
    ax_tva = fig.add_subplot(gs[1, 0], projection=proj)
    ax_leg = fig.add_subplot(gs[1, 1])
    ax_leg.axis("off")

    ax_us.set_extent([-130, -65, 22, 50], crs=ccrs.PlateCarree())
    ax_us.add_feature(cfeature.STATES,    linewidth=0.6, edgecolor="#666666", facecolor="none")
    ax_us.add_feature(cfeature.COASTLINE, linewidth=0.5)

    for ax, labels_left, labels_bottom in [(ax_us, True, True), (ax_tva, True, True)]:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, alpha=0.5, color="gray", linestyle="--")
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = labels_left
        gl.bottom_labels = labels_bottom
        gl.xlabel_style  = {"size": 9}
        gl.ylabel_style  = {"size": 9}

    tva_box = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor="red", facecolor="none",
        transform=ccrs.PlateCarree(), zorder=4,
    )
    ax_us.add_patch(tva_box)

    ax_tva.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
    ax_tva.add_feature(cfeature.STATES,    linewidth=0.8, edgecolor="#BBBBBB", facecolor="none")
    ax_tva.add_feature(cfeature.COASTLINE, linewidth=0.5)

    for v in voltage_values:
        gdf_v = lines_within_bbox[lines_within_bbox["VOLTAGE"] == v]
        segs  = [np.asarray(geom.coords) for geom in gdf_v.geometry if geom is not None]
        if not segs:
            continue
        coll = mpl.collections.LineCollection(segs, linewidths=0.7, colors=voltage_colors[v], alpha=0.85, transform=ccrs.PlateCarree(), zorder=1)
        ax_tva.add_collection(coll)

    temp_transform = ccrs.PlateCarree()._as_mpl_transform(ax_tva)

    for i, sub in enumerate(filtered_subs):
        sub_lat, sub_lon = trafo_unique[trafo_unique.sub_id == sub][["latitude", "longitude"]].values[0].tolist()
        gic_monitor      = gic_monitors[i][0] if isinstance(gic_monitors[i], list) else gic_monitors[i]
        ax_tva.scatter(sub_lon, sub_lat, s=100, marker="s", c="tab:red", edgecolor="black", linewidth=0.5, zorder=3, transform=ccrs.PlateCarree())
        ax_tva.annotate(gic_monitor[0], xy=(sub_lon, sub_lat), xytext=(sub_lon + 0.15, sub_lat + 0.15), xycoords=temp_transform, color="black", fontsize=10)

    for pair in pairs:
        mag_name = pair["mag"]
        mt_idx   = np.where(mt_names == pair["mt"])[0][0]
        mag_lat  = float(np.atleast_1d(ds_tva_mag.sel(device=mag_name).latitude.values)[0])
        mag_lon  = float(np.atleast_1d(ds_tva_mag.sel(device=mag_name).longitude.values)[0])
        mt_lat   = float(mt_lats[mt_idx])
        mt_lon   = float(mt_lons[mt_idx])

        ax_tva.scatter(mag_lon, mag_lat, s=100, marker="o", c="tab:purple", edgecolor="black", linewidth=1, zorder=3, transform=ccrs.PlateCarree())
        ax_tva.scatter(mt_lon,  mt_lat,  s=80,  marker="^", c="tab:blue",   edgecolor="black", linewidth=1, zorder=2, transform=ccrs.PlateCarree())

    fig.canvas.draw()

    for lon, lat in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
        con = ConnectionPatch(
            xyA=(lon, lat), xyB=(lon, lat),
            coordsA=ax_us.transData, coordsB=ax_tva.transData,
            axesA=ax_us, axesB=ax_tva,
            color="red", lw=1.2, ls="--", alpha=0.9, clip_on=False,
        )
        con.set_zorder(20)
        ax_us.add_artist(con)

    ax_us.set_zorder(1)
    ax_leg.set_zorder(2)
    ax_tva.set_zorder(10)
    ax_tva.patch.set_facecolor("white")
    ax_tva.patch.set_edgecolor("0.2")
    ax_tva.patch.set_linewidth(1.2)
    ax_tva.patch.set_clip_on(False)
    ax_tva.patch.set_path_effects([pe.SimplePatchShadow(offset=(5, -5), alpha=0.28, rho=0.9), pe.Normal()])
    for sp in ax_tva.spines.values():
        sp.set_zorder(11)

    transmission_elements = [Line2D([], [], color=voltage_colors[v], linewidth=2, label=f"{v} kV") for v in voltage_values]
    site_elements = [
        Line2D([], [], marker="s", markersize=10, markerfacecolor="tab:red",    markeredgecolor="black", linestyle="None", label="GIC Monitor/Substation"),
        Line2D([], [], marker="o", markersize=8,  markerfacecolor="tab:purple", markeredgecolor="black", linestyle="None", label="Magnetometer"),
        Line2D([], [], marker="^", markersize=7,  markerfacecolor="tab:blue",   markeredgecolor="black", linestyle="None", label="MT Station"),
    ]
    ax_leg.legend(handles=transmission_elements + site_elements, loc="center", frameon=False, fontsize=10)

    for ext in ["png", "pdf"]:
        fig.savefig(figures_dir / f"sites_map.{ext}", dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved sites map to {figures_dir}")


def main():
    """Run site mapping pipeline for TVA validation region."""
    df_lines_EHV, tva_data, ds_tva_mag, simulated_ds = load_data()
    create_sites_map(df_lines_EHV, tva_data, ds_tva_mag, simulated_ds)


if __name__ == "__main__":
    main()