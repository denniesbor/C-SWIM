"""
Script to prepare extra-high voltage (EHV) substation and transmission lines.

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025

"""

# --- standard library imports ------------------------------------------------
import os
import gc
import sys
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import bezpy
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import split, substring

# sys.path.append("..")
from configs import setup_logger, get_data_dir

warnings.simplefilter("ignore")
PROJ = "EPSG:5070"  # projected CRS for distance work

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/p_power_grid.log")


# %%
def load_and_process_transmission_lines(transmission_lines_path, ferc_gdf_path):
    """
    Load and process transmission line data, filtering for extra-high voltage
    (EHV) lines and associating them with FERC regions.

    Parameters:
    transmission_lines_path : str
        Path to the transmission lines GeoJSON or shapefile.
    ferc_gdf_path : str
        Path to the FERC region boundaries GeoJSON or shapefile.

    Returns:
    gpd.GeoDataFrame
        Processed transmission lines with unique IDs, voltage filtering,
        FERC regions, and computed lengths.
    """
    gdf = gpd.read_file(transmission_lines_path).to_crs("EPSG:4326")
    gdf.rename(columns={"ID": "line_id"}, inplace=True)
    gdf = gdf.reset_index(drop=True).explode(index_parts=True).reset_index(level=1)
    gdf["line_id"] = gdf.apply(
        lambda row: (
            f"{row['line_id']}_{row['level_1']}"
            if row["level_1"] > 0
            else row["line_id"]
        ),
        axis=1,
    )
    gdf = gdf[gdf["VOLTAGE"] >= 200].drop(columns=["level_1"])
    ferc_gdf = process_ferc_gdf(ferc_gdf_path)
    gdf = gpd.sjoin(gdf, ferc_gdf, how="inner", predicate="intersects").drop(
        columns="index_right"
    )
    gdf["length"] = gdf.apply(lambda row: bezpy.tl.TransmissionLine(row).length, axis=1)
    del ferc_gdf
    gc.collect()

    return gdf


def load_and_process_substations(substations_data_path, ferc_gdf_path):
    """
    Load and process OSM substation data, adapting to available columns.
    """
    substations_gdf = gpd.read_file(substations_data_path).to_crs("EPSG:4326")

    # Handle OSM osmid vs expected ss_id
    if "osmid" in substations_gdf.columns:
        substations_gdf["ss_id"] = substations_gdf["osmid"].astype(str)
    elif "SS_ID" in substations_gdf.columns:
        substations_gdf["ss_id"] = substations_gdf["SS_ID"]
    else:
        # Create a unique ID if none exists
        substations_gdf["ss_id"] = substations_gdf.index.astype(str)

    # Ensure geometry is buffered properly
    substations_gdf["geometry"] = substations_gdf["geometry"].apply(
        lambda geom: (
            geom.buffer(0.00001) if isinstance(geom, (Point, LineString)) else geom
        )
    )

    substations_gdf.dropna(subset=["geometry"], inplace=True)

    # Process FERC regions
    ferc_gdf = process_ferc_gdf(ferc_gdf_path)
    substations_gdf = gpd.sjoin(
        substations_gdf, ferc_gdf, how="inner", predicate="intersects"
    )
    substations_gdf.drop(columns="index_right", inplace=True)

    # Map OSM columns to expected columns
    column_mapping = {
        "name": "SS_NAME",
        "operator": "SS_OPERATOR",
        "voltage": "SS_VOLTAGE",
        "substation": "SS_TYPE",
        "element_type": "SS_TYPE",  # fallback
        "power": "SS_TYPE",  # another fallback
    }

    for osm_col, expected_col in column_mapping.items():
        if (
            osm_col in substations_gdf.columns
            and expected_col not in substations_gdf.columns
        ):
            substations_gdf[expected_col] = substations_gdf[osm_col]

    # Ensure required columns exist with defaults if necessary
    required_cols = ["SS_NAME", "SS_OPERATOR", "SS_VOLTAGE", "SS_TYPE"]
    for col in required_cols:
        if col not in substations_gdf.columns:
            substations_gdf[col] = "Unknown"

    substations_gdf = prepare_substations_for_grid_analysis(substations_gdf)

    return substations_gdf


def process_ferc_gdf(ferc_gdf_path):
    """
    Process FERC region GeoDataFrame by standardizing region names and
    filtering out non-relevant regions.

    Parameters:
    ferc_gdf_path : str
        Path to the FERC region boundaries GeoJSON or shapefile.

    Returns:
    gpd.GeoDataFrame
        Processed FERC GeoDataFrame with standardized names and filtering.
    """

    ferc_gdf = gpd.read_file(ferc_gdf_path).to_crs("EPSG:4326")
    rename_mapping = {
        "NorthernGridConnected": "NorthGC",
        "WestConnect": "WestC",
        "NorthernGridUnconnected": "NorthGUC",
        "WestConnectNonEnrolled": "WestCNE",
    }
    ferc_gdf["REGIONS"] = ferc_gdf["REGIONS"].replace(rename_mapping)
    ferc_gdf = ferc_gdf[ferc_gdf["REGIONS"] != "NotOrder1000"]

    return ferc_gdf


def prepare_substations_for_grid_analysis(substation_gdf):
    """Prepare OSM substations for grid analysis."""

    # Ensure ss_id exists
    if "osmid" in substation_gdf.columns and "ss_id" not in substation_gdf.columns:
        substation_gdf["ss_id"] = substation_gdf.index.astype(str)

    # Map OSM columns to expected columns
    column_mapping = {
        "name": "SS_NAME",
        "operator": "SS_OPERATOR",
        "voltage": "SS_VOLTAGE",
        "substation": "SS_TYPE",
        "power": "SS_TYPE",  # fallback
    }

    for osm_col, grid_col in column_mapping.items():
        if osm_col in substation_gdf.columns and grid_col not in substation_gdf.columns:
            substation_gdf[grid_col] = substation_gdf[osm_col]

    # Add defaults for missing columns
    defaults = {
        "SS_NAME": "Unknown",
        "SS_OPERATOR": "Unknown",
        "SS_VOLTAGE": "Unknown",
        "SS_TYPE": "substation",
    }

    for col, default in defaults.items():
        if col not in substation_gdf.columns:
            substation_gdf[col] = default

    # Handle geometry conversion if needed
    try:
        assert all(substation_gdf.geometry.geom_type == "Point")
    except AssertionError:
        # Store original CRS
        original_crs = substation_gdf.crs

        # Convert to projected CRS for accurate centroid calculation
        substation_gdf = substation_gdf.to_crs("EPSG:5070")  # US Albers Equal Area

        # Convert non-point geometries to centroids
        non_point_mask = substation_gdf.geometry.geom_type != "Point"
        substation_gdf.loc[non_point_mask, "geometry"] = substation_gdf.loc[
            non_point_mask, "geometry"
        ].centroid

        # Convert back to original CRS
        substation_gdf = substation_gdf.to_crs(original_crs)

    return substation_gdf


# Intersect transmission lines with substations
def graph_node_edges(
    substation_gdf: gpd.GeoDataFrame,
    translines_gdf: gpd.GeoDataFrame,
    buffer_distance: float = 250,
    wgs84: str = "EPSG:4326",
    proj: str = "EPSG:5070",
):
    """
    Build node/edge GeoDataFrames for EHV grid analysis.
    Returns (substations_points, transmission_lines, substations_buffers).
    """

    # ----- CRS -----
    substation_gdf = substation_gdf.to_crs(wgs84).to_crs(proj)
    translines_gdf = translines_gdf.to_crs(wgs84).to_crs(proj)

    # ----- buffer + spatial join -----
    substation_gdf["buffered"] = substation_gdf.geometry.buffer(buffer_distance)
    buffered_ss = substation_gdf.set_geometry("buffered")

    intersection = gpd.sjoin(
        translines_gdf, buffered_ss, how="inner", predicate="intersects"
    )

    print(intersection.columns)

    # if ggeometry col rename to geom_left for downstream compatibility
    if "geometry" in intersection.columns:
        intersection.rename(columns={"geometry": "geometry_left"}, inplace=True)

    inter_gdf = gpd.GeoDataFrame(intersection, geometry="geometry_left", crs=proj)

    # ----- rename helpers -----
    rename = {
        "name": "SS_NAME",
        "operator": "SS_OPERATOR",
        "voltage": "SS_VOLTAGE",
        "substation": "SS_TYPE",
        "ss_id": "SS_ID",
        "REGION_ID_left": "REGION_ID",
        "REGIONS_left": "REGION",
        "line_id": "LINE_ID",
        "NAICS_CODE": "LINE_NAICS_CODE",
        "VOLTAGE": "LINE_VOLTAGE",
        "VOLT_CLASS": "LINE_VOLT_CLASS",
        "INFERRED": "INFERRED",
        "SUB_1": "SUB_1",
        "SUB_2": "SUB_2",
        "length": "LINE_LEN",
    }

    # ----- transmission‑line GeoDF -----
    tl_gdf = inter_gdf.copy()
    tl_cols = [c for c in rename if c in tl_gdf.columns]
    tl_cols.append(tl_gdf.geometry.name)  # append 'geometry_left'
    tl_gdf = tl_gdf[tl_cols].rename(columns=rename)

    # ----- buffered‑substation GeoDF -----
    ss_gdf = inter_gdf.set_geometry("geometry_right")
    ss_cols = [c for c in rename if c in ss_gdf.columns]
    ss_cols.append(ss_gdf.geometry.name)  # append 'geometry_right'
    ss_gdf = ss_gdf[ss_cols].rename(columns=rename)

    # ----- representative point GeoDF -----
    substation_pts = ss_gdf.copy()
    substation_pts["rep_point"] = substation_pts.geometry.centroid
    substation_pts = substation_pts.set_geometry("rep_point")

    # ----- back to WGS84 -----
    tl_gdf = tl_gdf.to_crs(wgs84)
    ss_gdf = ss_gdf.to_crs(wgs84)
    substation_pts = substation_pts.to_crs(wgs84)

    gc.collect()
    return substation_pts, tl_gdf, ss_gdf


def get_nodes_edges(tl_gdf: gpd.GeoDataFrame, ss_gdf: gpd.GeoDataFrame):
    """Build dictionaries that describe substation/line topology."""
    tl_p = tl_gdf.to_crs(PROJ)
    ss_p = ss_gdf.to_crs(PROJ)

    rep_pt = ss_p.set_index("SS_ID")["rep_point"].to_dict()
    line_geom = tl_p.set_index("LINE_ID")["geometry_left"].to_dict()

    sub2lines, sub2volts, line2subs = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    # basic mappings ----------------------------------------------------------
    for ss, lid, v in tl_gdf[["SS_ID", "LINE_ID", "LINE_VOLTAGE"]].itertuples(False):
        if lid not in sub2lines[ss]:
            sub2lines[ss].append(lid)
        sub2volts[ss].append(v)
        if ss not in line2subs[lid]:
            line2subs[lid].append(ss)

    sub2subs = defaultdict(list)
    line_pairs = {}

    # ordered substation pairs per line ---------------------------------------
    for lid, subs in line2subs.items():
        g = line_geom.get(lid)
        if g is None or len(subs) < 2 or not g.is_valid:
            line_pairs[lid] = []
            continue

        proj = []
        for ss in subs:
            p = rep_pt.get(ss)
            if p is None or not p.is_valid:
                continue
            snap = g.interpolate(g.project(p))  # snap to line
            d = g.project(snap)
            if np.isfinite(d):
                proj.append((d, ss))

        if len(proj) < 2:
            line_pairs[lid] = []
            continue

        proj.sort()
        ordered = [s for d, s in proj]
        pairs = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
        line_pairs[lid] = pairs

        for a, b in pairs:  # mutual adjacency
            if b not in sub2subs[a]:
                sub2subs[a].append(b)
            if a not in sub2subs[b]:
                sub2subs[b].append(a)

    return (
        dict(sub2lines),
        dict(line2subs),
        dict(sub2volts),
        dict(sub2subs),
        line_pairs,
    )


def _prep_substations(substation_gdf, substation_to_lines, substation_to_line_voltages):
    """Return (ss_df, ss_gt300, region_dict) exactly as original block."""
    g = substation_gdf.drop_duplicates("SS_ID").to_crs(4326)
    g["lat"] = g.centroid.y
    g["lon"] = g.centroid.x
    g["connected_tl_id"] = g["SS_ID"].map(substation_to_lines)
    g["LINE_VOLTS"] = g["SS_ID"].map(substation_to_line_voltages)

    cols = [
        "SS_ID",
        "SS_NAME",
        "SS_OPERATOR",
        "SS_VOLTAGE",
        "SS_TYPE",
        "REGION_ID",
        "REGION",
        "lat",
        "lon",
        "connected_tl_id",
        "LINE_VOLTS",
    ]
    ss_df = g[cols]
    ss_big = ss_df.copy().reset_index(drop=True)
    region = dict(zip(ss_big["SS_ID"], ss_big["REGION"]))
    return ss_df, ss_big, region


def _filter_ferc(trans_lines_within_FERC, ids):
    """Subset & trim FERC table – unchanged."""
    keep = trans_lines_within_FERC[trans_lines_within_FERC["LINE_ID"].isin(ids)]
    cols = [
        "LINE_ID",
        "OWNER",
        "VOLTAGE",
        "VOLT_CLASS",
        "SUB_1",
        "SUB_2",
        "SHAPE__Len",
        "geometry",
        "REGION_ID",
        "REGIONS",
        "length",
    ]
    return keep[cols].reset_index(drop=True)


def _split_multi_sub_lines(tl_gdf_subset, pair_map, substation_gdf):
    """Replicates original splitting loop verbatim, returns GeoDataFrame."""
    proj_crs = "EPSG:5070"
    tl_p = tl_gdf_subset.to_crs(proj_crs)
    ss_p = substation_gdf.to_crs(proj_crs)

    line_geom = dict(zip(tl_p["LINE_ID"], tl_p["geometry_left"]))
    ss_pt = dict(zip(ss_p["SS_ID"], ss_p["rep_point"]))

    recs = []
    for lid, pairs in pair_map.items():
        if not pairs:  # skip empties
            continue
        g_line = line_geom.get(lid)
        if g_line is None:
            continue
        for a, b in pairs:
            pa, pb = ss_pt.get(a), ss_pt.get(b)
            if pa is None or pb is None:
                continue
            d1, d2 = g_line.project(pa), g_line.project(pb)
            seg = substring(g_line, min(d1, d2), max(d1, d2))
            recs.append(
                dict(
                    LINE_ID=f"{lid}_{a}_{b}",
                    geometry=seg,
                    substation_a=a,
                    substation_b=b,
                    original_line_id=lid,
                )
            )
    gdf = gpd.GeoDataFrame(recs, crs=proj_crs).to_crs("EPSG:4326")
    gdf.reset_index(drop=True, inplace=True)
    gdf["new_line_id"] = (
        gdf["LINE_ID"] + "_" + gdf.groupby("LINE_ID").cumcount().astype(str)
    )
    return gdf


def get_EHV_lines(substation_gdf, tl_gdf_subset, translines_gdf):
    """Wrapper that calls the original blocks now tucked in helpers."""
    # ------------------------------------------------------------------ input
    trans_lines_within_FERC = translines_gdf.copy()
    trans_lines_within_FERC.rename(columns={"line_id": "LINE_ID"}, inplace=True)

    (sub2lines, line2subs, sub2volts, sub2subs, pair_map) = get_nodes_edges(
        tl_gdf_subset, substation_gdf
    )

    # ----------------------------------------------------------- substations
    ss_df, tm_ss_gt300, region_dict = _prep_substations(
        substation_gdf, sub2lines, sub2volts
    )

    # ----------------------------------------------------------- FERC filter
    connected_ids = tm_ss_gt300["connected_tl_id"].explode().unique()
    trans_ferc_filt = _filter_ferc(trans_lines_within_FERC, connected_ids)

    # ----------------------------------------------------------- split lines
    split_tl = _split_multi_sub_lines(tl_gdf_subset, pair_map, substation_gdf)

    split_tl = split_tl.merge(
        trans_ferc_filt[["LINE_ID", "VOLTAGE"]], on="LINE_ID", how="left"
    )

    trans_ferc_minus_split = trans_ferc_filt[
        ~trans_ferc_filt["LINE_ID"].isin(split_tl["LINE_ID"])
    ][["LINE_ID", "VOLTAGE", "geometry"]]

    split_tl.drop(columns=["LINE_ID"], inplace=True)
    split_tl.rename(columns={"new_line_id": "LINE_ID"}, inplace=True)

    trans_lines_within_FERC_filtered_ = gpd.GeoDataFrame(
        pd.concat(
            [trans_ferc_minus_split, split_tl[["LINE_ID", "VOLTAGE", "geometry"]]]
        ),
        crs=trans_ferc_minus_split.crs,
    )

    # ---------------------------------------------------------------- graphs
    df_ss_graph = pd.DataFrame()
    df_ss_graph["SS_IDs"] = sub2volts.keys()
    df_ss_graph["Connecting_Lines"] = df_ss_graph["SS_IDs"].map(sub2lines)
    df_ss_graph["Connecting_Substations"] = df_ss_graph["SS_IDs"].map(sub2subs)
    df_ss_graph["REGION"] = df_ss_graph["SS_IDs"].map(region_dict)
    df_ss_graph = df_ss_graph[df_ss_graph["SS_IDs"].isin(tm_ss_gt300["SS_ID"])]

    # ---------------------------------------------------------------- pairs
    filtered_line2subs = {lid: s for lid, s in pair_map.items() if len(s) == 1}
    edge_rows = [
        {"LINE_ID": lid, "substation_a": a, "substation_b": b}
        for lid, subs in filtered_line2subs.items()
        for a, b in subs
    ]
    line_to_substations_df = pd.concat(
        [pd.DataFrame(edge_rows), split_tl[["LINE_ID", "substation_a", "substation_b"]]]
    )

    # ---------------------------------------------------------------- length
    trans_lines_within_FERC_filtered_["obj"] = trans_lines_within_FERC_filtered_.apply(
        bezpy.tl.TransmissionLine, axis=1
    )
    trans_lines_within_FERC_filtered_["length"] = trans_lines_within_FERC_filtered_[
        "obj"
    ].apply(lambda x: x.length)
    trans_lines_within_FERC_filtered_ = trans_lines_within_FERC_filtered_[
        ~(trans_lines_within_FERC_filtered_.VOLTAGE == -9.99999e05)
    ].dropna()

    # ---------------------------------------------------------------- EHV set
    df_lines_EHV = line_to_substations_df[
        line_to_substations_df.substation_a.isin(tm_ss_gt300.SS_ID)
    ]
    df_lines_EHV = df_lines_EHV[df_lines_EHV.substation_b.isin(tm_ss_gt300.SS_ID)]

    df_lines_EHV = df_lines_EHV.merge(
        trans_lines_within_FERC_filtered_[["LINE_ID", "VOLTAGE", "length"]],
        on="LINE_ID",
        how="left",
    ).drop_duplicates()

    df_lines_EHV = df_lines_EHV[~df_lines_EHV["VOLTAGE"].isna()]
    df_lines_EHV["VOLTAGE"] = df_lines_EHV["VOLTAGE"].astype(int)
    df_lines_EHV = df_lines_EHV[df_lines_EHV.VOLTAGE >= 200]

    df_lines_EHV = df_lines_EHV.merge(
        trans_lines_within_FERC_filtered_[["LINE_ID", "geometry"]],
        on="LINE_ID",
        how="left",
    )
    df_lines_EHV = gpd.GeoDataFrame(df_lines_EHV, geometry="geometry").drop_duplicates()

    df_lines_EHV["from_bus_id"] = (
        df_lines_EHV["substation_a"].astype(str)
        + "_"
        + df_lines_EHV["VOLTAGE"].astype(int).astype(str)
    )
    df_lines_EHV["to_bus_id"] = (
        df_lines_EHV["substation_b"].astype(str)
        + "_"
        + df_lines_EHV["VOLTAGE"].astype(int).astype(str)
    )

    sub1_pairs = df_lines_EHV[
        ["substation_a", "VOLTAGE", "from_bus_id"]
    ].drop_duplicates()
    sub2_pairs = df_lines_EHV[
        ["substation_b", "VOLTAGE", "to_bus_id"]
    ].drop_duplicates()
    sub1_pairs.columns = ["substation", "voltage", "bus_id"]
    sub2_pairs.columns = ["substation", "voltage", "bus_id"]

    unique_sub_voltage_pairs = (
        pd.concat([sub1_pairs, sub2_pairs]).drop_duplicates().reset_index(drop=True)
    )
    unique_sub_voltage_pairs = unique_sub_voltage_pairs.merge(
        substation_gdf[["SS_ID", "rep_point", "SS_TYPE", "REGION"]],
        left_on="substation",
        right_on="SS_ID",
        how="left",
    )
    unique_sub_voltage_pairs = gpd.GeoDataFrame(
        unique_sub_voltage_pairs, geometry="rep_point"
    )
    unique_sub_voltage_pairs.drop("SS_ID", axis=1, inplace=True)

    # mapping to standard volt classes ---------------------------------------
    line_voltage_ratings = {
        345: 345,
        230: 230,
        450: np.nan,
        500: 500,
        765: 765,
        250: 230,
        400: np.nan,
        232: 230,
        1000: np.nan,
        220: 230,
        273: np.nan,
        218: 230,
        236: 230,
        287: np.nan,
        238: 230,
        200: np.nan,
    }
    df_lines_EHV["VOLTAGE"] = df_lines_EHV.VOLTAGE.map(line_voltage_ratings)
    df_lines_EHV.dropna(subset=["VOLTAGE"], inplace=True)

    # ------------------------------------------------------------------ return
    return (
        unique_sub_voltage_pairs,
        sub2volts,
        ss_df,
        df_lines_EHV,
        trans_lines_within_FERC_filtered_,
        tl_gdf_subset[["LINE_ID", "LINE_VOLTAGE", "geometry_left"]],
    )


# %%
if __name__ == "__main__":
    logger.info("[A] Starting EHV grid processing pipeline")

    dir_out = DATA_LOC / "grid_processed"
    os.makedirs(dir_out, exist_ok=True)

    # Generate 'processed_transmission_lines.pkl'
    logger.info("[B] Loading and processing transmission lines")
    filename = "Electric__Power_Transmission_Lines.shp"
    folder = DATA_LOC / "Electric__Power_Transmission_Lines"
    transmission_lines_path = folder / filename
    ferc_gdf_path = DATA_LOC / "nerc_gdf.geojson"
    translines_gdf = load_and_process_transmission_lines(
        transmission_lines_path, ferc_gdf_path
    )
    logger.info("[C] Saving processed transmission lines")
    translines_gdf.to_file(dir_out / "processed_transmission_lines.gpkg", driver="GPKG")
    with open(dir_out / "processed_transmission_lines.pkl", "wb") as f:
        pickle.dump(translines_gdf, f)

    # Generate 'processed_substations.pkl'
    logger.info("[D] Loading and processing substations")
    substations_data_path = (
        DATA_LOC / "substation_locations" / "us_substations_full.geojson"
    )
    substation_gdf = load_and_process_substations(substations_data_path, ferc_gdf_path)
    logger.info("[E] Saving processed substations")
    substation_gdf.to_file(dir_out / "processed_substations.gpkg", driver="GPKG")
    with open(dir_out / "processed_substations.pkl", "wb") as f:
        pickle.dump(substation_gdf, f)

    # Graph nodes and edges
    logger.info("[F] Building graph topology")
    substation_gdf, tl_gdf_subset, ss_gdf_subset = graph_node_edges(
        substation_gdf, translines_gdf
    )

    # Get EHV lines
    logger.info("[G] Extracting EHV lines and connections")
    (
        unique_sub_voltage_pairs,
        substation_to_line_voltages,
        ss_df,
        df_lines_EHV,
        trans_lines_within_FERC_filtered_,
        tl_gdf_subset,
    ) = get_EHV_lines(substation_gdf, tl_gdf_subset, translines_gdf)

    print(df_lines_EHV.shape)

    # Pickle unique_sub_voltage_pairs
    logger.info("[H] Saving results - unique_sub_voltage_pairs")
    with open(dir_out / "unique_sub_voltage_pairs.pkl", "wb") as f:
        pickle.dump(unique_sub_voltage_pairs, f)

    # Pickle substation to line voltages
    logger.info("[I] Saving results - substation_to_line_voltages")
    with open(dir_out / "substation_to_line_voltages.pkl", "wb") as f:
        pickle.dump(substation_to_line_voltages, f)

    # Pickle substations ss_df
    logger.info("[J] Saving results - ss_df")
    with open(dir_out / "ss_df.pkl", "wb") as f:
        pickle.dump(ss_df, f)

    # Pickle df_lines_EHV
    logger.info("[K] Saving results - df_lines_EHV")
    with open(dir_out / "df_lines_EHV.pkl", "wb") as f:
        pickle.dump(df_lines_EHV, f)

    # Save tranlines within FERC filtered as pkl
    logger.info("[L] Saving results - trans_lines_within_FERC_filtered")
    with open(dir_out / "trans_lines_within_FERC_filtered.pkl", "wb") as f:
        pickle.dump(trans_lines_within_FERC_filtered_, f)

    # Rename geometry_left to geometry
    tl_gdf_subset.rename(columns={"geometry_left": "geometry"}, inplace=True)

    # Pickle tl_gdf_subset line_id and geometry only
    with open(dir_out / "tl_gdf_subset.pkl", "wb") as f:
        pickle.dump(
            tl_gdf_subset[["LINE_ID", "LINE_VOLTAGE", "geometry"]].drop_duplicates(
                subset="LINE_ID"
            ),
            f,
        )
