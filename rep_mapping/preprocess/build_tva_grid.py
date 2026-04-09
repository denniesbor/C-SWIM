"""
Build TVA transmission network from OSM substations and HIFLD lines.
Author: Dennies
"""

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import Point

from rep_mapping.rep_config import (
    HIFLD_PATH,
    OSM_SUB_PATH,
    TVA_BOUNDARY,
    DEVICES_NC,
    GRID_MAPPING,
    TVA_DIR,
    CRS_PROJ,
    CRS_GEO,
    VOLTAGE_THRESHOLD,
    SNAP_DISTANCE,
    KEEP_SYNTHETIC_ABOVE,
    LINE_RESISTANCE,
    TRAFO_WINDING_R,
    RG_REAL,
    RG_SYNTHETIC,
    POOL_GEN,
    POOL_TRANS,
    setup_logger,
)

import xarray as xr

logger = setup_logger(log_file="logs/build_tva_grid.log")


def load_raw_data():
    logger.info("Loading raw data")
    tl_hifld = gpd.read_file(HIFLD_PATH)
    sub_gdf = gpd.read_file(OSM_SUB_PATH)
    tva_boundary = gpd.read_file(TVA_BOUNDARY)
    devices_ds = xr.open_dataset(DEVICES_NC)
    devices_gdf = gpd.GeoDataFrame(
        {
            "device": devices_ds.device.values,
            "latitude": devices_ds.latitude.values,
            "longitude": devices_ds.longitude.values,
            "type": devices_ds.type.values,
        },
        geometry=gpd.points_from_xy(
            devices_ds.longitude.values, devices_ds.latitude.values
        ),
        crs=CRS_GEO,
    )
    return tl_hifld, sub_gdf, tva_boundary, devices_gdf


def clip_to_tva(tl_hifld, sub_gdf, tva_boundary):
    logger.info("Clipping to TVA footprint")
    tva_proj = tva_boundary.to_crs(CRS_PROJ)
    tva_service_area = tva_proj.union_all()

    tl_hifld_gdf = tl_hifld[tl_hifld["VOLTAGE"] > VOLTAGE_THRESHOLD].copy()
    tl_proj = tl_hifld_gdf.to_crs(CRS_PROJ)
    sub_proj = sub_gdf.to_crs(CRS_PROJ)

    tva_lines = tl_proj[tl_proj.intersects(tva_service_area)].copy()
    tva_subs = sub_proj[sub_proj.intersects(tva_service_area)].copy()
    tva_lines = tva_lines[tva_lines.geom_type == "LineString"].copy()

    logger.info(f"HIFLD lines: {len(tva_lines)}  OSM subs: {len(tva_subs)}")

    hifld_union = tva_lines.union_all().buffer(SNAP_DISTANCE)
    tva_subs_filt = tva_subs[tva_subs.intersects(hifld_union)].copy()

    logger.info(f"OSM subs near HIFLD lines: {len(tva_subs_filt)}")
    return tva_lines, tva_subs_filt


def build_endpoints(tva_lines):
    records = []
    for idx, row in tva_lines.iterrows():
        coords = list(row.geometry.coords)
        records.append(
            {
                "line_idx": idx,
                "end": "from",
                "geometry": Point(coords[0]),
                "VOLTAGE": row["VOLTAGE"],
            }
        )
        records.append(
            {
                "line_idx": idx,
                "end": "to",
                "geometry": Point(coords[-1]),
                "VOLTAGE": row["VOLTAGE"],
            }
        )
    return gpd.GeoDataFrame(records, crs=CRS_PROJ)


def match_endpoints_to_subs(ep_gdf, tva_subs_filt):
    ep_join = gpd.sjoin_nearest(
        ep_gdf,
        tva_subs_filt[["osmid", "geometry"]],
        how="left",
        max_distance=SNAP_DISTANCE,
        distance_col="dist_to_sub",
    )
    ep_join["node_id"] = ep_join["osmid"].where(
        ep_join["osmid"].notna(),
        other=ep_join["line_idx"].astype(str) + "_" + ep_join["end"],
    )
    logger.info(
        f"Matched: {ep_join['osmid'].notna().sum()}  "
        f"Synthetic: {ep_join['osmid'].isna().sum()}"
    )
    return ep_join


def match_monitored_devices(tva_subs_filt, devices_gdf, ep_join):
    devices_proj = devices_gdf.to_crs(CRS_PROJ).copy()
    devices_buf = devices_proj.copy()
    devices_buf["geometry"] = devices_buf.geometry.buffer(5000)

    matched_devices = gpd.sjoin(
        tva_subs_filt,
        devices_buf[["device", "type", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    monitored_osmids = set(matched_devices["osmid"].astype(float).unique())
    already_matched = set(
        ep_join[ep_join["osmid"].notna()]["osmid"].astype(float).unique()
    )
    missing_naturally = monitored_osmids - already_matched

    logger.info(
        f"Monitored devices: {len(monitored_osmids)}  "
        f"In HIFLD coverage: {len(monitored_osmids - missing_naturally)}  "
        f"Missing: {len(missing_naturally)}"
    )
    for osmid in missing_naturally:
        device = matched_devices[matched_devices["osmid"] == osmid]["device"].values[0]
        logger.warning(f"  Not in HIFLD: {osmid} — {device}")

    return matched_devices, monitored_osmids


def build_graph(tva_lines, ep_join):
    G = nx.Graph()
    for idx, row in tva_lines.iterrows():
        ep_line = ep_join[ep_join["line_idx"] == idx]
        if len(ep_line) != 2:
            continue
        from_node = ep_line[ep_line["end"] == "from"]["node_id"].values[0]
        to_node = ep_line[ep_line["end"] == "to"]["node_id"].values[0]
        if from_node == to_node:
            continue
        G.add_edge(
            from_node,
            to_node,
            line_idx=idx,
            voltage=row["VOLTAGE"],
            length_m=row.geometry.length,
        )
    logger.info(
        f"Initial graph — nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}"
    )
    return G


def prune_graph(G, min_voltage_keep=230, max_iterations=20):
    G = G.copy()
    for i in range(max_iterations):
        dangles = [
            n
            for n, d in G.degree()
            if d == 1
            and isinstance(n, str)
            and "_" in str(n)
            and G.get_edge_data(n, list(G.neighbors(n))[0]).get("voltage", 0)
            < min_voltage_keep
        ]
        if not dangles:
            break
        G.remove_nodes_from(dangles)
    return G


def clean_graph(G, monitored_osmids):
    G_pruned = prune_graph(G, min_voltage_keep=KEEP_SYNTHETIC_ABOVE)

    # Restore stranded monitored nodes
    stranded = [
        n for n in monitored_osmids if n in G_pruned.nodes and G_pruned.degree(n) == 0
    ]
    for n in stranded:
        for neighbor in G.neighbors(n):
            if isinstance(neighbor, str) and "_" in str(neighbor):
                G_pruned.add_node(neighbor)
                G_pruned.add_edge(n, neighbor, **G.get_edge_data(n, neighbor))

    singletons = [
        n for n, d in G_pruned.degree() if d == 0 and float(n) not in monitored_osmids
    ]
    G_pruned.remove_nodes_from(singletons)

    components = list(nx.connected_components(G_pruned))
    logger.info(
        f"Cleaned graph — nodes: {G_pruned.number_of_nodes()}  "
        f"edges: {G_pruned.number_of_edges()}  "
        f"components: {len(components)}  "
        f"largest: {max(len(c) for c in components)}"
    )

    G_backbone = G_pruned.subgraph(
        max(nx.connected_components(G_pruned), key=len)
    ).copy()

    in_backbone = monitored_osmids & set(G_backbone.nodes)
    logger.info(
        f"Monitored devices in backbone: {len(in_backbone)} / {len(monitored_osmids)}"
    )

    return G_backbone


def _snap_voltage(v):
    classes = sorted(LINE_RESISTANCE.keys())
    return min(classes, key=lambda c: abs(c - v)) if not pd.isna(v) else 230


def build_substation_buses(
    G_backbone, tva_subs_filt, ep_gdf, tva_lines, grid_mapping_df, rng_seed=42
):
    rng = np.random.default_rng(rng_seed)

    gm = grid_mapping_df.copy()
    gm["SS_ID"] = gm["SS_ID"].astype(float)
    trafo_rows = gm[gm["Marker Label"] == "Transformer"].copy()
    trafo_rows["Role"] = trafo_rows["Attributes"].apply(
        lambda x: x["role"] if isinstance(x, dict) else "transmission"
    )
    osm_counts = trafo_rows.groupby("SS_ID").size().to_dict()
    osm_role = (
        trafo_rows.groupby("SS_ID")["Role"]
        .agg(
            lambda s: (
                "generation" if (s == "generation").mean() > 0.5 else "transmission"
            )
        )
        .to_dict()
    )

    sub_geo = tva_subs_filt.to_crs(CRS_GEO).set_index("osmid")["geometry"]
    ep_geo_ll = ep_gdf.to_crs(CRS_GEO)

    edge_lookup = {}
    for u, v, d in G_backbone.edges(data=True):
        edge_lookup.setdefault(u, []).append((v, d))
        edge_lookup.setdefault(v, []).append((u, d))

    substation_buses = {}
    substations_rows = []

    for node in G_backbone.nodes:
        is_synthetic = isinstance(node, str) and "_" in str(node)

        if is_synthetic:
            ep_match = ep_geo_ll[
                (ep_geo_ll["line_idx"].astype(str) + "_" + ep_geo_ll["end"]) == node
            ]
            lat, lon = (
                (ep_match.iloc[0].geometry.y, ep_match.iloc[0].geometry.x)
                if len(ep_match)
                else (35.5, -86.5)
            )
        else:
            try:
                pt = sub_geo.loc[float(node)]
                lat, lon = pt.y, pt.x
            except KeyError:
                lat, lon = 35.5, -86.5

        voltages = [d.get("voltage", np.nan) for _, d in edge_lookup.get(node, [])]
        voltages = [v for v in voltages if not pd.isna(v)]
        max_kv = max(voltages) if voltages else 230.0

        hv_bus = str(node)
        lv_bus = hv_bus + "_lv"
        osmid_f = float(node) if not is_synthetic else None

        if is_synthetic:
            trafo_count = 1
            trafo_types = ["GSU"]
            rg = RG_SYNTHETIC
        elif osmid_f in osm_counts:
            trafo_count = int(osm_counts[osmid_f])
            pool = POOL_GEN if osm_role.get(osmid_f) == "generation" else POOL_TRANS
            trafo_types = list(rng.choice(pool, size=trafo_count, replace=True))
            rg = RG_REAL
        else:
            trafo_count = int(rng.integers(1, 4))
            trafo_types = list(rng.choice(POOL_TRANS, size=trafo_count, replace=True))
            rg = RG_REAL

        substation_buses[hv_bus] = {
            "SS_ID": hv_bus,
            "hv_bus": hv_bus,
            "lv_bus": lv_bus,
            "buses": [hv_bus, lv_bus],
            "HV_voltage": max_kv,
            "LV_voltage": 0,
            "Transformer_type": trafo_types[0],
            "trafo_types": trafo_types,
            "trafo_count": trafo_count,
            "is_synthetic": is_synthetic,
            "external_bus_to_hv_bus": [],
            "external_bus_to_lv_bus": [],
        }
        substations_rows.append(
            {
                "name": hv_bus,
                "latitude": lat,
                "longitude": lon,
                "grounding_resistance": rg,
                "buses": [hv_bus, lv_bus],
                "kv_max": max_kv,
            }
        )

    substations_df = pd.DataFrame(substations_rows)

    line_rows = []
    seen = set()
    for u, v, d in G_backbone.edges(data=True):
        key = tuple(sorted([str(u), str(v)]))
        if key in seen:
            continue
        seen.add(key)
        length_m = d.get("length_m", 0.0)
        voltage = d.get("voltage", 230.0)
        kv_cls = _snap_voltage(voltage if not pd.isna(voltage) else 230.0)
        R = max((length_m / 1000.0) * LINE_RESISTANCE.get(kv_cls, 0.05), 1e-3)
        line_rows.append(
            {
                "name": f"{u}_{v}",
                "from_bus": str(u),
                "to_bus": str(v),
                "V": voltage,
                "length": length_m / 1000.0,
                "R": R,
            }
        )
    df_lines = pd.DataFrame(line_rows)

    all_buses = sorted({b for info in substation_buses.values() for b in info["buses"]})
    bus_ids_map = {b: i + 1 for i, b in enumerate(all_buses)}

    sub_look_up = {}
    idx = 0
    for info in substation_buses.values():
        for bus in sorted(info["buses"]):
            sub_look_up[bus] = idx
            idx += 1
    for hv_bus in substation_buses:
        sub_look_up[hv_bus + "_neutral"] = idx
        idx += 1

    logger.info(
        f"Substations: {len(substation_buses)}  Lines: {len(df_lines)}  "
        f"Matrix: {idx}x{idx}"
    )

    return substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df


def main():
    logger.info("Building TVA grid")

    tl_hifld, sub_gdf, tva_boundary, devices_gdf = load_raw_data()
    tva_lines, tva_subs_filt = clip_to_tva(tl_hifld, sub_gdf, tva_boundary)
    ep_gdf = build_endpoints(tva_lines)
    ep_join = match_endpoints_to_subs(ep_gdf, tva_subs_filt)
    matched_devices, monitored_osmids = match_monitored_devices(
        tva_subs_filt, devices_gdf, ep_join
    )
    G = build_graph(tva_lines, ep_join)
    G_backbone = clean_graph(G, monitored_osmids)

    grid_mapping_df = pd.read_csv(GRID_MAPPING)
    grid_mapping_df["Attributes"] = grid_mapping_df["Attributes"].apply(eval)

    substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df = (
        build_substation_buses(
            G_backbone, tva_subs_filt, ep_gdf, tva_lines, grid_mapping_df
        )
    )

    # Save
    with open(TVA_DIR / "G_backbone.pkl", "wb") as f:
        pickle.dump(G_backbone, f)
    with open(TVA_DIR / "substation_buses.pkl", "wb") as f:
        pickle.dump(substation_buses, f)
    with open(TVA_DIR / "sub_look_up.pkl", "wb") as f:
        pickle.dump(sub_look_up, f)
    with open(TVA_DIR / "bus_ids_map.pkl", "wb") as f:
        pickle.dump(bus_ids_map, f)
    df_lines.to_parquet(TVA_DIR / "df_lines.parquet")
    substations_df.to_parquet(TVA_DIR / "substations_df.parquet")
    matched_devices.to_parquet(TVA_DIR / "matched_devices.parquet")
    tva_lines.to_crs(CRS_GEO).to_parquet(TVA_DIR / "tva_lines_hifld.parquet")

    logger.info("TVA grid saved to data/tva/")


if __name__ == "__main__":
    main()
