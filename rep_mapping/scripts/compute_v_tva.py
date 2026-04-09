"""
Compute line voltages from Gannon E-field using Delaunay interpolation.
Author: Dennies
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import bezpy
from tqdm import tqdm

from rep_mapping.rep_config import TVA_DIR, HIFLD_PATH, GANNON_DS, CRS_GEO, setup_logger

logger = setup_logger(log_file="logs/compute_voltages.log")


def load_gannon():
    logger.info("Loading Gannon E-field")
    gannon_ds = xr.open_dataset(GANNON_DS)
    E_pred = gannon_ds.E_pred.values
    site_xys = list(zip(gannon_ds.site_x.values, gannon_ds.site_y.values))
    start = np.datetime64("2024-05-09T14:00:00")
    time_axis = gannon_ds.time.values[gannon_ds.time.values >= start]
    n_times = len(time_axis)
    E_pred = E_pred[gannon_ds.time.values >= start]
    logger.info(f"E_pred shape: {E_pred.shape}  MT sites: {len(site_xys)}")
    return E_pred, site_xys, time_axis, n_times


def build_tl_objects(G_backbone, hifld_path):
    logger.info("Loading HIFLD geometries")
    tl_hifld = gpd.read_file(hifld_path).to_crs(CRS_GEO)
    hifld_id_map = dict(zip(tl_hifld.index, tl_hifld["ID"].astype(str)))
    hifld_id_to_geom = dict(zip(tl_hifld["ID"].astype(str), tl_hifld.geometry))

    tl_objects = []
    tl_line_ids = []
    seen = set()

    for u, v, d in tqdm(list(G_backbone.edges(data=True)), desc="Building TL objects"):
        key = tuple(sorted([str(u), str(v)]))
        if key in seen:
            continue
        seen.add(key)
        lid = d.get("line_idx")
        hifld_id = hifld_id_map.get(lid)
        if hifld_id is None:
            continue
        geom = hifld_id_to_geom.get(str(hifld_id))
        if geom is None or geom.is_empty:
            continue
        line_row = type("Row", (), {"geometry": geom})()
        try:
            tl_obj = bezpy.tl.TransmissionLine(line_row)
            tl_objects.append(tl_obj)
            tl_line_ids.append(f"{u}_{v}")
        except Exception as e:
            logger.warning(f"Failed {hifld_id}: {e}")

    logger.info(f"TransmissionLine objects: {len(tl_objects)}")
    return tl_objects, tl_line_ids


def compute_voltages(tl_objects, tl_line_ids, E_pred, site_xys, n_times, df_lines):
    for tl in tqdm(tl_objects, desc="Setting Delaunay weights"):
        tl.set_delaunay_weights(site_xys)

    arr_v = np.zeros((n_times, len(tl_objects)))
    for i, tl in enumerate(tqdm(tl_objects, desc="Calculating voltages")):
        arr_v[:, i] = tl.calc_voltages(E_pred, how="delaunay")
    arr_v = np.nan_to_num(arr_v, nan=0.0)

    logger.info(
        f"Voltage array shape: {arr_v.shape}  "
        f"Max: {np.abs(arr_v).max():.2f} V  "
        f"Nonzero lines: {np.any(arr_v != 0, axis=0).sum()}"
    )

    v_gannon_cols = [f"V_gannon_{i+1}" for i in range(n_times)]
    name_to_pos = {row["name"]: pos for pos, (_, row) in enumerate(df_lines.iterrows())}

    v_arr = np.zeros((len(df_lines), n_times))
    for col_i, name in enumerate(tqdm(tl_line_ids, desc="Mapping voltages")):
        pos = name_to_pos.get(name)
        if pos is not None:
            v_arr[pos, :] = arr_v[:, col_i]

    v_df = pd.DataFrame(v_arr, columns=v_gannon_cols, index=df_lines.index)
    df_lines = pd.concat([df_lines.copy(), v_df], axis=1)

    logger.info(
        f"Lines with Gannon voltage: {(v_arr.any(axis=1)).sum()} / {len(df_lines)}"
    )
    return df_lines, v_gannon_cols, v_arr


def main():
    with open(TVA_DIR / "G_backbone.pkl", "rb") as f:
        G_backbone = pickle.load(f)

    df_lines = pd.read_parquet(TVA_DIR / "df_lines.parquet")

    E_pred, site_xys, time_axis, n_times = load_gannon()
    tl_objects, tl_line_ids = build_tl_objects(G_backbone, HIFLD_PATH)
    df_lines, v_gannon_cols, v_arr = compute_voltages(
        tl_objects, tl_line_ids, E_pred, site_xys, n_times, df_lines
    )

    np.save(TVA_DIR / "arr_v_tva.npy", v_arr)
    np.save(TVA_DIR / "time_axis.npy", time_axis)
    df_lines.to_parquet(TVA_DIR / "df_lines_with_voltages.parquet")

    logger.info("Saved voltages to data/tva/")


if __name__ == "__main__":
    main()
