"""
Compute Gannon line voltages for UIUC150 using bezpy Delaunay interpolation.
Author: Dennies
"""

import pickle
import numpy as np
import pandas as pd
import xarray as xr
import bezpy
from shapely.geometry import LineString
from tqdm import tqdm

from rep_mapping.rep_config import UIUC_DIR, GANNON_DS, setup_logger

logger = setup_logger(log_file="logs/compute_voltages_uiuc.log")


def load_gannon():
    logger.info("Loading Gannon E-field")
    gannon_ds = xr.open_dataset(GANNON_DS)
    E_pred = gannon_ds.E_pred.values
    site_xys = list(zip(gannon_ds.site_x.values, gannon_ds.site_y.values))
    start = np.datetime64("2024-05-09T14:00:00")
    mask = gannon_ds.time.values >= start
    time_axis = gannon_ds.time.values[mask]
    E_pred = E_pred[mask]
    n_times = len(time_axis)
    logger.info(f"E_pred shape: {E_pred.shape}  MT sites: {len(site_xys)}")
    return E_pred, site_xys, time_axis, n_times


def build_tl_objects(df_lines, bus_coords):
    tl_objects = []
    tl_line_ids = []

    for _, row in tqdm(
        df_lines.iterrows(), total=len(df_lines), desc="Building TL objects"
    ):
        fb = (
            row["from_bus"][1]
            if isinstance(row["from_bus"], tuple)
            else row["from_bus"]
        )
        tb = row["to_bus"][1] if isinstance(row["to_bus"], tuple) else row["to_bus"]

        c1 = bus_coords.get(int(fb))
        c2 = bus_coords.get(int(tb))
        if c1 is None or c2 is None:
            continue

        geom = LineString([c1, c2])
        line_row = type("Row", (), {"geometry": geom})()

        try:
            tl_obj = bezpy.tl.TransmissionLine(line_row)
            tl_objects.append(tl_obj)
            tl_line_ids.append(row["name"])
        except Exception as e:
            logger.warning(f"Failed {row['name']}: {e}")

    logger.info(f"TransmissionLine objects: {len(tl_objects)}")
    return tl_objects, tl_line_ids


def main():
    logger.info("Loading UIUC150 grid")

    with open(UIUC_DIR / "df_lines.pkl", "rb") as f:
        df_lines = pickle.load(f)

    with open(UIUC_DIR / "bus_coords.pkl", "rb") as f:
        bus_coords = pickle.load(f)

    E_pred, site_xys, time_axis, n_times = load_gannon()

    tl_objects, tl_line_ids = build_tl_objects(df_lines, bus_coords)

    for tl in tqdm(tl_objects, desc="Setting Delaunay weights"):
        tl.set_delaunay_weights(site_xys)

    arr_v = np.zeros((n_times, len(tl_objects)))
    for i, tl in enumerate(tqdm(tl_objects, desc="Calculating voltages")):
        arr_v[:, i] = tl.calc_voltages(E_pred, how="delaunay")
    arr_v = np.nan_to_num(arr_v, nan=0.0)

    logger.info(f"Voltage array shape: {arr_v.shape}")
    logger.info(f"Max voltage:         {np.abs(arr_v).max():.2f} V")
    logger.info(
        f"Lines nonzero:       {np.any(arr_v != 0, axis=0).sum()} / {len(tl_objects)}"
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

    matched = (v_arr.any(axis=1)).sum()
    logger.info(f"Lines with Gannon voltage: {matched} / {len(df_lines)}")

    np.save(UIUC_DIR / "arr_v_uiuc.npy", v_arr)
    np.save(UIUC_DIR / "time_axis_gannon.npy", time_axis)

    with open(UIUC_DIR / "df_lines_with_voltages.pkl", "wb") as f:
        pickle.dump(df_lines, f)

    logger.info(f"Saved to {UIUC_DIR}")


if __name__ == "__main__":
    main()
