"""
Calculate geomagnetically induced currents (GICs) along transmission lines and transformers.
Requires output from build_admittance_matrix.py.
Authors: Dennies and Ed
"""

# %%
import os
import gc
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import torch
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
import geopandas as gpd

from scipy.sparse.linalg import spsolve
from shapely.geometry import LineString
from scipy.ndimage import gaussian_filter1d

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as dense_solve, lstsq as dense_lstsq
from scipy.sparse.linalg import splu, lsqr

from build_admittance_matrix import (
    process_substation_buses,
    random_admittance_matrix,
    get_transformer_samples,
    randomize_grounding_resistance,
    apply_random_line_dc_blocking,
)

from configs import setup_logger, get_data_dir, GROUND_GIC_DIR

logger = setup_logger(log_file="logs/gic_calculation.log")

DATA_LOC = get_data_dir()
processed_gic_path = DATA_LOC / "gic"
processed_gnd_gic_path = DATA_LOC / "gnd_gic"
os.makedirs(processed_gic_path, exist_ok=True)
os.makedirs(processed_gnd_gic_path, exist_ok=True)

# Return periods for GIC calculations
return_periods = np.arange(50, 251, 25)

seed_base = 42
P_LINE_BLOCK = 0.01  # Probability of line blocking device


def find_substation_name(bus, sub_ref):
    """Find the substation name for a given bus from reference dictionary."""
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    return None


def load_and_process_gic_data(DATA_LOC, df_lines, results_path):
    """Load and process geomagnetically induced current (GIC) data from HDF5 files."""

    logger.info("Loading and processing GIC data...")

    with h5py.File(DATA_LOC / results_path, "r") as f:

        logger.info("Reading geomagnetic data from geomagnetic_data.h5")
        mt_names = f["sites/mt_sites/names"][:]
        mt_coords = f["sites/mt_sites/coordinates"][:]

        line_ids = f["sites/transmission_lines/line_ids"][:]
        line_ids_str = [
            id.decode("utf-8") if isinstance(id, bytes) else str(id) for id in line_ids
        ]

        halloween_e = f["events/halloween/E"][:] / 1000
        halloween_b = f["events/halloween/B"][:]
        halloween_v = f["events/halloween/V"][:]

        st_patricks_e = f["events/st_patricks/E"][:] / 1000
        st_patricks_b = f["events/st_patricks/B"][:]
        st_patricks_v = f["events/st_patricks/V"][:]

        gannon_e = f["events/gannon/E"][:] / 1000
        gannon_b = f["events/gannon/B"][:]
        gannon_v = f["events/gannon/V"][:]

        e_fields, b_fields, v_fields = {}, {}, {}

        for period in return_periods:
            e_fields[period] = f[f"predictions/E/{period}_year"][:]
            b_fields[period] = f[f"predictions/B/{period}_year"][:]
            v_fields[period] = f[f"predictions/V/{period}_year"][:]

    v_cols = ["V_halloween", "V_st_patricks", "V_gannon"] + [
        f"V_{period}" for period in return_periods
    ]

    id_to_index = {id: i for i, id in enumerate(line_ids_str)}
    indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])
    mask = indices != -1

    logger.info(f"Mask coverage: {mask.sum()}/{len(mask)}")

    df_lines.loc[mask, "V_halloween"] = halloween_v[indices[mask]]
    df_lines.loc[mask, "V_st_patricks"] = st_patricks_v[indices[mask]]
    df_lines.loc[mask, "V_gannon"] = gannon_v[indices[mask]]

    for period in return_periods:
        df_lines.loc[mask, f"V_{period}"] = v_fields[period][indices[mask]]

    df_lines[v_cols] = df_lines[v_cols].fillna(0)

    tmpl_dir = DATA_LOC / "storm_maxes"

    # Read the 1989 delaunay array
    arr_v_march89 = np.load(tmpl_dir / "delaunay_v_march89.npy")

    # Read the gannon array data
    arr_v_gannon = np.load(tmpl_dir / "delaunay_v_gannon.npy")

    # Append signage
    sgn_gannon = v_direction(arr_v_gannon)  # (L,)
    sgn_march89 = v_direction(arr_v_march89)  # (L,)

    # Map to df_lines row order using existing index mapping
    sgn_gannon_df = np.full(df_lines.shape[0], np.nan, dtype=float)
    sgn_march89_df = np.full(df_lines.shape[0], np.nan, dtype=float)
    sgn_gannon_df[mask] = sgn_gannon[indices[mask]]
    sgn_march89_df[mask] = sgn_march89[indices[mask]]

    df_lines["sgn_gannon"] = sgn_gannon_df
    df_lines["sgn_march89"] = sgn_march89_df

    sgn = df_lines["sgn_gannon"].to_numpy()
    if "sgn_march89" in df_lines.columns:
        sgn = np.where(
            np.isfinite(sgn) & (sgn != 0), sgn, df_lines["sgn_march89"].to_numpy()
        )
    sgn = np.where(np.isfinite(sgn) & (sgn != 0), sgn, 1.0)

    for period in return_periods:
        col = f"V_{period}"
        if col in df_lines.columns:
            df_lines[col] = df_lines[col].to_numpy() * sgn

    logger.info("GIC data loaded and processed successfully.")

    return (
        df_lines,
        mt_coords,
        mt_names,
        e_fields,
        b_fields,
        v_fields,
        gannon_e,
        v_cols,
    )


def v_direction(V):
    # Sign of the max voltage
    absV = np.abs(V)
    p95 = np.nanpercentile(absV, 95, axis=0)
    top = absV >= p95[None, :]
    sgn = np.sign(np.nanmedian(np.where(top, V, np.nan), axis=0))
    bad = (~np.isfinite(sgn)) | (sgn == 0)
    if np.any(bad):
        idx = np.nanargmax(absV[:, bad], axis=0)
        s_alt = np.sign(V[idx, np.arange(bad.sum())])
        sgn[bad] = np.where(s_alt == 0, 1.0, s_alt)
    return sgn.astype(float)


def calculate_GIC(df, V_nodal, col, non_zero_indices, n_nodes):
    """Calculate the Ground Induced Current (GIC) for transmission lines."""
    logger.info(f"Calculating GIC for {col}...")

    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal

    bus_n = df["from_bus"].values
    bus_k = df["to_bus"].values

    y_nk = 1 / df["R"].values
    j_nk = (df[col].values) * y_nk

    vn = V_all[bus_n]
    vk = V_all[bus_k]

    i_nk = np.round(j_nk + (vn - vk) * y_nk, 2)

    df[f"{col.split('_')[1]}_i_nk"] = i_nk

    logger.info(f"GIC calculation for {col} completed.")

    return df


def calc_trafo_gic(
    sub_look_up, df_transformers, V_nodal, sub_ref, n_nodes, non_zero_indices, title=""
):
    """Calculate transformer winding GIC."""
    gic = {}
    if len(df_transformers) == 0:
        return gic

    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal

    valid_buses = df_transformers["bus1"].isin(sub_look_up.keys())
    if sub_ref is not None:
        valid_subs = df_transformers["bus1"].apply(
            lambda b: find_substation_name(b, sub_ref) != "Substation 7"
        )
        valid = valid_buses & valid_subs
    else:
        valid = valid_buses

    if not valid.any():
        return gic

    df_valid = df_transformers[valid].copy()

    bus1_idx = df_valid["bus1"].map(sub_look_up).to_numpy()
    bus2_idx = df_valid["bus2"].map(sub_look_up).to_numpy()
    neutral_idx = df_valid["sub"].map(sub_look_up).to_numpy()

    valid_indices = ~(pd.isna(bus1_idx) | pd.isna(bus2_idx) | pd.isna(neutral_idx))
    if not valid_indices.any():
        return gic

    bus1_idx = bus1_idx[valid_indices].astype(int)
    bus2_idx = bus2_idx[valid_indices].astype(int)
    neutral_idx = neutral_idx[valid_indices].astype(int)
    df_valid = df_valid[valid_indices]

    V_bus1, V_bus2, V_neu = V_all[bus1_idx], V_all[bus2_idx], V_all[neutral_idx]

    W1_vals = df_valid["W1"].to_numpy()
    W2_vals = df_valid["W2"].to_numpy()
    Y_W1 = np.divide(1.0, W1_vals, out=np.zeros_like(W1_vals), where=W1_vals != 0)
    Y_W2 = np.divide(1.0, W2_vals, out=np.zeros_like(W2_vals), where=W2_vals != 0)

    t_type = df_valid["type"].to_numpy()
    t_names = df_valid["name"].to_numpy()

    mask_gsu = np.isin(t_type, ["GSU", "GSU w/ GIC BD", "GY-D"])
    mask_auto = t_type == "Auto"
    mask_gy = np.isin(t_type, ["GY-GY-D", "GY-GY"])

    if mask_gsu.any():
        gsu_indices = np.where(mask_gsu)[0]
        hv_vals = (V_bus1[gsu_indices] - V_neu[gsu_indices]) * Y_W1[gsu_indices]
        gic.update(
            {t_names[i]: {"HV": hv_vals[idx]} for idx, i in enumerate(gsu_indices)}
        )

    if mask_auto.any():
        auto_indices = np.where(mask_auto)[0]
        ser_vals = (V_bus1[auto_indices] - V_bus2[auto_indices]) * Y_W1[auto_indices]
        com_vals = (V_bus2[auto_indices] - V_neu[auto_indices]) * Y_W2[auto_indices]
        gic.update(
            {
                t_names[i]: {"Series": ser_vals[idx], "Common": com_vals[idx]}
                for idx, i in enumerate(auto_indices)
            }
        )

    if mask_gy.any():
        gy_indices = np.where(mask_gy)[0]
        hv_vals = (V_bus1[gy_indices] - V_neu[gy_indices]) * Y_W1[gy_indices]
        lv_vals = (V_bus2[gy_indices] - V_neu[gy_indices]) * Y_W2[gy_indices]
        gic.update(
            {
                t_names[i]: {"HV": hv_vals[idx], "LV": lv_vals[idx]}
                for idx, i in enumerate(gy_indices)
            }
        )

    logger.info(f"GIC calculation for transformers {title} completed.")
    return gic


def get_injection_currents(
    df_lines,
    n_nodes,
    non_zero_indices,
    sub_look_up,
    DATA_LOC,
    gannon_storm_only=False,
    gic_test_case=False,
):
    """Calculate injection currents for power system nodes using vectorized operations."""
    logger.info("Calculating injection currents (fully vectorized)...")

    if gannon_storm_only:
        cols = [col for col in df_lines.columns if "V_gannon" in col]
    elif gic_test_case:
        cols = [
            col
            for col in df_lines.columns
            if col.startswith("V_") and col in ["V_northward", "V_eastward"]
        ]
    else:
        cols = ["V_halloween", "V_st_patricks", "V_gannon"] + [
            f"V_{period}" for period in return_periods
        ]

    R = df_lines["R"].to_numpy()
    from_idx = df_lines["from_bus"].map(sub_look_up).to_numpy()
    to_idx = df_lines["to_bus"].map(sub_look_up).to_numpy()

    I_eff_mat = df_lines[cols].to_numpy() / R[:, None]
    I_eff_mat = np.nan_to_num(I_eff_mat, nan=0.0)

    n_cols = len(cols)
    injection_currents_all = np.zeros((n_nodes, n_cols), dtype=I_eff_mat.dtype)

    for j in range(n_cols):
        pos = np.bincount(to_idx, weights=I_eff_mat[:, j], minlength=n_nodes)
        neg = np.bincount(from_idx, weights=I_eff_mat[:, j], minlength=n_nodes)
        injection_currents_all[:, j] = pos - neg

    logger.info(
        f"Injection matrix: shape={injection_currents_all.shape}, "
        f"finite={np.isfinite(injection_currents_all).all()}, "
        f"nonzero_rows={np.any(injection_currents_all != 0, axis=1).sum()}"
    )

    injections_data = {
        col: injection_currents_all[non_zero_indices, j] for j, col in enumerate(cols)
    }

    return injections_data


def can_use_float16(arr):
    """Determine if an array can be safely represented in float16 or float32."""
    try:
        if hasattr(arr, "get"):
            arr = arr.get()
        max_abs = np.max(np.abs(arr))
        min_abs = np.min(np.abs(arr[arr != 0])) if np.any(arr != 0) else np.float32(0)
    except Exception:
        if isinstance(arr, torch.Tensor):
            tensor_arr = arr
        else:
            tensor_arr = torch.tensor(arr)
        max_abs = torch.max(torch.abs(tensor_arr)).item()
        min_abs = (
            torch.min(torch.abs(tensor_arr[tensor_arr != 0])).item()
            if torch.any(tensor_arr != 0)
            else 0.0
        )
    return (max_abs < 65504 and min_abs > 6.1e-5) or max_abs == 0


def nodal_voltage_calculation(Y_total, injections_data):
    """Calculate nodal voltages using PyTorch with robust fallbacks."""
    logger.info("Calculating nodal voltages using PyTorch...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        matrix_size_bytes = Y_total.nbytes
        total_required = matrix_size_bytes * 2.0
        free_memory = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        if total_required > free_memory:
            logger.info(
                f"Matrix too large for GPU ({total_required/1e9:.1f}GB needed, {free_memory/1e9:.1f}GB free); using CPU"
            )
            device = torch.device("cpu")
    else:
        gc.collect()

    logger.info(f"Using device: {device}")

    Y_tensor = torch.tensor(Y_total, dtype=torch.float32, device=device)

    # Symmetrize (admittance should be symmetric up to numerical noise)
    Y_tensor = 0.5 * (Y_tensor + Y_tensor.T)

    valid_keys = [k for k, v in injections_data.items() if v is not None]
    if not valid_keys:
        return {k: None for k in injections_data}

    injection_batch = torch.stack(
        [
            torch.tensor(injections_data[k], dtype=torch.float32, device=device)
            for k in valid_keys
        ],
        dim=-1,
    )

    # Sanitize inputs
    if not torch.isfinite(Y_tensor).all():
        logger.warning("Y_tensor contains non-finite values, cleaning...")
        Y_tensor = torch.nan_to_num(Y_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if not torch.isfinite(injection_batch).all():
        logger.warning("injection_batch contains non-finite values, cleaning...")
        injection_batch = torch.nan_to_num(
            injection_batch, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Always solve in float64 for stability
    Y64 = Y_tensor.to(torch.float64)
    B64 = injection_batch.to(torch.float64)

    I = torch.eye(Y64.shape[0], dtype=torch.float64, device=Y64.device)
    V_n = None

    # 1) Try Cholesky with small regularization
    for reg in (1e-6, 1e-5, 1e-4, 1e-3):
        try:
            Y_reg = Y64 + reg * I
            L = torch.linalg.cholesky(Y_reg)
            V_try = torch.cholesky_solve(B64, L)
            if torch.isfinite(V_try).all():
                V_n = V_try
                logger.info(f"Cholesky succeeded with reg={reg}")
                break
        except Exception as e:
            logger.debug(f"Cholesky failed (reg={reg}): {e}")

    # 2) If Cholesky failed, try direct solve (LU w/ pivoting)
    if V_n is None:
        for reg in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                Y_reg = Y64 + reg * I
                V_try = torch.linalg.solve(Y_reg, B64)
                if torch.isfinite(V_try).all():
                    V_n = V_try
                    logger.info(f"Direct solve succeeded with reg={reg}")
                    break
            except Exception as e:
                logger.debug(f"Direct solve failed (reg={reg}): {e}")

    # 3) Final fallback: damped least squares
    if V_n is None:
        logger.info("Using damped least squares fallback")
        lam = 1e-4
        V_n = torch.linalg.lstsq(Y64 + lam * I, B64).solution

    V_n = V_n.to(torch.float32)
    results = {k: V_n[:, i].detach().cpu().numpy() for i, k in enumerate(valid_keys)}

    # Final guard: replace any residual NaNs with zero
    for k in results:
        if not np.isfinite(results[k]).all():
            logger.warning(f"Found non-finite values in {k}, setting to zero")
            r = results[k]
            r[~np.isfinite(r)] = 0.0
            results[k] = r

    logger.info("Nodal voltage calculation completed.")
    return results


def find_storm_maxima(E_pred, gannon_times, window_hours=(5 / 60), num_peaks=500):
    """Find the maximum E-field magnitudes during a geomagnetic storm."""
    window_samples = int(window_hours * 60)
    site_maxE_mags = np.sqrt(np.sum(E_pred**2, axis=2))
    total_magnitude = np.nansum(site_maxE_mags, axis=1)

    smoothed_magnitude = gaussian_filter1d(total_magnitude, sigma=window_samples // 5)

    peaks, _ = signal.find_peaks(
        smoothed_magnitude,
        distance=window_samples // 5,
        prominence=0.1 * np.max(smoothed_magnitude),
    )

    if len(peaks) > num_peaks:
        peak_prominences = signal.peak_prominences(smoothed_magnitude, peaks)[0]
        top_peaks = peaks[np.argsort(peak_prominences)[-num_peaks:]]
    else:
        top_peaks = peaks

    return [
        {
            "peak_time": gannon_times[peak],
            "site_magnitudes": E_pred[peak],
            "total_magnitude": total_magnitude[peak],
            "smoothed_magnitude": smoothed_magnitude[peak],
        }
        for peak in np.sort(top_peaks)
    ]


def process_gannon(DATA_LOC, df_lines, results_path, peak_time=False, res=30):
    """Process Gannon storm data and extract voltage values at peak times."""
    logger.info("Loading and processing Gannon GIC data...")

    gannon_ds = xr.open_dataset(DATA_LOC / "storm_maxes" / "ds_gannon.nc")
    gannon_times = gannon_ds.time.values

    if peak_time:
        E_pred = gannon_ds.E_pred.values
        peak_data = find_storm_maxima(E_pred, gannon_times)
        peak_times = [data["peak_time"] for data in peak_data]

        np.save(DATA_LOC / "peak_times.npy", peak_times)

    else:
        start_date = np.datetime64("2024-05-10")

        mask = (gannon_times >= start_date) & (
            gannon_times.astype("datetime64[m]").view("int64") % res == 0
        )
        peak_times = gannon_times[mask]

        logger.info(f"Peak times loaded successfully... shape: {peak_times.shape}")

        np.save(DATA_LOC / "peak_times_1.npy", peak_times)

    peak_indices = np.searchsorted(gannon_times, peak_times)

    arr_v = np.load(DATA_LOC / "storm_maxes" / "delaunay_v_gannon.npy")

    arr_v_peaks = arr_v[peak_indices]

    with h5py.File(DATA_LOC / results_path, "r") as f:
        logger.info("Reading geomagnetic data from geomagnetic_data.h5")
        mt_names = f["sites/mt_sites/names"][:]
        mt_coords = f["sites/mt_sites/coordinates"][:]

        line_ids = f["sites/transmission_lines/line_ids"][:]
        line_ids_str = [
            id.decode("utf-8") if isinstance(id, bytes) else str(id) for id in line_ids
        ]

        halloween_e = f["events/halloween/E"][:] / 1000
        halloween_b = f["events/halloween/B"][:]
        halloween_v = f["events/halloween/V"][:]

        st_patricks_e = f["events/st_patricks/E"][:] / 1000
        st_patricks_b = f["events/st_patricks/B"][:]
        st_patricks_v = f["events/st_patricks/V"][:]

        gannon_e = f["events/gannon/E"][:] / 1000
        gannon_v = f["events/gannon/V"][:]

    v_cols = ["V_gannon_peak_max"]
    num_peaks = len(peak_indices)
    v_gannon_cols = [f"V_gannon_{i+1}" for i in range(num_peaks)]

    v_cols.extend(v_gannon_cols)

    id_to_index = {id: i for i, id in enumerate(line_ids_str)}
    indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])
    mask = indices != -1

    df_lines.loc[mask, "V_gannon_peak_max"] = gannon_v[indices[mask]]

    new_cols = {}
    for i, peak_idx in enumerate(peak_indices):
        col_name = f"V_gannon_{i+1}"
        col = np.full(df_lines.shape[0], np.nan, dtype=float)
        col[mask] = arr_v_peaks[i][indices[mask]]
        new_cols[col_name] = col

    df_lines = pd.concat(
        [df_lines, pd.DataFrame(new_cols, index=df_lines.index)], axis=1
    )
    df_lines[v_cols] = df_lines[v_cols].fillna(0)

    logger.info("Gannon Voltage values loaded and processed successfully.")

    return (df_lines, mt_coords, mt_names, gannon_e, v_cols)


def keep_nodes_with_ground(Y_n, Y_e):
    """Keep only nodes that are in components with at least one grounded node."""
    # Adjacency from network admittance (ignore self-diagonal)
    A = (Y_n != 0).astype(bool)
    np.fill_diagonal(A, False)

    # Component labels by BFS/DFS
    N = A.shape[0]
    comp = -np.ones(N, dtype=int)
    cid = 0
    for s in range(N):
        if comp[s] != -1:
            continue
        # BFS
        q = [s]
        comp[s] = cid
        for u in q:
            nbrs = np.flatnonzero(A[u])
            for v in nbrs:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        cid += 1

    grounded = np.diag(Y_e) > 0
    comp_has_ground = np.zeros(cid, dtype=bool)
    np.logical_or.at(comp_has_ground, comp[grounded], True)

    keep = comp_has_ground[comp]
    return np.flatnonzero(keep)


def solve_total_nodal_gic(Y_e_reduced, V_nodal, non_zero_indices, n_nodes):

    # Ensure 1D nodal vector for the reduced system
    v_red = np.asarray(V_nodal, dtype=float).ravel()  # (N_reduced,)
    y_diag = np.diag(Y_e_reduced).astype(float)  # (N_reduced,)

    # Per-phase ground current on reduced nodes (diagonal multiply), then total 3-phase
    i_e_red = y_diag * v_red
    i_e_red *= 3.0

    # Scatter back to full node indexing
    full = np.zeros(n_nodes, dtype=i_e_red.dtype)
    full[non_zero_indices] = i_e_red

    return full


def main(generate_grid_only=False, gannon_storm_only=False):
    """Compute GICs and optionally export storm-only results."""
    results_path = "statistical_analysis/geomagnetic_data_return_periods.h5"

    substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info = (
        process_substation_buses(DATA_LOC)
    )

    df_lines.drop(columns=["geometry"], inplace=True)
    df_lines["name"] = df_lines["name"].astype(str)

    transmission_line_path = DATA_LOC / "grid_processed" / "trans_lines_pickle.pkl"
    with open(transmission_line_path, "rb") as p:
        trans_lines_gdf = pickle.load(p)

    trans_lines_gdf["line_id"] = trans_lines_gdf["line_id"].astype(str)
    df_lines = df_lines.merge(
        trans_lines_gdf[["line_id", "geometry"]], right_on="line_id", left_on="name"
    )

    sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))
    trafos_data = get_transformer_samples(
        substation_buses, new_trafos=False
    )  # To use new transformers, set new_trafos=True

    if gannon_storm_only:
        df_lines, mt_coords, mt_names, gannon_e, v_cols = process_gannon(
            DATA_LOC, df_lines, results_path, res=1
        )
        logger.info("Gannon storm data processed successfully.")
    else:
        (
            df_lines,
            mt_coords,
            mt_names,
            e_fields,
            b_fields,
            v_fields,
            gannon_e,
            v_cols,
        ) = load_and_process_gic_data(DATA_LOC, df_lines, results_path)

    n_nodes = len(sub_look_up)

    logger.info("Starting GIC calculations...")
    for i, trafo_data in enumerate(trafos_data):

        df_ss_iter = randomize_grounding_resistance(
            df_substations_info, seed=seed_base + i
        )
        df_lines_iter = apply_random_line_dc_blocking(
            df_lines, p_block=P_LINE_BLOCK, seed=seed_base + i
        )

        if i < 0:
            continue

        out_path = (
            GROUND_GIC_DIR / f"ground_gic_gannon_{i}.csv"
            if gannon_storm_only
            else processed_gic_path / f"winding_gic_rand_{i}.csv"
        )
        if out_path.is_file():
            continue

        logger.info(f"Processing iteration {i}...")
        logger.info("Generating random admittance matrix...")

        Y_n, Y_e, df_transformers = random_admittance_matrix(
            substation_buses,
            trafo_data,
            bus_ids_map,
            sub_look_up,
            df_lines_iter,
            df_ss_iter,
        )

        if can_use_float16(Y_n) and can_use_float16(Y_e):
            dtype = np.float32
        elif (np.max(np.abs(Y_n)) < 3.4e38) and (np.max(np.abs(Y_e)) < 3.4e38):
            dtype = np.float32
        else:
            dtype = np.float64

        Y_n = Y_n.astype(dtype, copy=False)
        Y_e = Y_e.astype(dtype, copy=False)

        row_active = np.any(Y_n != 0, axis=1) | np.any(Y_e != 0, axis=1)
        nz = np.flatnonzero(row_active)
        keep = keep_nodes_with_ground(Y_n[np.ix_(nz, nz)], Y_e[np.ix_(nz, nz)])
        non_zero_indices = nz[keep]

        # Reduce
        Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
        Y_e = Y_e[np.ix_(non_zero_indices, non_zero_indices)]
        Y_total = Y_n + Y_e

        del Y_n
        gc.collect()

        injections_data = get_injection_currents(
            df_lines_iter,
            n_nodes,
            non_zero_indices,
            sub_look_up,
            DATA_LOC,
            gannon_storm_only=gannon_storm_only,
        )

        nodal_voltages = nodal_voltage_calculation(Y_total, injections_data)

        if gannon_storm_only:
            # Build ground GIC output for all voltage columns
            v_keys = [
                c
                for c in v_cols
                if c in nodal_voltages and nodal_voltages[c] is not None
            ]
            if not v_keys:
                logger.error("No valid voltage keys for ground GIC.")
                continue

            # Compute ground GIC for each key, using reduced Y_e
            ig_cols = []
            for k in v_keys:
                ig_full = solve_total_nodal_gic(
                    Y_e, nodal_voltages[k], non_zero_indices, n_nodes
                )
                ig_cols.append(ig_full)

            ig_all = np.vstack(ig_cols).T

            idx_series = df_ss_iter["name"].map(sub_look_up)
            mask = idx_series.notna()
            valid_substations = df_ss_iter.loc[mask, "name"].to_numpy()
            valid_indices = idx_series[mask].astype(int).to_numpy()

            gic_values_T = ig_all[valid_indices].T  # (K, n_valid_substations)

            data = {"Substation": valid_substations}
            for col, vals in zip(v_keys, gic_values_T):
                data[f"GIC_{col.split('V_')[1]}"] = vals

            pd.DataFrame(data).to_csv(out_path, index=False)
            
            del Y_e
            gc.collect()
            continue

        else:
            df_lines_copy = df_lines_iter.copy()
            df_lines_copy["from_bus"] = df_lines_copy["from_bus"].apply(
                lambda x: sub_look_up.get(x)
            )
            df_lines_copy["to_bus"] = df_lines_copy["to_bus"].apply(
                lambda x: sub_look_up.get(x)
            )

            def _calc_period(period):
                try:
                    if f"V_{period}" not in nodal_voltages:
                        logger.warning(f"Nodal voltages missing for {period}")
                        return period, None, None
                    df_lines_local = df_lines_copy.copy()
                    df_transformers_local = df_transformers.copy()
                    V_nodal = nodal_voltages[f"V_{period}"]
                    df_gic = calculate_GIC(
                        df_lines_local,
                        V_nodal,
                        f"V_{period}",
                        non_zero_indices,
                        n_nodes,
                    )
                    gic = calc_trafo_gic(
                        sub_look_up,
                        df_transformers_local,
                        V_nodal,
                        sub_ref,
                        n_nodes,
                        non_zero_indices,
                        f"{period}-year-hazard",
                    )
                    return period, df_gic, gic
                except Exception as e:
                    logger.error(f"Error calculating GIC for period {period}: {str(e)}")
                    return period, None, None

            gic_data = {}
            calculation_success = True

            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        pool.submit(_calc_period, p): p
                        for p in ["gannon", *return_periods]
                    }
                    for fut in as_completed(futures):
                        try:
                            period, df_gic_res, gic_res = fut.result()
                            if df_gic_res is not None and gic_res is not None:
                                gic_data[period] = gic_res
                            else:
                                logger.warning(
                                    f"Skipping period {period} due to calculation error"
                                )
                        except Exception as e:
                            failed_period = futures[fut]
                            logger.error(
                                f"Failed to get result for period {failed_period}: {str(e)}"
                            )
                            calculation_success = False

                if not gic_data or not calculation_success:
                    logger.error(f"Calculation failed for iteration {i}, skipping...")
                    continue

                expected_periods = ["gannon"] + list(return_periods)
                missing_periods = [p for p in expected_periods if p not in gic_data]
                if missing_periods:
                    logger.warning(f"Missing data for periods: {missing_periods}")
                    if len(missing_periods) > len(expected_periods) / 2:
                        logger.error(
                            f"Too many periods failed for iteration {i}, skipping..."
                        )
                        continue

                winding_gic_df_list = []
                for period, gic_values in gic_data.items():
                    try:
                        if not gic_values or not isinstance(gic_values, dict):
                            logger.warning(
                                f"Invalid GIC data for period {period}, skipping period"
                            )
                            continue
                        hash_gic_period = [
                            (trafo, winding, gic)
                            for trafo, windings in gic_values.items()
                            for winding, gic in windings.items()
                            if isinstance(windings, dict)
                        ]
                        if not hash_gic_period:
                            logger.warning(f"No valid GIC data for period {period}")
                            continue
                        winding_gic_df = pd.DataFrame(
                            hash_gic_period,
                            columns=[
                                "Transformer",
                                "Winding",
                                f"{period}-year-hazard A/ph",
                            ],
                        )
                        winding_gic_df_list.append(winding_gic_df)
                    except Exception as e:
                        logger.error(
                            f"Error processing GIC data for period {period}: {str(e)}"
                        )
                        continue

                if not winding_gic_df_list:
                    logger.error(
                        f"No valid GIC dataframes created for iteration {i}, skipping..."
                    )
                    continue

                try:
                    winding_gic_df = pd.concat(winding_gic_df_list, axis=1).loc[
                        :, ~pd.concat(winding_gic_df_list, axis=1).columns.duplicated()
                    ]
                    if winding_gic_df.empty:
                        logger.error(
                            f"Empty dataframe after merge for iteration {i}, skipping..."
                        )
                        continue

                    df_transformers["Transformer"] = df_transformers["name"]
                    winding_gic_df = winding_gic_df.merge(
                        df_transformers[
                            ["sub_id", "Transformer", "latitude", "longitude"]
                        ],
                        on="Transformer",
                        how="inner",
                    )
                    if winding_gic_df.empty:
                        logger.error(
                            f"Empty dataframe after transformer merge for iteration {i}, skipping..."
                        )
                        continue

                    winding_gic_df.to_csv(out_path, index=False)
                    logger.info(f"Successfully saved GIC data for iteration {i}")
                except Exception as e:
                    logger.error(
                        f"Error during dataframe processing for iteration {i}: {str(e)}"
                    )
                    continue
            except Exception as e:
                logger.error(f"Critical error in iteration {i}: {str(e)}")
                continue

            filename_gic = processed_gnd_gic_path / f"ground_gic{i}.csv"
            if os.path.exists(filename_gic):
                continue

            v_keys = [
                c
                for c in v_cols
                if (c in nodal_voltages) and (nodal_voltages[c] is not None)
            ]
            if not v_keys:
                logger.error("No valid voltage keys for ground GIC.")
                continue

            ig_cols = []
            for k in v_keys:
                ig_full = solve_total_nodal_gic(
                    Y_e, nodal_voltages[k], non_zero_indices, n_nodes
                )
                ig_cols.append(ig_full)

            ig_all = np.vstack(ig_cols).T  # (n_nodes, K)
            
            del Y_e
            gc.collect()

            idx_series = df_ss_iter["name"].map(sub_look_up)
            mask = idx_series.notna()
            valid_substations = df_ss_iter.loc[mask, "name"].to_numpy()
            valid_indices = idx_series[mask].astype(int).to_numpy()

            gic_values_T = ig_all[valid_indices].T  # (K, n_valid_substations)

            data = {"Substation": valid_substations}
            for col, vals in zip(v_keys, gic_values_T):
                data[f"GIC_{col.split('V_')[1]}"] = vals

            pd.DataFrame(data).to_csv(filename_gic, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate GICs.")
    parser.add_argument(
        "--gannon-only",
        action="store_true",
        help="Run Gannon-storm-only workflow.",
    )
    args = parser.parse_args()
    main(gannon_storm_only=args.gannon_only)

# %%
