# ...............................................................................
# Description: Levereages the admittance matrix to calculate the GICs along the
# transmission lines and transformers in the network.
# Dependency: Output from build_admittance_matrix.py
# ...............................................................................

# %%
import os
import gc
from pathlib import Path
from tqdm import tqdm
import pickle
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from shapely.ops import unary_union
from scipy.interpolate import griddata
import xarray as xr
import torch
from shapely.geometry import LineString
from shapely.geometry import Point
import geopandas as gpd
from scipy.ndimage import gaussian_filter1d
from scipy import signal

# Import custom functions
from build_admittance_matrix import process_substation_buses, random_admittance_matrix, get_transformer_samples
from configs import setup_logger, get_data_dir

# Set up logger
logger = setup_logger(log_file="logs/gic_calculation.log")

# Get data data log and configure logger
DATA_LOC = get_data_dir()
processed_gic_path = DATA_LOC / "gic"
processed_gnd_gic_path = DATA_LOC / "gnd_gic"
os.makedirs(processed_gic_path, exist_ok=True)
os.makedirs(processed_gnd_gic_path, exist_ok=True)

# Define return periods for GIC calculations
return_periods = np.arange(50, 251, 25)

def find_substation_name(bus, sub_ref):
    """
    Find the substation name for a given bus from the substation reference dictionary.
    """
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    # If not found, return None
    return None


def load_and_process_gic_data(DATA_LOC, df_lines, results_path):
    """
    Load and process geomagnetically induced current (GIC) data from HDF5 files.
    """

    logger.info("Loading and processing GIC data...")

    # Load the data from storm maxes
    with h5py.File(DATA_LOC / results_path, "r") as f:

        logger.info("Reading geomagnetic data from geomagnetic_data.h5")
        # Read MT site information
        mt_names = f["sites/mt_sites/names"][:]
        mt_coords = f["sites/mt_sites/coordinates"][:]

        # Read transmission line IDs
        line_ids = f["sites/transmission_lines/line_ids"][:]
        line_ids_str = [id.decode('utf-8') if isinstance(id, bytes) else str(id) for id in line_ids]

        # Read Halloween storm data
        halloween_e = f["events/halloween/E"][:] / 1000
        halloween_b = f["events/halloween/B"][:]
        halloween_v = f["events/halloween/V"][:]

        # Read st_patricks storm data
        st_patricks_e = f["events/st_patricks/E"][:] / 1000
        st_patricks_b = f["events/st_patricks/B"][:]
        st_patricks_v = f["events/st_patricks/V"][:]

        # Read the Gannon storm data
        gannon_e = f["events/gannon/E"][:] / 1000
        gannon_b = f["events/gannon/B"][:]
        gannon_v = f["events/gannon/V"][:]

        e_fields, b_fields, v_fields = {}, {}, {}

        # Load E, B, and V fields dynamically for each return period
        for period in return_periods:
            e_fields[period] = f[f"predictions/E/{period}_year"][:]
            b_fields[period] = f[f"predictions/B/{period}_year"][:]
            v_fields[period] = f[f"predictions/V/{period}_year"][:]

    # Voltage columns for all events
    v_cols = ["V_halloween", "V_st_patricks", "V_gannon"] + [
        f"V_{period}" for period in return_periods
    ]

    # Mapping IDs to indices
    id_to_index = {id: i for i, id in enumerate(line_ids_str)}
    indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])
    mask = indices != -1
    
    print("mask", mask)

    # Use boolean indexing to handle missing values
    df_lines.loc[mask, "V_halloween"] = halloween_v[indices[mask]]
    df_lines.loc[mask, "V_st_patricks"] = st_patricks_v[indices[mask]]
    df_lines.loc[mask, "V_gannon"] = gannon_v[indices[mask]]

    # Assign dynamic voltage columns
    for period in return_periods:
        df_lines.loc[mask, f"V_{period}"] = v_fields[period][indices[mask]]

    # Set a default value for all missing values
    df_lines[v_cols] = df_lines[v_cols].fillna(0)

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


def calculate_injection_currents_vectorized(
    df, n_nodes, col, non_zero_indices, sub_look_up
):
    """
    Calculation of injection currents for a power system network.
    """
    logger.info(f"Calculating injection currents for {col} (vectorized)...")
    injection_currents = np.zeros(n_nodes, dtype=np.float64)

    # Compute effective current
    I_eff = df[col] / df["R"]
    valid = ~I_eff.isna()

    # Map bus names to node indices for valid rows
    i_idx = df["from_bus"].loc[valid].map(sub_look_up).to_numpy()
    j_idx = df["to_bus"].loc[valid].map(sub_look_up).to_numpy()
    I_eff_valid = I_eff.loc[valid].to_numpy()

    # Subtract at 'from_bus' indices, add at 'to_bus' indices
    np.add.at(injection_currents, i_idx, -I_eff_valid)
    np.add.at(injection_currents, j_idx, I_eff_valid)

    return injection_currents[non_zero_indices]


def calculate_GIC(df, V_nodal, col, non_zero_indices, n_nodes):
    """
    Calculate the Ground Induced Current (GIC) for transmission lines.
    """
    logger.info(f"Calculating GIC for {col}...")

    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal

    bus_n = df["from_bus"].values
    bus_k = df["to_bus"].values

    y_nk = 1 / df["R"].values
    j_nk = (df[col].values) * y_nk

    # Get the nodal voltages
    vn = V_all[bus_n]
    vk = V_all[bus_k]

    # Solving for transmission lines GIC
    i_nk = np.round(j_nk + (vn - vk) * y_nk, 2)

    df[f"{col.split('_')[1]}_i_nk"] = i_nk

    logger.info(f"GIC calculation for {col} completed.")

    return df

def calc_trafo_gic(
    sub_look_up, df_transformers, V_nodal, sub_ref, n_nodes, non_zero_indices, title=""
):
    """
    Transformer winding gic calculations.
    """
    gic = {}
    if len(df_transformers) == 0:
        return gic
    
    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal
    
    # Pre-filter for valid buses and exclude Substation 7
    valid_buses = df_transformers["bus1"].isin(sub_look_up.keys())
    if sub_ref is not None:
        # Keep original approach - can't optimize without knowing sub_ref structure
        valid_subs = df_transformers["bus1"].apply(
            lambda b: find_substation_name(b, sub_ref) != "Substation 7"
        )
        valid = valid_buses & valid_subs
    else:
        valid = valid_buses
    
    if not valid.any():
        return gic
    
    # Filter dataframe once
    df_valid = df_transformers[valid].copy()
    
    # Pre-index with error handling
    bus1_idx = df_valid["bus1"].map(sub_look_up).to_numpy()
    bus2_idx = df_valid["bus2"].map(sub_look_up).to_numpy() 
    neutral_idx = df_valid["sub"].map(sub_look_up).to_numpy()
    
    # Handle missing indices -- use pd.isna for safety
    valid_indices = ~(pd.isna(bus1_idx) | pd.isna(bus2_idx) | pd.isna(neutral_idx))
    if not valid_indices.any():
        return gic
    
    bus1_idx = bus1_idx[valid_indices].astype(int)
    bus2_idx = bus2_idx[valid_indices].astype(int) 
    neutral_idx = neutral_idx[valid_indices].astype(int)
    df_valid = df_valid[valid_indices]
    
    # Vectorized voltages and admittances
    V_bus1, V_bus2, V_neu = V_all[bus1_idx], V_all[bus2_idx], V_all[neutral_idx]
    
    # Safe division with zero handling
    W1_vals = df_valid["W1"].to_numpy()
    W2_vals = df_valid["W2"].to_numpy()
    Y_W1 = np.divide(1.0, W1_vals, out=np.zeros_like(W1_vals), where=W1_vals!=0)
    Y_W2 = np.divide(1.0, W2_vals, out=np.zeros_like(W2_vals), where=W2_vals!=0)
    
    # Type masks
    t_type = df_valid["type"].to_numpy()
    t_names = df_valid["name"].to_numpy()
    
    mask_gsu = np.isin(t_type, ["GSU", "GSU w/ GIC BD", "GY-D"])
    mask_auto = (t_type == "Auto")
    mask_gy = np.isin(t_type, ["GY-GY-D", "GY-GY"])
    
    # Only compute what we need - slice first then compute
    if mask_gsu.any():
        gsu_indices = np.where(mask_gsu)[0]
        hv_vals = (V_bus1[gsu_indices] - V_neu[gsu_indices]) * Y_W1[gsu_indices]
        gic.update({t_names[i]: {"HV": hv_vals[idx]} for idx, i in enumerate(gsu_indices)})
    
    if mask_auto.any():
        auto_indices = np.where(mask_auto)[0]
        ser_vals = (V_bus1[auto_indices] - V_bus2[auto_indices]) * Y_W1[auto_indices]
        com_vals = (V_bus2[auto_indices] - V_neu[auto_indices]) * Y_W2[auto_indices]
        gic.update({t_names[i]: {"Series": ser_vals[idx], "Common": com_vals[idx]} 
                   for idx, i in enumerate(auto_indices)})
    
    if mask_gy.any():
        gy_indices = np.where(mask_gy)[0]
        hv_vals = (V_bus1[gy_indices] - V_neu[gy_indices]) * Y_W1[gy_indices]
        lv_vals = (V_bus2[gy_indices] - V_neu[gy_indices]) * Y_W2[gy_indices]
        gic.update({t_names[i]: {"HV": hv_vals[idx], "LV": lv_vals[idx]} 
                   for idx, i in enumerate(gy_indices)})
    
    logger.info(f"GIC calculation for transformers {title} completed.")
    return gic


def get_conus_polygon():
    """
    Retrieves the polygon representing the continental United States (CONUS).
    """
    logger.info("Retrieving CONUS polygon...")

    try:
        import cartopy.io.shapereader as shpreader
        shapename = "admin_1_states_provinces_lakes"
        us_states = shpreader.natural_earth(
            resolution="110m", category="cultural", name=shapename
        )

        conus_states = []
        for state in shpreader.Reader(us_states).records():
            if state.attributes["admin"] == "United States of America":
                # Exclude Alaska and Hawaii
                if state.attributes["name"] not in ["Alaska", "Hawaii"]:
                    conus_states.append(state.geometry)

        conus_polygon = unary_union(conus_states)
        logger.info("CONUS polygon retrieved.")
        return conus_polygon
    except Exception as e:
        logger.error(f"Failed to retrieve CONUS polygon: {e}")
        return None


def generate_grid_and_mask(
    e_fields,
    mt_coordinates,
    resolution=(500, 1000),
    filename="grid.pkl",
):
    """
    Generate a grid and mask for interpolating E-field data, then pickle the results.
    """
    if mt_coordinates.shape[0] != e_fields.shape[0]:
        logger.warning("Warning: Number of points and values don't match!")

    logger.info("Generating grid and mask...")
    lon_min, lon_max = np.min(mt_coordinates[:, 1]), np.max(mt_coordinates[:, 1])
    lat_min, lat_max = np.min(mt_coordinates[:, 0]), np.max(mt_coordinates[:, 0])

    grid_x, grid_y = np.mgrid[
        lon_min : lon_max : complex(0, resolution[0]),
        lat_min : lat_max : complex(0, resolution[1]),
    ]

    conus_polygon = get_conus_polygon()

    if conus_polygon is not None:
        mask = np.array(
            [
                conus_polygon.contains(Point(x, y))
                for x, y in zip(grid_x.ravel(), grid_y.ravel())
            ]
        ).reshape(grid_x.shape)
    else:
        mask = (
            (grid_x >= lon_min)
            & (grid_x <= lon_max)
            & (grid_y >= lat_min)
            & (grid_y <= lat_max)
        )

    # Interpolate the E-field data onto the grid
    grid_z = griddata(
        mt_coordinates[:, [1, 0]], e_fields, (grid_x, grid_y), method="linear"
    )

    # Apply mask to grid_z only
    grid_z = np.ma.array(grid_z, mask=~mask)

    with open(filename, "wb") as f:
        pickle.dump((grid_x, grid_y, grid_z, e_fields), f)
        logger.info("Grid and mask saved to file.")


def extract_line_coordinates(
    df, geometry_col="geometry", source_crs=None, target_crs="EPSG:4326", filename=None
):
    """
    Extract line coordinates from a DataFrame with a geometry column, optionally transforming coordinates.
    """
    logger.info("Extracting line coordinates...")
    line_coordinates = []
    valid_indices = []

    # Ensure df is a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=geometry_col)

    # Ensure the GeoDataFrame has a CRS
    if df.crs is None:
        if source_crs is None:
            raise ValueError("GeoDataFrame has no CRS and source_crs is not provided")
        df = df.set_crs(source_crs, allow_override=True)

    # Transform to target CRS if needed
    if df.crs.to_string() != target_crs:
        df = df.to_crs(target_crs)

    for idx, geometry in enumerate(df[geometry_col]):
        if isinstance(geometry, LineString):
            coords = np.array(geometry.coords)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                line_coordinates.append(coords[:, :2])
                valid_indices.append(idx)
            else:
                logger.error(
                    f"Skipping linestring at index {idx} with unexpected shape: {coords.shape}"
                )
        else:
            logger.error(f"Invalid LineString at index {idx}: {geometry}")

    logger.info("Line coordinates extracted.")

    if filename:
        with open(filename, "wb") as f:
            pickle.dump((line_coordinates, valid_indices), f)
            logger.info("Line coordinates saved to file.")

    return line_coordinates, valid_indices


def get_injection_currents_vectorized(
    df_lines, n_nodes, non_zero_indices, sub_look_up, DATA_LOC, gannon_storm_only=False
):
    """
    Calculate injection currents for power system nodes using vectorized operations
    """
    logger.info("Calculating injection currents (fully vectorized)...")

    if gannon_storm_only:
        cols = [col for col in df_lines.columns if "V_gannon" in col]
    else:
        cols = ["V_halloween", "V_st_patricks", "V_gannon"] + [
            f"V_{period}" for period in return_periods
        ]

    # Precompute necessary arrays
    R = df_lines["R"].to_numpy()  # (n_lines,)
    from_idx = df_lines["from_bus"].map(sub_look_up).to_numpy()  # (n_lines,)
    to_idx = df_lines["to_bus"].map(sub_look_up).to_numpy()  # (n_lines,)

    # Compute effective currents for all columns at once (n_lines x n_cols)
    I_eff_mat = df_lines[cols].to_numpy() / R[:, None]
    I_eff_mat = np.nan_to_num(I_eff_mat, nan=0.0)

    n_cols = len(cols)
    # Initialize injection currents matrix (n_nodes x n_cols)
    injection_currents_all = np.zeros((n_nodes, n_cols), dtype=I_eff_mat.dtype)

    for j in range(n_cols):
        pos = np.bincount(to_idx, weights=I_eff_mat[:, j], minlength=n_nodes)
        neg = np.bincount(from_idx, weights=I_eff_mat[:, j], minlength=n_nodes)
        injection_currents_all[:, j] = pos - neg

    # Collect results only for non-zero nodes
    injections_data = {
        col: injection_currents_all[non_zero_indices, j] for j, col in enumerate(cols)
    }

    return injections_data


def can_use_float16(arr):
    """
    Determine if an array can be safely represented in float16 or float 32
    """
    try:
        # Use numpy first
        if hasattr(arr, "get"):
            arr = arr.get()
        max_abs = np.max(np.abs(arr))
        min_abs = np.min(np.abs(arr[arr != 0])) if np.any(arr != 0) else np.float32(0)
    except Exception:
        # Fall back to using torch
        if isinstance(arr, torch.Tensor):
            tensor_arr = arr
        else:
            tensor_arr = torch.tensor(arr)
        max_abs = torch.max(torch.abs(tensor_arr)).item()
        min_abs = torch.min(torch.abs(tensor_arr[tensor_arr != 0])).item() if torch.any(tensor_arr != 0) else 0.0
    return (max_abs < 65504 and min_abs > 6.1e-5) or max_abs == 0


def nodal_voltage_calculation_torch_vectorized(Y_total, injections_data):
    """
    Calculate nodal voltages using PyTorch's linear algebra solvers with vectorization.
    """
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
        
        # Check if matrix fits in GPU memory
        matrix_size_bytes = Y_total.nbytes
        identity_size_bytes = Y_total.shape[0] * Y_total.shape[1] * 4  # float32
        total_required = (matrix_size_bytes + identity_size_bytes) * 1.2  # 20% buffer
        
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        
        if total_required > free_memory:
            logger.info(f"Matrix too large for GPU ({total_required/1e9:.1f}GB needed, {free_memory/1e9:.1f}GB free), using CPU")
            device = torch.device("cpu")
        else:
            logger.info(f"Using GPU ({total_required/1e9:.1f}GB needed, {free_memory/1e9:.1f}GB free)")
    else:
        gc.collect()

    logger.info(f"Using device: {device}")

    Y_tensor = torch.tensor(Y_total, dtype=torch.float32, device=device)

    # Adaptive regularization strategy
    success = False
    reg_values = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    for reg in reg_values:
        Y_reg = Y_tensor + torch.eye(Y_tensor.shape[0], dtype=torch.float32, device=device) * reg
        
        try:
            L = torch.linalg.cholesky(Y_reg)
            print(f"Cholesky succeeded with regularization: {reg}")
            success = True
            break
        except Exception as e:
            print(f"Cholesky failed with reg={reg}: {e}")
            continue

    if not success:
        # Try more aggressive strategies
        logger.info("Trying alternative decomposition methods...")
        
        # Method 1: Add regularization only to zero diagonal elements
        diag_mask = torch.diag(Y_tensor) < 1e-10
        Y_reg = Y_tensor.clone()
        Y_reg[diag_mask, diag_mask] += 1e-3
        
        try:
            L = torch.linalg.cholesky(Y_reg)
            logger.info("Cholesky succeeded with selective regularization")
            success = True
        except:
            logger.error("Cholesky failed with selective regularization")
            pass
        
        if not success:
            # Method 2: LU decomposition (more stable but slower)
            logger.info("Cholesky failed, trying LU decomposition...")
            try:
                LU, pivots = torch.linalg.lu_factor(Y_tensor)
                # Modified solve process for LU
                valid_keys = [k for k, v in injections_data.items() if v is not None]
                injection_batch = torch.stack(
                    [torch.tensor(injections_data[k], dtype=torch.float32, device=device) 
                        for k in valid_keys], dim=-1
                )
                V_n = torch.linalg.lu_solve(LU, pivots, injection_batch)
                
                results = {}
                for idx, key in enumerate(valid_keys):
                    results[key] = V_n[:, idx].cpu().numpy()
                return results
            except Exception as e:
                logger.error(f"LU decomposition also failed: {e}")
                return {k: None for k in injections_data}

    # Continue with Cholesky solve
    valid_keys = [k for k, v in injections_data.items() if v is not None]
    if not valid_keys:
        return {k: None for k in injections_data}

    # Stack injections
    injection_batch = torch.stack(
        [torch.tensor(injections_data[k], dtype=torch.float32, device=device) 
            for k in valid_keys], dim=-1
    )

    # Solve using Cholesky decomposition
    try:
        # Use solve_triangular for better performance
        P = torch.linalg.solve_triangular(L, injection_batch, upper=False)
        V_n = torch.linalg.solve_triangular(L.t(), P, upper=True)
    except Exception as e:
        print(f"Triangular solve failed: {e}")
        V_n = torch.linalg.solve(Y_reg, injection_batch)

    # Convert results
    results = {}
    for idx, key in enumerate(valid_keys):
        results[key] = V_n[:, idx].cpu().numpy()
        
    logger.info("Nodal voltage calculation completed.")

    return results


def find_storm_maxima(E_pred, gannon_times, window_hours=(5 / 60), num_peaks=500):
    """
    Find the maximum E-field magnitudes during a geomagnetic storm.
    """
    window_samples = int(window_hours * 60)
    site_maxE_mags = np.sqrt(np.sum(E_pred**2, axis=2))
    total_magnitude = np.nansum(site_maxE_mags, axis=1)

    smoothed_magnitude = gaussian_filter1d(total_magnitude, sigma=window_samples // 5)

    peaks, _ = signal.find_peaks(
        smoothed_magnitude,
        distance=window_samples // 5,  # Reduce min distance between peaks
        prominence=0.1 * np.max(smoothed_magnitude),  # Reduce prominence threshold
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
    """
    Process Gannon storm data and extract voltage values at peak times.
    """
    logger.info("Loading and processing Gannon GIC data...")

    gannon_ds = xr.open_dataset(DATA_LOC / "storm_maxes" / "ds_gannon.nc")
    gannon_times = gannon_ds.time.values

    if peak_time:
        # Find the peak times
        E_pred = gannon_ds.E_pred.values
        peak_data = find_storm_maxima(E_pred, gannon_times)
        peak_times = [data["peak_time"] for data in peak_data]

        # Save peak times
        np.save(DATA_LOC / "peak_times.npy", peak_times)

    else:
        # Convert the start date to numpy datetime64
        start_date = np.datetime64("2024-05-10")

        # Create the mask with proper datetime64 comparison
        mask = (gannon_times >= start_date) & (
            gannon_times.astype("datetime64[m]").view("int64") % res == 0
        )
        peak_times = gannon_times[mask]

        logger.info(f"Peak times loaded successfully... shape: {peak_times.shape}")

        # Save peak times
        np.save(DATA_LOC / "peak_times_1.npy", peak_times)

    # Get indices of peak times in gannon_times
    peak_indices = np.searchsorted(gannon_times, peak_times)

    # Load delaunay array - Vs
    arr_v = np.load(DATA_LOC / "storm_maxes" / "delaunay_v_gannon.npy")

    # Index Vs during peak E
    arr_v_peaks = arr_v[peak_indices]

    # Load the data from storm maxes
    with h5py.File(DATA_LOC / results_path, "r") as f:
        logger.info("Reading geomagnetic data from geomagnetic_data.h5")
        # Read MT site information
        mt_names = f["sites/mt_sites/names"][:]
        mt_coords = f["sites/mt_sites/coordinates"][:]

        # Read transmission line IDs
        line_ids = f["sites/transmission_lines/line_ids"][:]
        line_ids_str = [id.decode('utf-8') if isinstance(id, bytes) else str(id) for id in line_ids]

        # Read Halloween storm data
        halloween_e = f["events/halloween/E"][:] / 1000
        halloween_b = f["events/halloween/B"][:]
        halloween_v = f["events/halloween/V"][:]

        # Read st_patricks storm data
        st_patricks_e = f["events/st_patricks/E"][:] / 1000
        st_patricks_b = f["events/st_patricks/B"][:]
        st_patricks_v = f["events/st_patricks/V"][:]

        # Read the Gannon storm data
        gannon_e = f["events/gannon/E"][:] / 1000
        gannon_v = f["events/gannon/V"][:]

    # Voltage columns for all events
    v_cols = ["V_gannon_peak_max"]
    num_peaks = len(peak_indices)
    v_gannon_cols = [f"V_gannon_{i+1}" for i in range(num_peaks)]

    # Extend the voltage column list
    v_cols.extend(v_gannon_cols)

    # Mapping IDs to indices
    id_to_index = {id: i for i, id in enumerate(line_ids_str)}
    indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])
    mask = indices != -1

    df_lines.loc[mask, "V_gannon_peak_max"] = gannon_v[indices[mask]]

    # Add voltage values for each peak
    for i, peak_idx in enumerate(peak_indices):
        col_name = f"V_gannon_{i+1}"
        df_lines[col_name] = np.nan  # Initialize column with NaNs
        df_lines.loc[mask, col_name] = arr_v_peaks[i][indices[mask]]

    # Set a default value for all missing values
    df_lines[v_cols] = df_lines[v_cols].fillna(0)

    logger.info("Gannon Voltage values loaded and processed successfully.")

    return (df_lines, mt_coords, mt_names, gannon_e, v_cols)


def pre_compute_ze(Y):
    """
    Compute impedance matrix from admittance matrix with zero handling.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.reciprocal(Y, where=Y != 0)  # Avoids explicit np.isinf check
        Z[Y == 0] = 0
        return Z


def solve_total_nodal_gic_optimized(Z, V_all):
    """
    Vectorized nodal GIC solution for stacked voltage components.
    """
    Y_reg = Z.T @ Z + 1e-20 * np.eye(Z.shape[1])
    return spsolve(Y_reg, Z.T @ V_all.T) * 3


def main(generate_grid_only=False, gannon_storm_only=False): 
    """
    Main function to calculate GICs in the power system.
    """
    results_path = "statistical_analysis/geomagnetic_data_return_periods.h5"

    # Get substation buses data
    substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info = (
        process_substation_buses(DATA_LOC)
    )

    # Load and process transmission line data
    df_lines.drop(columns=["geometry"], inplace=True)
    df_lines["name"] = df_lines["name"].astype(str)

    transmission_line_path = (
        DATA_LOC / "grid_processed" / "trans_lines_pickle.pkl"
    )
    with open(transmission_line_path, "rb") as p:
        trans_lines_gdf = pickle.load(p)

    trans_lines_gdf["line_id"] = trans_lines_gdf["line_id"].astype(str)
    df_lines = df_lines.merge(
        trans_lines_gdf[["line_id", "geometry"]], right_on="line_id", left_on="name"
    )

    # Create a dictionary for quick substation lookup
    sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))

    trafos_data = get_transformer_samples(substation_buses)

    if gannon_storm_only:
        df_lines, mt_coords, mt_names, gannon_e, v_cols = process_gannon(
            DATA_LOC, df_lines, results_path, res=1
        )
        logger.info("Gannon storm data processed successfully.")
    else:
        # Load and process GIC data
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

    n_nodes = len(sub_look_up)  # Number of nodes in the network
    
    # Get 1000 dfs of winding GICs and np gics
    logger.info("Starting GIC calculations...")
    for i, trafo_data in enumerate(trafos_data):
        
        if i < 500:
            continue

        # Save the GIC DataFrame
        filename = processed_gic_path / f"winding_gic_rand_{i}.csv"

        if os.path.exists(filename) and not gannon_storm_only:
            continue
        
        if gannon_storm_only:
            filename_gic = (
                DATA_LOC / "gic" / "ground_gic" / f"ground_gic_gannon_{i}.csv"
            )

            if os.path.exists(filename_gic):
                continue

        logger.info(f"Processing iteration {i}...")
        # Generate a random admittance matrix
        logger.info("Generating random admittance matrix...")

        Y_n, Y_e, df_transformers = random_admittance_matrix(
            substation_buses,
            trafo_data,
            bus_ids_map,
            sub_look_up,
            df_lines,
            df_substations_info,
        )

        # Find indices of rows/columns where all elements are zero in the admittance matrix
        zero_row_indices = np.where(np.all(Y_n == 0, axis=1))[0]  # Zero rows

        # Get the non-zero row/col indices
        non_zero_indices = np.setdiff1d(np.arange(Y_n.shape[0]), zero_row_indices)

        # Convert to smaller data types BEFORE operations
        if can_use_float16(Y_n) and can_use_float16(Y_e):
            Y_n = Y_n.astype(np.float16)
            Y_e = Y_e.astype(np.float16)
            dtype = np.float16
        elif (np.max(np.abs(Y_n)) < 3.4e38 and np.max(np.abs(Y_e)) < 3.4e38):
            Y_n = Y_n.astype(np.float32) 
            Y_e = Y_e.astype(np.float32)
            dtype = np.float32
        else:
            dtype = np.float64

        logger.info(f"Using {dtype} for matrices")

        Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
        Y_e = Y_e[np.ix_(non_zero_indices, non_zero_indices)]

        Y_total = np.add(Y_n, Y_e, dtype=dtype)
    
        del Y_n; import gc; gc.collect()
        
        # Get injections data
        injections_data = get_injection_currents_vectorized(
            df_lines,
            n_nodes,
            non_zero_indices,
            sub_look_up,
            DATA_LOC,
            gannon_storm_only=gannon_storm_only,
        )

        nodal_voltages = nodal_voltage_calculation_torch_vectorized(Y_total, injections_data)

        # If gannon only, let's solve for ground currents
        if gannon_storm_only:
            filename_gic = (
                Path("/data/archives/nfs/spw-geophy/data") / "gic" / "ground_gic" / f"ground_gic_gannon_{i}.csv"
            )

            if os.path.exists(filename_gic):
                continue

            Z_e = pre_compute_ze(Y_e)
            v_all = np.vstack([nodal_voltages[c] for c in v_cols])
            ig_all = solve_total_nodal_gic_optimized(Z_e, v_all)

            idx_series = df_substations_info["name"].map(sub_look_up)
            mask = idx_series.notna() & (df_substations_info["name"] != "Substation 1")
            valid_substations = df_substations_info.loc[mask, "name"].to_numpy()
            valid_indices = idx_series[mask].astype(int).to_numpy()

            non_reduced_mat = np.zeros((n_nodes, ig_all.shape[1]), dtype=ig_all.dtype)
            non_reduced_mat[non_zero_indices] = ig_all
            gic_values_T = non_reduced_mat[valid_indices].T  # (T, S)

            data = {"Substation": valid_substations}
            for col, vals in zip(v_cols, gic_values_T):
                data[f"GIC_{col.split('V_')[1]}"] = vals

            pd.DataFrame(data).to_csv(filename_gic, index=False)

            continue
        else:
            df_lines_copy = df_lines.copy()
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
                    
                    # Each thread gets its own copy to avoid race conditions
                    df_lines_local = df_lines_copy.copy()
                    df_transformers_local = df_transformers.copy()
                    
                    V_nodal = nodal_voltages[f"V_{period}"]
                    
                    df_gic = calculate_GIC(
                        df_lines_local, V_nodal, f"V_{period}", non_zero_indices, n_nodes
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
                                logger.warning(f"Skipping period {period} due to calculation error")
                        except Exception as e:
                            failed_period = futures[fut]
                            logger.error(f"Failed to get result for period {failed_period}: {str(e)}")
                            calculation_success = False

                # Check if we have valid results before processing
                if not gic_data or not calculation_success:
                    logger.error(f"Calculation failed for iteration {i}, skipping...")
                    continue  # Skip this iteration entirely

                # Validate that we have the expected periods
                expected_periods = ["gannon"] + list(return_periods)
                missing_periods = [p for p in expected_periods if p not in gic_data]
                if missing_periods:
                    logger.warning(f"Missing data for periods: {missing_periods}")
                    # Decide if you want to continue with partial data or skip
                    if len(missing_periods) > len(expected_periods) / 2:  # If more than half failed
                        logger.error(f"Too many periods failed for iteration {i}, skipping...")
                        continue

                # Prepare GIC DataFrames for each period
                winding_gic_df_list = []
                for period, gic_values in gic_data.items():
                    try:
                        # Validate gic_values structure
                        if not gic_values or not isinstance(gic_values, dict):
                            logger.warning(f"Invalid GIC data for period {period}, skipping period")
                            continue
                            
                        hash_gic_period = [
                            (trafo, winding, gic)
                            for trafo, windings in gic_values.items()
                            for winding, gic in windings.items()
                            if isinstance(windings, dict)  # Safety check
                        ]
                        
                        if not hash_gic_period:  # No valid data
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
                        logger.error(f"Error processing GIC data for period {period}: {str(e)}")
                        continue

                # Check if we have any valid dataframes
                if not winding_gic_df_list:
                    logger.error(f"No valid GIC dataframes created for iteration {i}, skipping...")
                    continue

                # Merge all GIC dataframes
                try:
                    winding_gic_df = pd.concat(winding_gic_df_list, axis=1).loc[
                        :, ~pd.concat(winding_gic_df_list, axis=1).columns.duplicated()
                    ]
                    
                    # Check if the resulting dataframe is valid
                    if winding_gic_df.empty:
                        logger.error(f"Empty dataframe after merge for iteration {i}, skipping...")
                        continue

                    # Finalize transformer data merge
                    df_transformers["Transformer"] = df_transformers["name"]
                    winding_gic_df = winding_gic_df.merge(
                        df_transformers[["sub_id", "Transformer", "latitude", "longitude"]],
                        on="Transformer",
                        how="inner",
                    )
                    
                    # Final validation before saving
                    if winding_gic_df.empty:
                        logger.error(f"Empty dataframe after transformer merge for iteration {i}, skipping...")
                        continue

                    # Save the GIC DataFrame only if it's valid
                    winding_gic_df.to_csv(filename, index=False)
                    logger.info(f"Successfully saved GIC data for iteration {i}")
                    
                except Exception as e:
                    logger.error(f"Error during dataframe processing for iteration {i}: {str(e)}")
                    continue

            except Exception as e:
                logger.error(f"Critical error in iteration {i}: {str(e)}")
                continue  # Skip to next iteration
            
            filename_gic = (
                processed_gnd_gic_path / f"ground_gic{i}.csv"
            )

            if os.path.exists(filename_gic):
                continue

            Z_e = pre_compute_ze(Y_e)
            del Y_e; gc.collect()
            
            v_all = np.vstack([nodal_voltages[c] for c in v_cols])
            ig_all = solve_total_nodal_gic_optimized(Z_e, v_all)
            
            del Z_e; gc.collect()

            idx_series = df_substations_info["name"].map(sub_look_up)
            mask = idx_series.notna() & (df_substations_info["name"] != "Substation 1")
            valid_substations = df_substations_info.loc[mask, "name"].to_numpy()
            valid_indices = idx_series[mask].astype(int).to_numpy()

            non_reduced_mat = np.zeros((n_nodes, ig_all.shape[1]), dtype=ig_all.dtype)
            non_reduced_mat[non_zero_indices] = ig_all
            gic_values_T = non_reduced_mat[valid_indices].T  # (T, S)

            data = {"Substation": valid_substations}
            for col, vals in zip(v_cols, gic_values_T):
                data[f"GIC_{col.split('V_')[1]}"] = vals

            pd.DataFrame(data).to_csv(filename_gic, index=False)


if __name__ == "__main__":
    # Set Torch fall back to cpu... although slower
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    gannon_storm_only = True

    main(gannon_storm_only=gannon_storm_only)
