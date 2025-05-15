# ...............................................................................
# Description: Levereages the admittance matrix to calculate the GICs along the
# transmission lines and transformers in the network.
# Dependency: Output from build_admittance_matrix.py and data from data_preprocessing.ipynb
# ...............................................................................

# %%
import os
import gc
import pickle
from functools import reduce
import h5py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from shapely.ops import unary_union
from scipy.interpolate import griddata
import xarray as xr
import torch
from shapely.geometry import LineString
from shapely.geometry import Point  # Point class
import geopandas as gpd
from scipy.ndimage import gaussian_filter1d
from scipy import signal

# Import custom functions
from build_admittance_matrix import process_substation_buses, random_admittance_matrix, get_transformer_samples
from configs import setup_logger, get_data_dir


def find_substation_name(bus, sub_ref):
    """
    Find the substation name for a given bus from the substation reference dictionary.
    
    Parameters
    ----------
    bus : str
        The bus ID to search for
    sub_ref : dict
        Dictionary mapping substation names to lists of bus IDs
    
    Returns
    -------
    str or None
        The name of the substation containing the bus, or None if not found
    """
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    # If not found, return None
    return None


def load_and_process_gic_data(DATA_LOC, df_lines, results_path):
    """
    Load and process geomagnetically induced current (GIC) data from HDF5 files.

    Parameters
    ----------
    DATA_LOC : Path
        Path to the data directory
    df_lines : DataFrame
        DataFrame containing transmission line data
    results_path : str
        Path to the HDF5 file containing geomagnetic data

    Returns
    -------
    tuple
        Tuple containing:
        - df_lines: Updated DataFrame with transmission line data
        - mt_coords: Coordinates of MT sites
        - mt_names: Names of MT sites
        - e_fields: Dictionary of E-field data for different return periods
        - b_fields: Dictionary of B-field data for different return periods
        - v_fields: Dictionary of voltage data for different return periods
        - gannon_e: E-field data for Gannon storm
        - v_cols: List of voltage column names
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
    Vectorized calculation of injection currents for a power system network.
    
    This function computes the nodal injection currents resulting from GIC for each node
    in the network using vectorized operations for improved performance.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing transmission line data
    n_nodes : int
        Number of nodes in the network
    col : str
        Column name in df representing the voltage source
    non_zero_indices : array
        Indices of non-zero nodes to consider
    sub_look_up : dict
        Dictionary mapping bus names to node indices
        
    Returns
    -------
    ndarray
        Array of injection currents for non-zero nodes
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
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing transmission line data
    V_nodal : ndarray
        Nodal voltages calculated from the network model
    col : str
        Column name representing the GIC source voltages
    non_zero_indices : array
        Indices of non-zero nodes in the network
    n_nodes : int
        Total number of nodes in the network
        
    Returns
    -------
    DataFrame
        Input DataFrame with an additional column containing calculated GIC values
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
    Calculate GIC in transformer windings based on nodal voltages.
    
    This function computes GIC in different types of transformers (GSU, Auto, GY-GY-D, etc.)
    based on their configuration and the nodal voltages from the network solution.
    
    Parameters
    ----------
    sub_look_up : dict
        Dictionary mapping bus names to node indices
    df_transformers : DataFrame
        DataFrame containing transformer data
    V_nodal : ndarray
        Nodal voltages calculated from the network model
    sub_ref : dict
        Dictionary mapping substation names to bus lists
    n_nodes : int
        Total number of nodes in the network
    non_zero_indices : array
        Indices of non-zero nodes in the network
    title : str, optional
        Title for logging purposes
        
    Returns
    -------
    dict
        Dictionary mapping transformer names to dictionaries of winding GIC values
    """
    logger.info(f"Calculating GIC for transformers {title}...")
    gic = {}

    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal

    # Process transformers and build admittance matrix
    for bus, bus_idx in sub_look_up.items():
        sub = find_substation_name(bus, sub_ref)

        # Filter transformers for current bus
        trafos = df_transformers[df_transformers["bus1"] == bus]

        if len(trafos) == 0 or sub == "Substation 7":
            continue

        # Process each transformer
        for _, trafo in trafos.iterrows():
            # Extract parameters
            bus1 = trafo["bus1"]
            bus2 = trafo["bus2"]
            neutral_point = trafo["sub"]  # Neutral point node (for auto-transformers)
            W1 = trafo["W1"]  # Impedance for Winding 1 (Primary, Series)
            W2 = trafo["W2"]  # Impedance for Winding 2 (Secondary, if available)

            trafo_name = trafo["name"]
            trafo_type = trafo["type"]
            bus1_idx = sub_look_up[bus1]
            neutral_idx = sub_look_up.get(neutral_point)
            bus2_idx = sub_look_up[bus2]

            if trafo_type == "GSU" or trafo_type == "GSU w/ GIC BD":
                Y_w1 = 1 / W1  # Primary winding admittance
                i_k = (V_all[bus1_idx] - V_all[neutral_idx]) * Y_w1
                gic[trafo_name] = {"HV": i_k}

            elif trafo_type == "Tee":
                # Commented out code, consider removing if not needed
                continue

            elif trafo_type == "Auto":
                Y_series = 1 / W1
                Y_common = 1 / W2
                I_s = (V_all[bus1_idx] - V_all[bus2_idx]) * Y_series
                I_c = (V_all[bus2_idx] - V_all[neutral_idx]) * Y_common
                gic[trafo_name] = {"Series": I_s, "Common": I_c}

            elif trafo_type in ["GY-GY-D", "GY-GY"]:
                Y_primary = 1 / W1
                Y_secondary = 1 / W2
                I_w1 = (V_all[bus1_idx] - V_all[neutral_idx]) * Y_primary
                I_w2 = (V_all[bus2_idx] - V_all[neutral_idx]) * Y_secondary
                gic[trafo_name] = {"HV": I_w1, "LV": I_w2}

    logger.info(f"GIC calculation for transformers {title} completed.")
    return gic


def get_conus_polygon():
    """
    Retrieves the polygon representing the continental United States (CONUS).

    Returns
    -------
    shapely.geometry.Polygon or None
        A polygon object representing the boundary of the continental United States,
        or None if retrieval fails.
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
    
    Parameters
    ----------
    e_fields : ndarray
        Array containing E-field data values
    mt_coordinates : ndarray
        Array containing coordinates of MT sites, shape (n_sites, 2)
    resolution : tuple, optional
        Grid resolution as (x_points, y_points), default (500, 1000)
    filename : str, optional
        Filename to save the pickled grid data, default "grid.pkl"
        
    Notes
    -----
    This function preprocesses grid data for later use in visualization,
    creating a masked grid confined to the continental US boundary.
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

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        DataFrame with geometry column containing LineString objects
    geometry_col : str, optional
        Name of the geometry column, default 'geometry'
    source_crs : str, optional
        The source CRS of the geometries (e.g., 'EPSG:4326' for WGS84)
    target_crs : str, optional
        The target CRS, default 'EPSG:4326' for WGS84
    filename : str, optional
        If provided, save the extracted coordinates to this file

    Returns
    -------
    tuple
        Tuple containing:
        - line_coordinates: list of numpy arrays containing line coordinates
        - valid_indices: list of indices of valid LineStrings
        
    Raises
    ------
    ValueError
        If GeoDataFrame has no CRS and source_crs is not provided
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
    Calculate injection currents for power system nodes using vectorized operations.
    
    This optimized function computes injection currents for all voltage sources
    in a single operation, which is significantly faster than iterative methods.
    
    Parameters
    ----------
    df_lines : DataFrame
        DataFrame containing transmission line data
    n_nodes : int
        Number of nodes in the network
    non_zero_indices : ndarray
        Indices of non-zero nodes in the network
    sub_look_up : dict
        Dictionary mapping bus names to node indices
    DATA_LOC : Path
        Path to data directory (for compatibility with original function)
    gannon_storm_only : bool, optional
        If True, only calculate for Gannon storm voltage sources, default False
        
    Returns
    -------
    dict
        Dictionary mapping voltage column names to arrays of injection currents
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

    # Use np.bincount for each injection column (typically only a few columns)
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
    Determine if an array can be safely represented in float16 format.
    
    Parameters
    ----------
    arr : ndarray or tensor
        Array to check for float16 compatibility
        
    Returns
    -------
    bool
        True if the array can be safely represented in float16, False otherwise
        
    Notes
    -----
    This function evaluates whether an array has values within the range that
    can be accurately represented by float16 (approx. 6.1e-5 to 65504).
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
    
    This function computes nodal voltages by solving the system Y*V = I using 
    Cholesky decomposition and vectorized operations, with fallbacks for numerical stability.
    
    Parameters
    ----------
    Y_total : ndarray
        Total admittance matrix (Y_n + Y_e)
    injections_data : dict
        Dictionary mapping voltage source names to injection current arrays
        
    Returns
    -------
    dict
        Dictionary mapping voltage source names to nodal voltage arrays
        
    Notes
    -----
    The function automatically selects the best device (CUDA > MPS > CPU)
    and precision (float16/float32) based on hardware availability and 
    numerical requirements.
    """
    logger.info("Calculating nodal voltages using PyTorch...")
    # Select device: cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    logger.info(f"Using device: {device}")

    # Use float32 for numerical stability
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
        # Fallback to standard solve
        V_n = torch.linalg.solve(Y_reg, injection_batch)

    # Convert results
    results = {}
    for idx, key in enumerate(valid_keys):
        results[key] = V_n[:, idx].cpu().numpy()
        
    logger.info("Nodal voltage calculation completed.")

    return results



def format_np_gic(substations_df, sub_look_up, non_zero_indices, ig, n_nodes, v_col):
    """
    Format ground-induced current (GIC) data for substations into a DataFrame.
    
    Parameters
    ----------
    substations_df : DataFrame
        DataFrame containing substation information
    sub_look_up : dict
        Dictionary mapping bus names to node indices
    non_zero_indices : ndarray
        Indices of non-zero nodes in the network
    ig : ndarray
        Array of ground-induced currents for non-zero nodes
    n_nodes : int
        Total number of nodes in the network
    v_col : str
        Voltage column name for labeling
        
    Returns
    -------
    DataFrame
        DataFrame with columns 'Substation' and a GIC value column named based on v_col
    """
    # Initialize full GIC array with zeros
    non_reduced_matrix = np.zeros(n_nodes)

    # Assign computed ground currents to the corresponding substations
    for reduced_idx, full_idx in enumerate(non_zero_indices):
        non_reduced_matrix[full_idx] = ig[reduced_idx]

    # Extract valid indices (excluding "Substation 1")
    indices = [
        sub_look_up[sub]
        for sub in substations_df["name"]
        if sub_look_up.get(sub) is not None and sub != "Substation 1"
    ]

    # Get corresponding substations
    valid_substations = [
        sub for sub in substations_df["name"] if sub_look_up.get(sub) in indices
    ]

    # Create and return DataFrame
    return pd.DataFrame(
        {
            "Substation": valid_substations,
            f"GIC_{v_col.split('V_')[1]}": non_reduced_matrix[indices],
        }
    )


def find_storm_maxima(E_pred, gannon_times, window_hours=(5 / 60), num_peaks=500):
    """
    Find the maximum E-field magnitudes during a geomagnetic storm.
    
    Parameters
    ----------
    E_pred : ndarray
        Predicted E-field values, shape (time_steps, sites, components)
    gannon_times : ndarray
        Array of timestamps corresponding to E_pred
    window_hours : float, optional
        Window size in hours for smoothing and peak detection, default 5 minutes (5/60)
    num_peaks : int, optional
        Maximum number of peaks to return, default 500
        
    Returns
    -------
    list of dict
        List of dictionaries, each containing:
        - peak_time: timestamp of the peak
        - site_magnitudes: E-field magnitudes at all sites for this peak
        - total_magnitude: sum of magnitudes across all sites
        - smoothed_magnitude: smoothed total magnitude value
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
    
    Parameters
    ----------
    DATA_LOC : Path
        Path to the data directory
    df_lines : DataFrame
        DataFrame containing transmission line data
    results_path : str
        Path to the HDF5 file containing geomagnetic data
    peak_time : bool, optional
        If True, find peaks in E-field data; if False, use time-based sampling, default False
    res : int, optional
        Time resolution in minutes for sampling when peak_time=False, default 30
        
    Returns
    -------
    tuple
        Tuple containing:
        - df_lines: Updated DataFrame with voltage values
        - mt_coords: Coordinates of MT sites
        - mt_names: Names of MT sites
        - gannon_e: E-field data for Gannon storm
        - v_cols: List of voltage column names
    """
    logger.info("Loading and processing Gannon GIC data...")

    gannon_ds = xr.open_dataset(DATA_LOC / "ds_gannon.nc")
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
    arr_v = np.load(DATA_LOC / "delaunay_v_gannon.npy")

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
    
    Parameters
    ----------
    Y : ndarray
        Admittance matrix
        
    Returns
    -------
    ndarray
        Impedance matrix with zeros where admittance is zero
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.reciprocal(Y, where=Y != 0)  # Avoids explicit np.isinf check
        Z[Y == 0] = 0
        return Z


def solve_total_nodal_gic_optimized(Z, V_all):
    """
    Vectorized nodal GIC solution for stacked voltage components.
    
    Parameters
    ----------
    Z : ndarray
        Impedance matrix
    V_all : ndarray
        Matrix of voltage solutions stacked by column
        
    Returns
    -------
    ndarray
        Optimized solution for nodal GICs for all voltage sources
        
    Notes
    -----
    This function solves the regularized system (Z^T·Z)·I = Z^T·V for all
    voltage sources simultaneously, applying a scaling factor of 3 for
    three-phase representation.
    """
    Y_reg = Z.T @ Z + 1e-20 * np.eye(Z.shape[1])
    return spsolve(Y_reg, Z.T @ V_all.T) * 3


def main(generate_grid_only=False, gannon_storm_only=False): 
    """
    Main function to calculate GICs in the power system.
    
    Parameters
    ----------
    generate_grid_only : bool, optional
        If True, only generate grid data for plotting, default False
    gannon_storm_only : bool, optional
        If True, only process Gannon storm data, default False
        
    Notes
    -----
    This function orchestrates the entire GIC calculation workflow:
    1. Load power system data
    2. Process geomagnetic field data
    3. Generate admittance matrices
    4. Calculate nodal voltages
    5. Compute GICs in transmission lines and transformers
    6. Save results to CSV files
    
    When generate_grid_only=True, it only creates interpolated grids
    of E-field data for visualization purposes.
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

    if not generate_grid_only:
        # Get 1000 dfs of winding GICs and np gics
        logger.info("Starting GIC calculations...")
        for i, trafo_data in enumerate(trafos_data):

            # if i % 2 == 0:
            #     continue

            # Save the GIC DataFrame
            filename = processed_gic_path / f"winding_gic_rand_{i}.csv"

            if os.path.exists(filename) and not gannon_storm_only:
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

            # Reduce the Y_n and Y_e matrices
            Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
            Y_e = Y_e[np.ix_(non_zero_indices, non_zero_indices)]

            # Y total is sum of earthing and network impedances
            Y_total = Y_n + Y_e

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
                dfs_np_gic = []  # List to store np gic dataframes

                Z_e = pre_compute_ze(Y_e)
                v_all = np.vstack([nodal_voltages[v_col] for v_col in v_cols])

                ig_all = solve_total_nodal_gic_optimized(Z_e, v_all)
                # Format each time step's GIC results
                for timestep in range(ig_all.shape[1]):
                    ig = ig_all[:, timestep]  # Get GIC for this timestep

                    # Format using your existing function
                    df_ground_gic = format_np_gic(
                        df_substations_info,
                        sub_look_up,
                        non_zero_indices,
                        ig,
                        n_nodes,
                        v_cols[timestep],  # Use corresponding voltage column name
                    )
                    dfs_np_gic.append(df_ground_gic)

                df_final_gic = reduce(
                    lambda left, right: pd.merge(
                        left, right, on="Substation", how="outer"
                    ),
                    dfs_np_gic,
                )

                filename_gic = (
                    processed_gic_path / "ground_gic" / f"ground_gic_gannon_{i}.csv"
                )  # Adjust filename as needed

                if os.path.exists(filename_gic):
                    continue

                df_final_gic.to_csv(filename_gic, index=False)
                continue
            else:
                df_lines_copy = df_lines.copy()
                df_lines_copy["from_bus"] = df_lines_copy["from_bus"].apply(
                    lambda x: sub_look_up.get(x)
                )
                df_lines_copy["to_bus"] = df_lines_copy["to_bus"].apply(
                    lambda x: sub_look_up.get(x)
                )
                # Calculate GIC for each return period
                gic_data = {}
                for period in ["gannon"] + list(return_periods):
                    # Check if V_gannon exists in nodal voltages
                    if f"V_{period}" not in nodal_voltages:
                        logger.warning(f"Nodal voltages missing for {period}")
                        continue
                    V_nodal = nodal_voltages[f"V_{period}"]
                    df_gic = calculate_GIC(
                        df_lines_copy, V_nodal, f"V_{period}", non_zero_indices, n_nodes
                    )
                    gic_data[period] = calc_trafo_gic(
                        sub_look_up,
                        df_transformers.copy(),
                        V_nodal,
                        sub_ref,
                        n_nodes,
                        non_zero_indices,
                        f"{period}-year-hazard",
                    )

                # Prepare GIC DataFrames for each period
                winding_gic_df_list = []
                for period, gic_values in gic_data.items():
                    hash_gic_period = [
                        (trafo, winding, gic)
                        for trafo, windings in gic_values.items()
                        for winding, gic in windings.items()
                    ]
                    winding_gic_df = pd.DataFrame(
                        hash_gic_period,
                        columns=[
                            "Transformer",
                            "Winding",
                            f"{period}-year-hazard A/ph",
                        ],
                    )
                    winding_gic_df_list.append(winding_gic_df)

                # Merge all GIC dataframes
                winding_gic_df = pd.concat(winding_gic_df_list, axis=1).loc[
                    :, ~pd.concat(winding_gic_df_list, axis=1).columns.duplicated()
                ]

                # Finalize transformer data merge
                df_transformers["Transformer"] = df_transformers["name"]
                winding_gic_df = winding_gic_df.merge(
                    df_transformers[["sub_id", "Transformer", "latitude", "longitude"]],
                    on="Transformer",
                    how="inner",
                )

                # Save the GIC DataFrame
                winding_gic_df.to_csv(filename, index=False)
    else:
        logger.info("Generating grid and mask for plotting...")
        # Specify the file paths for grid 75, 100, 150, 200 and 250
        grid_e_75_path = DATA_LOC / "grid_e_75.pkl"
        grid_e_100_path = DATA_LOC / "grid_e_100.pkl"
        grid_e_150_path = DATA_LOC / "grid_e_150.pkl"
        grid_e_200_path = DATA_LOC / "grid_e_200.pkl"
        grid_e_250_path = DATA_LOC / "grid_e_250.pkl"
        grid_e_500_path = DATA_LOC / "grid_e_500.pkl"
        grid_e_1000_path = DATA_LOC / "grid_e_1000.pkl"
        grid_e_gannon_path = DATA_LOC / "grid_e_gannon.pkl"

        grid_file_paths = [
            grid_e_75_path,
            grid_e_100_path,
            grid_e_150_path,
            grid_e_200_path,
            grid_e_250_path,
            grid_e_500_path,
            grid_e_1000_path,
            grid_e_gannon_path,
        ]

        # Prepare the e fields for plotting
        e_field_75 = e_fields[75]
        e_field_100 = e_fields[100]
        e_field_150 = e_fields[150]
        e_field_200 = e_fields[200]
        e_field_250 = e_fields[250]
        e_field_500 = e_fields[500]
        e_field_1000 = e_fields[1000]
        e_field_gannon = gannon_e

        e_fields_period = [
            e_field_75,
            e_field_100,
            e_field_150,
            e_field_200,
            e_field_250,
            e_field_500,
            e_field_1000,
            e_field_gannon,
        ]

        # Prepare the transmission lines data for plotting
        for grid_filename, e_field in zip(grid_file_paths, e_fields_period):
            if not os.path.exists(grid_filename):
                # Generate and save the grid and mask
                generate_grid_and_mask(
                    e_field,
                    mt_coords,
                    resolution=(500, 1000),
                    filename=grid_filename,
                )

        logger.info("Grid and mask generated and saved.")

        # Save df_lines too
        df_lines.to_pickle(DATA_LOC / "df_lines.pkl")

    line_coords_file = DATA_LOC / "line_coords.pkl"
    source_crs = "EPSG:4326"
    if not os.path.exists(line_coords_file):
        line_coordinates, valid_indices = extract_line_coordinates(
            df_lines, filename=line_coords_file
        )


if __name__ == "__main__":
    # Set Torch fall back to cpu... although slower
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Get data data log and configure logger
    DATA_LOC = get_data_dir()
    processed_gic_path = DATA_LOC / "gic"
    os.makedirs(processed_gic_path, exist_ok=True)
    
    logger = setup_logger(log_file="logs/gic_calculation.log")

    # Define return periods
    return_periods = np.arange(50, 251, 25)

    gannon_storm_only = False

    main(generate_grid_only=False, gannon_storm_only=False)
