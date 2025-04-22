# ...............................................................................
# Description: Levereages the admittance matrix to calculate the GICs along the
# transmission lines and transformers in the network.
# Dependency: Output from build_admittance_matrix.py and data from data_preprocessing.ipynb
# ...............................................................................

# %%
import warnings
import sys
import os
import pickle
import logging
import time
from pathlib import Path
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve, cholesky
from shapely import wkt
from scipy import stats
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from memory_profiler import profile
from shapely.ops import transform
from scipy.interpolate import griddata
import pyproj
import cupy as cp
from shapely.geometry import LineString
from shapely.geometry import Point  # Point class
import geopandas as gpd

# Import custom functions
from build_admittance_matrix_eo import process_substation_buses, random_admittance_matrix

# print(cp.__version__)
# print(cp.cuda.runtime.runtimeGetVersion())  # Check runtime version
# print(cp.cuda.runtime.driverGetVersion())  # Check driver version

# Custom logger
# Set up logging, a reusable function
def setup_logging():
    """
    Set up logging to both file and console.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler("final_data_analysis.log")
    file_handler.setLevel(logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
data_loc = Path.cwd() / "data"


# %%
# Function to find the substation name for a given bus
def find_substation_name(bus, sub_ref):
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    # If not found, return None
    return None


def main(generate_grid=False):

    # Load the data
    # Data loc
    data_loc = Path.cwd() / "data" 
    results_path = data_loc / "statistical_analysis" / "geomagnetic_data_return_periods.h5"

    # Get substation buses data
    substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info = (
        process_substation_buses(data_loc)
    )

    # Load and process transmission line data
    df_lines.drop(columns=["geometry"], inplace=True)
    df_lines["name"] = df_lines["name"].astype(np.int32)

    filename = "trans_lines_pickle.pkl"
    folder = data_loc / "Electric__Power_Transmission_Lines" 
    transmission_line_path = folder / filename
    with open(transmission_line_path, "rb") as p:
        trans_lines_gdf = pickle.load(p)

    trans_lines_gdf["line_id"] = trans_lines_gdf["line_id"].astype(np.int32)
    df_lines = df_lines.merge(
        trans_lines_gdf[["line_id", "geometry"]], right_on="line_id", left_on="name"
    )

    # Create a dictionary for quick substation lookup
    sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))

    trafos_data = get_transformer_samples(substation_buses)

    # Load and process GIC data
    (
        df_lines,
        mt_coords,
        mt_names,
        e_fields,
        b_fields,
        v_fields,
        # gannon_e,
    ) = load_and_process_gic_data(data_loc, df_lines, results_path)

    n_nodes = len(sub_look_up)  # Number of nodes in the network
    # cLear gpu memory"""  """

    # Get 1000 dfs of winding GICs and np gics
    for i, trafo_data in enumerate(trafos_data):

        # Save the GIC DataFrame
        filename = data_loc / "admittance_matrix" / f"winding_gic_rand_{i}.csv"
        
        if os.path.exists(filename):
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

        # Find indices of rows/columns where all elements are zero in the admittance mat
        zero_row_indices = np.where(np.all(Y_n == 0, axis=1))[0]  # Zero rows
        zero_col_indices = np.where(np.all(Y_n == 0, axis=0))[0]  # Zero columns

        # Get the non-zero row/col indices
        non_zero_indices = np.setdiff1d(np.arange(Y_n.shape[0]), zero_row_indices)

        # Reduce the Y_n and Y_e matrices
        Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
        Y_e = Y_e[np.ix_(non_zero_indices, non_zero_indices)]

        # Y total is sum of earthing and network impedances
        Y_total = Y_n + Y_e
        # Get injections data
        injections_data = get_injection_currents(
            df_lines, n_nodes, non_zero_indices, sub_look_up, data_loc
        )

        nodal_voltages = nodal_voltage_calculation(Y_total, injections_data)

        # df_lines_copy = df_lines.copy()
        # df_lines_copy["from_bus"] = df_lines_copy["from_bus"].apply(
        #     lambda x: sub_look_up.get(x)
        # )
        # df_lines_copy["to_bus"] = df_lines_copy["to_bus"].apply(
        #     lambda x: sub_look_up.get(x)
        # )

        # # Calculate GIC for each return period
        # gic_data = {}
        # # for period in ["gannon"] + list(return_periods):
        # for period in list(return_periods):
        #     # Check if V_gannon exists in nodal voltages
        #     if f"V_{period}" not in nodal_voltages:
        #         print("Nodal voltages", f"V_{period}", nodal_voltages)
        #         continue
        #     V_nodal = nodal_voltages[f"V_{period}"]
        #     df_gic = calculate_GIC(
        #         df_lines_copy, V_nodal, f"V_{period}", non_zero_indices, n_nodes
        #     )
        #     gic_data[period] = calc_trafo_gic(
        #         sub_look_up,
        #         df_transformers.copy(),
        #         V_nodal,
        #         sub_ref,
        #         n_nodes,
        #         non_zero_indices,
        #         f"{period}-year-hazard",
        #     )

        # # Prepare GIC DataFrames for each period
        # winding_gic_df_list = []
        # for period, gic_values in gic_data.items():
        #     hash_gic_period = [
        #         (trafo, winding, gic)
        #         for trafo, windings in gic_values.items()
        #         for winding, gic in windings.items()
        #     ]
        #     winding_gic_df = pd.DataFrame(
        #         hash_gic_period,
        #         columns=["Transformer", "Winding", f"{period}-year-hazard A/ph"],
        #     )
        #     winding_gic_df_list.append(winding_gic_df)

        # # Merge all GIC dataframes
        # winding_gic_df = pd.concat(winding_gic_df_list, axis=1).loc[
        #     :, ~pd.concat(winding_gic_df_list, axis=1).columns.duplicated()
        # ]
        # print(len(winding_gic_df))
        # # Finalize transformer data merge
        # df_transformers["Transformer"] = df_transformers["name"]
        # winding_gic_df = winding_gic_df.merge(
        #     df_transformers[["sub_id", "Transformer", "latitude", "longitude"]],
        #     on="Transformer",
        #     how="inner",
        # )

        # # Save the GIC DataFrame
        # winding_gic_df.to_csv(filename, index=False)

    #     # Calculate the total GIC for the network
    #     parallel_gic_calculation_and_processing(
    #         Y_e, nodal_voltages, non_zero_indices, n_nodes, data_loc, filename
    #     )

    #     # cLear gpu memory

    #     # Prepare the grid and mask for plotting
    #     if generate_grid:
    #         grid_e_100_path = data_loc / "grid_e_100.pkl"
    #         grid_e_500_path = data_loc / "grid_e_500.pkl"
    #         grid_e_1000_path = data_loc / "grid_e_1000.pkl"
    #         grid_e_gannon_path = data_loc / "grid_e_gannon.pkl"

    #         grid_file_paths = [
    #             grid_e_100_path,
    #             grid_e_500_path,
    #             grid_e_1000_path,
    #             grid_e_gannon_path,
    #         ]

    #         e_field_100 = e_fields[100]
    #         e_field_500 = e_fields[500]
    #         e_field_1000 = e_fields[1000]
    #         e_field_gannon = gannon_e

    #         e_fields_period = [e_field_100, e_field_500, e_field_1000, e_field_gannon]

    #         # Prepare the transmission lines data for plotting
    #         for grid_filename, e_field in zip(grid_file_paths, e_fields_period):
    #             if not os.path.exists(grid_filename):
    #                 # Generate and save the grid and mask
    #                 generate_grid_and_mask(
    #                     e_field,
    #                     mt_coords,
    #                     resolution=(500, 1000),
    #                     filename=grid_filename,
    #                 )

    #         logging.info("Grid and mask generated and saved.")

    # line_coords_file = data_loc / "line_coords.pkl"
    # source_crs = "EPSG:4326"
    # if not os.path.exists(line_coords_file):
    #     line_coordinates, valid_indices = extract_line_coordinates(
    #         df_lines, filename=line_coords_file
    #     )


def load_and_process_gic_data(data_loc, df_lines, results_path):
    """
    Load and process geomagnetically induced current (GIC) data.

    Parameters
    ----------
    data_loc : Path
        Path to the data directory.
    consider_real_transformer_data : bool, optional
        Whether to use real transformer data (default is False).

    Returns
    -------
    dict
        A dictionary containing processed data:
        - Y_total: Total admittance matrix
        - df_lines: DataFrame with transmission line data
        - df_substations_info: DataFrame with substation information
        - df_transformers: DataFrame with transformer data
        - transformer_counts_dict: Dictionary of transformer counts
        - sub_look_up: Dictionary for substation lookup
        - sub_ref: Dictionary for quick substation reference
    """
    logger.info("Loading and processing GIC data...")

    # Load the data from storm maxes
    with h5py.File(results_path, "r") as f:

        logger.info("Reading geomagnetic data from geomagnetic_data.h5")
        # Read MT site information
        mt_names = f["sites/mt_sites/names"][:]
        mt_coords = f["sites/mt_sites/coordinates"][:]

        # Read transmission line IDs
        line_ids = f["sites/transmission_lines/line_ids"][:]

        # Read Halloween storm data
        halloween_e = f["events/halloween/E"][:] / 1000
        halloween_b = f["events/halloween/B"][:]
        halloween_v = f["events/halloween/V"][:]

        # Read st_patricks storm data
        st_patricks_e = f["events/st_patricks/E"][:] / 1000
        st_patricks_b = f["events/st_patricks/B"][:]
        st_patricks_v = f["events/st_patricks/V"][:]

        # # Read the Gannon storm data
        # gannon_e = f["events/gannon/E"][:] / 1000
        # gannon_b = f["events/gannon/B"][:]
        # gannon_v = f["events/gannon/V"][:]

        e_fields, b_fields, v_fields = {}, {}, {}

        # Load E, B, and V fields dynamically for each return period
        for period in return_periods:
            e_fields[period] = f[f"predictions/E/{period}_year"][:]
            b_fields[period] = f[f"predictions/B/{period}_year"][:]
            v_fields[period] = f[f"predictions/V/{period}_year"][:]

    # Voltage columns for all events
    v_cols = ["V_halloween", "V_st_patricks"] + [ #, "V_gannon"
        f"V_{period}" for period in return_periods
    ]

    # Mapping IDs to indices
    id_to_index = {id: i for i, id in enumerate(line_ids)}
    indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])
    mask = indices != -1

    # Use boolean indexing to handle missing values
    mask = indices != -1
    df_lines.loc[mask, "V_halloween"] = halloween_v[indices[mask]]
    df_lines.loc[mask, "V_st_patricks"] = st_patricks_v[indices[mask]]
    # df_lines.loc[mask, "V_gannon"] = gannon_v[indices[mask]]

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
        # gannon_e,
    )


# Calculate the injection currents for the network
def calculate_injection_currents(df, n_nodes, col, non_zero_indices, sub_look_up):
    """
    Calculate the injection currents for each node in a network based on the given dataframe.
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the network data.
    - n_nodes (int): The number of nodes in the network.
    - col (str): The column name representing the source currents in the dataframe.
    Returns:
    - injection_currents (numpy.ndarray): An array of injection currents for each node in the network.
    """
    logger.info(f"Calculating injection currents for {col}...")

    # Initialize injection currents for x and y components
    injection_currents = np.zeros(n_nodes)

    # Calculate line currents and injection currents
    for i, line in df.iterrows():

        # Get the source currents
        I_eff = line[col] / line["R"]

        if np.isnan(I_eff):
            # logger.error(f"NaN current encountered for line {line['name']}")
            continue

        # Get i and j
        j, i = sub_look_up.get(line["to_bus"]), sub_look_up.get(line["from_bus"])

        # Currents into and out of the nodes
        injection_currents[i] -= I_eff
        injection_currents[j] += I_eff

    # Eliminate the zero rows and columns
    injection_currents = injection_currents[non_zero_indices]

    return injection_currents


# Solve for nodal voltages
# Using sparse to solve for np gics
def solve_eqn(Y, injection_currents):
    # Solve for nodal voltages
    # Due to size of the matrix add regularization term
    # Skews the results slightly but is fairly faste
    logger.info("Solving for nodal voltages...")
    Y_reg = Y.T @ Y + 1e-20 * eye(Y.shape[1])
    V_nodal = spsolve(Y_reg, Y.T @ injection_currents)
    logger.info("Nodal voltages solved successfully.")
    # Return nodal voltages
    return V_nodal


# Using lPm to solve for nodal voltages
def get_and_solve_cholesky(Y, Je):
    regularization = 1e-6
    Y_reg = Y + np.eye(Y.shape[0]) * regularization
    # Cholesky decomposition of Y_n
    L = cholesky(Y_reg, lower=True)

    # Step 1: Solve L * P = Je
    P = solve(L, Je)

    # Step 2: Solve L^T * V_n = P to get nodal voltages V_n
    V_n = solve(L.T, P)

    return V_n


# GPU implementations
def solve_eqn_gpu(Y, injection_currents):
    """GPU version of equation solver."""
    logger.info("GPU: Solving for nodal voltages...")
    start = time.perf_counter()

    # Transfer to GPU
    Y_gpu = cp.asarray(Y)
    injection_currents_gpu = cp.asarray(injection_currents)

    # Solve on GPU
    Y_reg = Y_gpu.T @ Y_gpu + 1e-20 * cp.eye(Y_gpu.shape[1])
    V_nodal = cp.linalg.solve(Y_reg, Y_gpu.T @ injection_currents_gpu)

    # Transfer back to CPU
    result = cp.asnumpy(V_nodal)

    elapsed = time.perf_counter() - start
    logger.info(f"GPU: Nodal voltages solved in {elapsed:.4f} seconds")
    return result


def get_and_solve_cholesky_gpu(Y, Je):
    """GPU version of Cholesky solver."""
    try:
        # Transfer to GPU
        Y_gpu = cp.asarray(Y)
        Je_gpu = cp.asarray(Je)

        regularization = 1e-6
        Y_reg = Y_gpu + cp.eye(Y_gpu.shape[0]) * regularization
        L = cp.linalg.cholesky(Y_reg)
        P = cp.linalg.solve(L, Je_gpu)
        V_n = cp.linalg.solve(L.T, P)

        # Transfer back to CPU
        result = cp.asnumpy(V_n)
        return result

    except Exception as e:
        logger.error(f"GPU: Error during Cholesky solution: {str(e)}")
        return None


# %%
# Calculate the GIC for the transmission lines
def calculate_GIC(df, V_nodal, col, non_zero_indices, n_nodes):
    """
    Calculate the Ground Induced Current (GIC) for a given dataframe.
    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the transmission line data.
    - V_nodal (numpy.ndarray): The nodal voltages.
    - col (str): The column name representing the GIC values.
    Returns:
    - df (pandas.DataFrame): The input dataframe with an additional column representing the calculated GIC values.
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


# %%
def calc_trafo_gic(
    sub_look_up, df_transformers, V_nodal, sub_ref, n_nodes, non_zero_indices, title=""
):

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


# %%
# Solve for the currents to the ground through the neutral points
def solve_total_nodal_gic(Y_e, Vn, title=""):

    logger.info(f"Solving for total NP GIC {title}...")

    # Current flowing to the ground due to eastward/northward geolectric field
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_e = 1 / Y_e
        Z_e[np.isinf(Z_e)] = 0  # Replace inf with 0

    i_g = solve_eqn(Z_e, Vn)  # Per phase
    return i_g * 3  # Total


# %%
# PLot gridded countour map
def get_conus_polygon():
    """
    Retrieves the polygon representing the continental United States (CONUS).

    Returns:
        A polygon object representing the boundary of the continental United States.

    Raises:
        None.
    """
    logger.info("Retrieving CONUS polygon...")

    try:
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


# For efficieny, we can preprocess tghe grid x, y, and gridz and pickle them
# Then load them in the carto function
def generate_grid_and_mask(
    e_fields,
    mt_coordinates,
    resolution=(500, 1000),
    filename="grid.pkl",
):

    if mt_coordinates.shape[0] != e_fields.shape[0]:
        print("Warning: Number of points and values still don't match!")

    logging.info("Generating grid and mask...")
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
        logging.info("Grid and mask saved to file.")


# %%
# Prepare transmission lines data for plotting
def extract_line_coordinates(
    df, geometry_col="geometry", source_crs=None, target_crs="EPSG:4326", filename=None
):
    """
    Extract line coordinates from a DataFrame with a geometry column, optionally transforming coordinates.

    Parameters:
    - df: DataFrame with geometry column containing LineString objects
    - geometry_col: name of the geometry column (default: 'geometry')
    - source_crs: The source CRS of the geometries (e.g., 'EPSG:4326' for WGS84)
    - target_crs: The target CRS (default: 'EPSG:4326' for WGS84)

    Returns:
    - line_coordinates: list of numpy arrays containing line coordinates
    - valid_indices: list of indices of valid LineStrings
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

    with open(filename, "wb") as f:
        pickle.dump((line_coordinates, valid_indices), f)
        logger.info("Line coordinates saved to file.")

    return line_coordinates, valid_indices


def get_injection_currents(df_lines, n_nodes, non_zero_indices, sub_look_up, data_loc):
    """
    Get or calculate injection currents for each event and return period.

    Parameters
    ----------
    df_lines : DataFrame
        DataFrame containing transmission line data.
    n_nodes : int
        Number of nodes.
    non_zero_indices : list
        Indices for non-zero injection currents.
    sub_look_up : dict
        Dictionary for substation lookups.
    data_loc : Path
        Path to the data directory.

    Returns
    -------
    dict
        Dictionary with injection currents for each event and return period.
    """
    logger.info("Calculating injection currents...")

    # Initialize injections data dictionary
    injections_data = {
        "V_halloween": calculate_injection_currents(
            df_lines, n_nodes, "V_halloween", non_zero_indices, sub_look_up
        ),
        "V_st_patricks": calculate_injection_currents(
            df_lines, n_nodes, "V_st_patricks", non_zero_indices, sub_look_up
        ),
        # "V_gannon": calculate_injection_currents(
        #     df_lines, n_nodes, "V_gannon", non_zero_indices, sub_look_up
        # ),
    }

    # Add dynamic return period injections
    for period in return_periods:
        injections_data[f"V_{period}"] = calculate_injection_currents(
            df_lines, n_nodes, f"V_{period}", non_zero_indices, sub_look_up
        )

    return injections_data


def calculate_GIC_multiple(
    df_lines_copy, sub_look_up, V_nodals, non_zero_indices, n_nodes
):
    """
    Calculate GIC for multiple scenarios, including dynamically defined return periods.

    Parameters
    ----------
    df_lines_copy : DataFrame
        Copy of the DataFrame containing transmission line data.
    sub_look_up : dict
        Dictionary for substation lookups.
    V_nodals : list
        List of nodal voltages for each scenario.
    non_zero_indices : list
        Indices for non-zero GIC values.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    DataFrame
        Concatenated DataFrame with GIC calculations for each scenario.
    """

    # Scenarios
    scenarios = ["Gannon"] + [f"V_{period}" for period in return_periods]

    # Initialize list for DataFrames
    df_list = []

    # Calculate GIC for each nodal voltage and scenario
    for V_nodal, scenario in zip(V_nodals, scenarios):
        df = calculate_GIC(
            df_lines_copy.copy(), V_nodal, scenario, non_zero_indices, n_nodes
        )
        df_list.append(df)

    # Concatenate all DataFrames, removing duplicated columns
    return pd.concat(df_list, axis=1).loc[
        :, ~pd.concat(df_list, axis=1).columns.duplicated()
    ]


def get_and_solve_cholesky_wrapper(args):
    Y_total, injections = args
    # time_elapsed = time.time()
    # print("Solving for nodal voltages...", Y_total.shape, injections.shape)
    # v_n = get_and_solve_cholesky_gpu(Y_total, injections)
    # print(f"Time taken GPU: {time.time() - time_elapsed}")

    # No gpus
    v_n = get_and_solve_cholesky(Y_total, injections)

    return v_n


def parallel_nodal_voltage_calculation(Y_total, injections_data):
    """
    Perform parallel nodal voltage calculations for multiple injection datasets.

    Parameters:
    - Y_total: The admittance matrix (Y matrix) for the system.
    - injections_data: Dictionary containing various injection data.

    Returns:
    - results: Dictionary of calculated voltages for each dataset.
    """
    logger.info("Preparing tasks for parallel nodal voltage calculation...")

    # List of tasks for parallel execution (filter out None tasks)
    tasks = [
        (name, injections_data.get(name))
        for name in ["gannon", "V_100", "V_500", "V_1000"]
    ]
    tasks = [(name, data) for name, data in tasks if data is not None]

    if not tasks:
        logger.error("No valid injection data provided.")
        return {}
    # Prepare the data for multiprocessing
    task_data = [(Y_total, t[1]) for t in tasks]

    logger.info("Solving for nodal voltages...")

    # Use Pool for parallel computation
    with Pool(processes=min(cpu_count(), len(tasks))) as pool:
        try:
            # Solve the system for each injection dataset in parallel
            results = {
                name: result
                for name, result in zip(
                    [t[0] for t in tasks],
                    pool.map(get_and_solve_cholesky_wrapper, task_data),
                )
            }
            logger.info("Nodal voltages solved successfully.")
            return results
        except Exception as e:
            logger.error(f"Error during parallel execution: {str(e)}")
            return {}


def parallel_gic_calculation_and_processing(
    Y_e, nodal_voltages, non_zero_indices, n_nodes, data_loc, filename
):
    # Create a partial function with Y_e fixed
    solve_func = partial(solve_total_nodal_gic, Y_e)

    # Define tasks for Gannon and dynamically generated return periods
    tasks = [("Gannon", nodal_voltages["V_gannon"], "Gannon")]

    # Append tasks for each return period
    tasks += [
        (
            f"{period}-year-hazard",
            nodal_voltages[f"V_{period}"],
            f"{period}-year-hazard",
        )
        for period in return_periods
    ]

    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(solve_func, V_nodal, scenario): name
            for name, V_nodal, scenario in tasks
        }

        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_task):
            name = future_to_task[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"{name} generated an exception: {exc}")

    # Process the results
    data = np.stack([results[name] for name, _, _ in tasks], axis=1)

    data_zeros = np.zeros((n_nodes, 4))
    data_zeros[non_zero_indices, :] = data
    df_ig = pd.DataFrame(data_zeros, columns=[name for name, _, _ in tasks])

    # Save the dataframe
    df_ig.to_csv(data_loc / f"np_gic_rand_{filename}.csv", index=False)

    return df_ig


def nodal_voltage_calculation(Y_total, injections_data):
    """
    Optimized sequential GPU solver with batching and memory reuse.
    Uses float16 where possible for memory efficiency while maintaining numerical stability.
    """
    # cp.get_default_memory_pool().free_all_blocks()

    def can_use_float16(arr):
        """Check if array values are within float16 range"""
        if isinstance(arr, cp.ndarray):
            max_abs = cp.max(cp.abs(arr))
            min_abs = (
                cp.min(cp.abs(arr[arr != 0])) if cp.any(arr != 0) else cp.float32(0)
            )
        else:
            max_abs = np.max(np.abs(arr))
            min_abs = (
                np.min(np.abs(arr[arr != 0])) if np.any(arr != 0) else np.float32(0)
            )

        # Check if values are within float16 range
        return (max_abs < 65504 and min_abs > 6.1e-5) or max_abs == 0

    # try:
    # Check if Y_total can use float16
    y_dtype = cp.float16 if can_use_float16(Y_total) else cp.float32
    Y_gpu = cp.asarray(Y_total, dtype=y_dtype)

    # Regularization - keep in float32 for stability
    regularization = cp.float32(1e-6)
    # Convert to float32 for matrix operations
    Y_reg = (
        Y_gpu.astype(cp.float32)
        + cp.eye(Y_gpu.shape[0], dtype=cp.float32) * regularization
    )

    # Cholesky decomposition needs float32 for stability
    L = cp.linalg.cholesky(Y_reg)

    scenarios = ["V_gannon"] + [f"V_{period}" for period in return_periods]
    tasks = [(name, injections_data.get(name)) for name in scenarios]
    tasks = [(name, data) for name, data in tasks if data is not None]

    results = {}

    # Preallocate GPU memory
    if tasks:
        sample_shape = tasks[0][1].shape
        sample_data = tasks[0][1]

        # Check if injections can use float16
        inj_dtype = cp.float16 if can_use_float16(sample_data) else cp.float32
        injection_gpu = cp.empty(sample_shape, dtype=inj_dtype)

    for name, injections in tasks:
        try:
            # Check if current injection can use float16
            curr_dtype = cp.float16 if can_use_float16(injections) else cp.float32

            # Reallocate if shape or dtype doesn't match
            if (
                injections.shape != sample_shape
                or injection_gpu.dtype != curr_dtype
            ):
                logger.warning(
                    f"Shape or dtype mismatch for {name}, reallocating GPU memory."
                )
                injection_gpu = cp.asarray(injections, dtype=curr_dtype)
            else:
                cp.copyto(injection_gpu, cp.asarray(injections, dtype=curr_dtype))

            # Convert to float32 for solving the system
            injection_float32 = injection_gpu.astype(cp.float32)

            # Solve system using pre-computed Cholesky decomposition
            P = cp.linalg.solve(L, injection_float32)
            V_n = cp.linalg.solve(L.T, P)

            # Check if result can be stored in float16
            if can_use_float16(V_n):
                V_n = V_n.astype(cp.float16)

            # Transfer result back to CPU
            results[name] = cp.asnumpy(V_n)

        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            results[name] = None
            continue

        # # Cleanup GPU memory
        # del Y_gpu, Y_reg, L, injection_gpu, P, V_n
        # cp.get_default_memory_pool().free_all_blocks()

        # logger.info("All nodal voltage computations completed.")
        # return results

    # except Exception as e:
    #     logger.error(f"{str(e)} Switching to CPU computation...")
    #     return {}


def get_transformer_samples(
    substation_buses,
    sample_net_name="sample_network.pkl",
    n_samples=1,
    seed=42,
    ):
    """
    Generate multiple samples of network configurations:
    - Each substation gets assigned a transformer count
    - Each transformer count gets assigned types
    - Returns a list of DataFrames, each with one sample configuration per substation.
    """

    # Define file path
    data_path = data_loc / 'admittance_matrix' / sample_net_name

    # Check if pre-generated samples exist
    if os.path.exists(data_path):  # Changed from 'not os.path.exists'
        with open(data_path, "rb") as f:
            all_samples = pickle.load(f)
    else:
        # Set seed for reproducibility
        np.random.seed(seed)

        # Transformer types and sample storage
        transformer_types = ["GY-GY", "GY-GY-D", "Auto"]
        all_samples = []

        for i in range(n_samples):
            # Create configuration data for each substation in this sample
            # Transformer generator number
            transformer_gen_num = 0
            transformers_data = []

            # Randomly assign transformer count and types for each substation
            for sub_id, values in substation_buses.items():
                # Assign a random transformer count (1 to 3)
                trafo_count = np.random.randint(
                    1, 4
                )  # Changed from 4 to match your description

                # Select random transformer types for the count
                selected_types = list(
                    np.random.choice(transformer_types, size=trafo_count, replace=True)
                )

                # Add transformers
                for transformer_type in selected_types:
                    transformer_gen_num += 1
                    transformer_number = (
                        "T" + "_" + str(sub_id) + "_" + str(transformer_gen_num)
                    )

                    transformer_data = {
                        "sub_id": sub_id,
                        "name": transformer_number,
                        "type": transformer_type,
                        "bus1_id": values["hv_bus"],
                        "bus2_id": values["lv_bus"],
                    }

                    transformers_data.append(transformer_data)

            # Append the sample as a DataFrame
            all_samples.append(pd.DataFrame(transformers_data))

        logger.info(f"Generated {len(all_samples)} samples. Saving to {data_path}")
        # Save the generated samples to a file for reuse
        with open(data_path, "wb") as f:
            pickle.dump(all_samples, f)

    return all_samples

    # %%
# # Load the data
# # Data loc
# data_loc = Path.cwd() / "data"
# results_path = "geomagnetic_data_return_periods.h5"

# # Get substation buses data
# substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info = (
# process_substation_buses(data_loc)
# )

# # Load and process transmission line data
# df_lines.drop(columns=["geometry"], inplace=True)
# df_lines["name"] = df_lines["name"].astype(np.int32)

# transmission_line_path = (
# data_loc / "Electric__Power_Transmission_Lines" / "trans_lines_pickle.pkl"
# )

# # %%
# with open(transmission_line_path, "rb") as p:
#     trans_lines_gdf = pickle.load(p)


# # %%
# trans_lines_gdf["line_id"] = trans_lines_gdf["line_id"].astype(np.int32)
# df_lines = df_lines.merge(
# trans_lines_gdf[["line_id", "geometry"]], right_on="line_id", left_on="name"
# )

# # Create a dictionary for quick substation lookup
# sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))

# trafos_data = get_transformer_samples(substation_buses)

# # Load and process GIC data
# (
# df_lines,
# mt_coords,
# mt_names,
# e_fields,
# b_fields,
# v_fields,
# gannon_e,
# ) = load_and_process_gic_data(data_loc, df_lines, results_path)

# # %%
# n_nodes = len(sub_look_up)  # Number of nodes in the network
# # cLear gpu memory"""  """

# # Get 1000 dfs of winding GICs and np gics
# for i, trafo_data in enumerate(trafos_data):

#     # Save the GIC DataFrame
#     filename = data_loc / f"winding_gic_rand_{i}.csv"

#     logger.info(f"Processing iteration {i}...")
#     # Generate a random admittance matrix
#     logger.info("Generating random admittance matrix...")

#     Y_n, Y_e, df_transformers = random_admittance_matrix(
#         substation_buses,
#         trafo_data,
#         bus_ids_map,
#         sub_look_up,
#         df_lines,
#         df_substations_info,
#     )

#     # Find indices of rows/columns where all elements are zero in the admittance mat
#     zero_row_indices = np.where(np.all(Y_n == 0, axis=1))[0]  # Zero rows
#     zero_col_indices = np.where(np.all(Y_n == 0, axis=0))[0]  # Zero columns

#     # Get the non-zero row/col indices
#     non_zero_indices = np.setdiff1d(np.arange(Y_n.shape[0]), zero_row_indices)

#     # Reduce the Y_n and Y_e matrices
#     Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
#     Y_e = Y_e[np.ix_(non_zero_indices, non_zero_indices)]

#     # Y total is summ of earthing and network impedances
#     Y_total = Y_n + Y_e
#     # Get injections data
#     injections_data = get_injection_currents(
#         df_lines, n_nodes, non_zero_indices, sub_look_up, data_loc
#     )

#     break


# # %%
# # Clear memnory
# cp.get_default_memory_pool().free_all_blocks()

# nodal_voltages = nodal_voltage_calculation(Y_total, injections_data)



if __name__ == "__main__":

    import time
    start_time = time.time()

    # Define return periods
    return_periods = np.arange(25, 201, 50)
    # return_periods = np.array([100, 500, 1000])

    main(generate_grid=False)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
