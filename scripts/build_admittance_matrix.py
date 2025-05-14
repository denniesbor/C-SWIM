"""
Script to build a US EHV admittance matrix. 

Make sure you have first run the preprocessing scripts. 

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025
"""
import os
import pickle
import sys
import os
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve


from configs import setup_logger, get_data_dir

# Get data data log and configure logger
DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/build_adm_matrix.log")

def process_substation_buses(DATA_LOC, evan_data=False):

    (
        unique_sub_voltage_pairs,
        df_lines_EHV,
        trans_lines_gdf,
        substation_to_line_voltages,
        ss_df,
        trans_lines_within_FERC_filtered,
        ss_type_dict,
        transformer_counts_dict,
        ss_role_dict,
    ) = load_and_process_data(DATA_LOC)

    # File paths
    substation_buses_pkl = DATA_LOC / "admittance_matrix" / "substation_buses.pkl"
    bus_ids_map_pkl = DATA_LOC / "admittance_matrix" / "bus_ids_map.pkl"
    sub_look_up_pkl = DATA_LOC / "admittance_matrix" / "sub_look_up.pkl"
    transmission_lines_csv = DATA_LOC / "admittance_matrix" / "transmission_lines.csv"
    substation_info_csv = DATA_LOC / "admittance_matrix" / "substation_info.csv"

    # Check if pickled data exists to avoid reprocessing
    if (
        os.path.exists(substation_buses_pkl)
        and os.path.exists(bus_ids_map_pkl)
        and os.path.exists(sub_look_up_pkl)
    ):
        # Load data from pickle
        with open(substation_buses_pkl, "rb") as f:
            substation_buses = pickle.load(f)

        with open(bus_ids_map_pkl, "rb") as f:
            bus_ids_map = pickle.load(f)

        with open(sub_look_up_pkl, "rb") as f:
            sub_look_up = pickle.load(f)

        # Load CSV files
        df_lines = pd.read_csv(transmission_lines_csv)
        substations_df = pd.read_csv(substation_info_csv)

        # Try to eval buses
        substations_df["buses"] = substations_df["buses"].apply(eval)

    else:
        if evan_data:
            substation_buses = []
        else:
            # Process substation_buses and bus ids
            substation_buses = build_substation_buses(
                unique_sub_voltage_pairs,
                df_lines_EHV,
                ss_type_dict,
                substation_to_line_voltages,
            )

        # Create bus ids from buses within the substations
        buses = []
        for sub_info in substation_buses.values():
            buses.extend(sub_info["buses"])
        buses = np.unique(buses)

        # Create a dictionary to map buses to integers
        bus_ids_map = {bus: i + 1 for i, bus in enumerate(buses)}

        # Typically substation grounding resistances range from 0.1 -> 0.2 (Horton, et al., 2013)
        grounding_resistances = [0.1, 0.2]

        # Use from dict to load substation_buses
        df_substations_info = pd.DataFrame(substation_buses).T.reset_index()
        df_substations_info = df_substations_info[
            ["SS_ID", "Transformer_type", "buses"]
        ]

        # Apply bus_id map to buses
        df_substations_info["buses"] = df_substations_info["buses"].apply(
            lambda x: [bus_ids_map[bus] for bus in x]
        )

        # Merge with substations info to get latitude and longitude
        df_substations_info = df_substations_info.merge(
            ss_df[["SS_ID", "lat", "lon"]], on="SS_ID", how="left"
        )
        df_substations_info["grounding_resistance"] = grounding_resistances[1]

        # Rename SSID, lat, and lon columns
        df_substations_info.rename(
            columns={"SS_ID": "name", "lat": "latitude", "lon": "longitude"},
            inplace=True,
        )

        # Flatten the dictionary
        flattened_data = flatten_substation_dict(substation_buses, df_lines_EHV, buses)

        # Create DataFrame
        df_lines_final = pd.DataFrame(
            flattened_data, columns=["from_bus_id", "to_bus_id", "utility", "name"]
        )

        # Filter df_lines_EHV
        df_lines_EHV[~(df_lines_EHV.VOLTAGE.isin([345, 230, 765, 500]))].shape

        # Approximate line resistances
        line_resistance = {
            765: 0.01,  # Ω/km (converted from 0.0227 Ω/mi)
            500: 0.0141,  # Ω/km (converted from 0.0227 Ω/mi)
            345: 0.0283,  # Ω/km (converted from 0.0455 Ω/mi)
            232: 0.0450,  # Ω/km (estimated)
            230: 0.0500,  # Ω/km (estimated)
            220: 0.0700,  # Ω/km (estimated)
            138: 0.0800,  # Ω/km (estimated)
            69: 0.1200,  # Ω/km (estimated)
        }

        # Estimate line resistances
        substations_df = df_substations_info.copy()

        df_lines = calculate_line_resistances(
            df_lines_final,
            df_lines_EHV,
            line_resistance,
            trans_lines_within_FERC_filtered,
            bus_ids_map,
        )

        # Build neutral points and transformer lookup
        sub_look_up = {}
        index = 0
        for i, row in substations_df.iterrows():
            buses = row["buses"]
            for bus in sorted(buses):
                sub_look_up[bus] = index
                index += 1

        for i, row in substations_df.iterrows():
            if row["name"] != "Substation 7":
                sub_look_up[row["name"]] = index
                index += 1

        # Pickle the substation_buses, bus_ids_map, and sub_look_up
        with open(substation_buses_pkl, "wb") as f:
            pickle.dump(substation_buses, f)

        with open(bus_ids_map_pkl, "wb") as f:
            pickle.dump(bus_ids_map, f)

        with open(sub_look_up_pkl, "wb") as f:
            pickle.dump(sub_look_up, f)

        # Save transmission lines and substations info to CSV
        df_lines.to_csv(transmission_lines_csv, index=False)
        df_substations_info.to_csv(substation_info_csv, index=False)

    return substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df


def load_and_process_data(DATA_LOC):
    """
    Load and process various data files for power system analysis.

    Parameters
    ----------
    DATA_LOC : Path
        Path to the data directory.

    Returns
    -------
    tuple
        A tuple containing processed data:
        - unique_sub_voltage_pairs
        - df_lines_EHV
        - trans_lines_gdf
        - substation_to_line_voltages
        - ss_df
        - trans_lines_within_FERC_filtered
        - ss_type_dict
        - transformer_counts_dict
        - ss_role_dict
    """
    logger.info("Loading and processing data...")

    # Load pickled data
    with open(DATA_LOC / "grid_processed" / "unique_sub_voltage_pairs.pkl", "rb") as f:
        unique_sub_voltage_pairs = pickle.load(f)

    with open(DATA_LOC / "grid_processed" / "df_lines_EHV.pkl", "rb") as f:
        df_lines_EHV = pickle.load(f)

    filename = "trans_lines_pickle.pkl"
    folder = DATA_LOC / "Electric__Power_Transmission_Lines"
    with open(folder / filename, "rb") as f:
        trans_lines_gdf = pickle.load(f)
        trans_lines_gdf.rename(
            columns={"line_id": "LINE_ID"}, inplace=True)

    with open(DATA_LOC / "grid_processed" / "substation_to_line_voltages.pkl", "rb") as f:
        substation_to_line_voltages = pickle.load(f)

    with open(DATA_LOC / "grid_processed" / "ss_df.pkl", "rb") as f:
        ss_df = pickle.load(f)

    with open(DATA_LOC / "grid_processed" / "trans_lines_within_FERC_filtered.pkl", "rb") as f:
        trans_lines_within_FERC_filtered = pickle.load(f)

    # Process data
    df_lines_EHV = df_lines_EHV[df_lines_EHV["LINE_ID"].isin(trans_lines_gdf.LINE_ID)]
    ss_type_dict = dict(zip(ss_df["SS_ID"], ss_df["SS_TYPE"]))

    # Process grid mapping data
    grid_mapping = pd.read_csv(DATA_LOC / "grid_mapping.csv")
    grid_mapping["Attributes"] = grid_mapping["Attributes"].apply(eval)

    # Fix: Add .copy() to avoid SettingWithCopyWarning
    transformers_df = grid_mapping[grid_mapping["Marker Label"] == "Transformer"].copy()
    transformers_df["Role"] = transformers_df["Attributes"].apply(lambda x: x["role"])
    transformers_df["type"] = transformers_df["Attributes"].apply(lambda x: x["type"])

    transformer_counts = (
        transformers_df.groupby("SS_ID").size().reset_index(name="transformer_count")
    )
    transformer_counts_dict = dict(
        zip(transformer_counts["SS_ID"], transformer_counts["transformer_count"])
    )

    ss_role = transformers_df[["SS_ID", "Role"]].drop_duplicates()
    ss_role_dict = dict(zip(ss_role["SS_ID"], ss_role["Role"]))

    admittance_dir = DATA_LOC / "admittance_matrix"
    os.makedirs(admittance_dir, exist_ok=True)

    # Save transformer counts
    with open(admittance_dir / "transformer_counts.pkl", "wb") as f:
        pickle.dump(transformer_counts_dict, f)

    logger.info("Data loaded and processed.")

    return (
        unique_sub_voltage_pairs,
        df_lines_EHV,
        trans_lines_gdf,
        substation_to_line_voltages,
        ss_df,
        trans_lines_within_FERC_filtered,
        ss_type_dict,
        transformer_counts_dict,
        ss_role_dict,
    )


def build_substation_buses(
    unique_sub_voltage_pairs,
    df_lines_EHV,
    ss_type_dict,
    substation_to_line_voltages,
    ):
    """
    Build substation buses based on unique substation-voltage pairs and a dataframe of EHV lines.
    Parameters:
    unique_sub_voltage_pairs (DataFrame): A DataFrame containing unique substation-voltage pairs.
    df_lines_EHV (DataFrame): A DataFrame containing EHV lines.
    Returns:
    tuple: A tuple containing two elements:
        - substation_buses (dict): A dictionary mapping substation IDs to substation information.
        - transformers_data (list): A list of transformer data.
    Raises:
    None
    Notes:
    - This function approximates substation buses based on spatially intersected transmission lines and substations.
    - The function estimates injection currents for the designed low voltage and high voltage buses.
    - The function makes assumptions about transformer characteristics based on the voltage ratings of the connected lines.
    - The function assigns transformer types such as "GY-D", "GY-GY", "GY-GY-D", "Auto", or "Unknown" based on the number and ratios of voltage ratings.
    - The function identifies external buses connected to the low voltage bus
    """

    logger.info("Building substation buses...")
    # Approximate substation buses
    substation_buses = {}

    # .........................................................................
    # Design the low voltage and high voltage buses using spatially intersected tls and substations
    # This approach allows for the estimation of the injection currents
    # .........................................................................
    for i, row in unique_sub_voltage_pairs.iterrows():
        substation = row.substation
        line_voltage = row.voltage
        ss_type = ss_type_dict[substation]

        # Get how many trafos are connected to the substation
        if substation not in substation_buses:

            # We will focus now on distribution and transmission substations
            if ss_type not in ["distribution", "transmission"]:
                continue
            # print('here')
            # Voltages in substation
            sub_connected_lines = np.array(
                [float(v) for v in substation_to_line_voltages[substation]]
            )
            substation_buses_unique = np.unique(np.sort(sub_connected_lines))[::-1]

            # Max voltage in substation
            max_voltage_substation = np.max(substation_buses_unique)

            # Get the bus in the substation with the highest voltage
            sub_a_maxV_bus_series = unique_sub_voltage_pairs.query(
                "substation == @substation and voltage == @max_voltage_substation"
            )["bus_id"]

            external_bus_to_hv_bus = []
            # If we have multiple lines intersecting in a substation
            if len(sub_a_maxV_bus_series.values) >= 1:
                sub_a_maxV_bus = sub_a_maxV_bus_series.values[0]

                # Find connected to_bus_ids and from_bus_ids
                sub_a_maxV_bus_series_to = df_lines_EHV.query(
                    "from_bus_id == @sub_a_maxV_bus"
                )["to_bus_id"].values
                sub_a_maxV_bus_series_from = df_lines_EHV.query(
                    "to_bus_id == @sub_a_maxV_bus"
                )["from_bus_id"].values

                # Combine and get unique bus ids
                all_connected_buses = np.unique(
                    np.concatenate(
                        (sub_a_maxV_bus_series_to, sub_a_maxV_bus_series_from)
                    )
                )
                external_bus_to_hv_bus = list(all_connected_buses)

            # If the max bus is not available in the filtered substation or not connected to any other bus
            else:
                sub_a_maxV_bus = f"{substation}_{int(max_voltage_substation)}"

            # ...............................................
            # Make assumptions of transformer characteristics
            # In some generating stations, two of the generators are connected to one transformer
            # In some generations, power might be exported in two voltage levels
            # ...............................................

            # If the lines have single voltage rating, the gic doesn't flow in the secondary (assign a D-Wye)
            if len(substation_buses_unique) == 1:
                low_voltage_bus = sub_a_maxV_bus + "lv"
                transformer_type = "GY-D"

            else:
                # If lines are multiple rated, could be three windings (Gy-Gy-D), Gy-Gy or Auto transformer if closely rated
                low_voltage_bus = f"{substation}_{int(substation_buses_unique[1])}"

                # If only two unique voltage ratings, check for their ratios
                # Most def an auto if they are close
                if len(substation_buses_unique) == 2:
                    transformer_type = (
                        "GY-GY"
                        if np.max(sub_connected_lines) / np.min(sub_connected_lines) > 2
                        else "Auto"
                    )

                # If three unique voltage ratings, could be a three winding transformer
                elif len(substation_buses_unique) == 3:
                    transformer_type = "GY-GY-D"
                # If multiple voltage ratings, we are interested in HV buses (ignore others)
                elif len(substation_buses_unique) > 3:
                    # Filter those with greater than 200, else assign Gy-D
                    filtered_sub_bus_unique = substation_buses_unique[
                        substation_buses_unique >= 200
                    ]

                    if len(filtered_sub_bus_unique) == 1:
                        transformer_type = "GY-GY"
                    elif len(filtered_sub_bus_unique) == 2:
                        transformer_type = (
                            "GY-GY"
                            if np.max(sub_connected_lines) / np.min(sub_connected_lines)
                            > 2
                            else "Auto"
                        )
                    elif len(filtered_sub_bus_unique) == 3:
                        transformer_type = "GY-GY-D"
                    else:
                        transformer_type = "Unknown"

                else:
                    transformer_type = "Unknown"

            # Substation information
            substation_info = {
                "SS_ID": substation,
                "buses": [sub_a_maxV_bus, low_voltage_bus],
                "hv_bus": sub_a_maxV_bus,
                "lv_bus": low_voltage_bus,
                "HV_voltage": max_voltage_substation,
                "LV_voltage": (
                    int(substation_buses_unique[1])
                    if len(substation_buses_unique) > 1
                    else 0
                ),
                "Transformer_type": transformer_type,
                "external_bus_to_hv_bus": external_bus_to_hv_bus,
                "external_bus_to_lv_bus": [],
            }

            # ......................................................
            # If the transformer type not a gsu, GY-D or Tee, get external buses connected to the low voltage bus
            # Interested in buses > 200 v. if no lines, then ignore
            # ......................................................

            # Get other buses in substations except max_buses
            if transformer_type not in ["GY-D", "Tee", "GSU"]:
                lv_bus_v = int(substation_buses_unique[1])
                lv_bus_id = f"{substation}_{lv_bus_v}"
                if lv_bus_v >= 200:
                    # Find sub bus connected to_bus_ids and from_bus_ids
                    sub_bus_series_to = df_lines_EHV.query("from_bus_id == @lv_bus_id")[
                        "to_bus_id"
                    ].values
                    sub_bus_series_from = df_lines_EHV.query("to_bus_id == @lv_bus_id")[
                        "from_bus_id"
                    ].values

                    # Combine and get unique bus ids
                    sub_bus_connected_buses = np.unique(
                        np.concatenate((sub_bus_series_to, sub_bus_series_from))
                    )

                    substation_info["external_bus_to_lv_bus"] = list(
                        sub_bus_connected_buses
                    )

            substation_buses[substation] = substation_info

    logger.info("Substation buses built.")

    return substation_buses


def flatten_substation_dict(data, df_lines_EHV, buses):
    """
    Flatten the substation dictionary and generate records for building an admittance matrix.
    Parameters:
        data (dict): A dictionary containing substation details.
    Returns:
        list: A list of records for building an admittance matrix.
        
    """
    # Format lines into a format to build an admittance matrix
    records = []

    logger.info("Flattening substation dictionary...")

    for substation, details in data.items():
        hv_bus = details["hv_bus"]
        lv_bus = details["lv_bus"]

        # Connections from HV bus to external buses
        for ext_bus in details["external_bus_to_hv_bus"]:
            line_connections = df_lines_EHV[
                (df_lines_EHV["from_bus_id"] == hv_bus)
                & (df_lines_EHV["to_bus_id"] == ext_bus)
                | (df_lines_EHV["from_bus_id"] == ext_bus)
                & (df_lines_EHV["to_bus_id"] == hv_bus)
            ][["LINE_ID", "from_bus_id", "to_bus_id"]].values.tolist()
            for line in line_connections:
                LINE_ID = line[0]
                sub1 = line[1]
                sub2 = line[2]

                if sub1 in buses and sub2 in buses:
                    records.append((hv_bus, ext_bus, "line", LINE_ID))

        # Connections from LV buses to external buses
        for ext_bus_2_lv_bus in details["external_bus_to_lv_bus"]:
            line_connections = df_lines_EHV[
                (df_lines_EHV["from_bus_id"] == lv_bus)
                & (df_lines_EHV["to_bus_id"] == ext_bus_2_lv_bus)
                | (df_lines_EHV["from_bus_id"] == ext_bus_2_lv_bus)
                & (df_lines_EHV["to_bus_id"] == lv_bus)
            ][["LINE_ID", "from_bus_id", "to_bus_id"]].values.tolist()

            for line in line_connections:
                LINE_ID = line[0]
                sub1 = line[1]
                sub2 = line[2]

                if sub1 in buses and sub2 in buses:
                    records.append((lv_bus, ext_bus_2_lv_bus, "line", LINE_ID))
    return records


def calculate_line_resistances(df, df_ehv, line_resistance, 
    trans_lines_within_FERC_filtered_, bus_ids_map):
    """
    Calculate line resistances for each line in the dataframe.
    Parameters:
    - df (pandas.DataFrame): The input dataframe containing line information.
    - df_ehv (pandas.DataFrame): The input dataframe containing EHV line information.
    - line_resistance (dict): A dictionary mapping voltage levels to line resistances.
    Returns:
    - df (pandas.DataFrame): The modified dataframe with calculated line resistances.
    Raises:
    - None
    Example:
    df = pd.DataFrame(...)
    df_ehv = pd.DataFrame(...)
    line_resistance = {110: 0.1, 220: 0.2, 400: 0.3}
    result = calculate_line_resistances(df, df_ehv, line_resistance)
    """
    # Remove duplicates
    df.drop_duplicates(subset=["name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Map substation IDs to unique numbers
    df["from_bus"] = df["from_bus_id"].map(bus_ids_map)
    df["to_bus"] = df["to_bus_id"].map(bus_ids_map)

    # Merge with the lines to get the line length and Votage rating
    df = df.merge(
        df_ehv[["LINE_ID", "length", "VOLTAGE"]],
        left_on="name",
        right_on="LINE_ID",
        how="left",
    )

    # Merge df_lines_unique with transmission lines to get geometries for plotting
    df = df.merge(
        trans_lines_within_FERC_filtered_[["LINE_ID", "geometry"]],
        left_on="name",
        right_on="LINE_ID",
        how="left",
    )

    # Get the useful buses
    df = df[["name", "from_bus", "to_bus", "length", "VOLTAGE", "geometry"]]

    # Rename Voltage to V
    df.rename(columns={"VOLTAGE": "V"}, inplace=True)

    # Increase length by 3%
    df["length"] = df["length"] * 1.03

    # Apply line resistance to get R per km
    df["R_per_km"] = df["V"].map(line_resistance)

    # Get R for the entire length
    df["R"] = df["length"] * df["R_per_km"]

    df.drop_duplicates(subset=["name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_transformer_samples(
    substation_buses,
    sample_net_name="sample_network.pkl",
    n_samples=2000,
    seed=42,
    ):
    """
    Generate multiple samples of network configurations:
    - Each substation gets assigned a transformer count
    - Each transformer count gets assigned types
    - Returns a list of DataFrames, each with one sample configuration per substation.
    """

    # Define file path
    data_path = DATA_LOC / sample_net_name

    # Check if pre-generated samples exist
    if os.path.exists(data_path):  # Changed from 'not os.path.exists'
        with open(data_path, "rb") as f:
            all_samples = pickle.load(f)
            logger.info(
                f"Loaded {len(all_samples)} pre-generated samples from {data_path}"
            )
    else:
        logger.info(f"Generating {n_samples} new samples...")
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


def random_admittance_matrix(substation_buses, df_transformers, bus_ids_map,
    sub_look_up, df_lines, substations_df, evan_data=False,
    transformer_counts_dict=None):
    # ..................................................
    # Create a dictionary to store the resistance values
    # Only HV distribution and transmission substations are taken into consideration
    # We consider a Delta-Wye architecture for substations with a single bus
    # Although, a dummy LV bus is introduced where GIC flow is zero
    # ..................................................
    transformer_winding_resistances = {
        "GY-GY-D": {"pri": 0.2, "sec": 0.1},
        "GY-GY": {"pri": 0.04, "sec": 0.06},
        "Auto": {"pri": 0.04, "sec": 0.06},
        "GSU": {"pri": 0.15, "sec": float("inf")},
        "Tee": {"pri": 0.01, "sec": 0.01},
        "GY-D": {"pri": 0.05, "sec": 0.1},
    }

    if evan_data:
        transformers_data = get_transformer_data_evan(
            substation_buses, transformer_counts_dict
        )
        # Create a transformer df
        df_transformers = pd.DataFrame(transformers_data)

    # Transformer W1 and W2
    df_transformers["W1"] = df_transformers["type"].apply(
        lambda x: transformer_winding_resistances[x]["pri"]
    )
    df_transformers["W2"] = df_transformers["type"].apply(
        lambda x: transformer_winding_resistances[x]["sec"]
    )

    # Apply bus_id map
    df_transformers["bus1"] = df_transformers["bus1_id"].map(bus_ids_map)
    df_transformers["bus2"] = df_transformers["bus2_id"].map(bus_ids_map)

    # Merge transformers with lat and lon data
    df_transformers = df_transformers.merge(
        substations_df[["name", "latitude", "longitude"]],
        left_on="sub_id",
        right_on="name",
        how="left",
    )

    # Rename name_x to name
    df_transformers.rename(columns={"name_x": "name"}, inplace=True)

    # Drop name_y
    df_transformers.drop("name_y", axis=1, inplace=True)

    # Useful cols
    df_transformers = df_transformers[
        ["sub_id", "name", "type", "bus1", "bus2", "W1", "W2", "latitude", "longitude"]
    ]

    sub_ref = dict(zip(substations_df.name, substations_df.buses))

    # Neutral points in a trafor
    df_transformers["sub"] = df_transformers.bus1.apply(
        lambda x: find_substation_name(x, sub_ref)
    )
    df_transformers["neutral_point"] = df_transformers["sub"].apply(
        lambda x: sub_look_up.get(x, None)
    )

    # Build network admittance
    Y = network_admittance(sub_look_up, sub_ref, df_transformers, df_lines)
    Y_e = earthing_impedance(sub_look_up, substations_df)

    return Y, Y_e, df_transformers


def get_transformer_data_evan(substation_buses, transformer_counts_dict):

    # Get transformer data using Evan's verified data
    logger.info("Getting transformer data using Evan's verified data...")

    transformer_types = ["GY-D", "GY-GY", "GY-GY-D", "Auto"]

    # Tranformer generator number
    transformer_gen_num = 0
    transformers_data = []
    count = 0
    for substation, values in substation_buses.items():

        tf_count = transformer_counts_dict.get(substation, 1)

        # Increment transformer generator number
        tf_nos = []
        for _ in range(min(tf_count, 3)):
            transformer_gen_num += 1
            transformer_number = "T" + str(transformer_gen_num)

            transformer_data = {
                "sub_id": substation,
                "name": transformer_number,
                "type": values["Transformer_type"],
                "bus1_id": values["hv_bus"],
                "bus2_id": values["lv_bus"],
            }

            transformers_data.append(transformer_data)
            tf_nos.append(transformer_number)

    logger.info("Transformer data generated.")
    return transformers_data

# --------------------------------Build Admittance Matrix--------------------------------
# We shall us LPm formulation to build the admittance matrix
# The matrix is made of the network admittance and earthing impedances
# The method is verified with Horton et al., 2013 GIC test case
def find_substation_name(bus, sub_ref):
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name
    return None


# Build Y^n
def add_admittance(Y, from_bus, to_bus, admittance):
    """
    Add admittance to the Y matrix.
    Parameters:
    - Y (numpy.ndarray): The admittance matrix.
    - from_bus (int): The index of the "from" bus.
    - to_bus (int): The index of the "to" bus.
    - admittance (float): The admittance value to be added.
    Returns:
    - None
    Notes:
    - This function modifies the admittance matrix in-place by adding the given admittance value to the appropriate elements.
    - The admittance value is added to the diagonal elements corresponding to the "from" and "to" buses.
    - If the "from" bus is different from the "to" bus, the admittance value is also subtracted from the off-diagonal elements.
    """

    i, j = from_bus, to_bus
    Y[i, i] += admittance
    if i != j:
        Y[j, j] += admittance
        Y[i, j] -= admittance
        Y[j, i] -= admittance


def add_admittance_auto(Y, from_bus, to_bus, neutral_bus, Y_series, Y_common):
    """
    Add admittance values to the admittance matrix if an Auto transformer.
    Parameters:
    - Y (numpy.ndarray): The admittance matrix.
    - from_bus (int): The index of the "from" bus.
    - to_bus (int): The index of the "to" bus.
    - neutral_bus (int): The index of the neutral bus.
    - Y_series (float): The admittance value for the series connection.
    - Y_common (float): The admittance value for the common connection.
    Returns:
    None
    """

    i, j, k = to_bus, from_bus, neutral_bus
    add_admittance(Y, from_bus, neutral_bus, Y_common)
    add_admittance(Y, from_bus, to_bus, Y_series)

    Y[i, i] += Y_common
    Y[i, i] += Y_series
    Y[j, j] += Y_series
    Y[i, j] -= Y_series
    Y[j, i] -= Y_series


def network_admittance(sub_look_up, sub_ref, df_transformers, df_lines):
    """
    Calculates the network admittance matrix.

    Parameters:
    - sub_look_up (dict): A dictionary mapping bus names to their corresponding indices.
    - sub_ref (str): References to the substations.
    - df_transformers (DataFrame): A DataFrame containing information about transformers.
    - df_lines (DataFrame): A DataFrame containing information about transmission lines.

    Returns:
    - Y (ndarray): The network admittance matrix.

    This function calculates the network admittance matrix based on the given bus lookup dictionary,
    substation name, transformer DataFrame, and transmission line DataFrame. The admittance matrix
    represents the conductance and susceptance of the network elements, including transformers and
    transmission lines. The matrix is returned as a numpy ndarray. The matrix is sparse and positive semi definite
    """
    # Function implementation goes here
    pass

    # Number of unique nodes (buses + neutral points)
    n_nodes = len(sub_look_up)

    # Initialize the admittance matrix Y
    Y = np.zeros((n_nodes, n_nodes))

    phases = 1

    # Process transformers and build admittance matrix
    for bus, bus_idx in sub_look_up.items():
        sub = find_substation_name(bus, sub_ref)

        # Filter transformers for current bus
        trafos = df_transformers[(df_transformers["bus1"] == bus)]

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

            trafo_type = trafo["type"]
            bus1_idx = sub_look_up[bus1]
            neutral_idx = (
                sub_look_up[neutral_point] if neutral_point in sub_look_up else None
            )
            bus2_idx = sub_look_up[bus2]

            if trafo_type == "GSU":
                Y_w1 = 1 / W1  # Primary winding admittance
                add_admittance(Y, bus1_idx, neutral_idx, Y_w1)

            elif trafo_type == "Tee":
                # Y_pri = phases / (row['W1'])
                # Y_sec = phases / (row['W2'])
                # add_admittance(Y, bus_n, np_bus, Y_pri)
                # add_admittance(Y, bus_k, np_bus, Y_sec)
                continue

            elif trafo_type == "GSU w/ GIC BD":
                Y_w1 = 1 / W1  # Primary winding admittance
                add_admittance(Y, bus1_idx, neutral_idx, Y_w1)

            elif trafo_type == "Auto":
                Y_series = 1 / W1
                Y_common = 1 / W2
                add_admittance(Y, bus2_idx, bus1_idx, Y_series)
                add_admittance(Y, bus2_idx, neutral_idx, Y_common)

            elif trafo_type in ["GY-GY-D", "GY-GY"]:
                Y_primary = 1 / W1
                Y_secondary = 1 / W2
                add_admittance(Y, bus1_idx, neutral_idx, Y_primary)
                add_admittance(Y, bus2_idx, neutral_idx, Y_secondary)

    # Add transmission line admittances
    for i, line in df_lines.iterrows():
        Y_line = phases / line["R"]
        bus_n = sub_look_up[line["from_bus"]]
        bus_k = sub_look_up[line["to_bus"]]
        add_admittance(Y, bus_n, bus_k, Y_line)

    return Y


def earthing_impedance(sub_look_up, substations_df):
    """
    Calculate the earthing impedance matrix.

    Parameters:
    - sub_look_up (dict): A dictionary containing the mapping of substation names to indices.
    - substations_df (pandas.DataFrame): A DataFrame containing information about substations.

    Returns:
    - Y_e (numpy.ndarray): The earthing impedance matrix.

    Notes:
    - The earthing impedance matrix is a square matrix with dimensions equal to the number of unique nodes (buses + neutral points).
    - The diagonal elements of the matrix represent the admittance per phase of each substation's grounding resistance.
    - The off-diagonal elements of the matrix are zero.
    """
    # Code implementation goes here
    pass
    # Number of unique nodes (buses + neutral points)
    n_nodes = len(sub_look_up)
    # Build earthing impedance Y^e
    Y_e = np.zeros((n_nodes, n_nodes))

    for i, row_sub in substations_df.iterrows():
        sub = row_sub["name"]

        # Get index in look up table
        index = sub_look_up.get(sub, None)

        if index is None or sub == "Substation 1":
            continue

        Rg = row_sub.grounding_resistance
        Y_rg = 1 / (3 * Rg)  # Divide by 3 to get admittance per phase

        Y_e[index, index] = Y_rg

    return Y_e


if __name__ == "__main__":

    admittances = []
    # Get substation buses
    substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df = (
        process_substation_buses(
            DATA_LOC,
        )
    )

    df_transformers = get_transformer_samples(substation_buses)

    for i, df_transformer in enumerate(df_transformers):
        logger.info(f"Building admittance matrix {i + 1}...")
        Y, Y_e = random_admittance_matrix(
            substation_buses, df_transformer, bus_ids_map, sub_look_up, df_lines, substations_df,
        )
        admittances.append(Y)
        break
