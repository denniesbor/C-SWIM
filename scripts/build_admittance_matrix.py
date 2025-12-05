# %%
"""
Build US EHV admittance matrix.
Ensure preprocessing scripts have been run first.
Authors: Dennies and Ed
"""
import os
import pickle
from random import seed

import numpy as np
import pandas as pd

from configs import setup_logger, get_data_dir, P_TRAFO_BD

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/build_adm_matrix.log")

# Seed base
seed_base = 42  # For reproducibility


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

    substation_buses_pkl = DATA_LOC / "admittance_matrix" / "substation_buses.pkl"
    bus_ids_map_pkl = DATA_LOC / "admittance_matrix" / "bus_ids_map.pkl"
    sub_look_up_pkl = DATA_LOC / "admittance_matrix" / "sub_look_up.pkl"
    transmission_lines_csv = DATA_LOC / "admittance_matrix" / "transmission_lines.csv"
    substation_info_csv = DATA_LOC / "admittance_matrix" / "substation_info.csv"

    if (
        os.path.exists(substation_buses_pkl)
        and os.path.exists(bus_ids_map_pkl)
        and os.path.exists(sub_look_up_pkl)
    ):
        with open(substation_buses_pkl, "rb") as f:
            substation_buses = pickle.load(f)

        with open(bus_ids_map_pkl, "rb") as f:
            bus_ids_map = pickle.load(f)

        with open(sub_look_up_pkl, "rb") as f:
            sub_look_up = pickle.load(f)

        df_lines = pd.read_csv(transmission_lines_csv)
        substations_df = pd.read_csv(substation_info_csv)

        substations_df["buses"] = substations_df["buses"].apply(eval)

    else:
        if evan_data:
            substation_buses = []
        else:
            substation_buses = build_substation_buses(
                unique_sub_voltage_pairs,
                df_lines_EHV,
                ss_type_dict,
                substation_to_line_voltages,
            )

        buses = []
        for sub_info in substation_buses.values():
            buses.extend(sub_info["buses"])
        buses = np.unique(buses)

        bus_ids_map = {bus: i + 1 for i, bus in enumerate(buses)}

        # Substation grounding resistances range from 0.1 -> 0.2 (Horton, et al., 2013)
        grounding_resistances = [0.1, 0.2]

        df_substations_info = pd.DataFrame(substation_buses).T.reset_index()
        df_substations_info = df_substations_info[
            ["SS_ID", "Transformer_type", "buses"]
        ]

        df_substations_info["buses"] = df_substations_info["buses"].apply(
            lambda x: [bus_ids_map[bus] for bus in x]
        )

        df_substations_info = df_substations_info.merge(
            ss_df[["SS_ID", "lat", "lon"]], on="SS_ID", how="left"
        )
        df_substations_info["grounding_resistance"] = grounding_resistances[1]

        df_substations_info.rename(
            columns={"SS_ID": "name", "lat": "latitude", "lon": "longitude"},
            inplace=True,
        )

        flattened_data = flatten_substation_dict(substation_buses, df_lines_EHV, buses)

        df_lines_final = pd.DataFrame(
            flattened_data, columns=["from_bus_id", "to_bus_id", "utility", "name"]
        )

        df_lines_EHV[~(df_lines_EHV.VOLTAGE.isin([345, 230, 765, 500]))].shape

        # Line resistance approximations (Ω/km)
        line_resistance = {
            765: 0.01,
            500: 0.0141,
            345: 0.0283,
            232: 0.0450,
            230: 0.0500,
            220: 0.0700,
            161: 0.0800,
            138: 0.0900,
            69: 0.1200,
        }

        substations_df = df_substations_info.copy()

        df_lines = calculate_line_resistances(
            df_lines_final,
            df_lines_EHV,
            line_resistance,
            trans_lines_within_FERC_filtered,
            bus_ids_map,
        )

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

        with open(substation_buses_pkl, "wb") as f:
            pickle.dump(substation_buses, f)

        with open(bus_ids_map_pkl, "wb") as f:
            pickle.dump(bus_ids_map, f)

        with open(sub_look_up_pkl, "wb") as f:
            pickle.dump(sub_look_up, f)

        df_lines.to_csv(transmission_lines_csv, index=False)
        df_substations_info.to_csv(substation_info_csv, index=False)

    return substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df


def load_and_process_data(DATA_LOC):
    """Load and process power system data files."""
    logger.info("Loading and processing data...")

    with open(DATA_LOC / "grid_processed" / "unique_sub_voltage_pairs.pkl", "rb") as f:
        unique_sub_voltage_pairs = pickle.load(f)

    with open(DATA_LOC / "grid_processed" / "df_lines_EHV.pkl", "rb") as f:
        df_lines_EHV = pickle.load(f)

    folder = DATA_LOC / "grid_processed"

    with open(folder / "trans_lines_pickle.pkl", "rb") as f:
        trans_lines_gdf = pickle.load(f)
        trans_lines_gdf.rename(columns={"line_id": "LINE_ID"}, inplace=True)

    with open(folder / "substation_to_line_voltages.pkl", "rb") as f:
        substation_to_line_voltages = pickle.load(f)

    with open(folder / "ss_df.pkl", "rb") as f:
        ss_df = pickle.load(f)

    with open(folder / "trans_lines_within_FERC_filtered.pkl", "rb") as f:
        trans_lines_within_FERC_filtered = pickle.load(f)

    df_lines_EHV = df_lines_EHV[df_lines_EHV["LINE_ID"].isin(trans_lines_gdf.LINE_ID)]
    ss_type_dict = dict(zip(ss_df["SS_ID"], ss_df["SS_TYPE"]))

    logger.info(f"DF lines EHV shape: {df_lines_EHV.shape}")

    grid_mapping = pd.read_csv(DATA_LOC / "grid_mapping.csv")
    grid_mapping["Attributes"] = grid_mapping["Attributes"].apply(eval)

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
    """Build substation buses based on voltage pairs and EHV lines."""

    logger.info("Building substation buses...")
    substation_buses = {}

    # Design low/high voltage buses using spatially intersected transmission lines and substations
    # This approach allows for estimation of injection currents -- advised by CT Gaunt
    for i, row in unique_sub_voltage_pairs.iterrows():
        substation = row.substation
        line_voltage = row.voltage
        ss_type = ss_type_dict[substation]

        if substation not in substation_buses:

            # Focus on distribution and transmission substations
            if ss_type not in ["distribution", "transmission"]:
                pass  # Do nothing for now

            sub_connected_lines = np.array(
                [float(v) for v in substation_to_line_voltages[substation]]
            )
            substation_buses_unique = np.unique(np.sort(sub_connected_lines))[::-1]

            max_voltage_substation = np.max(substation_buses_unique)

            sub_a_maxV_bus_series = unique_sub_voltage_pairs.query(
                "substation == @substation and voltage == @max_voltage_substation"
            )["bus_id"]

            external_bus_to_hv_bus = []
            # If multiple lines intersect at substation
            if len(sub_a_maxV_bus_series.values) >= 1:
                sub_a_maxV_bus = sub_a_maxV_bus_series.values[0]

                sub_a_maxV_bus_series_to = df_lines_EHV.query(
                    "from_bus_id == @sub_a_maxV_bus"
                )["to_bus_id"].values
                sub_a_maxV_bus_series_from = df_lines_EHV.query(
                    "to_bus_id == @sub_a_maxV_bus"
                )["from_bus_id"].values

                all_connected_buses = np.unique(
                    np.concatenate(
                        (sub_a_maxV_bus_series_to, sub_a_maxV_bus_series_from)
                    )
                )
                external_bus_to_hv_bus = list(all_connected_buses)

            else:
                sub_a_maxV_bus = f"{substation}_{int(max_voltage_substation)}"

            # Transformer characteristics assumptions:
            # - Generating stations may have two generators per transformer
            # - Some generation plants export power at multiple voltage levels

            # Single voltage rating means GIC doesn't flow in secondary (assign D-Wye)
            if len(substation_buses_unique) == 1:
                low_voltage_bus = sub_a_maxV_bus + "lv"
                transformer_type = "GY-D"

            else:
                # Multiple ratings could be three windings (Gy-Gy-D), Gy-Gy or Auto if closely rated
                low_voltage_bus = f"{substation}_{int(substation_buses_unique[1])}"

                # Two unique voltage ratings - check ratios
                if len(substation_buses_unique) == 2:
                    transformer_type = (
                        "GY-GY"
                        if np.max(sub_connected_lines) / np.min(sub_connected_lines) > 2
                        else "Auto"
                    )

                elif len(substation_buses_unique) == 3:
                    transformer_type = "GY-GY-D"
                # Multiple voltage ratings - focus on HV buses only
                elif len(substation_buses_unique) > 3:
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

            # Get external buses connected to low voltage bus for non-GSU, GY-D or Tee transformers
            # Only interested in buses >= 150V
            if transformer_type not in ["GY-D", "Tee", "GSU"]:
                lv_bus_v = int(substation_buses_unique[1])
                lv_bus_id = f"{substation}_{lv_bus_v}"
                if lv_bus_v >= 150:
                    sub_bus_series_to = df_lines_EHV.query("from_bus_id == @lv_bus_id")[
                        "to_bus_id"
                    ].values
                    sub_bus_series_from = df_lines_EHV.query("to_bus_id == @lv_bus_id")[
                        "from_bus_id"
                    ].values

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
    """Flatten substation dictionary to generate admittance matrix records."""
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


def calculate_line_resistances(
    df, df_ehv, line_resistance, trans_lines_within_FERC_filtered_, bus_ids_map
):
    """Calculate line resistances for admittance matrix."""
    df.drop_duplicates(subset=["name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["from_bus"] = df["from_bus_id"].map(bus_ids_map)
    df["to_bus"] = df["to_bus_id"].map(bus_ids_map)

    df = df.merge(
        df_ehv[["LINE_ID", "length", "VOLTAGE"]],
        left_on="name",
        right_on="LINE_ID",
        how="left",
    )

    df = df.merge(
        trans_lines_within_FERC_filtered_[["LINE_ID", "geometry"]],
        left_on="name",
        right_on="LINE_ID",
        how="left",
    )

    df = df[["name", "from_bus", "to_bus", "length", "VOLTAGE", "geometry"]]

    df.rename(columns={"VOLTAGE": "V"}, inplace=True)

    # Increase length by 3%
    df["length"] = df["length"] * 1.03

    df["R_per_km"] = df["V"].map(line_resistance)

    df["R"] = df["length"] * df["R_per_km"]

    df.drop_duplicates(subset=["name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_transformer_samples(
    substation_buses,
    sample_net_name="admittance_matrix/sample_network.pkl",
    n_samples=3000,
    seed=42,
    new_trafos=True,
):
    """Generate Monte Carlo samples of substation-transformer compositions (no BD)."""
    data_path = DATA_LOC / sample_net_name

    if (not new_trafos) and os.path.exists(data_path):
        with open(data_path, "rb") as f:
            all_samples = pickle.load(f)
            logger.info(f"Loaded {len(all_samples)} samples")
        return all_samples

    logger.info(f"Generating {n_samples} samples...")
    np.random.seed(seed)

    transformer_types_real = ["GY-GY", "GY-GY-D", "Auto"]
    transformer_types_artificial = ["GY-D"]

    all_samples = []
    for i in range(n_samples):
        transformer_gen_num = 0
        transformers_data = []

        for sub_id, values in substation_buses.items():
            lv_bus = values["lv_bus"]
            is_artificial = lv_bus.endswith("_lv") or lv_bus.endswith("lv")

            if is_artificial:
                trafo_count = np.random.randint(1, 3)  # 1 or 2
                available_types = transformer_types_artificial
                weights = [1.0]
            else:
                trafo_count = np.random.randint(1, 4)  # 1 to 3
                available_types = transformer_types_real
                weights = [1 / 3, 1 / 3, 1 / 3]

            selected_types = np.random.choice(
                available_types, size=trafo_count, replace=True, p=weights
            )

            for transformer_type in selected_types:
                transformer_gen_num += 1
                transformers_data.append(
                    {
                        "sub_id": sub_id,
                        "name": f"T_{sub_id}_{transformer_gen_num}",
                        "type": transformer_type,
                        "bus1_id": values["hv_bus"],
                        "bus2_id": values["lv_bus"],
                    }
                )

        all_samples.append(pd.DataFrame(transformers_data))

    os.makedirs(data_path.parent, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(all_samples, f)
    logger.info(f"Saved {len(all_samples)} samples")
    return all_samples


def random_admittance_matrix(
    substation_buses,
    df_transformers,
    bus_ids_map,
    sub_look_up,
    df_lines,
    substations_df,
    evan_data=False,
    transformer_counts_dict=None,
):

    # Resistance values for different transformer types
    transformer_winding_resistances = {
        "GY-GY-D": {"pri": 0.2, "sec": 0.1},
        "GY-GY": {"pri": 0.04, "sec": 0.06},
        "Auto": {"pri": 0.04, "sec": 0.06},
        "GSU": {"pri": 0.15, "sec": float("inf")},
        "GSU w/ GIC BD": {"pri": 0.1, "sec": float("inf")},  # neutral DC-blocked
        "Tee": {"pri": 0.01, "sec": 0.01},
        "GY-D": {"pri": 0.05, "sec": float("inf")},
    }

    if evan_data:
        transformers_data = get_transformer_data_evan(
            substation_buses, transformer_counts_dict
        )
        df_transformers = pd.DataFrame(transformers_data)

    df_transformers["W1"] = df_transformers["type"].apply(
        lambda x: transformer_winding_resistances[x]["pri"]
    )
    df_transformers["W2"] = df_transformers["type"].apply(
        lambda x: transformer_winding_resistances[x]["sec"]
    )

    df_transformers["bus1"] = df_transformers["bus1_id"].map(bus_ids_map)
    df_transformers["bus2"] = df_transformers["bus2_id"].map(bus_ids_map)

    substations_df["name"] = substations_df["name"].astype(str)

    df_transformers = df_transformers.merge(
        substations_df[["name", "latitude", "longitude"]],
        left_on="sub_id",
        right_on="name",
        how="left",
    )

    df_transformers.rename(columns={"name_x": "name"}, inplace=True)

    df_transformers.drop("name_y", axis=1, inplace=True)

    df_transformers = df_transformers[
        ["sub_id", "name", "type", "bus1", "bus2", "W1", "W2", "latitude", "longitude"]
    ]

    sub_ref = dict(zip(substations_df.name, substations_df.buses))

    df_transformers["sub"] = df_transformers.bus1.apply(
        lambda x: find_substation_name(x, sub_ref)
    )
    df_transformers["neutral_point"] = df_transformers["sub"].apply(
        lambda x: sub_look_up.get(x, None)
    )

    Y = network_admittance(sub_look_up, sub_ref, df_transformers, df_lines)
    Y_e = earthing_impedance(sub_look_up, substations_df)

    return Y, Y_e, df_transformers


def get_transformer_data_evan(substation_buses, transformer_counts_dict):
    """Generate transformer data using Evan's verified data."""

    logger.info("Getting transformer data using Evan's verified data...")

    transformer_types = ["GY-D", "GY-GY", "GY-GY-D", "Auto"]

    transformer_gen_num = 0
    transformers_data = []
    count = 0
    for substation, values in substation_buses.items():

        tf_count = transformer_counts_dict.get(substation, 1)

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


def find_substation_name(bus, sub_ref):
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name
    return None


def add_admittance(Y, from_bus, to_bus, admittance):
    """Add admittance to the Y matrix."""

    i, j = from_bus, to_bus
    Y[i, i] += admittance
    if i != j:
        Y[j, j] += admittance
        Y[i, j] -= admittance
        Y[j, i] -= admittance


def add_admittance_auto(Y, from_bus, to_bus, neutral_bus, Y_series, Y_common):
    """Add admittance values for auto transformers."""

    i, j, k = to_bus, from_bus, neutral_bus
    add_admittance(Y, from_bus, neutral_bus, Y_common)
    add_admittance(Y, from_bus, to_bus, Y_series)

    Y[i, i] += Y_common
    Y[i, i] += Y_series
    Y[j, j] += Y_series
    Y[i, j] -= Y_series
    Y[j, i] -= Y_series


def network_admittance(sub_look_up, sub_ref, df_transformers, df_lines):
    """Calculate network admittance matrix for transformers and transmission lines."""
    n_nodes = len(sub_look_up)

    Y = np.zeros((n_nodes, n_nodes))

    phases = 1

    for bus, bus_idx in sub_look_up.items():
        sub = find_substation_name(bus, sub_ref)

        trafos = df_transformers[(df_transformers["bus1"] == bus)]

        if len(trafos) == 0 or sub == "Substation 7":
            continue

        for _, trafo in trafos.iterrows():
            bus1 = trafo["bus1"]
            bus2 = trafo["bus2"]
            neutral_point = trafo["sub"]  # Neutral point node
            W1 = trafo["W1"]  # Winding 1 impedance
            W2 = trafo["W2"]  # Winding 2 impedance

            trafo_type = trafo["type"]
            bus1_idx = sub_look_up[bus1]
            neutral_idx = (
                sub_look_up[neutral_point] if neutral_point in sub_look_up else None
            )
            bus2_idx = sub_look_up[bus2]

            if trafo_type in ["GSU", "GY-D", "GSU w/ GIC BD"]:
                Y_w1 = 1 / W1  # Primary winding admittance
                add_admittance(Y, bus1_idx, neutral_idx, Y_w1)

            elif trafo_type == "Tee":
                continue

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

    for i, line in df_lines.iterrows():
        Y_line = phases / line["R"]
        bus_n = sub_look_up[line["from_bus"]]
        bus_k = sub_look_up[line["to_bus"]]
        add_admittance(Y, bus_n, bus_k, Y_line)

    return Y


def earthing_impedance(sub_look_up, substations_df):
    """Calculate earthing impedance matrix."""
    n_nodes = len(sub_look_up)
    Y_e = np.zeros((n_nodes, n_nodes))

    for i, row_sub in substations_df.iterrows():
        sub = row_sub["name"]
        index = sub_look_up.get(sub, None)

        if index is None:
            continue

        Rg = row_sub.grounding_resistance

        if np.isinf(Rg):
            Y_e[index, index] = 0.0
        else:
            Y_rg = 1 / (3 * Rg)
            Y_e[index, index] = Y_rg

    return Y_e


def randomize_grounding_resistance(
    df_substations_info,
    low=0.1,
    high=0.2,
    seed=None,
    skip=(),
    p_open=0.01,
):
    """Randomly assign grounding resistances; optionally set a p_open share to ∞."""
    rng = np.random.default_rng(seed)
    out = df_substations_info.copy()

    if "grounding_resistance" not in out.columns:
        out["grounding_resistance"] = np.nan
    out["name"] = out["name"].astype(str)

    # all substations eligible unless caller passes a non-empty `skip`
    mask = ~out["name"].isin(set(skip))
    n_eligible = int(mask.sum())
    if n_eligible == 0:
        return out

    out.loc[mask, "grounding_resistance"] = rng.uniform(low, high, size=n_eligible)

    k = int(np.floor(p_open * n_eligible))
    if k > 0:
        eligible_idx = np.flatnonzero(mask)
        open_idx = rng.choice(eligible_idx, size=k, replace=False)
        out.loc[open_idx, "grounding_resistance"] = np.inf

    return out


def apply_random_line_dc_blocking(df_lines, p_block=0.15, seed=None):
    """Randomly apply DC blocking to a fraction of transmission lines."""
    rng = np.random.default_rng(seed)
    out = df_lines.copy()
    if "dc_block" not in out.columns:
        out["dc_block"] = False

    sel = rng.random(len(out)) < float(p_block)
    idx = np.flatnonzero(sel).astype(np.int64)

    if idx.size:
        out.loc[idx, "dc_block"] = True
        out.loc[idx, "R"] = np.inf
        vcols = [c for c in out.columns if c.startswith("V_")]
        if vcols:
            out.loc[idx, vcols] = 0.0

    return out


if __name__ == "__main__":

    admittances = []
    substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df = (
        process_substation_buses(
            DATA_LOC,
        )
    )

    df_transformers = get_transformer_samples(substation_buses)

    for i, df_transformer in enumerate(df_transformers):
        logger.info(f"Building admittance matrix {i + 1}...")

        Y, Y_e, df_transformers = random_admittance_matrix(
            substation_buses,
            df_transformer,
            bus_ids_map,
            sub_look_up,
            df_lines,
            df_transformer,
        )
        logger.info(f"Max value in Y: {np.max(Y)}")
        logger.info(f"Min value in Y: {np.min(Y)}")
        admittances.append(Y)
        logger.info(f"Admittance matrix of shape {Y.shape} built for sample {i + 1}.")
        break

# %%
