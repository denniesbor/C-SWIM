# %%
"""
GIC Test Suite - IEEE Benchmark Validation
Tests all GIC calculation modules against IEEE test case data.
Also performs sensitivity analysis via Monte Carlo variations of the grid configuration.
Author Dennies and Ed
"""
import sys
import os
import pickle
import copy
import math
import numpy as np
import pandas as pd
from collections import defaultdict

# sys.path.insert(0, "../")

from build_admittance_matrix import (
    network_admittance,
    earthing_impedance,
    find_substation_name,
)

from configs import get_data_dir

from est_gic import (
    nodal_voltage_calculation,
    calculate_GIC,
    calc_trafo_gic,
    get_injection_currents,
    can_use_float16,
    keep_nodes_with_ground,
    solve_total_nodal_gic,
)

DATA_LOC = get_data_dir()


# Utility functions
def haversine_distance_and_components(lat1, lon1, lat2, lon2):
    """Return distance and E-field components (Ex, Ey) for 1 V/km fields."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    dx = R * np.cos(lat1) * (lon2 - lon1)
    dy = R * (lat2 - lat1)
    return distance, dx, dy


def _ground_gic_table(
    sub_look_up,
    df_substations_info,
    non_zero_indices,
    n_nodes,
    Y_e_reduced,
    V_nodal,
    label,
):
    """
    Return DataFrame of ground GIC per substation for one scenario label.

    Notes:
      - Uses I_e = Y_e * V_n (per phase), then multiplies by 3 for total 3-phase ground current.
      - Ungrounded nodes (diagonal Y_e == 0) are exactly zero.
    """

    full = solve_total_nodal_gic(Y_e_reduced, V_nodal, non_zero_indices, n_nodes)

    # Build the table
    idx_series = df_substations_info["name"].map(sub_look_up)
    mask = idx_series.notna()
    valid_idx = idx_series[mask].astype(int).to_numpy()

    return pd.DataFrame(
        {
            "Substation": df_substations_info.loc[mask, "name"].to_numpy(),
            f"GIC_{label}": full[valid_idx],
        }
    )


# IEEE Test Case Data
large_vals = float("inf")  # Use 'inf' for open-circuit (DC-blocked) windings`
ORIGINAL_DATA = {
    # Transmission line data
    "lines": [
        {"name": "1", "from_bus": 2, "to_bus": 3, "R": 3.512, "V": 345},
        {"name": "2", "from_bus": 2, "to_bus": 17, "R": 3.525, "V": 345},
        {"name": "3", "from_bus": 15, "to_bus": 4, "R": 1.986, "V": 500},
        {"name": "4", "from_bus": 17, "to_bus": 16, "R": 4.665, "V": 345},
        {"name": "5", "from_bus": 4, "to_bus": 5, "R": 2.345, "V": 500},
        {"name": "6", "from_bus": 4, "to_bus": 5, "R": 2.345, "V": 500},
        {"name": "7", "from_bus": 5, "to_bus": 6, "R": 2.975, "V": 500},
        {"name": "8", "from_bus": 5, "to_bus": 11, "R": 3.509 + large_vals, "V": 500},
        {"name": "9", "from_bus": 6, "to_bus": 11, "R": 1.444, "V": 500},
        {"name": "10", "from_bus": 4, "to_bus": 6, "R": 4.666, "V": 500},
        {"name": "11", "from_bus": 15, "to_bus": 6, "R": 2.924, "V": 500},
        {"name": "12", "from_bus": 15, "to_bus": 6, "R": 2.924, "V": 500},
        {"name": "13", "from_bus": 11, "to_bus": 12, "R": 2.324, "V": 500},
        {"name": "14", "from_bus": 16, "to_bus": 20, "R": 4.049, "V": 345},
        {"name": "15", "from_bus": 17, "to_bus": 20, "R": 6.940, "V": 345},
    ],
    # Transformer data
    "transformers": [
        {
            "name": "T1",
            "type": "GSU w/ GIC BD",
            "W1": 0.1,
            "bus1": 2,
            "W2": large_vals,
            "bus2": 1,
        },
        {"name": "T2", "type": "GY-GY-D", "W1": 0.2, "bus1": 4, "W2": 0.1, "bus2": 3},
        {
            "name": "T3",
            "type": "GSU",
            "W1": 0.1,
            "bus1": 17,
            "W2": large_vals,
            "bus2": 18,
        },
        {
            "name": "T4",
            "type": "GSU",
            "W1": 0.1,
            "bus1": 17,
            "W2": large_vals,
            "bus2": 19,
        },
        {"name": "T5", "type": "Auto", "W1": 0.04, "bus1": 15, "W2": 0.06, "bus2": 16},
        {
            "name": "T6",
            "type": "GSU",
            "W1": 0.15,
            "bus1": 6,
            "W2": large_vals,
            "bus2": 7,
        },
        {
            "name": "T7",
            "type": "GSU",
            "W1": 0.15,
            "bus1": 6,
            "W2": large_vals,
            "bus2": 8,
        },
        {"name": "T8", "type": "GY-GY", "W1": 0.04, "bus1": 5, "W2": 0.06, "bus2": 20},
        {"name": "T9", "type": "GY-GY", "W1": 0.04, "bus1": 5, "W2": 0.06, "bus2": 20},
        {
            "name": "T10",
            "type": "GSU",
            "W1": 0.1,
            "bus1": 12,
            "W2": large_vals,
            "bus2": 13,
        },
        {
            "name": "T11",
            "type": "GSU",
            "W1": 0.1,
            "bus1": 12,
            "W2": large_vals,
            "bus2": 14,
        },
        {"name": "T12", "type": "Auto", "W1": 0.04, "bus1": 4, "W2": 0.06, "bus2": 3},
        {"name": "T13", "type": "GY-GY-D", "W1": 0.2, "bus1": 4, "W2": 0.1, "bus2": 3},
        {"name": "T14", "type": "Auto", "W1": 0.04, "bus1": 4, "W2": 0.06, "bus2": 3},
        {"name": "T15", "type": "Auto", "W1": 0.04, "bus1": 15, "W2": 0.06, "bus2": 16},
        # {"name": "T16", "type": "Tee", "W1":0.01, "bus1": 6, "W2":0.01, "bus2": 11}
    ],
    # Substation data
    "substations": {
        1: {
            "name": "Substation 1",
            "buses": [1, 2],
            "transformers": ["T1"],
            "grounding_resistance": float("inf"),
            "latitude": 33.6135,
            "longitude": -87.3737,
        },
        2: {
            "name": "Substation 2",
            "buses": [17, 18, 19],
            "transformers": ["T3", "T4"],
            "grounding_resistance": 0.2,
            "latitude": 34.3104,
            "longitude": -86.3658,
        },
        3: {
            "name": "Substation 3",
            "buses": [15, 16],
            "transformers": ["T15", "T5"],
            "grounding_resistance": 0.2,
            "latitude": 33.9551,
            "longitude": -84.6794,
        },
        4: {
            "name": "Substation 4",
            "buses": [3, 4],
            "transformers": ["T2", "T12", "T13", "T14"],
            "grounding_resistance": 1,
            "latitude": 33.5479,
            "longitude": -86.0746,
        },
        5: {
            "name": "Substation 5",
            "buses": [5, 20],
            "transformers": ["T8", "T9"],
            "grounding_resistance": 0.1,
            "latitude": 32.7051,
            "longitude": -84.6634,
        },
        6: {
            "name": "Substation 6",
            "buses": [6, 7, 8],
            "transformers": ["T6", "T7"],
            "grounding_resistance": 0.1,
            "latitude": 33.3773,
            "longitude": -82.6188,
        },
        7: {
            "name": "Substation 7",
            "buses": [11],
            "transformers": ["T16"],
            "grounding_resistance": 0.1,
            "latitude": 34.2522,
            "longitude": -82.8363,
        },
        8: {
            "name": "Substation 8",
            "buses": [12, 13, 14],
            "transformers": ["T10", "T11"],
            "grounding_resistance": 0.1,
            "latitude": 34.1956,
            "longitude": -81.098,
        },
    },
}


def setup_dataframes_and_network(system_data):
    """
    Initialize dataframes and create network topology mappings
    """
    df_lines = pd.DataFrame(system_data["lines"])
    df_transformers = pd.DataFrame(system_data["transformers"])
    substations_df = pd.DataFrame(system_data["substations"]).T

    # Create substation reference dictionary
    sub_ref = dict(zip(substations_df.name, substations_df.buses))

    return df_lines, df_transformers, substations_df, sub_ref


def calculate_line_electric_fields(df_lines, substations_df, sub_ref):
    """
    Calculate electric field components for transmission lines
    """
    # Map lines to origin/destination substations
    df_lines["origin_sub"] = df_lines.from_bus.apply(
        lambda x: find_substation_name(x, sub_ref)
    )
    df_lines["to_sub"] = df_lines.to_bus.apply(
        lambda x: find_substation_name(x, sub_ref)
    )

    # Get substation coordinates for distance calculations
    sub_coords = substations_df.set_index("name")[["latitude", "longitude"]]

    # Calculate distance and field components using existing utility
    df_lines[["distance", "Ex", "Ey"]] = df_lines.apply(
        lambda row: haversine_distance_and_components(
            sub_coords.loc[row.origin_sub, "latitude"],
            sub_coords.loc[row.origin_sub, "longitude"],
            sub_coords.loc[row.to_sub, "latitude"],
            sub_coords.loc[row.to_sub, "longitude"],
        ),
        axis=1,
        result_type="expand",
    )

    # Set voltage scenarios for 1 V/km fields
    df_lines["V_northward"] = df_lines["Ey"]
    df_lines["V_eastward"] = df_lines["Ex"]

    return df_lines


def build_substation_lookup(substations_df):
    """
    Create node indexing for admittance matrix
    """
    sub_look_up = {}
    index = 0

    # Map all buses first
    for _, row in substations_df.iterrows():
        for bus in sorted(row["buses"]):
            sub_look_up[bus] = index
            index += 1

    # Map substation names (neutral points) - skip Substation 7
    for _, row in substations_df.iterrows():
        if row["name"] != "Substation 7":
            sub_look_up[row["name"]] = index
            index += 1

    n_nodes = len(sub_look_up)
    return sub_look_up, n_nodes


def prepare_transformer_data(df_transformers, sub_ref, sub_look_up):
    """
    Add required transformer mappings for admittance calculation
    """
    df_transformers["sub"] = df_transformers.bus1.apply(
        lambda x: find_substation_name(x, sub_ref)
    )
    df_transformers["neutral_point"] = df_transformers["sub"].apply(
        lambda x: sub_look_up.get(x, None)
    )
    return df_transformers


def build_system_admittance(
    sub_look_up, sub_ref, df_transformers, df_lines, substations_df, dtype=np.float64
):
    """
    Construct network (Y_n) and earthing (Y_e) admittance matrices, identify active nodes,
    and return reduced forms for solving.
    """
    # Build admittances
    Y_n = network_admittance(sub_look_up, sub_ref, df_transformers, df_lines).astype(
        dtype, copy=False
    )
    Y_e = earthing_impedance(sub_look_up, substations_df).astype(dtype, copy=False)

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

    return Y_n, Y_e, Y_total, non_zero_indices


def solve_nodal_voltages(df_lines, Y_reduced, sub_look_up, non_zero_indices, n_nodes):
    """
    Calculate nodal voltages from injection currents
    """
    # Calculate injection currents for both scenarios
    injections_data = get_injection_currents(
        df_lines,
        n_nodes,
        non_zero_indices,
        sub_look_up,
        None,
        gannon_storm_only=False,
        gic_test_case=True,
    )

    # Solve for nodal voltages
    nodal_voltages = nodal_voltage_calculation(Y_reduced, injections_data)

    return nodal_voltages


def compute_line_gics(df_lines, nodal_voltages, sub_look_up, non_zero_indices, n_nodes):
    """
    Calculate GIC in transmission lines for all scenarios
    """
    line_tables = []

    # Map bus numbers to matrix indices for GIC calculation
    df_lines["from_bus"] = df_lines["from_bus"].apply(lambda x: sub_look_up.get(x))
    df_lines["to_bus"] = df_lines["to_bus"].apply(lambda x: sub_look_up.get(x))

    # Calculate line GICs for each field scenario
    for scenario in ["northward", "eastward"]:
        col = f"V_{scenario}"
        if col in nodal_voltages and nodal_voltages[col] is not None:
            V_nodal = nodal_voltages[col]
            df_gic = calculate_GIC(
                df_lines.copy(), V_nodal, col, non_zero_indices, n_nodes
            )

            # Extract results table
            line_tables.append(
                df_gic[["name", f"{col.split('_')[1]}_i_nk"]].rename(
                    columns={f"{col.split('_')[1]}_i_nk": f"Line_I_{scenario} [A]"}
                )
            )

    # Merge northward and eastward results
    df_line_results = line_tables[0].merge(line_tables[1], on="name", how="outer")
    df_line_results.rename(columns={"name": "Line"}, inplace=True)

    return df_line_results


def compute_transformer_gics(
    df_transformers, nodal_voltages, sub_look_up, sub_ref, non_zero_indices, n_nodes
):
    """
    Calculate GIC in transformer windings for all scenarios
    """
    trafo_tables = []

    # Calculate transformer GICs for each field scenario
    for scenario in ["northward", "eastward"]:
        col = f"V_{scenario}"
        if col in nodal_voltages and nodal_voltages[col] is not None:
            V_nodal = nodal_voltages[col]

            # Get transformer winding GICs
            trafo_gic = calc_trafo_gic(
                sub_look_up,
                df_transformers,
                V_nodal,
                sub_ref,
                n_nodes,
                non_zero_indices,
                scenario,
            )

            # Convert to flat table format
            records = [
                (trafo, winding, val)
                for trafo, winds in trafo_gic.items()
                for winding, val in winds.items()
            ]
            trafo_tables.append(
                pd.DataFrame(
                    records,
                    columns=["Transformer", "Winding", f"Trafo_I_{scenario} [A]"],
                )
            )

    # Merge northward and eastward results
    df_trafo_results = trafo_tables[0].merge(
        trafo_tables[1], on=["Transformer", "Winding"], how="outer"
    )

    return df_trafo_results


def compute_ground_gics(
    Y_e_reduced,
    nodal_voltages,
    sub_look_up,
    df_substations_info,
    non_zero_indices,
    n_nodes,
):
    tables = []
    for scenario in ("northward", "eastward"):
        key = f"V_{scenario}"
        V_nodal = nodal_voltages.get(key)
        if V_nodal is not None:
            tbl = _ground_gic_table(
                sub_look_up=sub_look_up,
                df_substations_info=df_substations_info,
                non_zero_indices=non_zero_indices,
                n_nodes=n_nodes,
                Y_e_reduced=Y_e_reduced,
                V_nodal=V_nodal,
                label=scenario,
            )
            tables.append(tbl)

    if not tables:
        return pd.DataFrame(columns=["Substation", "GIC_northward", "GIC_eastward"])

    out = tables[0]
    for t in tables[1:]:
        out = out.merge(t, on="Substation", how="outer")

    # Force presence of every substation; fill missing with 0 A
    all_subs = pd.Index(
        (
            df_substations_info["name"]
            if "name" in df_substations_info.columns
            else df_substations_info.index
        ),
        name="Substation",
    )
    out = out.set_index("Substation").reindex(all_subs, fill_value=0).reset_index()

    return out


def calculate_all_gics(system_data):
    """
    Calculate all GIC types for given system configuration

    Args:
        system_data: Dictionary with 'lines', 'transformers', 'substations' keys

    Returns:
        tuple: (line_gics_df, transformer_gics_df, ground_gics_df)
    """
    # Setup system structure
    df_lines, df_transformers, substations_df, sub_ref = setup_dataframes_and_network(
        system_data
    )

    # Calculate transmission line electric field components
    df_lines = calculate_line_electric_fields(df_lines, substations_df, sub_ref)

    # Build node indexing
    sub_look_up, n_nodes = build_substation_lookup(substations_df)

    # Prepare transformer mappings
    df_transformers = prepare_transformer_data(df_transformers, sub_ref, sub_look_up)

    # Build system admittance matrices
    Y_n, Y_e_reduced, Y_reduced, non_zero_indices = build_system_admittance(
        sub_look_up, sub_ref, df_transformers, df_lines, substations_df
    )

    # Solve for nodal voltages
    nodal_voltages = solve_nodal_voltages(
        df_lines, Y_reduced, sub_look_up, non_zero_indices, n_nodes
    )

    # Calculate GICs for all components
    line_gics = compute_line_gics(
        df_lines, nodal_voltages, sub_look_up, non_zero_indices, n_nodes
    )
    transformer_gics = compute_transformer_gics(
        df_transformers, nodal_voltages, sub_look_up, sub_ref, non_zero_indices, n_nodes
    )
    ground_gics = compute_ground_gics(
        Y_e_reduced,
        nodal_voltages,
        sub_look_up,
        substations_df,
        non_zero_indices,
        n_nodes,
    )

    return line_gics, transformer_gics, ground_gics


# %%
# Parameters for Monte Carlo variations
PARAMS = {
    "p_block_500": 0.20,
    "p_block_345": 0.10,
    "hub_sub": 6,
    "hub_bump": 0.05,
    "very_large_R": 1e15,
    "p_keep2": 0.60,
    "p_keep1": 0.30,
    "p_keep0": 0.10,
    "max_gsu_per_site": 2,
    "p_bd_on": 0.35,
    "p_bd_on_t1": None,
    "p_swap_delta": 0.30,
    "p_auto_to_wye": 0.15,
    "new_nongsu_min": 1,
    "new_nongsu_max": 3,
    "new_nongsu_probs": [0.4, 0.4, 0.2],
    "lock_sites": (7,),
    "p_site_block": 0.15,
    "rg_if_no_bd_low": 0.1,
    "rg_if_no_bd_high": 0.2,
}


def _is_gsu(t):
    # Check if transformer type is a generator step-up unit.
    return str(t.get("type", "")).startswith("GSU")


def _bus_to_sub_map(substations):
    # Create mapping from bus numbers to substation IDs.
    m = {}
    for sid, s in substations.items():
        for b in s["buses"]:
            m[int(b)] = sid
    return m


def _unique_name(names, base):
    # Generate unique name by appending number if base name already exists.
    if base not in names:
        return base
    k = 1
    while f"{base}_{k}" in names:
        k += 1
    return f"{base}_{k}"


def _group_lines_by_corridor(lines):
    # Group transmission lines by corridor (same endpoints and voltage).
    g = defaultdict(list)
    for i, L in enumerate(lines):
        a, b, v = int(L["from_bus"]), int(L["to_bus"]), int(L["V"])
        key = (min(a, b), max(a, b), v)
        g[key].append(i)
    return g


def _apply_structured_line_blocking(lines, substations, rng, P):
    # Block entire transmission corridors based on voltage level and hub proximity.
    bus2sub = _bus_to_sub_map(substations)
    out = copy.deepcopy(lines)

    for (_, _, v), idxs in _group_lines_by_corridor(out).items():
        p = P["p_block_500"] if v >= 500 else P["p_block_345"]
        a, b = out[idxs[0]]["from_bus"], out[idxs[0]]["to_bus"]

        if bus2sub.get(int(a)) == P["hub_sub"] or bus2sub.get(int(b)) == P["hub_sub"]:
            p = min(1.0, p + P["hub_bump"])

        if rng.random() < p:
            for j in idxs:
                out[j] = dict(out[j])
                out[j]["dc_block"] = True
                out[j]["R"] = P["very_large_R"]
                for k in list(out[j].keys()):
                    if isinstance(k, str) and k.startswith("V_"):
                        out[j][k] = 0.0
        else:
            for j in idxs:
                out[j] = dict(out[j])
                out[j]["dc_block"] = (
                    bool(out[j].get("dc_block", False))
                    and out[j]["R"] == P["very_large_R"]
                )
    return out


def _toggle_types(transformers, rng, P):
    # Randomly change transformer types between compatible configurations.
    out = []
    for t in transformers:
        tt = dict(t)
        ty = str(tt["type"])

        if tt["name"] in {"T2", "T13"} and rng.random() < P["p_swap_delta"]:
            tt["type"] = "GY-GY" if ty == "GY-GY-D" else "GY-GY-D"
        elif ty == "Auto" and rng.random() < P["p_auto_to_wye"]:
            tt["type"] = "GY-GY"
        elif ty == "GY-GY" and rng.random() < P["p_auto_to_wye"]:
            tt["type"] = "Auto"
        out.append(tt)
    return out


def _limit_gsus_per_site(transformers, substations, rng, P):
    # Limit number of active GSU transformers per substation based on availability probabilities.
    p2, p1, p0 = P["p_keep2"], P["p_keep1"], P["p_keep0"]
    bus2sub = _bus_to_sub_map(substations)
    gsu_by_sub = defaultdict(list)
    non_gsu = []

    for t in transformers:
        if _is_gsu(t):
            gsu_by_sub[bus2sub.get(int(t["bus1"]), -1)].append(t)
        else:
            non_gsu.append(t)

    kept = []
    for sid, gsus in gsu_by_sub.items():
        if sid in P["lock_sites"]:
            kept.extend(gsus)
            continue

        draw = rng.random()
        desired = 2 if draw < p2 else (1 if draw < p2 + p1 else 0)
        target = min(desired, P["max_gsu_per_site"], len(gsus))

        if target == len(gsus):
            pick = gsus
        elif target == 0:
            pick = []
        else:
            idx = rng.choice(len(gsus), size=target, replace=False)
            pick = [gsus[i] for i in np.atleast_1d(idx)]
        kept.extend(pick)

    if not any(_is_gsu(t) for t in kept):
        all_g = [t for t in transformers if _is_gsu(t)]
        kept.append(rng.choice(all_g))

    return non_gsu + kept


def _apply_gsu_bd(transformers, rng, P):
    # Apply GIC blocking devices to GSU transformers based on probability settings.
    p_on = float(P.get("p_bd_on", 0.35))
    val = P.get("p_bd_on_t1", None)
    p_on_t1 = float(p_on if val is None else val)
    out = []

    for t in transformers:
        tt = dict(t)
        if _is_gsu(tt):
            p = p_on_t1 if tt.get("name") == "T1" else p_on
            tt["type"] = "GSU w/ GIC BD" if (rng.random() < p) else "GSU"
        out.append(tt)
    return out


def _add_new_nongsu(transformers, substations, rng, P):
    # Add random non-GSU transformers to the system based on existing templates.
    names = {t["name"] for t in transformers}
    pool = [t for t in transformers if not _is_gsu(t)]
    if not pool:
        return transformers

    K_vals = list(range(P["new_nongsu_min"], P["new_nongsu_max"] + 1))
    K = int(rng.choice(K_vals, p=np.array(P["new_nongsu_probs"])))
    out = list(transformers)
    avoid = set(substations[P["lock_sites"][0]]["buses"]) if P["lock_sites"] else set()

    tries, added = 0, 0
    while added < K and tries < 50:
        tries += 1
        base = dict(rng.choice(pool))
        if int(base.get("bus1", -1)) in avoid or int(base.get("bus2", -1)) in avoid:
            continue
        new_name = _unique_name(names, base["name"] + "X")
        names.add(new_name)
        base["name"] = new_name
        out.append(base)
        added += 1
    return out


def _enforce_sub_transformer_lists(substations, transformers):
    # Update substation transformer lists to match actual transformer connections.
    bus2sub = _bus_to_sub_map(substations)
    lists = {sid: [] for sid in substations}

    for t in transformers:
        sids = set()
        if "bus1" in t:
            sids.add(bus2sub.get(int(t["bus1"])))
        if "bus2" in t:
            sids.add(bus2sub.get(int(t["bus2"])))
        for sid in sids:
            if sid in lists:
                lists[sid].append(t["name"])

    out = copy.deepcopy(substations)
    for sid in out:
        out[sid] = dict(out[sid], transformers=lists.get(sid, []))
    return out


def generate_cfg_base(ORIGINAL_DATA):
    # Create base configuration by removing any existing DC blocking flags.
    D = copy.deepcopy(ORIGINAL_DATA)
    for L in D["lines"]:
        L.pop("dc_block", None)
    return D


def generate_cfg_var(ORIGINAL_DATA, seed=42, params=PARAMS):
    # Generate random network configuration variant with Monte Carlo parameter variations.
    rng = np.random.default_rng(seed)
    D = copy.deepcopy(ORIGINAL_DATA)
    P = params

    D["lines"] = _apply_structured_line_blocking(D["lines"], D["substations"], rng, P)

    tra = _toggle_types(D["transformers"], rng, P)
    tra = _limit_gsus_per_site(tra, D["substations"], rng, P)
    tra = _apply_gsu_bd(tra, rng, P)
    tra = _add_new_nongsu(tra, D["substations"], rng, P)

    VL = P["very_large_R"]
    lock = set(P["lock_sites"])
    for sid, s in D["substations"].items():
        if sid in lock:
            continue
        if rng.random() < P["p_site_block"]:
            s = dict(s)
            s["grounding_resistance"] = VL
            D["substations"][sid] = s

    b2s = _bus_to_sub_map(D["substations"])
    low, high = P["rg_if_no_bd_low"], P["rg_if_no_bd_high"]
    subs_need = set()
    for t in tra:
        if _is_gsu(t) and ("w/ GIC BD" not in str(t["type"])):
            sid = b2s.get(int(t["bus1"]))
            if sid is not None and sid not in lock:
                subs_need.add(sid)

    for sid in subs_need:
        s = dict(D["substations"][sid])
        if (
            not np.isfinite(s.get("grounding_resistance", np.inf))
            or s["grounding_resistance"] >= VL / 10
        ):
            s["grounding_resistance"] = float(rng.uniform(low, high))
        D["substations"][sid] = s

    D["transformers"] = tra
    D["substations"] = _enforce_sub_transformer_lists(
        D["substations"], D["transformers"]
    )
    return D


def diff_cfgs(base, var):
    # Compare two configurations and return summary of differences.
    import math

    out = {}
    bL = {l["name"]: l for l in base["lines"]}
    vL = {l["name"]: l for l in var["lines"]}
    common = bL.keys() & vL.keys()
    out["lines_blocked"] = sorted(
        [n for n in common if vL[n].get("dc_block") and not bL[n].get("dc_block")]
    )
    out["lines_R_changed"] = sorted(
        [
            n
            for n in common
            if (bL[n]["R"] != vL[n]["R"])
            and (math.isfinite(bL[n]["R"]) or math.isfinite(vL[n]["R"]))
        ]
    )
    bT = {t["name"]: t for t in base["transformers"]}
    vT = {t["name"]: t for t in var["transformers"]}
    out["trafos_added"] = sorted(set(vT) - set(bT))
    out["trafos_removed"] = sorted(set(bT) - set(vT))
    out["trafos_type_swaps"] = sorted(
        [n for n in (set(bT) & set(vT)) if bT[n]["type"] != vT[n]["type"]]
    )
    return out


def run_horton_monte_carlo_cfg(
    system_data, *, config="base", n_scenarios=5000, seed=42, cfg_params=None
):
    # Run Monte Carlo GIC analysis for either base configuration or random variants.
    cfg_params = cfg_params or PARAMS
    line_runs, trafo_runs, ground_runs = [], [], []

    if config.lower() == "base":
        sys_in = generate_cfg_base(system_data)

        df_lines, df_transformers, substations_df, sub_ref = (
            setup_dataframes_and_network(sys_in)
        )
        df_lines = calculate_line_electric_fields(df_lines, substations_df, sub_ref)
        sub_ref = dict(zip(substations_df.name, substations_df.buses))
        sub_look_up, n_nodes = build_substation_lookup(substations_df)
        df_transformers = prepare_transformer_data(
            df_transformers, sub_ref, sub_look_up
        )
        Y_n, Y_e_reduced, Y_reduced, non_zero_indices = build_system_admittance(
            sub_look_up, sub_ref, df_transformers, df_lines, substations_df
        )
        nodal_voltages = solve_nodal_voltages(
            df_lines, Y_reduced, sub_look_up, non_zero_indices, n_nodes
        )

        L = compute_line_gics(
            df_lines, nodal_voltages, sub_look_up, non_zero_indices, n_nodes
        )
        T = compute_transformer_gics(
            df_transformers,
            nodal_voltages,
            sub_look_up,
            sub_ref,
            non_zero_indices,
            n_nodes,
        )
        G = compute_ground_gics(
            Y_e_reduced,
            nodal_voltages,
            sub_look_up,
            substations_df,
            non_zero_indices,
            n_nodes,
        )

        for df in (L, T, G):
            df["scenario"] = 0
            df["config"] = "base"

        return L, T, G

    elif config.lower() == "var":
        for i in range(n_scenarios):
            s = int(seed + i)
            sys_in = generate_cfg_var(system_data, seed=s, params=cfg_params)

            df_lines, df_transformers, substations_df, sub_ref = (
                setup_dataframes_and_network(sys_in)
            )
            df_lines = calculate_line_electric_fields(df_lines, substations_df, sub_ref)
            sub_ref = dict(zip(substations_df.name, substations_df.buses))
            sub_look_up, n_nodes = build_substation_lookup(substations_df)
            df_transformers = prepare_transformer_data(
                df_transformers, sub_ref, sub_look_up
            )
            Y_n, Y_e_reduced, Y_reduced, non_zero_indices = build_system_admittance(
                sub_look_up, sub_ref, df_transformers, df_lines, substations_df
            )
            nodal_voltages = solve_nodal_voltages(
                df_lines, Y_reduced, sub_look_up, non_zero_indices, n_nodes
            )

            L = compute_line_gics(
                df_lines, nodal_voltages, sub_look_up, non_zero_indices, n_nodes
            )
            T = compute_transformer_gics(
                df_transformers,
                nodal_voltages,
                sub_look_up,
                sub_ref,
                non_zero_indices,
                n_nodes,
            )
            G = compute_ground_gics(
                Y_e_reduced,
                nodal_voltages,
                sub_look_up,
                substations_df,
                non_zero_indices,
                n_nodes,
            )

            for df in (L, T, G):
                df["scenario"] = i
                df["config"] = "var"

            line_runs.append(L)
            trafo_runs.append(T)
            ground_runs.append(G)

        return (
            pd.concat(line_runs, ignore_index=True),
            pd.concat(trafo_runs, ignore_index=True),
            pd.concat(ground_runs, ignore_index=True),
        )
    else:
        raise ValueError("config must be 'base' or 'var'")


def run_horton_monte_carlo_base_vs_var(
    system_data, *, n_scenarios=2000, seed=42, cfg_params=None
):
    # Run comparative Monte Carlo analysis between base and variant configurations.
    lines_b, trafos_b, grounds_b = run_horton_monte_carlo_cfg(
        system_data,
        config="base",
        n_scenarios=n_scenarios,
        seed=seed,
        cfg_params=cfg_params,
    )
    lines_v, trafos_v, grounds_v = run_horton_monte_carlo_cfg(
        system_data,
        config="var",
        n_scenarios=n_scenarios,
        seed=seed,
        cfg_params=cfg_params,
    )
    return (
        pd.concat([lines_b, lines_v], ignore_index=True),
        pd.concat([trafos_b, trafos_v], ignore_index=True),
        pd.concat([grounds_b, grounds_v], ignore_index=True),
    )


if __name__ == "__main__":
    line_gics, transformer_gics, ground_gics = calculate_all_gics(ORIGINAL_DATA)

    line_bs, trafo_bs, ground_bs = run_horton_monte_carlo_base_vs_var(
        ORIGINAL_DATA, n_scenarios=5000, seed=42, cfg_params=PARAMS
    )

    out_dir = os.path.join(DATA_LOC, "horton_grid")
    os.makedirs(out_dir, exist_ok=True)

    mc_results = {
        "base_lines_gic": line_gics,
        "base_trafos_gic": transformer_gics,
        "base_grounds_gic": ground_gics,
        "mc_lines_gic": line_bs,
        "mc_trafos_gic": trafo_bs,
        "mc_grounds_gic": ground_bs,
    }

    with open(os.path.join(out_dir, "mc_results.pkl"), "wb") as f:
        pickle.dump(mc_results, f)
