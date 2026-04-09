"""
Run GIC solver for UIUC150 or TVA OSM grid.
Author: Dennies
"""

import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rep_mapping.rep_config import UIUC_DIR, TVA_DIR, TRAFO_WINDING_R, setup_logger

import scripts.build_admittance_matrix as _bam
import scripts.est_gic as _eg

network_admittance = _bam.network_admittance
earthing_impedance = _bam.earthing_impedance
get_injection_currents = _eg.get_injection_currents
nodal_voltage_calculation = _eg.nodal_voltage_calculation
calc_trafo_gic = _eg.calc_trafo_gic
keep_nodes_with_ground = _eg.keep_nodes_with_ground
solve_total_nodal_gic = _eg.solve_total_nodal_gic

logger = setup_logger(log_file="logs/run_gic.log")


def build_admittance(sub_look_up, sub_ref, df_transformers, df_lines, substations_df):
    Y_n = network_admittance(sub_look_up, sub_ref, df_transformers, df_lines)
    Y_e = earthing_impedance(sub_look_up, substations_df)

    row_active = np.any(Y_n != 0, axis=1) | np.any(Y_e != 0, axis=1)
    nz = np.flatnonzero(row_active)
    keep = keep_nodes_with_ground(Y_n[np.ix_(nz, nz)], Y_e[np.ix_(nz, nz)])
    nzi = nz[keep]
    Y_total = (Y_n + Y_e)[np.ix_(nzi, nzi)]
    Y_e_red = Y_e[np.ix_(nzi, nzi)]

    logger.info(f"Y shape: {Y_total.shape}  Active nodes: {len(nzi)}")
    return Y_total, Y_e_red, nzi


def calc_effective_gic(gic_dict, trafos_df):
    records = []
    for name, windings in gic_dict.items():
        row = {"name": name}
        row.update(windings)
        records.append(row)

    gic_df = pd.DataFrame(records)

    trafo_meta = (
        trafos_df[["name", "type", "kv_high", "kv_low"]].copy()
        if "kv_high" in trafos_df.columns
        else None
    )

    if trafo_meta is not None:
        gic_df = gic_df.merge(trafo_meta, on="name", how="left")
        v_ratio = (gic_df["kv_low"] / gic_df["kv_high"]).fillna(1.0)
    else:
        v_ratio = pd.Series(1.0, index=gic_df.index)

    def eff(r, vr):
        if pd.notna(r.get("Series")) and pd.notna(r.get("Common")):
            return abs(r["Series"] + r["Common"] * vr)
        elif pd.notna(r.get("HV")) and pd.notna(r.get("LV")):
            return abs(r["HV"] + r["LV"] * vr)
        elif pd.notna(r.get("HV")):
            return abs(r["HV"])
        elif pd.notna(r.get("Series")):
            return abs(r["Series"])
        return np.nan

    gic_df["I_eff"] = [
        eff(row, vr) for row, vr in zip(gic_df.to_dict("records"), v_ratio)
    ]

    gic_df["hv_bus"] = gic_df.name.apply(
        lambda n: int(n.split("_")[1]) if "_" in str(n) else np.nan
    )
    gic_df["lv_bus"] = gic_df.name.apply(
        lambda n: int(n.split("_")[2]) if len(n.split("_")) > 2 else np.nan
    )

    return gic_df


def run_uiuc(gannon=False):
    if gannon:
        logger.info("Running UIUC150 GIC — Gannon storm")
    else:
        logger.info("Running UIUC150 GIC — uniform 1 V/km East + North")

    with open(UIUC_DIR / "sub_look_up.pkl", "rb") as f:
        sub_look_up = pickle.load(f)
    with open(UIUC_DIR / "sub_ref.pkl", "rb") as f:
        sub_ref = pickle.load(f)
    with open(UIUC_DIR / "df_transformers.pkl", "rb") as f:
        df_transformers = pickle.load(f)
    with open(UIUC_DIR / "substations_df.pkl", "rb") as f:
        substations_df = pickle.load(f)

    if gannon:
        with open(UIUC_DIR / "df_lines_with_voltages.pkl", "rb") as f:
            df_lines = pickle.load(f)
    else:
        with open(UIUC_DIR / "df_lines.pkl", "rb") as f:
            df_lines = pickle.load(f)

    n_nodes = int(np.load(UIUC_DIR / "n_nodes.npy")[0])

    Y_total, Y_e_red, nzi = build_admittance(
        sub_look_up, sub_ref, df_transformers, df_lines, substations_df
    )

    if gannon:
        injections = get_injection_currents(
            df_lines, n_nodes, nzi, sub_look_up, UIUC_DIR, gannon_storm_only=True
        )

        logger.info(f"Injection keys: {len(injections)}")

        nodal_voltages = nodal_voltage_calculation(Y_total, injections)
        logger.info(f"Solved {len(nodal_voltages)} voltage keys")

        v_gannon_cols = sorted(
            [c for c in df_lines.columns if c.startswith("V_gannon_")],
            key=lambda x: int(x.split("_")[-1]),
        )

        ground_gic_ts = {}
        for key, V_nodal in tqdm(nodal_voltages.items(), desc="Ground GIC"):
            ig = solve_total_nodal_gic(Y_e_red, V_nodal, nzi, n_nodes)
            for sub in range(1, 99):
                node_idx = sub_look_up.get(("sub", sub))
                if node_idx is not None:
                    ground_gic_ts.setdefault(sub, []).append(ig[node_idx])

        time_axis = np.load(UIUC_DIR / "time_axis_gannon.npy", allow_pickle=True)
        ground_gic_df = pd.DataFrame(ground_gic_ts).T
        ground_gic_df.columns = v_gannon_cols[: ground_gic_df.shape[1]]

        ground_gic_df.to_parquet(UIUC_DIR / "ground_gic_ts_gannon.parquet")
        np.save(UIUC_DIR / "time_axis_gannon.npy", time_axis)

        logger.info(f"Ground GIC shape: {ground_gic_df.shape}")
        logger.info(f"Max GIC:          {ground_gic_df.values.max():.2f} A")
        logger.info(f"Saved to {UIUC_DIR}/ground_gic_ts_gannon.parquet")

    else:
        gic_bench = pd.read_parquet(UIUC_DIR / "gic_bench.parquet")

        injections = get_injection_currents(
            df_lines, n_nodes, nzi, sub_look_up, UIUC_DIR, gic_test_case=True
        )

        nodal_voltages = nodal_voltage_calculation(Y_total, injections)

        results = {}
        for direction in ["V_eastward", "V_northward"]:
            logger.info(f"Computing GIC — {direction}")
            gic_raw = calc_trafo_gic(
                sub_look_up,
                df_transformers,
                nodal_voltages[direction],
                sub_ref,
                n_nodes,
                nzi,
                direction,
            )
            gic_df = calc_effective_gic(gic_raw, df_transformers)
            ig_full = solve_total_nodal_gic(
                Y_e_red, nodal_voltages[direction], nzi, n_nodes
            )
            results[direction] = {"gic_df": gic_df, "ig_full": ig_full}

        gic_df_east = results["V_eastward"]["gic_df"]
        comp = gic_df_east.merge(gic_bench, on=["hv_bus", "lv_bus"]).drop_duplicates(
            subset=["hv_bus", "lv_bus"]
        )
        comp["err_pct"] = (
            (comp.I_eff - comp.GIC_eff_A).abs() / comp.GIC_eff_A.abs() * 100
        )

        comp_filt = comp[comp.GIC_eff_A >= 0.5]
        r = np.corrcoef(comp.I_eff, comp.GIC_eff_A)[0, 1]
        rmse = np.sqrt(((comp.I_eff - comp.GIC_eff_A) ** 2).mean())
        mae = (comp.I_eff - comp.GIC_eff_A).abs().mean()

        logger.info(f"Transformers compared: {len(comp)}")
        logger.info(f"Pearson r: {r:.3f}  RMSE: {rmse:.2f}  MAE: {mae:.2f}")
        logger.info(
            f"Filtered (>=0.5A, n={len(comp_filt)}) — "
            f"Mean error: {comp_filt['err_pct'].mean():.1f}%  "
            f"Max: {comp_filt['err_pct'].max():.1f}%"
        )

        worst = comp.sort_values("err_pct", ascending=False).head(5)
        logger.info(
            f"Worst outliers:\n"
            f"{worst[['hv_bus','lv_bus','I_eff','GIC_eff_A','err_pct']].to_string()}"
        )

        comp.to_parquet(UIUC_DIR / "gic_comparison.parquet")

        ground_gic = pd.DataFrame(
            {
                "node_idx": nzi,
                "ig_eastward": results["V_eastward"]["ig_full"][nzi],
                "ig_northward": results["V_northward"]["ig_full"][nzi],
            }
        )
        ground_gic.to_parquet(UIUC_DIR / "ground_gic.parquet")
        logger.info(f"Saved gic_comparison and ground_gic to {UIUC_DIR}")


def run_tva():
    logger.info("Running TVA GIC — Gannon storm")

    with open(TVA_DIR / "substation_buses.pkl", "rb") as f:
        substation_buses = pickle.load(f)

    substations_df = pd.read_parquet(TVA_DIR / "substations_df.parquet")
    df_lines = pd.read_parquet(TVA_DIR / "df_lines_with_voltages.parquet")

    sub_look_up = {}
    idx = 0
    for info in substation_buses.values():
        for bus in sorted(info["buses"]):
            sub_look_up[bus] = idx
            idx += 1
    for hv_bus in substation_buses:
        sub_look_up[hv_bus + "_neutral"] = idx
        idx += 1
    n_nodes = idx

    sub_ref = {
        hv_bus + "_neutral": [info["hv_bus"], info["lv_bus"]]
        for hv_bus, info in substation_buses.items()
    }

    trafo_records = []
    t_num = 0
    for hv_bus, info in substation_buses.items():
        for t_type in info["trafo_types"]:
            W = TRAFO_WINDING_R[t_type]
            lat = substations_df[substations_df.name == hv_bus]["latitude"].values[0]
            lon = substations_df[substations_df.name == hv_bus]["longitude"].values[0]
            trafo_records.append(
                {
                    "name": f"T_{hv_bus}_{t_num}",
                    "sub_id": hv_bus,
                    "type": t_type,
                    "bus1": info["hv_bus"],
                    "bus2": info["lv_bus"],
                    "W1": W["pri"],
                    "W2": W["sec"],
                    "sub": hv_bus + "_neutral",
                    "latitude": lat,
                    "longitude": lon,
                }
            )
            t_num += 1
    df_transformers = pd.DataFrame(trafo_records)

    substations_df_n = substations_df.copy()
    substations_df_n["name"] = substations_df_n["name"].apply(lambda x: x + "_neutral")

    Y_total, Y_e_red, nzi = build_admittance(
        sub_look_up, sub_ref, df_transformers, df_lines, substations_df_n
    )

    v_gannon_cols = sorted(
        [c for c in df_lines.columns if c.startswith("V_gannon_")],
        key=lambda x: int(x.split("_")[-1]),
    )

    injections = get_injection_currents(
        df_lines, n_nodes, nzi, sub_look_up, TVA_DIR, gannon_storm_only=True
    )

    logger.info(f"Injection keys: {len(injections)}")
    logger.info(f"Nodes per key:  {len(list(injections.values())[0])}")
    logger.info(f"Timesteps:      {len(injections)}")

    nodal_voltages = nodal_voltage_calculation(Y_total, injections)
    logger.info(f"Solved {len(nodal_voltages)} voltage keys")

    ground_gic_ts = {}
    for key, V_nodal in tqdm(nodal_voltages.items(), desc="Ground GIC"):
        ig = solve_total_nodal_gic(Y_e_red, V_nodal, nzi, n_nodes)
        for hv_bus in substation_buses:
            node_idx = sub_look_up.get(hv_bus + "_neutral")
            if node_idx is not None:
                ground_gic_ts.setdefault(hv_bus, []).append(ig[node_idx])

    time_axis = np.load(TVA_DIR / "time_axis.npy", allow_pickle=True)
    ground_gic_df = pd.DataFrame(ground_gic_ts).T
    ground_gic_df.columns = v_gannon_cols[: ground_gic_df.shape[1]]

    ground_gic_df.to_parquet(TVA_DIR / "ground_gic_ts.parquet")
    np.save(TVA_DIR / "time_axis.npy", time_axis)

    logger.info(f"Ground GIC shape: {ground_gic_df.shape}")
    logger.info(f"Max GIC:          {ground_gic_df.values.max():.2f} A")
    logger.info(f"Saved to {TVA_DIR}/ground_gic_ts.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIC solver.")
    parser.add_argument("--grid", choices=["uiuc", "tva"], required=True)
    parser.add_argument(
        "--gannon", action="store_true", help="Use Gannon storm voltages (UIUC only)"
    )
    args = parser.parse_args()

    if args.grid == "uiuc":
        run_uiuc(gannon=args.gannon)
    else:
        run_tva()
