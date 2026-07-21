"""
Run GIC solver for UIUC150 or TVA OSM grid.
Author: Dennies
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rep_mapping.rep_config import (
    UIUC_DIR,
    TVA_DIR,
    TRAFO_WINDING_R,
    POOL_GEN,
    POOL_TRANS,
    GRID_MAPPING,
    setup_logger,
)

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

SEED_BASE = 42


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


def _sample_trafo_types(
    substation_buses: dict,
    osm_counts: dict,
    osm_role: dict,
    rng: np.random.Generator,
) -> dict:
    """Sample transformer types for all substations.

    Replicates the RNG logic in build_tva_grid.build_substation_buses so that
    seed=SEED_BASE for run i=0 reproduces the stored single-seed types exactly.
    Synthetic substations never advance the RNG.
    """
    typed: dict = {}
    for hv_bus, info in substation_buses.items():
        if info.get("is_synthetic"):
            typed[hv_bus] = list(info["trafo_types"])
            continue
        osmid_f: float | None = None
        try:
            osmid_f = float(hv_bus)
        except (ValueError, TypeError):
            pass
        if osmid_f is not None and osmid_f in osm_counts:
            count = int(osm_counts[osmid_f])
            pool = POOL_GEN if osm_role.get(osmid_f) == "generation" else POOL_TRANS
            typed[hv_bus] = list(rng.choice(pool, size=count, replace=True))
        else:
            count = int(rng.integers(1, 4))
            typed[hv_bus] = list(rng.choice(POOL_TRANS, size=count, replace=True))
    return typed


def _build_transformers_tva(
    substation_buses: dict,
    trafo_types: dict,
    substations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build df_transformers for TVA grid from a trafo_types mapping."""
    records = []
    t_num = 0
    for hv_bus, info in substation_buses.items():
        rows = substations_df[substations_df["name"] == hv_bus]
        lat = float(rows["latitude"].values[0]) if len(rows) else 35.5
        lon = float(rows["longitude"].values[0]) if len(rows) else -86.5
        for t_type in trafo_types[hv_bus]:
            W = TRAFO_WINDING_R[t_type]
            records.append(
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
    return pd.DataFrame(records)


def _precompute_injections(
    df_lines: pd.DataFrame,
    sub_look_up: dict,
    n_nodes: int,
    v_gannon_cols: list[str],
) -> np.ndarray:
    """Compute full injection matrix (n_nodes x n_times) once for all MC runs.

    The geoelectric field and line topology are deterministic; only the admittance
    matrix changes between runs, so this result is reused across the entire ensemble.
    Returns array of shape (n_nodes, n_times).
    """
    R = df_lines["R"].to_numpy(dtype=float)
    from_idx = np.array([sub_look_up[b] for b in df_lines["from_bus"]])
    to_idx = np.array([sub_look_up[b] for b in df_lines["to_bus"]])
    V_mat = df_lines[v_gannon_cols].to_numpy(dtype=float)
    I_mat = np.nan_to_num(V_mat / R[:, None], nan=0.0)
    n_times = len(v_gannon_cols)
    inj = np.zeros((n_nodes, n_times), dtype=float)
    for j in range(n_times):
        inj[:, j] += np.bincount(to_idx, weights=I_mat[:, j], minlength=n_nodes)
        inj[:, j] -= np.bincount(from_idx, weights=I_mat[:, j], minlength=n_nodes)
    return inj


def _randomize_grounding_tva(
    substations_df_n: pd.DataFrame,
    synthetic_neutral: set[str],
    seed: int,
    p_open: float = 0.01,
) -> pd.DataFrame:
    """Sample per-run grounding resistance, differentiated by substation type.

    Real/annotated substations: Uniform(0.1, 1.0) Ohm.
    Synthetic endpoints:        Uniform(2.0, 20.0) Ohm.
    A fraction p_open of real substations is set to inf (ungrounded path).
    A single RNG seeded by `seed` advances through real draws, then synthetic
    draws, then the open-set selection, so the sequence is fully determined.
    """
    rng = np.random.default_rng(seed)
    out = substations_df_n.copy()
    out["name"] = out["name"].astype(str)
    is_synth = out["name"].isin(synthetic_neutral)

    n_real = int((~is_synth).sum())
    if n_real > 0:
        out.loc[~is_synth, "grounding_resistance"] = rng.uniform(0.1, 1.0, size=n_real)

    n_synth = int(is_synth.sum())
    if n_synth > 0:
        out.loc[is_synth, "grounding_resistance"] = rng.uniform(2.0, 20.0, size=n_synth)

    k = int(np.floor(p_open * n_real))
    if k > 0:
        real_positions = np.flatnonzero(~is_synth.values)
        open_positions = rng.choice(real_positions, size=k, replace=False)
        out.iloc[open_positions, out.columns.get_loc("grounding_resistance")] = np.inf

    return out


def run_tva_mc(n_runs: int = 1000) -> None:
    """
    Role: Monte Carlo ensemble for TVA GIC over the Gannon storm.
    Description: Runs n_runs iterations with per-run seed = SEED_BASE + i.
    Each run resamples transformer types (POOL_GEN/POOL_TRANS) for all
    non-synthetic substations and independently randomizes grounding
    resistance: Uniform(0.1, 1.0) Ohm for real substations, Uniform(2.0,
    20.0) Ohm for synthetic endpoints, with p_open = 0.01 fraction of real
    substations set to inf. Line DC blocking is not applied (P_LINE_BLOCK =
    0). Seeds 42..42+n_runs-1 are fixed for reproducibility. The geoelectric
    field injection matrix is precomputed once; Y_n and Y_e are rebuilt per
    run. A deterministic baseline (seed-42 trafo types, fixed Rg = 0.2/10.0
    Ohm) is computed once at the end to confirm ensemble centering.

    Outputs:
      data/rep_data/tva/ground_gic_mc.parquet      rows=runs, cols=substations (peak |GIC|)
      data/rep_data/tva/ground_gic_mc_ts.npy       shape (n_runs, n_mon, n_times) float32
      data/rep_data/tva/ground_gic_mc_ts_subs.json list of monitored hv_bus names matching axis 1
      data/rep_data/tva/ground_gic_mc_summary.csv
    """
    logger.info("Running TVA GIC Monte Carlo — Gannon storm, %d runs", n_runs)

    with open(TVA_DIR / "substation_buses.pkl", "rb") as f:
        substation_buses = pickle.load(f)

    substations_df = pd.read_parquet(TVA_DIR / "substations_df.parquet")
    df_lines = pd.read_parquet(TVA_DIR / "df_lines_with_voltages.parquet")

    sub_look_up: dict = {}
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

    substations_df_n = substations_df.copy()
    substations_df_n["name"] = substations_df_n["name"].apply(lambda x: x + "_neutral")

    synthetic_neutral = {
        hv_bus + "_neutral"
        for hv_bus, info in substation_buses.items()
        if info.get("is_synthetic")
    }

    gm = pd.read_csv(GRID_MAPPING)
    gm["Attributes"] = gm["Attributes"].apply(eval)
    trafo_rows = gm[gm["Marker Label"] == "Transformer"].copy()
    trafo_rows["Role"] = trafo_rows["Attributes"].apply(
        lambda x: x["role"] if isinstance(x, dict) else "transmission"
    )
    trafo_rows["SS_ID"] = trafo_rows["SS_ID"].astype(float)
    osm_counts = trafo_rows.groupby("SS_ID").size().to_dict()
    osm_role = (
        trafo_rows.groupby("SS_ID")["Role"]
        .agg(
            lambda s: "generation" if (s == "generation").mean() > 0.5 else "transmission"
        )
        .to_dict()
    )

    v_gannon_cols = sorted(
        [c for c in df_lines.columns if c.startswith("V_gannon_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_times = len(v_gannon_cols)

    inj_full = _precompute_injections(df_lines, sub_look_up, n_nodes, v_gannon_cols)
    logger.info(
        "Injection matrix precomputed: %d nodes x %d timesteps", n_nodes, n_times
    )

    sub_names = list(substation_buses.keys())
    n_subs = len(sub_names)
    peak_gic = np.full((n_runs, n_subs), np.nan)

    matched_devices = pd.read_parquet(TVA_DIR / "matched_devices.parquet")
    sub_names_set = set(sub_names)
    mon_subs = [
        str(float(o))
        for o in matched_devices["osmid"].unique()
        if str(float(o)) in sub_names_set
    ]
    n_mon = len(mon_subs)
    mc_ts = np.full((n_runs, n_mon, n_times), np.nan, dtype=np.float32)
    logger.info("Tracking full time series for %d monitored substations", n_mon)

    for i in tqdm(range(n_runs), desc="MC runs"):
        seed = SEED_BASE + i
        rng = np.random.default_rng(seed)

        trafo_types_i = _sample_trafo_types(substation_buses, osm_counts, osm_role, rng)
        df_transformers_i = _build_transformers_tva(
            substation_buses, trafo_types_i, substations_df
        )

        substations_df_n_i = _randomize_grounding_tva(
            substations_df_n, synthetic_neutral, seed
        )

        Y_n = network_admittance(sub_look_up, sub_ref, df_transformers_i, df_lines)
        Y_e = earthing_impedance(sub_look_up, substations_df_n_i)

        row_active = np.any(Y_n != 0, axis=1) | np.any(Y_e != 0, axis=1)
        nz = np.flatnonzero(row_active)
        keep = keep_nodes_with_ground(Y_n[np.ix_(nz, nz)], Y_e[np.ix_(nz, nz)])
        nzi = nz[keep]

        Y_total = (Y_n + Y_e)[np.ix_(nzi, nzi)]
        Y_e_red = Y_e[np.ix_(nzi, nzi)]

        injections_data = {col: inj_full[nzi, j] for j, col in enumerate(v_gannon_cols)}

        nodal_voltages = nodal_voltage_calculation(Y_total, injections_data)

        V_matrix = np.stack(
            [nodal_voltages[col] for col in v_gannon_cols], axis=1
        )  # (len(nzi), n_times), float32
        y_e_diag = np.diag(Y_e_red)  # (len(nzi),)
        ig_reduced = 3.0 * y_e_diag[:, None] * V_matrix  # (len(nzi), n_times)

        for s, hv_bus in enumerate(sub_names):
            node_idx = sub_look_up.get(hv_bus + "_neutral")
            if node_idx is None:
                continue
            pos = np.searchsorted(nzi, node_idx)
            if pos < len(nzi) and nzi[pos] == node_idx:
                peak_gic[i, s] = float(np.max(np.abs(ig_reduced[pos, :])))

        for m_idx, hv_bus_m in enumerate(mon_subs):
            node_idx_m = sub_look_up.get(hv_bus_m + "_neutral")
            if node_idx_m is None:
                continue
            pos_m = np.searchsorted(nzi, node_idx_m)
            if pos_m < len(nzi) and nzi[pos_m] == node_idx_m:
                mc_ts[i, m_idx, :] = ig_reduced[pos_m, :].astype(np.float32)

    logger.info("MC runs complete")

    peak_df = pd.DataFrame(peak_gic, columns=sub_names)
    peak_df.index.name = "run"
    peak_df.to_parquet(TVA_DIR / "ground_gic_mc.parquet")
    logger.info(
        "Saved ground_gic_mc.parquet — shape %d runs x %d substations",
        n_runs,
        n_subs,
    )

    np.save(TVA_DIR / "ground_gic_mc_ts.npy", mc_ts)
    with open(TVA_DIR / "ground_gic_mc_ts_subs.json", "w") as _fh:
        json.dump(mon_subs, _fh)
    logger.info(
        "Saved ground_gic_mc_ts.npy — shape %d runs x %d monitored x %d timesteps",
        n_runs,
        n_mon,
        n_times,
    )

    p95_per_run = np.nanpercentile(peak_gic, 95, axis=1)
    pmax_per_run = np.nanmax(peak_gic, axis=1)

    # Deterministic baseline: seed-42 trafo types + fixed Rg = 0.2 / 10.0 Ohm
    rng_base = np.random.default_rng(SEED_BASE)
    tt_base = _sample_trafo_types(substation_buses, osm_counts, osm_role, rng_base)
    df_tf_base = _build_transformers_tva(substation_buses, tt_base, substations_df)
    Y_n_b = network_admittance(sub_look_up, sub_ref, df_tf_base, df_lines)
    Y_e_b = earthing_impedance(sub_look_up, substations_df_n)
    row_active_b = np.any(Y_n_b != 0, axis=1) | np.any(Y_e_b != 0, axis=1)
    nz_b = np.flatnonzero(row_active_b)
    keep_b = keep_nodes_with_ground(Y_n_b[np.ix_(nz_b, nz_b)], Y_e_b[np.ix_(nz_b, nz_b)])
    nzi_b = nz_b[keep_b]
    Y_total_b = (Y_n_b + Y_e_b)[np.ix_(nzi_b, nzi_b)]
    Y_e_red_b = Y_e_b[np.ix_(nzi_b, nzi_b)]
    inj_b = {col: inj_full[nzi_b, j] for j, col in enumerate(v_gannon_cols)}
    nv_b = nodal_voltage_calculation(Y_total_b, inj_b)
    V_mat_b = np.stack([nv_b[col] for col in v_gannon_cols], axis=1)
    ig_b = 3.0 * np.diag(Y_e_red_b)[:, None] * V_mat_b
    peak_base = np.full(n_subs, np.nan)
    for s, hv_bus in enumerate(sub_names):
        node_idx = sub_look_up.get(hv_bus + "_neutral")
        if node_idx is None:
            continue
        pos = np.searchsorted(nzi_b, node_idx)
        if pos < len(nzi_b) and nzi_b[pos] == node_idx:
            peak_base[s] = float(np.max(np.abs(ig_b[pos, :])))
    baseline_p95 = float(np.nanpercentile(peak_base, 95))
    logger.info(
        "Deterministic baseline (seed %d, fixed Rg=0.2/10.0): p95=%.2f A",
        SEED_BASE,
        baseline_p95,
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "headline_p95_ground_gic_A",
                "ensemble_peak_max_ground_gic_A",
            ],
            "median": [
                float(np.median(p95_per_run)),
                float(np.median(pmax_per_run)),
            ],
            "p5": [
                float(np.percentile(p95_per_run, 5)),
                float(np.percentile(pmax_per_run, 5)),
            ],
            "p95": [
                float(np.percentile(p95_per_run, 95)),
                float(np.percentile(pmax_per_run, 95)),
            ],
            "std": [
                float(np.std(p95_per_run)),
                float(np.std(pmax_per_run)),
            ],
            "deterministic_baseline": [baseline_p95, float(np.nanmax(peak_base))],
        }
    )
    summary.to_csv(TVA_DIR / "ground_gic_mc_summary.csv", index=False)

    logger.info(
        "Headline 95th-pct peak ground GIC: median=%.2f  [p5=%.2f, p95=%.2f]  std=%.2f A",
        summary.loc[0, "median"],
        summary.loc[0, "p5"],
        summary.loc[0, "p95"],
        summary.loc[0, "std"],
    )
    logger.info(
        "Ensemble peak max ground GIC: median=%.2f  [p5=%.2f, p95=%.2f]  std=%.2f A",
        summary.loc[1, "median"],
        summary.loc[1, "p5"],
        summary.loc[1, "p95"],
        summary.loc[1, "std"],
    )
    logger.info(
        "Ensemble median vs baseline: %.2f A (MC) vs %.2f A (fixed Rg=0.2)",
        float(np.median(p95_per_run)),
        baseline_p95,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIC solver.")
    parser.add_argument("--grid", choices=["uiuc", "tva"], required=True)
    parser.add_argument(
        "--gannon", action="store_true", help="Use Gannon storm voltages (UIUC only)"
    )
    parser.add_argument(
        "--mc",
        action="store_true",
        help="Run Monte Carlo ensemble (TVA only, seeds 42..42+n-runs-1)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1000,
        help="Number of MC runs (default: 1000)",
    )
    args = parser.parse_args()

    if args.grid == "uiuc":
        run_uiuc(gannon=args.gannon)
    elif args.mc:
        run_tva_mc(n_runs=args.n_runs)
    else:
        run_tva()
