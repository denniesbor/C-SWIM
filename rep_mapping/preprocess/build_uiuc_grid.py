"""
Parse UIUC150 xlsx and build GIC solver inputs.
Author: Dennies
"""

import pickle
import numpy as np
import pandas as pd

from rep_mapping.rep_config import UIUC_XLSX, UIUC_DIR, setup_logger

logger = setup_logger(log_file="logs/build_uiuc_grid.log")


def parse_uiuc150():
    logger.info(f"Parsing {UIUC_XLSX}")
    xl = pd.ExcelFile(UIUC_XLSX)

    subs = xl.parse("Substations")
    subs.columns = ["sub_num", "name", "lat", "lon", "kv_max", "Rg"]

    buses = xl.parse("Buses")
    buses.columns = ["bus_num", "bus_name", "sub_num", "kv"]
    buses = buses.merge(subs[["sub_num", "lat", "lon", "Rg"]], on="sub_num", how="left")

    lines = xl.parse("Lines")
    lines.columns = [
        "from_bus",
        "to_bus",
        "circuit",
        "R_pu",
        "X_pu",
        "B_pu",
        "mva_limit",
    ]
    kv_map = dict(zip(buses.bus_num, buses.kv))
    lines["kv"] = lines["from_bus"].map(kv_map)
    lines["R_ohm"] = lines["R_pu"] * (lines["kv"] ** 2) / 100.0

    trafos = xl.parse("Transformers")
    trafos.columns = [
        "hv_bus",
        "kv_high",
        "lv_bus",
        "kv_low",
        "circuit",
        "R_pu",
        "X_pu",
        "mva_limit",
        "type",
        "R_high_ohm",
        "R_low_ohm",
        "K",
    ]

    gic_bench = xl.parse("GIC results (1 V per km East)")
    gic_bench.columns = ["hv_bus", "lv_bus", "circuit", "GIC_eff_A"]

    logger.info(
        f"Buses: {len(buses)}  Lines: {len(lines)}  "
        f"Transformers: {len(trafos)}  Substations: {len(subs)}"
    )
    return buses, lines, trafos, subs, gic_bench


def build_uiuc_grid(buses, lines, trafos, subs):
    bus_to_sub = {int(r.bus_num): int(r.sub_num) for _, r in buses.iterrows()}
    bus_coords = {
        int(r.bus_num): (float(r.lon), float(r.lat)) for _, r in buses.iterrows()
    }

    sub_look_up = {}
    idx = 0
    for bus in sorted(buses.bus_num.astype(int).unique()):
        sub_look_up[("bus", bus)] = idx
        idx += 1
    for sub in sorted(subs.sub_num.astype(int).unique()):
        sub_look_up[("sub", sub)] = idx
        idx += 1
    n_nodes = idx

    sub_ref = {}
    for _, r in trafos.iterrows():
        s = bus_to_sub.get(int(r.hv_bus))
        k = ("sub", s)
        if s is not None and k not in sub_ref:
            sub_ref[k] = [("bus", int(r.hv_bus)), ("bus", int(r.lv_bus))]

    df_transformers = trafos.copy()
    df_transformers["sub_id"] = df_transformers.hv_bus.apply(
        lambda x: bus_to_sub.get(int(x))
    )
    df_transformers["name"] = df_transformers.apply(
        lambda r: f"T_{int(r.hv_bus)}_{int(r.lv_bus)}_{int(r.circuit)}", axis=1
    )
    df_transformers["type"] = df_transformers["type"].map(
        {"GWye-GWye Auto": "Auto", "GWye-Delta": "GY-D"}
    )
    df_transformers["bus1"] = df_transformers.hv_bus.apply(lambda x: ("bus", int(x)))
    df_transformers["bus2"] = df_transformers.lv_bus.apply(lambda x: ("bus", int(x)))
    df_transformers["W1"] = df_transformers["R_high_ohm"]
    df_transformers["W2"] = df_transformers["R_low_ohm"]
    df_transformers["sub"] = df_transformers["sub_id"].apply(lambda x: ("sub", int(x)))
    df_transformers["latitude"] = df_transformers.hv_bus.apply(
        lambda x: bus_coords.get(int(x), (None, None))[1]
    )
    df_transformers["longitude"] = df_transformers.hv_bus.apply(
        lambda x: bus_coords.get(int(x), (None, None))[0]
    )

    def line_emf(fb, tb, Ex, Ey):
        c1 = bus_coords.get(fb)
        c2 = bus_coords.get(tb)
        if c1 is None or c2 is None:
            return 0.0
        lat_mid = np.radians((c1[1] + c2[1]) / 2)
        dx = np.radians(c2[0] - c1[0]) * 6371 * np.cos(lat_mid)
        dy = np.radians(c2[1] - c1[1]) * 6371
        return Ex * dx + Ey * dy

    df_lines = lines[["from_bus", "to_bus", "R_ohm"]].copy()
    df_lines.rename(columns={"R_ohm": "R"}, inplace=True)
    df_lines["name"] = df_lines.apply(
        lambda r: f"{int(r.from_bus)}_{int(r.to_bus)}", axis=1
    )
    df_lines["V_eastward"] = df_lines.apply(
        lambda r: line_emf(int(r.from_bus), int(r.to_bus), -1.0, 0.0), axis=1
    )
    df_lines["V_northward"] = df_lines.apply(
        lambda r: line_emf(int(r.from_bus), int(r.to_bus), 0.0, -1.0), axis=1
    )
    df_lines["from_bus"] = df_lines["from_bus"].apply(lambda x: ("bus", int(x)))
    df_lines["to_bus"] = df_lines["to_bus"].apply(lambda x: ("bus", int(x)))

    substations_df = pd.DataFrame(
        {
            "name": subs.sub_num.astype(int).apply(lambda x: ("sub", x)),
            "latitude": subs.lat.values,
            "longitude": subs.lon.values,
            "grounding_resistance": subs.Rg.values,
        }
    )

    return (
        sub_look_up,
        sub_ref,
        n_nodes,
        df_transformers,
        df_lines,
        substations_df,
        bus_coords,
    )


def main():
    buses, lines, trafos, subs, gic_bench = parse_uiuc150()
    (
        sub_look_up,
        sub_ref,
        n_nodes,
        df_transformers,
        df_lines,
        substations_df,
        bus_coords,
    ) = build_uiuc_grid(buses, lines, trafos, subs)

    with open(UIUC_DIR / "sub_look_up.pkl", "wb") as f:
        pickle.dump(sub_look_up, f)
    with open(UIUC_DIR / "sub_ref.pkl", "wb") as f:
        pickle.dump(sub_ref, f)
    with open(UIUC_DIR / "df_transformers.pkl", "wb") as f:
        pickle.dump(df_transformers, f)
    with open(UIUC_DIR / "df_lines.pkl", "wb") as f:
        pickle.dump(df_lines, f)
    with open(UIUC_DIR / "substations_df.pkl", "wb") as f:
        pickle.dump(substations_df, f)
    with open(UIUC_DIR / "bus_coords.pkl", "wb") as f:
        pickle.dump(bus_coords, f)

    gic_bench.to_parquet(UIUC_DIR / "gic_bench.parquet")
    np.save(UIUC_DIR / "n_nodes.npy", np.array([n_nodes]))

    logger.info(f"n_nodes:      {n_nodes}")
    logger.info(f"Transformers: {len(df_transformers)}")
    logger.info(f"Lines:        {len(df_lines)}")
    logger.info(f"bus_coords:   {len(bus_coords)}")
    logger.info(f"Saved to {UIUC_DIR}")


if __name__ == "__main__":
    main()
