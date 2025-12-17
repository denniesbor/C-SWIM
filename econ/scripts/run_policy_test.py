"""
Economic Policy Testing Framework

Author: Dennies Bor
Date: 2025

This module compares economic impacts of electricity failures using both
Input-Output (IO) and Computable General Equilibrium (CGE) modeling approaches.
"""

from typing import Dict

import numpy as np
import pandas as pd
from pyomo.environ import value
import matplotlib.pyplot as plt

from models.io_model import InputOutputModel
from models.cge_data_model import SAMDataManager, run_cge_example
from configs import IPOPT_EXEC, setup_logger

logger = setup_logger("Policy Test Runner")


def compare_electricity_failure_impacts(
    affected_sectors: Dict[str, float],
    sam_type: str = "us",
    output_dir: str = "results",
):
    """Compare the economic impacts of electricity failures using both IO and CGE modeling approaches."""
    logger.info(
        f"Analyzing economic impacts of electricity failure on {len(affected_sectors)} sectors"
    )

    io_results = run_io_analysis(affected_sectors, sam_type)
    cge_results = run_cge_analysis(affected_sectors, sam_type)
    comparison = compare_model_results(io_results, cge_results)

    return {
        "io_results": io_results,
        "cge_results": cge_results,
        "comparison": comparison,
    }


def run_io_analysis(affected_sectors: Dict[str, float], sam_type: str = "us"):
    """Run Input-Output analysis for electricity failure impacts."""
    logger.info("Running Input-Output Analysis")

    io = InputOutputModel("10sector")

    va_data = None
    try:
        va_df = pd.read_csv(io.path / "value_added.csv", index_col=0)
        va_data = pd.Series(va_df.sum(axis=0), index=io.sectors)
        logger.info("Loaded value-added data successfully")
    except Exception as e:
        logger.warning(f"Could not load value-added data: {e}")
        va_data = pd.Series(io.X * 0.45, index=io.sectors)
        logger.info("Using estimated value-added data instead")

    d_va = pd.Series(0, index=io.sectors)

    for sector, impact_pct in affected_sectors.items():
        if sector in d_va.index:
            sector_va = va_data[sector]
            shock_value = -sector_va * impact_pct
            d_va[sector] = shock_value
            logger.info(
                f"Applied {impact_pct*100:.1f}% reduction to {sector} value-added → shock of {shock_value:.2f}"
            )

    logger.info(f"Value-added shock vector: {d_va.round(2).to_dict()}")

    dx_va = io.total_output_from_value_added(d_va)
    multipliers = io.output_multipliers()

    logger.info(
        f"IO Model Results - Output changes by sector: {dx_va.round(2).to_dict()}"
    )
    logger.info(f"Total output impact: {dx_va.sum():.2f}")

    total_impact = dx_va.sum()
    sector_contributions = (
        (dx_va / total_impact * 100)
        if total_impact != 0
        else pd.Series(0, index=io.sectors)
    )

    significant_contributions = {
        sector: sector_contributions[sector]
        for sector in io.sectors
        if abs(sector_contributions[sector]) > 1
    }
    if significant_contributions:
        logger.info(f"Sector contribution to total impact: {significant_contributions}")

    return {
        "output_changes": dx_va,
        "shock_vector": d_va,
        "value_added": va_data,
        "multipliers": multipliers,
        "sector_contributions": sector_contributions,
        "total_impact": total_impact,
        "model_type": "IO",
    }


def run_policy(model_instance, policy_params, scenario_name="Policy Simulation"):
    """Run a policy simulation using the standard CGE model."""
    from pyomo.opt import SolverFactory

    baseline = {
        "Y": {i: model_instance.Y[i].value for i in model_instance.i},
        "Z": {i: model_instance.Z[i].value for i in model_instance.i},
        "Q": {i: model_instance.Q[i].value for i in model_instance.i},
        "Xp": {i: model_instance.Xp[i].value for i in model_instance.i},
        "Xg": {i: model_instance.Xg[i].value for i in model_instance.i},
        "Xv": {i: model_instance.Xv[i].value for i in model_instance.i},
        "E": {i: model_instance.E[i].value for i in model_instance.i},
        "M": {i: model_instance.M[i].value for i in model_instance.i},
        "F": {
            (h, i): model_instance.F[h, i].value
            for h in model_instance.h
            for i in model_instance.i
        },
        "pq": {i: model_instance.pq[i].value for i in model_instance.i},
        "pf": {h: model_instance.pf[h].value for h in model_instance.h},
        "pz": {i: model_instance.pz[i].value for i in model_instance.i},
        "Td": model_instance.Td.value,
        "Sp": model_instance.Sp.value,
        "Sg": model_instance.Sg.value,
        "epsilon": model_instance.epsilon.value,
        "welfare": model_instance.obj.expr(),
    }

    policy_model = model_instance.clone()
    policy_model.epsilon.set_value(1.0)

    logger.info(f"Applying policy scenario: {scenario_name}")
    for param_name, param_value in policy_params.items():
        if "." in param_name:
            base_param, index = param_name.split(".")
            if hasattr(policy_model, base_param):
                if index in getattr(policy_model, base_param):
                    setattr(
                        getattr(policy_model, base_param)[index], "value", param_value
                    )
                    logger.info(f"Set {base_param}[{index}] = {param_value}")
                else:
                    logger.warning(f"Index {index} not found in {base_param}")
            else:
                logger.warning(f"Parameter {base_param} not found in model")
        else:
            if hasattr(policy_model, param_name):
                setattr(policy_model, param_name, param_value)
                logger.info(f"Set {param_name} = {param_value}")
            else:
                logger.warning(f"Parameter {param_name} not found in model")

    solver = SolverFactory("ipopt", executable=IPOPT_EXEC)
    solver.options["max_iter"] = 5000
    solver.options["tol"] = 1e-6

    logger.info("Solving policy scenario...")
    results = solver.solve(policy_model, tee=True)

    gov_LHS = (
        sum(value(policy_model.pq[i] * policy_model.Xg[i]) for i in policy_model.i)
        + value(policy_model.Tr)
        + value(policy_model.Sg)
    )
    gov_RHS = (
        value(policy_model.Td)
        + sum(value(policy_model.Tz[i]) for i in policy_model.i)
        + sum(value(policy_model.Tm[i]) for i in policy_model.i)
    )
    print("Gov budget gap:", gov_LHS - gov_RHS)

    for h in policy_model.h:
        gap = sum(value(policy_model.F[h, i]) for i in policy_model.i) - value(
            policy_model.FF[h]
        )
        print(f"Factor {h} market gap:", gap)

    bop_gap = (
        sum(value(policy_model.pWe[i] * policy_model.E[i]) for i in policy_model.i)
        + value(policy_model.Sf)
        - sum(value(policy_model.pWm[i] * policy_model.M[i]) for i in policy_model.i)
    )
    print("BoP gap:", bop_gap)

    baseline_gdp = sum(
        value(model_instance.py[i]) * value(model_instance.Y[i])
        for i in model_instance.i
    )
    policy_gdp = sum(
        value(policy_model.py[i]) * value(policy_model.Y[i]) for i in policy_model.i
    )
    baseline["gdp"] = baseline_gdp

    policy_results = {
        "Y": {i: policy_model.Y[i].value for i in policy_model.i},
        "Z": {i: policy_model.Z[i].value for i in policy_model.i},
        "Q": {i: policy_model.Q[i].value for i in policy_model.i},
        "Xp": {i: policy_model.Xp[i].value for i in policy_model.i},
        "Xg": {i: policy_model.Xg[i].value for i in policy_model.i},
        "Xv": {i: policy_model.Xv[i].value for i in policy_model.i},
        "E": {i: policy_model.E[i].value for i in policy_model.i},
        "M": {i: policy_model.M[i].value for i in policy_model.i},
        "F": {
            (h, i): policy_model.F[h, i].value
            for h in policy_model.h
            for i in policy_model.i
        },
        "pq": {i: policy_model.pq[i].value for i in policy_model.i},
        "pf": {h: policy_model.pf[h].value for h in policy_model.h},
        "pz": {i: policy_model.pz[i].value for i in policy_model.i},
        "Td": policy_model.Td.value,
        "Sp": policy_model.Sp.value,
        "Sg": policy_model.Sg.value,
        "epsilon": policy_model.epsilon.value,
        "welfare": policy_model.obj.expr(),
        "gdp": policy_gdp,
    }

    pct_changes = {}
    for var in ["Y", "Z", "Q", "Xp", "Xg", "Xv", "E", "M", "pq", "pz"]:
        pct_changes[var] = {
            i: (
                ((policy_results[var][i] / baseline[var][i] - 1) * 100)
                if baseline[var][i] != 0
                else float("nan")
            )
            for i in policy_model.i
        }

    pct_changes["pf"] = {
        h: (
            ((policy_results["pf"][h] / baseline["pf"][h] - 1) * 100)
            if baseline["pf"][h] != 0
            else float("nan")
        )
        for h in policy_model.h
    }

    pct_changes["F"] = {
        (h, i): (
            ((policy_results["F"][(h, i)] / baseline["F"][(h, i)] - 1) * 100)
            if baseline["F"][(h, i)] != 0
            else float("nan")
        )
        for h in policy_model.h
        for i in policy_model.i
    }

    for var in ["Td", "Sp", "Sg", "epsilon", "welfare", "gdp"]:
        pct_changes[var] = (
            ((policy_results[var] / baseline[var] - 1) * 100)
            if baseline[var] != 0
            else float("nan")
        )

    logger.info("=" * 80)
    logger.info(f"RESULTS FOR {scenario_name}")
    logger.info("=" * 80)

    return {
        "baseline": baseline,
        "policy": policy_results,
        "pct_changes": pct_changes,
        "solver_status": {
            "status": results.solver.status,
            "termination_condition": results.solver.termination_condition,
        },
    }


def run_cge_analysis(affected_sectors: Dict[str, float], sam_type: str):
    """Run CGE analysis for electricity failure impacts."""
    logger.info("Running CGE Analysis")

    sam_manager = SAMDataManager()
    sam_data = sam_manager.load_sam(sam_type)
    sam_df, h_list, i_list, u_list = sam_data

    logger.info("Running baseline model...")
    baseline_instance = run_cge_example(sam_type=sam_type, display_results=False)

    policy_params = {}

    for sector, impact in affected_sectors.items():
        if sector in baseline_instance.i:
            current_b = value(baseline_instance.b[sector])
            policy_params[f"b.{sector}"] = current_b * (1 - impact)
            logger.info(f"Reducing productivity in {sector} by {impact*100:.1f}%")

    logger.info("Running policy scenario...")
    results = run_policy(
        baseline_instance, policy_params, scenario_name="Electricity Failure Scenario"
    )
    output_changes = {}
    for i in baseline_instance.i:
        baseline_output = results["baseline"]["Z"][i]
        policy_output = results["policy"]["Z"][i]
        output_changes[i] = policy_output - baseline_output

    output_changes_series = pd.Series(output_changes)

    return {
        "output_changes": output_changes_series,
        "baseline": results["baseline"],
        "policy": results["policy"],
        "pct_changes": results["pct_changes"],
        "gdp_impact": results["policy"]["gdp"] - results["baseline"]["gdp"],
        "model_type": "CGE",
    }


def compare_model_results(io_results, cge_results):
    """Compare IO and CGE model results."""
    logger.info("Comparing IO and CGE Results")

    all_sectors = sorted(
        set(io_results["output_changes"].index)
        | set(cge_results["output_changes"].index)
    )

    comparison = pd.DataFrame(index=all_sectors)

    comparison["IO_Output_Change"] = io_results["output_changes"]
    comparison["CGE_Output_Change"] = cge_results["output_changes"]

    io_total = io_results["output_changes"].sum()
    cge_total = cge_results["output_changes"].sum()

    comparison["IO_Pct_of_Total"] = comparison["IO_Output_Change"] / io_total * 100
    comparison["CGE_Pct_of_Total"] = comparison["CGE_Output_Change"] / cge_total * 100

    comparison["Difference"] = (
        comparison["CGE_Output_Change"] - comparison["IO_Output_Change"]
    )
    comparison["Difference_Pct"] = (
        comparison["Difference"] / comparison["IO_Output_Change"].abs() * 100
    )

    comparison = comparison.fillna(0)

    logger.info(
        f"Sectoral impacts comparison (IO vs CGE): {comparison.round(2).to_dict()}"
    )

    logger.info(f"IO Total Output Impact: {io_total:.2f}")
    logger.info(f"CGE Total Output Impact: {cge_total:.2f}")
    logger.info(
        f"Difference: {cge_total - io_total:.2f} ({(cge_total - io_total)/io_total*100:.2f}%)"
    )

    if "gdp_impact" in cge_results:
        logger.info(f"CGE GDP Impact: {cge_results['gdp_impact']:.2f}")

    return comparison


def plot_comparison(comparison, output_dir, filename_base):
    import os

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    sectors = comparison.index
    x = np.arange(len(sectors))
    width = 0.4

    plt.bar(x - width / 2, comparison["IO_Output_Change"], width, label="IO Model")
    plt.bar(x + width / 2, comparison["CGE_Output_Change"], width, label="CGE Model")

    plt.xlabel("Sectors")
    plt.ylabel("Output Change")
    plt.title("IO vs CGE: Output Changes by Sector")
    plt.xticks(x, sectors, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/output_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.bar(sectors, comparison["Difference_Pct"], color="steelblue")
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Sectors")
    plt.ylabel("Difference (%)")
    plt.title("Percentage Difference: CGE vs IO Impacts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_base}_pct_difference.png")
    plt.close()

    logger.info(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    affected_sectors = {
        "UTIL_CONST": 0.20,
        "MANUF": 0.15,
        "INFO": 0.10,
    }

    results = compare_electricity_failure_impacts(
        affected_sectors, sam_type="us", output_dir="electricity_failure_results"
    )