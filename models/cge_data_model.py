"""
CGE Data Model Manager
Author: Dennies Bor
Date: August 2025

This module provides data management and execution framework for Computable General Equilibrium (CGE) models.
Handles Social Accounting Matrix (SAM) data preparation, model execution, and solution validation.
"""

import os
import argparse

import numpy as np
import pandas as pd
from models.cge_model import StdModelDef

from configs import setup_logger, get_data_dir

DATA_LOC = get_data_dir()
logger = setup_logger("SAMDataManager")

IPOPT_EXEC = "/home/pve_ubuntu/miniconda3/envs/spw-env/bin/ipopt"


class SAMDataManager:
    """
    Manages Social Accounting Matrix (SAM) data preparation and storage
    for CGE modeling.
    """

    def __init__(self, data_dir=None):

        self.data_dir = data_dir if data_dir else "sam_data"

    def load_sam(self, sam_type="japan"):

        if sam_type.lower() == "japan":
            return self.text_book_japan_sam()
        elif sam_type.lower() == "simple":
            return self.text_book_simple_sam()
        elif sam_type.lower() == "us":
            return self.get_us_sam()
        else:
            raise ValueError(f"Unknown SAM type: {sam_type}")

    def text_book_japan_sam(self):
        """Create textbook example SAM for Japan"""
        data = [
            [1558469, 8427693, 1496991, 0, 0, 0, 0, 3965927, 0, 967198, 72018],
            [
                2462949,
                130321011,
                68646492,
                0,
                0,
                0,
                0,
                64927161,
                459179,
                39070695,
                46597315,
            ],
            [
                2273437,
                63503715,
                160713811,
                0,
                0,
                0,
                0,
                231268309,
                85247038,
                90250845,
                10817384,
            ],
            [6167952, 33816669, 149889160, 0, 0, 0, 0, 0, 0, 0, 0],
            [1372650, 59034654, 234353029, 0, 0, 0, 0, 0, 0, 0, 0],
            [534232, 14436136, 19877410, 0, 0, 0, 0, 0, 34847778, 0, 0],
            [144478, 3690547, 911, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 189873781, 294760333, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 34847778, 3835936, 44207200, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 140265517, -2815303, 0, -7161476],
            [1974129, 39254377, 9096735, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        labels = [
            "AGR",
            "MAN",
            "SRV",
            "CAP",
            "LAB",
            "IDT",
            "TRF",
            "HOH",
            "GOV",
            "INV",
            "EXT",
        ]
        h_list = ["CAP", "LAB"]
        i_list = ["AGR", "MAN", "SRV"]
        u_list = labels

        sam_japan = pd.DataFrame(data, index=labels, columns=labels, dtype=np.float64)
        sam_japan.loc["IDT", "GOV"] = 0

        factor = 1e-5
        return sam_japan * factor, h_list, i_list, u_list

    def text_book_simple_sam(self):
        """
        Create a simple textbook example SAM.
        Data matrix represents flows between: BRD (Bread), MLK (Milk), CAP (Capital),
        LAB (Labor), IDT (Indirect Tax), TRF (Transfers), HOH (Households),
        GOV (Government), INV (Investment), EXT (External/Rest of World)
        """

        data = [
            [21, 8, None, None, None, None, 20, 19, 16, 8],  # BRD
            [17, 9, None, None, None, None, 30, 14, 15, 4],  # MLK
            [20, 30, None, None, None, None, None, None, None, None],  # CAP
            [15, 25, None, None, None, None, None, None, None, None],  # LAB
            [5, 4, None, None, None, None, None, None, None, None],  # IDT
            [1, 2, None, None, None, None, None, None, None, None],  # TRF
            [None, None, 50, 40, None, None, None, None, None, None],  # HOH
            [None, None, None, None, 9, 3, 23, None, None, None],  # GOV
            [None, None, None, None, None, None, 17, 2, None, 12],  # INV
            [13, 11, None, None, None, None, None, None, None, None],  # EXT
        ]

        labels = ["BRD", "MLK", "CAP", "LAB", "IDT", "TRF", "HOH", "GOV", "INV", "EXT"]

        h_list = ["CAP", "LAB"]
        i_list = ["BRD", "MLK"]
        u_list = ["BRD", "MLK", "CAP", "LAB", "IDT", "TRF", "HOH", "GOV", "INV", "EXT"]

        df = pd.DataFrame(data, index=labels, columns=labels)
        df = df.astype(np.float64)

        return df, h_list, i_list, u_list

    def get_us_sam(self):
        """
        Load the US SAM data.
        """

        # Sam US
        sam_us = pd.read_csv(DATA_LOC / "sam" / "us_balanced_sam.csv", index_col=0)

        print(sam_us)

        # convert just the numeric data
        sam_us = sam_us.astype(np.float64)

        # Define account lists
        h_list = ["CAP", "LAB"]
        i_list = [
            "AGR",
            "MINING",
            "UTIL_CONST",
            "MANUF",
            "TRADE_TRANSP",
            "INFO",
            "FIRE",
            "PROF_OTHER",
            "EDUC_ENT",
            "G",
        ]
        u_list = [
            "AGR",
            "MINING",
            "UTIL_CONST",
            "MANUF",
            "TRADE_TRANSP",
            "INFO",
            "FIRE",
            "PROF_OTHER",
            "EDUC_ENT",
            "G",
            "CAP",
            "LAB",
            "IDT",
            "TRF",
            "HOH",
            "GOV",
            "INV",
            "EXT",
        ]

        # Scale the SAM data
        factor = 1e-3
        return sam_us * factor, h_list, i_list, u_list

    def prepare_sam_data(self, sam_data, output_name=None):
        """
        Prepare and save SAM data files for CGE modeling.
        """
        sam_df, h_list, i_list, u_list = sam_data
        data_dir = f"{output_name or 'default'}_data"
        data_dir = os.path.join(self.data_dir, data_dir)

        os.makedirs(data_dir, exist_ok=True)

        # Write set files
        with open(os.path.join(data_dir, "set-h.csv"), "w") as f:
            f.write("h\n" + "\n".join(h_list))

        with open(os.path.join(data_dir, "set-i.csv"), "w") as f:
            f.write("i\n" + "\n".join(i_list))

        with open(os.path.join(data_dir, "set-u.csv"), "w") as f:
            f.write("u\n" + "\n".join(u_list))

        # Write param-sam.csv
        sam_df.to_csv(os.path.join(data_dir, "param-sam.csv"), index=True, header=True)

        logger.info(f"Created SAM data files in: {data_dir}")
        return data_dir


class CGEModelRunner:
    """
    Runs CGE models with prepared SAM data.
    """

    def __init__(self, model_def_class=StdModelDef):

        self.model_def_class = model_def_class

    def run_model(self, data_dir, solver_options=None, tee=True):
        """
        Run the CGE model
        """

        from pyomo.environ import DataPortal, SolverFactory

        # Default solver options
        if solver_options is None:
            solver_options = {
                "hessian_approximation": "limited-memory",
                "max_iter": 5000,
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
                "bound_push": 1e-2,
                "mu_strategy": "adaptive",
                "print_level": 5,
                "halt_on_ampl_error": "yes",
            }

        # Build model and load data
        model_def = self.model_def_class()
        abstract_model = model_def.model()
        data = DataPortal()

        # Load sets and data
        data.load(filename=os.path.join(data_dir, "set-h.csv"), format="set", set="h")
        data.load(filename=os.path.join(data_dir, "set-i.csv"), format="set", set="i")
        data.load(filename=os.path.join(data_dir, "set-u.csv"), format="set", set="u")
        data.load(
            filename=os.path.join(data_dir, "param-sam.csv"),
            param="sam",
            format="array",
        )

        # Create instance and solve
        instance = abstract_model.create_instance(data)
        solver = SolverFactory("ipopt", executable=IPOPT_EXEC)

        for option, value in solver_options.items():
            solver.options[option] = value

        results = solver.solve(instance, tee=tee)
        logger.info(f"Solver status: {results.solver.status}")
        logger.info(f"Termination condition: {results.solver.termination_condition}")

        return instance


def compare_solution_vs_sam(instance, sam_df, scale_factor=1.0):
    """
    Compare key flows from a solved Pyomo instance against SAM data.
    """
    comparison = []

    # Compare intermediate input flows: X[i,j]
    for i in instance.i:
        for j in instance.i:
            model_val = (instance.X[i, j].value or 0.0) / scale_factor
            sam_val = (
                sam_df.loc[i, j] if (i in sam_df.index and j in sam_df.columns) else 0.0
            )
            diff = model_val - sam_val
            pct = 100 * diff / sam_val if abs(sam_val) > 1e-12 else None
            comparison.append(
                {
                    "Flow": "X",
                    "From": i,
                    "To": j,
                    "SAM": sam_val,
                    "Model": model_val,
                    "Diff": diff,
                    "PctDiff": pct,
                }
            )

    # Compare factor usage: F[h,i]
    for h in instance.h:
        for i in instance.i:
            model_val = (instance.F[h, i].value or 0.0) / scale_factor
            sam_val = (
                sam_df.loc[h, i] if (h in sam_df.index and i in sam_df.columns) else 0.0
            )
            diff = model_val - sam_val
            pct = 100 * diff / sam_val if abs(sam_val) > 1e-12 else None
            comparison.append(
                {
                    "Flow": "F",
                    "From": h,
                    "To": i,
                    "SAM": sam_val,
                    "Model": model_val,
                    "Diff": diff,
                    "PctDiff": pct,
                }
            )

    # Compare Household Consumption: Xp[i]
    for i in instance.i:
        model_val = (instance.Xp[i].value or 0.0) / scale_factor
        sam_val = (
            sam_df.loc[i, "HOH"]
            if (i in sam_df.index and "HOH" in sam_df.columns)
            else 0.0
        )
        diff = model_val - sam_val
        pct = 100 * diff / sam_val if abs(sam_val) > 1e-12 else None
        comparison.append(
            {
                "Flow": "Xp",
                "From": i,
                "To": "HOH",
                "SAM": sam_val,
                "Model": model_val,
                "Diff": diff,
                "PctDiff": pct,
            }
        )

    # Compare Government Consumption: Xg[i]
    for i in instance.i:
        model_val = (instance.Xg[i].value or 0.0) / scale_factor
        sam_val = (
            sam_df.loc[i, "GOV"]
            if (i in sam_df.index and "GOV" in sam_df.columns)
            else 0.0
        )
        diff = model_val - sam_val
        pct = 100 * diff / sam_val if abs(sam_val) > 1e-12 else None
        comparison.append(
            {
                "Flow": "Xg",
                "From": i,
                "To": "GOV",
                "SAM": sam_val,
                "Model": model_val,
                "Diff": diff,
                "PctDiff": pct,
            }
        )

    # To be extended with otjer flows

    return pd.DataFrame(comparison)


def print_comparison_table(df_comp):

    # Reorder columns (if desired)
    df_comp = df_comp[["Flow", "From", "To", "SAM", "Model", "Diff", "PctDiff"]]

    # Round numeric columns for neatness
    df_comp["SAM"] = df_comp["SAM"].round(4)
    df_comp["Model"] = df_comp["Model"].round(4)
    df_comp["Diff"] = df_comp["Diff"].round(4)
    df_comp["PctDiff"] = df_comp["PctDiff"].round(2)

    # Print the table
    logger.info(df_comp.to_string(index=False))


def run_cge_example(sam_type="japan", display_results=True):

    # Initialize objects
    sam_manager = SAMDataManager(DATA_LOC / "sam_data")
    model_runner = CGEModelRunner()

    # Load and prepare data
    sam_data = sam_manager.load_sam(sam_type)
    data_dir = sam_manager.prepare_sam_data(sam_data, output_name=f"{sam_type}_example")
    # Logger info data dir
    print("Data dir", data_dir)

    # Run model
    instance = model_runner.run_model(data_dir)

    # Display results if requested
    if display_results:
        instance.display()

    return instance


def main():
    parser = argparse.ArgumentParser(
        description="Run the CGE model against a chosen SAM (Japan, US, or simple)."
    )
    parser.add_argument(
        "sam_type",
        choices=["japan", "us", "simple"],
        help="Which SAM to load and run: 'japan', 'us', or 'simple'.",
    )
    args = parser.parse_args()
    sam_type = args.sam_type.lower()

    # Initialize SAM manager
    sam_manager = SAMDataManager(DATA_LOC / "sam_data")
    sam_df, h_list, i_list, u_list = sam_manager.load_sam(sam_type)

    logger.info(f"Running CGE model with {sam_type.upper()} SAM...")
    instance = run_cge_example(sam_type=sam_type)

    factor = 1

    logger.info("\nValidating model solution against original SAM data:")
    df_comparison = compare_solution_vs_sam(instance, sam_df, scale_factor=factor)
    print_comparison_table(df_comparison)

    logger.info("\nKey Economic Indicators:")
    logger.info("-----------------------")
    gdp = sum(
        instance.pf[h].value * sum(instance.F[h, i].value for i in instance.i)
        for h in instance.h
    )
    logger.info(f"GDP: {gdp:.4f}")


if __name__ == "__main__":
    main()
