"""Compute effective GIC per transformer from winding GIC CSVs."""

import glob
import pickle
from pathlib import Path


import pandas as pd
import numpy as np
from tqdm import tqdm

from configs import setup_logger, get_data_dir

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/calc_eff_gic.log")

data_path = DATA_LOC / "admittance_matrix/sample_network.pkl"
with open(data_path, "rb") as f:
    df_transformers = pickle.load(f)
    logger.info(f"Loaded {len(df_transformers)} pre-generated samples from {data_path}")


def calculate_effective_gic(gic_path, df_transformers, output_dir):
    """Calculate effective GIC from winding GIC data."""
    try:
        df_gic = pd.read_csv(gic_path)
        df_transformer = df_transformers[int(gic_path.stem.split("_")[-1])]
        hazard_cols = [col for col in df_gic.columns if "year-hazard" in col]

        df_meta = (
            df_gic.groupby("Transformer")
            .agg(
                {
                    "sub_id": "first",
                    "latitude": lambda x: np.round(x.iloc[0], 2),
                    "longitude": lambda x: np.round(x.iloc[0], 2),
                }
            )
            .reset_index()
        )

        df_meta = df_meta.merge(
            df_transformer[["name", "type", "bus1_id", "bus2_id"]].rename(
                columns={"name": "Transformer"}
            ),
            on="Transformer",
        )

        df_pivot = df_gic.pivot_table(
            index="Transformer", columns="Winding", values=hazard_cols, aggfunc="first"
        ).reset_index()

        df_pivot.columns = [
            "_".join(col).strip("_") if col[1] else col[0] for col in df_pivot.columns
        ]

        df_result = df_meta.merge(df_pivot, on="Transformer")

        # Calculate voltage ratios for multi-winding transformers
        multi_winding_mask = df_result["type"].isin(["Auto", "GY-GY", "GY-GY-D"])
        if multi_winding_mask.any():
            try:
                bus1_voltages = (
                    df_result.loc[multi_winding_mask, "bus1_id"]
                    .str.split("_")
                    .str[-1]
                    .astype(float)
                )
                bus2_voltages = (
                    df_result.loc[multi_winding_mask, "bus2_id"]
                    .str.split("_")
                    .str[-1]
                    .astype(float)
                )
                df_result.loc[multi_winding_mask, "v_ratio"] = np.minimum(
                    bus1_voltages, bus2_voltages
                ) / np.maximum(bus1_voltages, bus2_voltages)
            except Exception as e:
                logger.warning(f"Error calculating voltage ratios: {e}")
                df_result.loc[multi_winding_mask, "v_ratio"] = 1.0

        # Compute effective GIC based on transformer type
        for hazard_col in hazard_cols:
            hv_col = f"{hazard_col}_HV"
            lv_col = f"{hazard_col}_LV"
            series_col = f"{hazard_col}_Series"
            common_col = f"{hazard_col}_Common"

            effective_gic = pd.Series(index=df_result.index, dtype=float, data=np.nan)

            # Single-winding: I_eff = I_HV
            single_mask = df_result["type"].isin(
                ["GSU", "GSU w/ GIC BD", "GY-D", "GY-D w/ GIC BD"]
            )
            if single_mask.any() and hv_col in df_result.columns:
                hv_data = pd.to_numeric(
                    df_result.loc[single_mask, hv_col], errors="coerce"
                )
                effective_gic[single_mask] = hv_data

            # Auto: I_eff = I_series + I_common * v_ratio
            auto_mask = df_result["type"] == "Auto"
            if (
                auto_mask.any()
                and series_col in df_result.columns
                and common_col in df_result.columns
            ):
                series_data = pd.to_numeric(
                    df_result.loc[auto_mask, series_col], errors="coerce"
                )
                common_data = pd.to_numeric(
                    df_result.loc[auto_mask, common_col], errors="coerce"
                )
                v_ratio_data = pd.to_numeric(
                    df_result.loc[auto_mask, "v_ratio"], errors="coerce"
                )
                if (
                    series_data.isna().any()
                    or common_data.isna().any()
                    or v_ratio_data.isna().any()
                ):
                    logger.warning(
                        f"Non-numeric data in auto transformer calculation for {hazard_col}"
                    )
                effective_gic[auto_mask] = series_data + (common_data * v_ratio_data)

            # GY-GY: I_eff = I_HV + I_LV * v_ratio
            gy_mask = df_result["type"].isin(["GY-GY", "GY-GY-D"])
            if (
                gy_mask.any()
                and hv_col in df_result.columns
                and lv_col in df_result.columns
            ):
                hv_data = pd.to_numeric(df_result.loc[gy_mask, hv_col], errors="coerce")
                lv_data = pd.to_numeric(df_result.loc[gy_mask, lv_col], errors="coerce")
                v_ratio_data = pd.to_numeric(
                    df_result.loc[gy_mask, "v_ratio"], errors="coerce"
                )
                if (
                    hv_data.isna().any()
                    or lv_data.isna().any()
                    or v_ratio_data.isna().any()
                ):
                    logger.warning(
                        f"Non-numeric data in GY-GY calculation for {hazard_col}"
                    )
                effective_gic[gy_mask] = hv_data + (lv_data * v_ratio_data)

            df_result[f"e_{hazard_col}"] = effective_gic

        final_cols = ["Transformer", "type", "sub_id", "latitude", "longitude"] + [
            f"e_{col}" for col in hazard_cols
        ]
        df_effective = df_result[final_cols].copy()

        output_path = (
            output_dir / f"effective_gic_rand_{gic_path.stem.split('_')[-1]}.csv"
        )
        df_effective.to_csv(output_path, index=False)

    except Exception as e:
        logger.error(f"Error processing {gic_path}: {e}")
        raise


def main():
    """Process all winding GIC files and calculate effective GIC."""
    gic_dir = DATA_LOC / "gic"
    output_dir = DATA_LOC / "gic_eff"
    output_dir.mkdir(parents=True, exist_ok=True)

    gic_files = glob.glob(str(gic_dir / "winding_gic_rand_*.csv"))

    for gic_file in tqdm(gic_files, desc="Processing GIC files"):
        gic_path = Path(gic_file)
        try:
            calculate_effective_gic(gic_path, df_transformers, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {gic_path}: {e}")
            continue


if __name__ == "__main__":
    main()
