"""
US Social Accounting Matrix (SAM) Generator and Balancer
Authors: Dennies and Ed
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from preprocess.sam_balancer import balance_sam

from configs import setup_logger, get_data_dir

logger = setup_logger("US SAM Builder")
DATA_LOC = get_data_dir()
tables_dir = DATA_LOC / "supply_use_tables"
sam_output_dir = DATA_LOC / "sam"


def preprocess_use_table(file_path: str) -> pd.DataFrame:
    """Load and clean the BEA Use Table."""
    data = pd.read_csv(file_path)
    data.columns = [
        col if not col.startswith("Unnamed:") else "code" for col in data.columns
    ]
    data = data.copy()
    if "code" in data.columns and "Commodities/Industries" in data.columns:
        data = data.set_index(["code", "Commodities/Industries"])
    data = data.map(lambda x: str(x).strip().replace("---", "0"))
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    return data


def make_sam_accounts(use_tb_path, sup_tb_path):
    use_df = preprocess_use_table(use_tb_path)
    supply_df = preprocess_use_table(sup_tb_path)

    industry_groups = {
        "AGR": ["11"],
        "MINING": ["21"],
        "UTIL_CONST": ["22", "23"],
        "MANUF": ["31G"],
        "TRADE_TRANSP": ["42", "44RT", "48TW"],
        "INFO": ["51"],
        "FIRE": ["FIRE"],
        "PROF_OTHER": ["PROF", "81"],
        "EDUC_ENT": ["6", "7"],
        "G": ["G"],
    }

    sam_accounts = list(industry_groups.keys()) + [
        "CAP",
        "LAB",
        "IDT",
        "TRF",
        "HOH",
        "GOV",
        "INV",
        "EXT",
    ]
    sam = pd.DataFrame(0.0, index=sam_accounts, columns=sam_accounts)

    for paying_group, paying_cols in industry_groups.items():
        for receiving_group, receiving_cols in industry_groups.items():
            sam.loc[receiving_group, paying_group] = (
                use_df.loc[receiving_cols, paying_cols].sum().sum()
            )

    factor_map = {"V001": "LAB", "V003": "CAP"}
    for factor_code, factor in factor_map.items():
        for activity, cols in industry_groups.items():
            sam.loc[factor, activity] = (
                use_df.loc[(factor_code, slice(None)), cols].sum().sum()
            )

    for activity, codes in industry_groups.items():
        product_taxes = supply_df.loc[codes, "TOP"].sum()
        subsidies = supply_df.loc[codes, "SUB"].sum()
        sam.at["IDT", activity] = product_taxes - subsidies

    for activity, codes in industry_groups.items():
        production_tax = use_df.loc["T00OTOP", codes].sum().sum()
        sam.at["IDT", activity] += production_tax

    fd_map = {
        "HOH": ["F010"],
        "GOV": ["F100"],
        "INV": ["F020", "F030"],
        "EXT": ["F040"],
    }
    for inst, fd_codes in fd_map.items():
        for activity, cols in industry_groups.items():
            sam.at[activity, inst] = use_df.loc[cols, fd_codes].sum().sum()

    for activity, codes in industry_groups.items():
        sam.at["EXT", activity] = supply_df.loc[codes, "MCIF"].sum()

    for activity, codes in industry_groups.items():
        sam.at["IDT", activity] += supply_df.loc[(codes, slice(None)), "MDTY"].sum()

    sam.at["GOV", "IDT"] = sam.loc["IDT"].sum()
    sam.at["GOV", "TRF"] = sam.loc["TRF"].sum()

    sam.at["HOH", "LAB"] = sam.loc["LAB"].sum()
    sam.at["HOH", "CAP"] = sam.loc["CAP"].sum()

    add_data_path = DATA_LOC / "add_data"

    def load_and_clean_csv(file_name, skiprows, desc_col_index=1):
        path = os.path.join(add_data_path, file_name)
        df = pd.read_csv(path, skiprows=skiprows).fillna(0)
        for col in df.columns:
            if col != df.columns[desc_col_index]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def get_average_of_last_four(row):
        return row.iloc[-4:].astype(float).mean() * 1000

    tax_data = load_and_clean_csv("personal_current_tax_payments.csv", skiprows=3)
    direct_tax = (
        tax_data[
            tax_data.iloc[:, 1].str.contains("Personal current taxes", na=False)
        ].iloc[0, -1]
        * 1000
    )

    transfers_data = load_and_clean_csv("govt_transfers.csv", skiprows=3)
    transfers = (
        transfers_data[
            transfers_data.iloc[:, 1].str.contains(
                "Government social benefits", na=False
            )
        ].iloc[0, -1]
        * 1000
    )

    savings_data = load_and_clean_csv("personal_income_and_disp.csv", skiprows=4)
    savings_row = savings_data[
        savings_data.iloc[:, 1].str.contains("Personal saving", na=False)
    ].iloc[0]
    total_savings = get_average_of_last_four(savings_row)
    logger.info(
        f"Household savings: {total_savings} million (average of quarterly values)"
    )

    govt_data = load_and_clean_csv("gov_receipts_expenditures.csv", skiprows=4)
    receipts_row = govt_data[
        govt_data.iloc[:, 1].str.contains("Current receipts", na=False)
    ].iloc[0]
    expend_row = govt_data[
        govt_data.iloc[:, 1].str.contains("Current expenditures", na=False)
    ].iloc[0]
    receipts_total = get_average_of_last_four(receipts_row)
    expend_total = get_average_of_last_four(expend_row)
    govt_deficit = expend_total - receipts_total
    logger.info(
        f"Government deficit: {govt_deficit} million (based on average quarterly values)"
    )

    trade_data = load_and_clean_csv(
        "Foreign_Transactions_National_Income_and_Product_Accounts.csv", skiprows=1
    )
    exports_row = trade_data[
        trade_data.iloc[:, 1].str.contains("Exports of goods and services", na=False)
    ].iloc[0]
    imports_row = trade_data[
        trade_data.iloc[:, 1].str.contains("Imports of goods and services", na=False)
    ].iloc[0]
    exports_total = get_average_of_last_four(exports_row)
    imports_total = get_average_of_last_four(imports_row)
    trade_deficit = imports_total - exports_total
    logger.info(
        f"Trade deficit: {trade_deficit} million (based on average quarterly values)"
    )

    logger.info("Key values for SAM (in millions):")
    logger.info(f"1. Direct taxes (HOH→GOV): {direct_tax}")
    logger.info(f"2. Government transfers (GOV→HOH): {transfers}")
    logger.info(f"3. Household savings (HOH→INV): {total_savings}")
    logger.info(f"4. Government deficit (GOV→INV): {govt_deficit}")
    logger.info(f"5. Trade deficit (EXT→INV): {trade_deficit}")

    try:
        sam.loc["GOV", "HOH"] = float(direct_tax)
        sam.loc["HOH", "GOV"] = float(transfers)
        sam.loc["INV", "HOH"] = float(total_savings)
        sam.loc["INV", "GOV"] = float(govt_deficit)
    except KeyError as e:
        logger.error(f"SAM update failed: {e}")

    subsidy_amount = abs(sam.at["TRF", "AGR"])
    sam.at["TRF", "AGR"] = 0
    if (
        "AGR" in sam.index
        and "GOV" in sam.columns
        and not pd.isna(sam.at["AGR", "GOV"])
    ):
        sam.at["AGR", "GOV"] += subsidy_amount
    else:
        sam.at["AGR", "GOV"] = subsidy_amount

    subsidy_amount = abs(sam.at["TRF", "G"])
    sam.at["TRF", "G"] = 0
    if "G" in sam.index and "GOV" in sam.columns and not pd.isna(sam.at["G", "GOV"]):
        sam.at["G", "GOV"] += subsidy_amount
    else:
        sam.at["G", "GOV"] = subsidy_amount

    if "AGR" in sam.index and "INV" in sam.columns and sam.at["AGR", "INV"] < 0:
        neg_inv = abs(sam.at["AGR", "INV"])
        sam.at["AGR", "INV"] = 0
        inv_sectors = [
            sector
            for sector in industry_groups.keys()
            if sector != "AGR" and sam.at[sector, "INV"] > 0
        ]
        total_pos_inv = sum(sam.at[sector, "INV"] for sector in inv_sectors)
        if total_pos_inv > 0:
            for sector in inv_sectors:
                share = sam.at[sector, "INV"] / total_pos_inv
                sam.at[sector, "INV"] -= neg_inv * share

    try:
        trade_deficit_bea = trade_deficit
        trade_deficit_sam = sam.loc["EXT", :].sum() - sam.loc[:, "EXT"].sum()
        if abs(trade_deficit_bea - trade_deficit_sam) > 1000:
            sam.loc["INV", "EXT"] = 0.0
            sam.loc["EXT", "INV"] = max(trade_deficit_bea, 0.0)
    except KeyError as e:
        logger.error(f"Trade adjustment failed: {e}")

    return sam


def balance_and_save_sam(
    sam, output_path="output/us_balanced_sam.csv", decimal_places=1
):
    """Balance a SAM and save it to a CSV file."""
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    balanced_sam = balance_sam(sam, decimal_places=decimal_places)
    balanced_sam.to_csv(output_path)
    logger.info(f"Balanced SAM saved to {output_path}")
    return balanced_sam


if __name__ == "__main__":
    data_dir = Path("data")
    use_tb_path = tables_dir / "use_tables.csv"
    sup_tb_path = tables_dir / "supply_tables.csv"

    sam = make_sam_accounts(use_tb_path, sup_tb_path)

    balanced_sam = balance_and_save_sam(
        sam, sam_output_dir / "us_balanced_sam.csv", decimal_places=1
    )

    logger.info(f"Original row sums: {sam.sum(axis=1).sum()}")
    logger.info(f"Original column sums: {sam.sum(axis=0).sum()}")
    logger.info(f"Balanced row sums: {balanced_sam.sum(axis=1).sum()}")
    logger.info(f"Balanced column sums: {balanced_sam.sum(axis=0).sum()}")