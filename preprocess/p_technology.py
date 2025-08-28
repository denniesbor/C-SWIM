"""
Production Technology Matrix Generator for Economic Analysis
Authors: Dennies and Ed

This module processes input-output data from Bureau of Economic Analysis (BEA)
tables to create production technology matrices for economic analysis and
computable general equilibrium (CGE) modeling.
"""

import pandas as pd

from configs import setup_logger, get_data_dir

logger = setup_logger("Production Technology Builder")
DATA_LOC = get_data_dir()
tables_dir = DATA_LOC / "supply_use_tables"
g_output_dir = DATA_LOC / "gross_output"


def preprocess_use_table(file_path: str) -> pd.DataFrame:
    """Loads and cleans the BEA Use Table, converting values to billions of dollars."""
    data = pd.read_csv(file_path)
    data.columns = [
        col if not col.startswith("Unnamed:") else "code" for col in data.columns
    ]
    data = data.copy()

    if "code" in data.columns and "Commodities/Industries" in data.columns:
        data = data.set_index(["code", "Commodities/Industries"])

    data = data.map(lambda x: str(x).strip().replace("---", "0"))
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data / 1000
    return data


def clean_gross_output_file(input_path, output_path):
    """Removes HTML entities from gross output data file and creates clean version."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("+ACI-", '"')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def create_production_technology(use_table_path, output_dir=DATA_LOC / "10sector"):
    """Creates production technology matrices from BEA input-output data aggregated into 10 sectors."""

    groups = {
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

    U = preprocess_use_table(use_table_path)[1:]
    U = U.droplevel(1)

    intermediate = U.loc[
        U.index.str.match(r"^\d{1,2}G?$|^FIRE$|^PROF$|^44RT$|^48TW$|^G$")
    ]
    output_row = U.loc["T018"]

    long = (
        pd.Series(groups)
        .explode()
        .rename_axis("group")
        .reset_index()
        .rename(columns={0: "code"})
    )

    code2grp = long.set_index("code")["group"]

    U_big_temp = intermediate.rename(index=code2grp).groupby(level=0).sum()

    U_big = (
        U_big_temp.T.rename(index=code2grp)
        .groupby(level=0)
        .sum()
        .T.reindex(index=groups.keys(), columns=groups.keys())
    )

    X_big = (
        output_row.rename(index=code2grp).groupby(level=0).sum().reindex(groups.keys())
    )

    A_big = U_big.div(X_big, axis=1).round(6)

    va_rows = ["V001", "V003", "T00OTOP", "T00OSUB"]

    VA_big = (
        U.loc[va_rows, intermediate.columns]
        .rename(columns=code2grp)
        .T.groupby(level=0)
        .sum()
        .T.reindex(columns=groups.keys())
    )

    fd_cols = ["F010", "F100", "F020", "F030", "F040"]

    FD_big = (
        U.loc[intermediate.index, fd_cols]
        .rename(index=code2grp)
        .groupby(level=0)
        .sum()
        .reindex(index=groups.keys())
    )

    output_dir.mkdir(exist_ok=True)
    A_big.to_csv(output_dir / "direct_requirements.csv")
    X_big.to_csv(output_dir / "gross_output.csv", header=["2023"])
    VA_big.to_csv(output_dir / "value_added.csv")
    FD_big.to_csv(output_dir / "final_demand.csv")

    total_va = VA_big.sum(axis=0)
    total_va.to_csv(output_dir / "total_value_added.csv", header=["2023"])

    total_interuse = U_big.sum(axis=0)
    total_interuse.to_csv(output_dir / "total_intermediate_use.csv", header=["2023"])

    logger.info(f"10-sector matrices saved to {output_dir}")
    logger.info(f"Direct requirements matrix: {output_dir/'direct_requirements.csv'}")
    logger.info(f"Gross output vector: {output_dir/'gross_output.csv'}")
    logger.info(f"Value added components: {output_dir/'value_added.csv'}")
    logger.info(f"Final demand components: {output_dir/'final_demand.csv'}")

    return A_big, X_big, VA_big, FD_big


if __name__ == "__main__":
    clean_gross_output_file(
        g_output_dir / "gross_output.csv", g_output_dir / "cleaned_gross_output.csv"
    )

    A_big, X_big, VA_big, FD_big = create_production_technology(
        tables_dir / "use_tables.csv"
    )

    logger.info(f"Number of sectors: {len(A_big)}")
    logger.info(f"GDP (total value added): {VA_big.sum().sum():.1f} billion dollars")
