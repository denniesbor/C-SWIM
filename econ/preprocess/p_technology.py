"""
Production Technology Matrix Generator for Economic Analysis
Authors: Dennies Bor & Edward Oughton

Aggregates the BEA sector-level Use table into the 10-sector scheme used
across c-swim, then derives the direct requirements matrix (A), gross
output vector (X), value added components, and final demand components.

The Use table is fetched from the BEA InputOutput API (TableID 258) by
fetch_bea_use_table; this module just consumes the resulting CSV and
produces the 10-sector matrices that downstream models read.
"""

import pandas as pd

from configs import setup_logger, get_data_dir
from econ.preprocess.fetch_bea_census import fetch_bea_use_table, DEFAULT_YEAR

logger = setup_logger("Production Technology Builder")
DATA_LOC = get_data_dir(econ=True)
TABLES_DIR = DATA_LOC / "supply_use_tables"

# Sector aggregation: maps c-swim 10-sector labels to the BEA sector codes
# that appear as both row and column codes in the Use table.
SECTOR_GROUPS = {
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


def preprocess_use_table(file_path):
    """Load the Use table CSV produced by fetch_bea_use_table and convert
    values from $ millions to $ billions to match the rest of the pipeline.
    """
    data = pd.read_csv(file_path, index_col=0)
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

    return data  # Left in millions for now


def create_production_technology(use_table_path, output_dir=None):
    """Build the 10-sector A matrix, gross output vector, value added, and
    final demand components from the Use table.
    """
    if output_dir is None:
        output_dir = DATA_LOC / "10sector"

    U = preprocess_use_table(use_table_path)

    # Intermediate transactions: rows whose code matches a BEA industry
    # (numeric, FIRE, PROF, 44RT, 48TW, G). Auxiliary rows like Used, Other,
    # T-totals, and value-added rows are excluded by this filter.
    intermediate = U.loc[
        U.index.str.match(r"^\d{1,2}G?$|^FIRE$|^PROF$|^44RT$|^48TW$|^G$")
    ]
    output_row = U.loc["T018"]

    # Build a 2-digit code -> 10-sector lookup once and reuse for both
    # row and column aggregation.
    long = (
        pd.Series(SECTOR_GROUPS)
        .explode()
        .rename_axis("group")
        .reset_index()
        .rename(columns={0: "code"})
    )
    code2grp = long.set_index("code")["group"]

    # Aggregate intermediate transactions on rows (commodities), then on
    # columns (industries). Result is a 10x10 transaction matrix.
    U_big_temp = intermediate.rename(index=code2grp).groupby(level=0).sum()
    U_big = (
        U_big_temp.T.rename(index=code2grp)
        .groupby(level=0)
        .sum()
        .T.reindex(index=SECTOR_GROUPS.keys(), columns=SECTOR_GROUPS.keys())
    )

    # Aggregate the gross output row from T018.
    X_big = (
        output_row.rename(index=code2grp)
        .groupby(level=0)
        .sum()
        .reindex(SECTOR_GROUPS.keys())
    )

    # Direct requirements matrix: A = U / X (column-wise).
    A_big = U_big.div(X_big, axis=1).round(6)

    # Value added components: compensation, taxes, gross operating surplus.
    va_rows = ["V001", "V003", "T00OTOP", "T00OSUB"]
    VA_big = (
        U.loc[va_rows, intermediate.columns]
        .rename(columns=code2grp)
        .T.groupby(level=0)
        .sum()
        .T.reindex(columns=SECTOR_GROUPS.keys())
    )

    # Final demand components: PCE, government, investment, inventory, exports.
    fd_cols = ["F010", "F100", "F020", "F030", "F040"]
    FD_big = (
        U.loc[intermediate.index, fd_cols]
        .rename(index=code2grp)
        .groupby(level=0)
        .sum()
        .reindex(index=SECTOR_GROUPS.keys())
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    A_big.to_csv(output_dir / "direct_requirements.csv")
    X_big.to_csv(output_dir / "gross_output.csv", header=[str(DEFAULT_YEAR)])
    VA_big.to_csv(output_dir / "value_added.csv")
    FD_big.to_csv(output_dir / "final_demand.csv")
    VA_big.sum(axis=0).to_csv(
        output_dir / "total_value_added.csv", header=[str(DEFAULT_YEAR)]
    )
    U_big.sum(axis=0).to_csv(
        output_dir / "total_intermediate_use.csv", header=[str(DEFAULT_YEAR)]
    )

    logger.info(f"10-sector matrices saved to {output_dir}")
    return A_big, X_big, VA_big, FD_big


if __name__ == "__main__":
    use_table_fp = fetch_bea_use_table(year=DEFAULT_YEAR)

    A_big, X_big, VA_big, FD_big = create_production_technology(use_table_fp)

    logger.info(f"Number of sectors: {len(A_big)}")
    logger.info(f"GDP (total value added): {VA_big.sum().sum():.1f} million dollars")
