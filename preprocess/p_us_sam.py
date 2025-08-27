"""
US Social Accounting Matrix (SAM) Generator and Balancer

This script processes Bureau of Economic Analysis (BEA) Input-Output tables 
to construct and balance a Social Accounting Matrix (SAM) for the US economy. 
The methodology follows the framework presented in "Textbook of Computable 
General Equilibrium Modelling: Programming and Simulations" by Hosoe, Gasawa, 
and Hashimoto (2010).

The SAM captures the circular flow of income among economic agents and provides 
a complete and consistent picture of economic transactions for CGE modeling.

The script performs the following operations:
1. Processes BEA Use and Supply tables
2. Aggregates detailed industries into 10 major sectors
3. Constructs a SAM with activities, factors, taxes, and institutions
4. Incorporates additional NIPA data on taxes, transfers, and balances
5. Applies adjustments for subsidies, negative investments, and trade deficit
6. Balances the SAM using a cross-entropy minimization algorithm
7. Saves the balanced SAM to a CSV file

Functions:
----------
preprocess_use_table(file_path)
    Clean and prepare BEA Use or Supply tables for processing
    
make_sam_accounts(use_tb_path, sup_tb_path)
    Create the SAM structure from BEA tables and additional data
    
balance_and_save_sam(sam, output_path, decimal_places)
    Balance SAM using the cross-entropy algorithm and save to CSV

Sectoral Aggregation:
--------------------
The script aggregates BEA industries into 10 sectors:
- AGR: Agriculture, forestry, fishing, and hunting (NAICS 11)
- MINING: Mining (NAICS 21)
- UTIL_CONST: Utilities and Construction (NAICS 22, 23)
- MANUF: Manufacturing (NAICS 31G)
- TRADE_TRANSP: Wholesale, Retail, Transportation (NAICS 42, 44RT, 48TW)
- INFO: Information (NAICS 51)
- FIRE: Finance, Insurance, Real Estate (NAICS FIRE)
- PROF_OTHER: Professional and Other services (NAICS PROF, 81)
- EDUC_ENT: Education, Health, Entertainment (NAICS 6, 7)
- G: Government (NAICS G)

SAM Accounts:
------------
The SAM includes the following account categories:
- Production activities (10 sectors listed above)
- Factors of production (CAP, LAB)
- Tax accounts (IDT, TRF)
- Institutions (HOH, GOV, INV, EXT)

Data Requirements:
----------------
- BEA Use and Supply tables in CSV format
- Additional data files in the data/add_data/ directory:
  * personal_current_tax_payments.csv
  * govt_transfers.csv
  * personal_income_and_disp.csv
  * gov_receipts_expenditures.csv
  * Foreign_Transactions_National_Income_and_Product_Accounts.csv

Dependencies:
------------
- pandas, numpy: Data processing
- pathlib: File path management
- sam_balancer: SAM balancing algorithm

Usage:
------
When run as main script, processes BEA data in the 'data' directory
and creates a balanced SAM in 'data/us_balanced_sam.csv'.

Example:
    python build_us_sam.py

References:
----------
Hosoe, N., Gasawa, K., & Hashimoto, H. (2010). Textbook of Computable General 
Equilibrium Modelling: Programming and Simulations. Palgrave Macmillan.
"""

# Import necessary modules
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
    """
    Load and clean the BEA Use Table.
    """
    # Read and clean data
    data = pd.read_csv(file_path)
    
    # Rename unnamed column to 'code'
    data.columns = [col if not col.startswith('Unnamed:') else 'code' for col in data.columns]
    
    # Make a copy before setting index to avoid modification warnings
    data = data.copy()
    
    # Set index after ensuring columns exist
    if 'code' in data.columns and 'Commodities/Industries' in data.columns:
        data = data.set_index(['code', 'Commodities/Industries'])
        
    # Convert all values to strings, clean manually, then convert to numeric
    data = data.map(lambda x: str(x).strip().replace('---', '0'))
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    return data

def make_sam_accounts(use_tb_path, sup_tb_path):

    # Use df
    use_df = preprocess_use_table(use_tb_path)
    supply_df = preprocess_use_table(sup_tb_path)


    # Read and clean data
    use_df = pd.read_csv(use_tb_path)

    # Rename unnamed column to 'code'
    use_df.columns = [col if not col.startswith('Unnamed:') else 'code' for col in use_df.columns]

    # Make a copy before setting index to avoid modification warnings
    use_df = use_df.copy()

    # Set index after ensuring columns exist
    if 'code' in use_df.columns and 'Commodities/Industries' in use_df.columns:
        use_df = use_df.set_index(['code', 'Commodities/Industries'])
        
    # Convert all values to strings, clean manually, then convert to numeric
    use_df = use_df.map(lambda x: str(x).strip().replace('---', '0'))
    use_df = use_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # industry_groups = {
    #     '11': ['11'],               # Agriculture, forestry, fishing, and hunting
    #     '21': ['21'],               # Mining
    #     '22': ['22'],               # Utilities
    #     '23': ['23'],               # Construction
    #     '31G': ['31G'],             # Manufacturing
    #     '42': ['42'],               # Wholesale trade
    #     '44RT': ['44RT'],           # Retail trade
    #     '48TW': ['48TW'],           # Transportation and warehousing
    #     '51': ['51'],               # Information
    #     'FIRE': ['FIRE'],           # Finance, insurance, real estate, rental, and leasing
    #     'PROF': ['PROF'],           # Professional and business services
    #     '6': ['6'],                 # Educational services, health care, and social assistance
    #     '7': ['7'],                 # Arts, entertainment, recreation, accommodation, and food services
    #     '81': ['81'],               # Other services, except government
    #     'G': ['G']                  # Government
    # }

    industry_groups = {
        'AGR': ['11'],                         # Agriculture, forestry, fishing, and hunting
        'MINING': ['21'],                      # Mining
        'UTIL_CONST': ['22', '23'],            # Utilities and Construction
        'MANUF': ['31G'],                      # Manufacturing
        'TRADE_TRANSP': ['42', '44RT', '48TW'], # Wholesale, Retail trade, and Transportation
        'INFO': ['51'],                        # Information
        'FIRE': ['FIRE'],                      # Finance, insurance, real estate
        'PROF_OTHER': ['PROF', '81'],          # Professional and Other services
        'EDUC_ENT': ['6', '7'],                # Education, Health, Entertainment, Food services
        'G': ['G']                           # Government
    }

    sam_accounts = list(industry_groups.keys()) + ['CAP', 'LAB', 'IDT', 'TRF', 'HOH', 'GOV', 'INV', 'EXT']

    # Initialize SAM
    sam = pd.DataFrame(0.0, index=sam_accounts, columns=sam_accounts)

    # Activity-to-activity
    for paying_group, paying_cols in industry_groups.items():
        for receiving_group, receiving_cols in industry_groups.items():
            sam.loc[receiving_group, paying_group] = use_df.loc[receiving_cols, paying_cols].sum().sum()

    # Factors (Value Added)
    factor_map = {'V001': 'LAB', 'V003': 'CAP'}
    for factor_code, factor in factor_map.items():
        for activity, cols in industry_groups.items():
            sam.loc[factor, activity] = use_df.loc[(factor_code, slice(None)), cols].sum().sum()

    # Other product taxes less subsidies (excluding import duties)
    for activity, codes in industry_groups.items():
        product_taxes = supply_df.loc[codes, 'TOP'].sum()
        subsidies = supply_df.loc[codes, 'SUB'].sum()
        sam.at['IDT', activity] = product_taxes + subsidies

    # Add other production taxes
    for activity, codes in industry_groups.items():
        production_tax = use_df.loc["T00OTOP", codes].sum().sum()
        sam.at['IDT', activity] += production_tax

    # Final demand
    fd_map = {'HOH': ['F010'], 'GOV': ['F100'], 'INV': ['F020', 'F030'], 'EXT': ['F040']}
    for inst, fd_codes in fd_map.items():
        for activity, cols in industry_groups.items():
            sam.at[activity, inst] = use_df.loc[cols, fd_codes].sum().sum()

    # Imports from supply table
    for activity, codes in industry_groups.items():
        sam.at['EXT', activity] = supply_df.loc[codes, 'MCIF'].sum()

    # Import tariffs from supply table
    for activity, codes in industry_groups.items():
        sam.at['TRF', activity] = supply_df.loc[codes, 'MDTY'].sum()
        
    # Taxes and tariffs to government
    sam.at['GOV', 'IDT'] = sam.loc['IDT'].sum()
    sam.at['GOV', 'TRF'] = sam.loc['TRF'].sum()

    # Factors to households
    sam.at['HOH', 'LAB'] = sam.loc['LAB'].sum()
    sam.at['HOH', 'CAP'] = sam.loc['CAP'].sum()

    # from external accpimts
    # Define the data directory path
    add_data_path = DATA_LOC / "add_data"

    def load_and_clean_csv(file_name, skiprows, desc_col_index=1):
        path = os.path.join(add_data_path, file_name)
        df = pd.read_csv(path, skiprows=skiprows).fillna(0)
        for col in df.columns:
            if col != df.columns[desc_col_index]:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df   

    def get_average_of_last_four(row):
        return row.iloc[-4:].astype(float).mean() * 1000

    # 1. Direct taxes
    tax_data = load_and_clean_csv("personal_current_tax_payments.csv", skiprows=3)
    direct_tax = tax_data[tax_data.iloc[:, 1].str.contains("Personal current taxes", na=False)].iloc[0, -1] * 1000

    # 2. Government transfers
    transfers_data = load_and_clean_csv("govt_transfers.csv", skiprows=3)
    transfers = transfers_data[transfers_data.iloc[:, 1].str.contains("Government social benefits", na=False)].iloc[0, -1] * 1000

    # 3. Household savings
    savings_data = load_and_clean_csv("personal_income_and_disp.csv", skiprows=4)
    savings_row = savings_data[savings_data.iloc[:, 1].str.contains("Personal saving", na=False)].iloc[0]
    total_savings = get_average_of_last_four(savings_row)
    logger.info(f"Household savings: {total_savings} million (average of quarterly values)")

    # 4. Government deficit
    govt_data = load_and_clean_csv("gov_receipts_expenditures.csv", skiprows=4)
    receipts_row = govt_data[govt_data.iloc[:, 1].str.contains("Current receipts", na=False)].iloc[0]
    expend_row = govt_data[govt_data.iloc[:, 1].str.contains("Current expenditures", na=False)].iloc[0]
    receipts_total = get_average_of_last_four(receipts_row)
    expend_total = get_average_of_last_four(expend_row)
    govt_deficit = expend_total - receipts_total
    logger.info(f"Government deficit: {govt_deficit} million (based on average quarterly values)")

    # 5. Trade deficit
    trade_data = load_and_clean_csv("Foreign_Transactions_National_Income_and_Product_Accounts.csv", skiprows=1)
    exports_row = trade_data[trade_data.iloc[:, 1].str.contains("Exports of goods and services", na=False)].iloc[0]
    imports_row = trade_data[trade_data.iloc[:, 1].str.contains("Imports of goods and services", na=False)].iloc[0]
    exports_total = get_average_of_last_four(exports_row)
    imports_total = get_average_of_last_four(imports_row)
    trade_deficit = imports_total - exports_total
    logger.info(f"Trade deficit: {trade_deficit} million (based on average quarterly values)")
    
    logger.info("Key values for SAM (in millions):")
    logger.info(f"1. Direct taxes (HOH→GOV): {direct_tax}")
    logger.info(f"2. Government transfers (GOV→HOH): {transfers}")
    logger.info(f"3. Household savings (HOH→INV): {total_savings}")
    logger.info(f"4. Government deficit (GOV→INV): {govt_deficit}")
    logger.info(f"5. Trade deficit (EXT→INV): {trade_deficit}")

    # Update SAM with these values
    try:
        sam.loc['GOV', 'HOH'] = int(direct_tax)
        sam.loc['HOH', 'GOV'] = int(transfers)
        sam.loc['INV', 'HOH'] = int(total_savings)
        sam.loc['INV', 'GOV'] = int(govt_deficit)
    except KeyError as e:
        logger.error(f"SAM update failed: {e}")

    # Remove subsidy in agric to part of govt payments to agric
    subsidy_amount = abs(sam.at['TRF', 'AGR'])
    sam.at['TRF', 'AGR'] = 0

    if 'AGR' in sam.index and 'GOV' in sam.columns and not pd.isna(sam.at['AGR', 'GOV']):
        sam.at['AGR', 'GOV'] += subsidy_amount
    else:
        sam.at['AGR', 'GOV'] = subsidy_amount

    # Remove subsidy in agric to part of govt payments to agric
    subsidy_amount = abs(sam.at['TRF', 'G'])
    sam.at['TRF', 'G'] = 0

    if 'G' in sam.index and 'GOV' in sam.columns and not pd.isna(sam.at['G', 'GOV']):
        sam.at['G', 'GOV'] += subsidy_amount
    else:
        sam.at['G', 'GOV'] = subsidy_amount

    # Handle negative investment in agric
    if 'AGR' in sam.index and 'INV' in sam.columns and sam.at['AGR', 'INV'] < 0:
        neg_inv = abs(sam.at['AGR', 'INV'])
        sam.at['AGR', 'INV'] = 0
        
        # Find all sectors with positive investment
        inv_sectors = [sector for sector in industry_groups.keys() 
                    if sector != 'AGR' and sam.at[sector, 'INV'] > 0]
        
        # Calculate total positive investment
        total_pos_inv = sum(sam.at[sector, 'INV'] for sector in inv_sectors)
        
        # Distribute proportionally
        if total_pos_inv > 0:
            for sector in inv_sectors:
                share = sam.at[sector, 'INV'] / total_pos_inv
                sam.at[sector, 'INV'] -= neg_inv * share

    # Adjust trade deficit
    try:
        trade_deficit_bea = trade_deficit
        trade_deficit_sam = sam.loc['EXT', :].sum() - sam.loc[:, 'EXT'].sum()
        if abs(trade_deficit_bea - trade_deficit_sam) > 1000:
            if trade_deficit_bea > 0:
                sam.loc['INV', 'EXT'] = trade_deficit_bea
                sam.loc['EXT', 'INV'] = 0
            else:
                sam.loc['INV', 'EXT'] = 0
                sam.loc['EXT', 'INV'] = -trade_deficit_bea
    except KeyError as e:
        logger.error(f"Trade adjustment failed: {e}")
        
    # Return sam
    return sam


def balance_and_save_sam(sam, output_path="output/us_balanced_sam.csv", decimal_places=1):
    """
    Balance a SAM and save it to a CSV file.
    
    Parameters:
    -----------
    sam : pd.DataFrame
        The Social Accounting Matrix to balance
    output_path : str or Path, optional
        Path where to save the balanced SAM, default is "output/us_balanced_sam.csv"
    decimal_places : int, optional
        Number of decimal places to round results to, default is 1
        
    Returns:
    --------
    balanced_sam : pd.DataFrame
        The balanced Social Accounting Matrix
    """
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Balance the SAM
    balanced_sam = balance_sam(sam, decimal_places=decimal_places)
    
    # Save to CSV
    balanced_sam.to_csv(output_path)
    logger.info(f"Balanced SAM saved to {output_path}")
    
    return balanced_sam

# Main execution block
if __name__ == "__main__":
    data_dir = Path("data")
    use_tb_path = tables_dir / "use_tables.csv"
    sup_tb_path = tables_dir / "supply_tables.csv"
    
    # Make sam
    sam = make_sam_accounts(use_tb_path, sup_tb_path)
    
    # Balance the SAM with 1 decimal place and save to CSV
    balanced_sam = balance_and_save_sam(sam, sam_output_dir / "us_balanced_sam.csv", decimal_places=1)

    # Show some information about the balanced SAM
    logger.info(f"Original row sums: {sam.sum(axis=1).sum()}")
    logger.info(f"Original column sums: {sam.sum(axis=0).sum()}")
    logger.info(f"Balanced row sums: {balanced_sam.sum(axis=1).sum()}")
    logger.info(f"Balanced column sums: {balanced_sam.sum(axis=0).sum()}")