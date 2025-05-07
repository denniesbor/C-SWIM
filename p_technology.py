"""
Production Technology Matrix Generator for Economic Analysis

This module processes input-output data from Bureau of Economic Analysis (BEA)
tables to create production technology matrices for economic analysis and
computable general equilibrium (CGE) modeling.

The module performs the following operations:
1. Cleans HTML entities from gross output data files
2. Aggregates detailed BEA industry data into 10 macro-sectors
3. Constructs direct requirements matrix (A-matrix) and related technology matrices
4. Exports the matrices for use in CGE modeling or input-output analysis

Functions:
---------
preprocess_use_table(file_path: str) -> pd.DataFrame
    Loads, cleans, and processes the BEA Use Table, including normalization to billions
    
clean_gross_output_file(input_path, output_path)
    Removes HTML entities from gross output data file and creates clean version
    
create_production_technology(use_table_path, output_dir="10sector")
    Processes Use Table to create production technology matrices including:
    - Direct requirements (A-matrix)
    - Gross output vector (X)
    - Value added components
    - Final demand components

Key Matrices:
------------
A_big : pd.DataFrame
    Direct requirements coefficients matrix showing intermediate input needs per unit output
    
X_big : pd.Series
    Gross output by sector
    
VA_big : pd.DataFrame
    Value added components (labor, capital, taxes, subsidies) by sector
    
FD_big : pd.DataFrame
    Final demand components (household, government, investment, exports) by sector

Sector Aggregation:
------------------
The module aggregates detailed BEA industries into 10 sectors:
- AGR        : Agriculture, forestry, fishing, and hunting (NAICS 11)
- MINING     : Mining (NAICS 21)
- UTIL_CONST : Utilities and Construction (NAICS 22, 23)
- MANUF      : Manufacturing (NAICS 31G)
- TRADE_TRANSP: Wholesale, Retail trade, and Transportation (NAICS 42, 44RT, 48TW)
- INFO       : Information (NAICS 51)
- FIRE       : Finance, insurance, real estate (NAICS FIRE)
- PROF_OTHER : Professional and Other services (NAICS PROF, 81)
- EDUC_ENT   : Education, Health, Entertainment, Food services (NAICS 6, 7)
- G          : Government (NAICS G)

This aggregation maintains sectoral distinctions while reducing dimensionality for
tractable economic modeling.

Notes:
-----
- Input data is expected in CSV format from standard BEA releases
- Monetary values are converted to billions of dollars for numerical stability
- A directory '10sector' is created to store output files
- The direct requirements matrix can be used for traditional input-output analysis
  or as inputs to more sophisticated CGE models

Usage:
-----
When run as main script, processes BEA data files in the 'data' directory and 
creates technology matrices in the '10sector' directory.

Example:
    python p_technology.py

References:
----------
- Bureau of Economic Analysis (BEA) Input-Output Tables
- Hosoe, N., Gasawa, K., & Hashimoto, H. (2010). Textbook of Computable General 
  Equilibrium Modelling: Programming and Simulations. Palgrave Macmillan.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def preprocess_use_table(file_path: str) -> pd.DataFrame:
    """Load and clean the BEA Use Table."""
    data = pd.read_csv(file_path)
    data.columns = [col if not col.startswith('Unnamed:') else 'code' for col in data.columns]
    data = data.copy()
    
    if 'code' in data.columns and 'Commodities/Industries' in data.columns:
        data = data.set_index(['code', 'Commodities/Industries'])
        
    data = data.map(lambda x: str(x).strip().replace('---', '0'))
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    data = data / 1000  # Convert from Mn to Bn dollars
    return data

def clean_gross_output_file(input_path, output_path):
    """Clean the gross output file of HTML entities."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace HTML entities with proper characters
    content = content.replace('+ACI-', '"')
    
    # Write to a clean file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path

def create_production_technology(use_table_path, output_dir="10sector"):
    """
    Create production technology matrices from BEA input-output data.
    
    Parameters:
    -----------
    use_table_path : str
        Path to the BEA Use Table CSV file
    output_dir : str
        Directory where output files will be saved
    
    Returns:
    --------
    A_big : pd.DataFrame
        Direct requirements matrix (technical coefficients)
    """
    # Define industry groups - 10-sector aggregation
    groups = {
        'AGR'        : ['11'],
        'MINING'     : ['21'],
        'UTIL_CONST' : ['22','23'],
        'MANUF'      : ['31G'],
        'TRADE_TRANSP':['42','44RT','48TW'],
        'INFO'       : ['51'],
        'FIRE'       : ['FIRE'],
        'PROF_OTHER' : ['PROF','81'],
        'EDUC_ENT'   : ['6','7'],
        'G'          : ['G'],
    }

    # Load and preprocess the use table
    U = preprocess_use_table(use_table_path)[1:]
    U = U.droplevel(1)  # index = industry code

    # Split the full table
    # Filter for industries in the intermediate use part
    intermediate = U.loc[U.index.str.match(r'^\d{1,2}G?$|^FIRE$|^PROF$|^44RT$|^48TW$|^G$')]
    output_row = U.loc['T018']  # total industry output
                                                      
    # Build concordance as a tidy two-column DataFrame
    long = (pd.Series(groups)
             .explode()
             .rename_axis('group')
             .reset_index()
             .rename(columns={0: 'code'}))

    code2grp = long.set_index('code')['group']

    # Re-label and aggregate intermediate flows
    # Rename rows and sum
    U_big_temp = (intermediate
                   .rename(index=code2grp)
                   .groupby(level=0).sum())

    # Rename columns and sum (using .T instead of axis=1)
    U_big = (U_big_temp
              .T
              .rename(index=code2grp)
              .groupby(level=0).sum()
              .T
              .reindex(index=groups.keys(), columns=groups.keys()))

    # Aggregate gross output using T018
    X_big = (output_row
              .rename(index=code2grp)
              .groupby(level=0).sum()
              .reindex(groups.keys()))

    # Technical-coefficients matrix A = U / X
    A_big = U_big.div(X_big, axis=1).round(6)

    # Value added components
    va_rows = ['V001',      # labour
              'V003',       # capital
              'T00OTOP',    # other production taxes
              'T00OSUB']    # subsidies (negative or zero)

    VA_big = (U.loc[va_rows, intermediate.columns]
               .rename(columns=code2grp)
               .T                          # Transpose
               .groupby(level=0).sum()     # Group by industry group
               .T                          # Transpose back
               .reindex(columns=groups.keys()))

    # Final demand components
    fd_cols = ['F010',   # Household consumption
              'F100',    # Government consumption
              'F020',    # Fixed capital formation
              'F030',    # Inventory change
              'F040']    # Exports

    FD_big = (U.loc[intermediate.index, fd_cols]
               .rename(index=code2grp)
               .groupby(level=0).sum()
               .reindex(index=groups.keys()))

    # Save matrices
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    A_big.to_csv(out/'direct_requirements.csv')
    X_big.to_csv(out/'gross_output.csv', header=['2023'])
    VA_big.to_csv(out/'value_added.csv')
    FD_big.to_csv(out/'final_demand.csv')
    
    # Calculate total value added
    total_va = VA_big.sum(axis=0)
    total_va.to_csv(out/'total_value_added.csv', header=['2023'])
    
    # Calculate total intermediate use
    total_interuse = U_big.sum(axis=0)
    total_interuse.to_csv(out/'total_intermediate_use.csv', header=['2023'])
    
    print(f"10-sector matrices saved to {output_dir}:")
    print(f"  Direct requirements matrix ➜ {out/'direct_requirements.csv'}")
    print(f"  Gross output vector ➜ {out/'gross_output.csv'}")
    print(f"  Value added components ➜ {out/'value_added.csv'}")
    print(f"  Final demand components ➜ {out/'final_demand.csv'}")
    
    return A_big, X_big, VA_big, FD_big


if __name__ == "__main__":
    # Clean the gross output file
    clean_gross_output_file('./data/gross_output.csv', './data/cleaned_gross_output.csv')
    
    # Create production technology matrices
    A_big, X_big, VA_big, FD_big = create_production_technology('./data/use_tables.csv')
    

    print(f"Number of sectors: {len(A_big)}")
    print(f"GDP (total value added): {VA_big.sum().sum():.1f} billion dollars")