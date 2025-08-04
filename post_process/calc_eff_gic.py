import os
import glob
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from configs import setup_logger, get_data_dir

# Get data data log and configure logger
DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/calc_eff_gic.log")

# open transformer pickle file
data_path = DATA_LOC / "admittance_matrix/sample_network.pkl"
with open(data_path, "rb") as f:
    df_transformers = pickle.load(f)
    logger.info(
        f"Loaded {len(df_transformers)} pre-generated samples from {data_path}"
    )

def calculate_effective_gic(gic_path, df_transformers, output_dir):
    """Calculate effective GIC from winding GIC data"""
    df_gic = pd.read_csv(gic_path)
    df_transformer = df_transformers[int(gic_path.stem.split("_")[-1])]
    
    hazard_cols = [col for col in df_gic.columns if "year-hazard" in col]
    
    df_meta = df_gic.groupby('Transformer').agg({
        'sub_id': 'first',
        'latitude': lambda x: np.round(x.iloc[0], 2),
        'longitude': lambda x: np.round(x.iloc[0], 2)
    }).reset_index()
    
    df_meta = df_meta.merge(
        df_transformer[['name', 'type', 'bus1_id', 'bus2_id']].rename(columns={'name': 'Transformer'}),
        on='Transformer'
    )
    
    df_pivot = df_gic.pivot_table(
        index='Transformer',
        columns='Winding', 
        values=hazard_cols,
        aggfunc='first'
    ).reset_index()
    
    df_pivot.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_pivot.columns]
    
    df_result = df_meta.merge(df_pivot, on='Transformer')
    
    multi_winding_mask = df_result['type'].isin(["Auto", "GY-GY", "GY-GY-D"])
    if multi_winding_mask.any():
        bus1_voltages = df_result.loc[multi_winding_mask, 'bus1_id'].str.split('_').str[-1].astype(float)
        bus2_voltages = df_result.loc[multi_winding_mask, 'bus2_id'].str.split('_').str[-1].astype(float)
        df_result.loc[multi_winding_mask, 'v_ratio'] = np.minimum(bus1_voltages, bus2_voltages) / np.maximum(bus1_voltages, bus2_voltages)
    
    for hazard_col in hazard_cols:
        hv_col = f"{hazard_col}_HV"
        lv_col = f"{hazard_col}_LV" 
        series_col = f"{hazard_col}_Series"
        common_col = f"{hazard_col}_Common"
        
        effective_gic = pd.Series(index=df_result.index, dtype=float)
        
        single_mask = df_result['type'].isin(["GSU", "GSU w/ GIC BD", "GY-D", "GY-D w/ GIC BD"])
        if single_mask.any() and hv_col in df_result.columns:
            effective_gic[single_mask] = df_result.loc[single_mask, hv_col]
        
        auto_mask = df_result['type'] == "Auto"
        if auto_mask.any() and series_col in df_result.columns and common_col in df_result.columns:
            effective_gic[auto_mask] = (df_result.loc[auto_mask, series_col] + 
                                       df_result.loc[auto_mask, common_col] * df_result.loc[auto_mask, 'v_ratio'])
        
        gy_mask = df_result['type'].isin(["GY-GY", "GY-GY-D"])
        if gy_mask.any() and hv_col in df_result.columns and lv_col in df_result.columns:
            effective_gic[gy_mask] = (df_result.loc[gy_mask, hv_col] + 
                                     df_result.loc[gy_mask, lv_col] * df_result.loc[gy_mask, 'v_ratio'])
        
        df_result[f"e_{hazard_col}"] = effective_gic
    
    final_cols = ['Transformer', 'type', 'sub_id', 'latitude', 'longitude'] + [f"e_{col}" for col in hazard_cols]
    df_effective = df_result[final_cols].copy()
    
    output_path = output_dir / f"effective_gic_rand_{gic_path.stem.split('_')[-1]}.csv"
    df_effective.to_csv(output_path, index=False)

def main():
    """Process all winding GIC files and calculate effective GIC."""
    gic_dir = Path("/data/archives/nfs/spw-geophy/data/final_gic/gic")
    output_dir = Path("/data/archives/nfs/spw-geophy/data/final_gic/gic_eff")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    gic_files = glob.glob(str(gic_dir / "winding_gic_rand_*.csv"))
    
    for gic_file in tqdm(gic_files, desc="Processing GIC files"):
        gic_path = Path(gic_file)
        calculate_effective_gic(gic_path, df_transformers, output_dir)

if __name__ == "__main__":
    main()