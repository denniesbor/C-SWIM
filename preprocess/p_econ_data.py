import pandas as pd
import geopandas as gpd
import pickle
import numpy as np
import os
import gc
import random
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Polygon
from configs import setup_logger, get_data_dir, DENNIES_DATA_LOC

DATA_LOC = get_data_dir()
data_zcta_folder = DATA_LOC / "econ_data"
processed_econ_dir = data_zcta_folder / "processed_econ"
processed_voronoi_dir = DATA_LOC / "processed_voronoi"
logger = setup_logger(log_file="logs/p_econ_data")

def load_socioeconomic_data(data_zcta_folder):
    processed_file = processed_econ_dir / "socioeconomic_data.pkl"
    
    # if processed_file.exists():
    #     logger.info("Loading existing processed socioeconomic data")
    #     with open(processed_file, 'rb') as f:
    #         return pickle.load(f)
    
    logger.info("Processing socioeconomic data")
    
    data_zcta = pd.read_csv(data_zcta_folder / "NAICS_EST_GDP2022_ZCTA.csv",
                          dtype={"ZCTA": "Int64"})
    population_df = pd.read_csv(data_zcta_folder / "2020_decennial_census_at_ZCTA_level.csv",
                              dtype={"ZCTA": "Int64"})
    regions = data_zcta[["ZCTA", "REGIONS", "STABBR"]].drop_duplicates()
    regions_pop_df = regions.merge(population_df, on=["ZCTA", "STABBR"])
    
    zcta_gdf = gpd.read_file(data_zcta_folder / "tl_2020_us_zcta520.zip")
    zcta_gdf = (zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"})
                       .astype({"ZCTA": "Int64"}))
    zcta_business_gdf = zcta_gdf.merge(data_zcta, on="ZCTA", how="inner")
    zcta_business_gdf["points"] = zcta_business_gdf.geometry.representative_point()
    zcta_business_gdf.set_geometry("points", inplace=True)
    zcta_business_gdf = zcta_business_gdf.to_crs(epsg=4326)
    zcta_business_gdf["GDP2022"] = zcta_business_gdf["GDP2022"] / 365.0

    keep_cols = ["ZCTA","GEOID20","NAICS","EST","GDP2022","geometry"]
    gdf_min = zcta_business_gdf[keep_cols].copy()
    gdf_min["ZCTA"]    = gdf_min["ZCTA"].astype(np.uint32)
    gdf_min["GEOID20"] = gdf_min["GEOID20"].astype(np.uint32)
    gdf_min["EST"]     = gdf_min["EST"].astype(np.uint16)
    gdf_min["GDP2022"] = gdf_min["GDP2022"].astype(np.float32)
    
    gdp_wide = gdf_min.pivot_table(index=['ZCTA', 'GEOID20', 'geometry'], columns='NAICS', values='GDP2022', fill_value=0).reset_index()
    gdp_wide.columns = [f'GDP_{col}' if col not in ['ZCTA', 'GEOID20', 'geometry'] else col for col in gdp_wide.columns]

    est_wide = gdf_min.pivot_table(index=['ZCTA', 'GEOID20', 'geometry'], columns='NAICS', values='EST', fill_value=0).reset_index()
    est_wide.columns = [f'EST_{col}' if col not in ['ZCTA', 'GEOID20', 'geometry'] else col for col in est_wide.columns]

    zcta_wide_gdf = pd.merge(gdp_wide, est_wide.drop('geometry', axis=1), on=['ZCTA', 'GEOID20'])
    zcta_wide_gdf = gpd.GeoDataFrame(zcta_wide_gdf, geometry='geometry', crs="EPSG:4326")
    zcta_wide_gdf = pd.merge(zcta_wide_gdf, regions_pop_df[['ZCTA', 'POP20']], on='ZCTA', how='left')
    zcta_wide_gdf['POP20'] = zcta_wide_gdf['POP20'].fillna(0).astype(np.uint32)
    zcta_wide_gdf = zcta_wide_gdf.drop_duplicates(subset="ZCTA", keep="first")
    
    economic_cols = [col for col in zcta_wide_gdf.columns if col.startswith('GDP_') or col.startswith('EST_')]
    zcta_wide_gdf, new_economic_cols = aggregate_economic_columns(zcta_wide_gdf, economic_cols)
    zcta_wide_gdf[new_economic_cols] = zcta_wide_gdf[new_economic_cols].astype(np.float32)

    other_cols = [c for c in zcta_business_gdf.columns if c not in keep_cols]
    df_other = zcta_business_gdf[["ZCTA"] + other_cols].copy()
    
    del zcta_business_gdf, gdf_min, gdp_wide, est_wide
    gc.collect()
    
    states_gdf = gpd.read_file(data_zcta_folder / "tl_2022_us_state.zip").to_crs(epsg=4326)
    
    os.makedirs(processed_econ_dir, exist_ok=True)
    result = (data_zcta, population_df, regions_pop_df, zcta_wide_gdf, states_gdf, df_other)
    with open(processed_file, 'wb') as f:
        pickle.dump(result, f)
    
    logger.info("Saved processed socioeconomic data")
    return result

def aggregate_economic_columns(df, economic_cols):
    missing_cols = [col for col in economic_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    result_df = df.copy()
    
    industry_groups = {
        'AGR': ['11'],
        'MINING': ['21'],
        'UTIL_CONST': ['22', '23'],
        'MANUF': ['31'],
        'TRADE_TRANSP': ['42', '44', '48'],
        'INFO': ['51'],
        'FIRE': ['FIRE'],
        'PROF_OTHER': ['PROF', '81'],
        'EDUC_ENT': ['6', '7'],
        'G': ['G']
    }
    
    for prefix in ['GDP_', 'EST_']:
        for new_sector, naics_codes in industry_groups.items():
            new_col = f"{prefix}{new_sector}"
            result_df[new_col] = 0
            
            matching_cols = [f"{prefix}{code}" for code in naics_codes if f"{prefix}{code}" in economic_cols]
            if matching_cols:
                result_df[new_col] = df[matching_cols].sum(axis=1)
    
    aggregated_cols = [f"{prefix}{sector}" for prefix in ['GDP_', 'EST_'] for sector in industry_groups.keys()]
    if 'POP20' in df.columns:
        aggregated_cols.append('POP20')
    
    non_economic_cols = [col for col in df.columns if col not in economic_cols 
                        or col in ['geometry', 'ZCTA', 'GEOID20']]
    
    columns_to_keep = list(set(non_economic_cols + aggregated_cols))
    final_df = result_df[columns_to_keep].copy()
    
    return final_df, aggregated_cols

def create_voronoi_polygons(sub_coordinates: Dict[str, Tuple[float, float]], states_gdf) -> gpd.GeoDataFrame:
    voronoi_file = processed_voronoi_dir / "voronoi_polygons_clipped.geojson"
    
    # if voronoi_file.exists():
    #     logger.info("Loading existing Voronoi polygons")
    #     return gpd.read_file(voronoi_file)
    
    logger.info("Creating Voronoi polygons")
    coords = list(sub_coordinates.values())
    vor = Voronoi(coords)

    polygons = []
    sub_ids = list(sub_coordinates.keys())

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and region:
            polygon = Polygon([vor.vertices[i] for i in region])
        else:
            polygon = None
        polygons.append(polygon)

    valid_data = [
        {"sub_id": sub_id, "geometry": polygon}
        for sub_id, polygon in zip(sub_ids, polygons) if polygon is not None
    ]

    voronoi_gdf = gpd.GeoDataFrame(valid_data, crs="EPSG:4326")
    
    continental_us_states = states_gdf[~states_gdf['STATEFP'].isin(['02', '15', '72'])]
    continental_us_boundary = continental_us_states.unary_union
    
    voronoi_gdf = voronoi_gdf[voronoi_gdf.is_valid & ~voronoi_gdf.is_empty]
    voronoi_gdf = gpd.overlay(
        voronoi_gdf,
        gpd.GeoDataFrame(geometry=[continental_us_boundary], crs="EPSG:4326"),
        how="intersection"
    )

    os.makedirs(processed_voronoi_dir, exist_ok=True)
    voronoi_gdf.to_file(voronoi_file, driver="GeoJSON")
    logger.info("Saved Voronoi polygons")
    
    return voronoi_gdf


if __name__ == "__main__":

    logger.info("Starting economic data processing")

    data_zcta, population_df, regions_pop_df, zcta_business_gdf, states_gdf, df_other = load_socioeconomic_data(data_zcta_folder)

    df_substation = pd.read_csv(DENNIES_DATA_LOC / "admittance_matrix" / "substation_info.csv")
    ehv_coordinates = dict(zip(df_substation['name'], 
                            zip(df_substation['longitude'], df_substation['latitude'])))

    voronoi_gdf = create_voronoi_polygons(ehv_coordinates, states_gdf)

    logger.info("Economic data processing completed")