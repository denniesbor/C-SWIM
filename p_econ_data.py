import pandas as pd
import geopandas as gpd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Polygon
import os
import gc
import random


# Import necessary modules
pwr_grd_path = Path(".")
data_loc = pwr_grd_path / "data"
data_zcta_folder = pwr_grd_path / "econ_data"
data_risk_ass = pwr_grd_path / "econ_data"

def load_socioeconomic_data(data_zcta_folder):
    """Load and process business, population, and ZCTA data."""
    # Load business and population data
    data_zcta = pd.read_csv(data_zcta_folder / "NAICS_EST_GDP2022_ZCTA.csv",
                          dtype={"ZCTA": "Int64"})
    population_df = pd.read_csv(data_zcta_folder / "2020_decennial_census_at_ZCTA_level.csv",
                              dtype={"ZCTA": "Int64"})
    regions = data_zcta[["ZCTA", "REGIONS", "STABBR"]].drop_duplicates()
    regions_pop_df = regions.merge(population_df, on=["ZCTA", "STABBR"])
    
    # Create ZCTA business GeoDataFrame
    zcta_gdf = gpd.read_file(data_zcta_folder / "tl_2020_us_zcta520.zip")
    zcta_gdf = (zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"})
                       .astype({"ZCTA": "Int64"}))
    zcta_business_gdf = zcta_gdf.merge(data_zcta, on="ZCTA", how="inner")
    zcta_business_gdf["points"] = zcta_business_gdf.geometry.representative_point()
    zcta_business_gdf.set_geometry("points", inplace=True)
    zcta_business_gdf = zcta_business_gdf.to_crs(epsg=4326)
    zcta_business_gdf["GDP2022"] = zcta_business_gdf["GDP2022"] / 365.0  # daily GDP
    
    keep_cols = ["ZCTA","GEOID20","NAICS","EST","GDP2022","geometry"]

    # Build the minimal GeoDataFrame and downcast it
    gdf_min = zcta_business_gdf[keep_cols].copy()
    gdf_min["ZCTA"]    = gdf_min["ZCTA"].astype(np.uint32)
    gdf_min["GEOID20"] = gdf_min["GEOID20"].astype(np.uint32)
    gdf_min["EST"]     = gdf_min["EST"].astype(np.uint16)
    gdf_min["GDP2022"] = gdf_min["GDP2022"].astype(np.float32)
    
    
    # Pivot GDP data
    gdp_wide = gdf_min.pivot_table(index=['ZCTA', 'GEOID20', 'geometry'], columns='NAICS', values='GDP2022', fill_value=0).reset_index()
    gdp_wide.columns = [f'GDP_{col}' if col not in ['ZCTA', 'GEOID20', 'geometry'] else col for col in gdp_wide.columns]

    # Pivot establishment data
    est_wide = gdf_min.pivot_table(index=['ZCTA', 'GEOID20', 'geometry'], columns='NAICS', values='EST', fill_value=0).reset_index()
    est_wide.columns = [f'EST_{col}' if col not in ['ZCTA', 'GEOID20', 'geometry'] else col for col in est_wide.columns]

    # Merge and add population data
    zcta_wide_gdf = pd.merge(gdp_wide, est_wide.drop('geometry', axis=1), on=['ZCTA', 'GEOID20'])
    zcta_wide_gdf = gpd.GeoDataFrame(zcta_wide_gdf, geometry='geometry', crs="EPSG:4326")
    zcta_wide_gdf = pd.merge(zcta_wide_gdf, regions_pop_df[['ZCTA', 'POP20']], on='ZCTA', how='left')
    zcta_wide_gdf['POP20'] = zcta_wide_gdf['POP20'].fillna(0).astype(np.uint32)
    zcta_wide_gdf = zcta_wide_gdf.drop_duplicates(subset="ZCTA", keep="first")
    
    # Aggregate economic columns into 10 sectors
    economic_cols = [col for col in zcta_wide_gdf.columns if col.startswith('GDP_') or col.startswith('EST_')]
    zcta_wide_gdf, new_economic_cols = aggregate_economic_columns(zcta_wide_gdf, economic_cols)
    zcta_wide_gdf[new_economic_cols] = zcta_wide_gdf[new_economic_cols].astype(np.float32)


    # Now pull out the “other” columns (everything not in keep_cols)
    other_cols = [c for c in zcta_business_gdf.columns if c not in keep_cols]
    df_other   = zcta_business_gdf[["ZCTA"] + other_cols].copy()
    
    del zcta_business_gdf
    del gdf_min
    del gdp_wide
    del est_wide    
    
    # Load state boundaries
    states_gdf = gpd.read_file(data_zcta_folder / "tl_2022_us_state.zip").to_crs(epsg=4326)
    
    return data_zcta, population_df, regions_pop_df, zcta_wide_gdf, states_gdf, df_other


def aggregate_economic_columns(df, economic_cols):
    """
    Aggregate economic columns (GDP and establishment counts) into 10 sectors
    based on the industry grouping and drop original columns.
    
    Parameters:
    -----------
    df : GeoDataFrame
        Input GeoDataFrame with economic columns
    economic_cols : list
        List of economic column names to aggregate
    
    Returns:
    --------
    tuple
        (DataFrame with only aggregated columns, list of new column names)
    """
    # Validate that all economic columns exist in the dataframe
    missing_cols = [col for col in economic_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")
    
    # Create new DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Define industry groups mapping
    industry_groups = {
        'AGR': ['11'],                          # Agriculture, forestry, fishing, and hunting
        'MINING': ['21'],                       # Mining
        'UTIL_CONST': ['22', '23'],             # Utilities and Construction
        'MANUF': ['31'],                        # Manufacturing
        'TRADE_TRANSP': ['42', '44', '48'],     # Wholesale, Retail trade, and Transportation
        'INFO': ['51'],                         # Information
        'FIRE': ['FIRE'],                       # Finance, insurance, real estate
        'PROF_OTHER': ['PROF', '81'],           # Professional and Other services
        'EDUC_ENT': ['6', '7'],                 # Education, Health, Entertainment, Food services
        'G': ['G']                              # Government
    }
    
    # Create aggregated columns for GDP and establishments
    for prefix in ['GDP_', 'EST_']:
        for new_sector, naics_codes in industry_groups.items():
            # Create column name for the aggregated sector
            new_col = f"{prefix}{new_sector}"
            
            # Initialize with zeros
            result_df[new_col] = 0
            
            # Sum all relevant columns for this sector
            matching_cols = []
            for code in naics_codes:
                col_name = f"{prefix}{code}"
                if col_name in economic_cols:
                    matching_cols.append(col_name)
            
            # Add all matching columns at once to avoid iterative addition
            if matching_cols:
                result_df[new_col] = df[matching_cols].sum(axis=1)
    
    # Create list of new aggregated columns
    aggregated_cols = []
    for prefix in ['GDP_', 'EST_']:
        for sector in industry_groups.keys():
            aggregated_cols.append(f"{prefix}{sector}")
    
    # Add population column if it exists
    if 'POP20' in df.columns:
        aggregated_cols.append('POP20')
    
    # Get non-economic columns to keep
    non_economic_cols = [col for col in df.columns if col not in economic_cols 
                        or col == 'geometry' or col == 'ZCTA' or col == 'GEOID20']
    
    # Create a new DataFrame with only the aggregated columns and essential non-economic columns
    columns_to_keep = list(set(non_economic_cols + aggregated_cols))
    
    # Create a more memory-efficient DataFrame
    final_df = result_df[columns_to_keep].copy()
    
    return final_df, aggregated_cols

def load_power_grid_data(data_risk_ass):
    """Load power grid related data."""
    # Load pickle files
    transformer_info = pickle.load(open(data_risk_ass / "df_transformers.pkl", "rb"))
    sub_2_sub = pickle.load(open(data_risk_ass / "substations_nodes.pkl", "rb"))
    G = pickle.load(open(data_risk_ass / "graph.pkl", "rb"))
    substation_to_lines = pickle.load(open(data_risk_ass / "substation_to_line_voltages.pkl", "rb"))
    
    # Load and process transmission lines
    transmission_lines = pickle.load(open(data_risk_ass / "tl_gdf_subset.pkl", "rb"))
    transmission_lines.rename(columns={"LINE_ID": "line_id", "LINE_VOLTAGE": "voltage"}, 
                             inplace=True)
    transmission_lines = transmission_lines.to_crs(epsg=4326)
    
    return transformer_info, sub_2_sub, G, substation_to_lines, transmission_lines

def load_substation_data(data_risk_ass):
    """Load substation data."""
    subs_all_gdf = gpd.read_file(data_risk_ass / "ss_within_ferc.geojson").to_crs(epsg=4326)
    subs_all_gdf["points"] = subs_all_gdf.geometry.representative_point()
    subs_all_gdf["latitude"] = subs_all_gdf.points.y
    subs_all_gdf["longitude"] = subs_all_gdf.points.x
    
    return subs_all_gdf

def get_connected_lines(sub: str, substation_to_lines: Dict[str, List[str]]) -> List[str]:
    """Get line IDs connected to a substation."""
    return substation_to_lines.get(sub, [])

def process_substation(sub: str, sub_2_sub_dict: Dict[str, Dict], substation_to_lines: Dict[str, List[str]]) -> Tuple[str, Dict]:
    """Process a single substation and return its data."""
    connected_subs = list(sub_2_sub_dict.get(sub, {}).keys())
    connected_lines = get_connected_lines(sub, substation_to_lines)
    
    sub_data = {
        "subs": connected_subs,
        "line_Ids": connected_lines
    }
    
    return sub, sub_data

def process_all_substations(sub_2_sub_dict: Dict[str, Dict], substation_to_lines: Dict[str, List[str]]) -> Dict[str, Dict]:
    """Process all substations and return the complete data structure."""
    return {
        sub: process_substation(sub, sub_2_sub_dict, substation_to_lines)[1]
        for sub in sub_2_sub_dict.keys()
    }
    
def analyze_failed_transformers(failed_transformers: pd.DataFrame, sub_2_sub_dict: Dict[str, Dict], substation_to_lines: Dict[str, List[str]]) -> Dict[str, Dict]:
    """Analyze failed transformers and their connected substations and lines."""
    return dict(map(
        lambda sub: process_substation(sub, sub_2_sub_dict, substation_to_lines),
        failed_transformers.sub_id.values
    ))


def prepare_substation_coordinates(subs_gdf):
    """Extract substation coordinates from GeoDataFrame."""
    sub_coordinates = {
        sub: (lon, lat) 
        for sub, lon, lat in zip(subs_gdf.ss_id, subs_gdf.longitude, subs_gdf.latitude)
    }
    return sub_coordinates

def enhance_substation_data(sub_2_sub_dict, sub_coordinates):
    """Add geographic coordinates to substation connection data."""
    enhanced_data = {}
    
    for sub, connections in sub_2_sub_dict.items():
        if sub in sub_coordinates:
            lon, lat = sub_coordinates[sub]
            enhanced_data[sub] = {
                "coordinates": (lon, lat),
                "connections": connections
            }
    
    return enhanced_data

def create_delaunay_network(sub_coordinates):
    """Generate Delaunay triangulation network for substations."""
    # Extract coordinates as array
    points = np.array(list(sub_coordinates.values()))
    sub_ids = list(sub_coordinates.keys())
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    # Create edges from triangulation
    edges = set()
    for simplex in tri.simplices:
        # Add all edges in the triangle
        edges.add(frozenset([simplex[0], simplex[1]]))
        edges.add(frozenset([simplex[1], simplex[2]]))
        edges.add(frozenset([simplex[2], simplex[0]]))
    
    # Convert to GeoDataFrame
    network_data = []
    for edge in edges:
        i, j = list(edge)
        line = LineString([points[i], points[j]])
        network_data.append({
            "from_id": sub_ids[i],
            "to_id": sub_ids[j],
            "geometry": line,
            "distance": line.length
        })
    
    network_gdf = gpd.GeoDataFrame(network_data, crs="EPSG:4326")
    return network_gdf

def create_delaunay_triangles(sub_coordinates):
    """Generate triangles from Delaunay triangulation."""
    # Extract coordinates as array
    points = np.array(list(sub_coordinates.values()))
    sub_ids = list(sub_coordinates.keys())
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    # Create triangles
    triangles = []
    for i, simplex in enumerate(tri.simplices):
        # Create polygon from triangle vertices
        triangle = Polygon([points[simplex[0]], points[simplex[1]], points[simplex[2]]])
        triangles.append({
            "tri_id": i,
            "vertices": [sub_ids[simplex[j]] for j in range(3)],  # Fixed indexing here
            "geometry": triangle
        })
    
    triangles_gdf = gpd.GeoDataFrame(triangles, crs="EPSG:4326")
    return triangles_gdf

def clip_to_continental_us(gdf, states_gdf):
    """Clip geometries to the continental US."""
    # Filter to continental US states (excluding Alaska, Hawaii, and territories)
    continental_states = states_gdf[~states_gdf.STUSPS.isin(['AK', 'HI', 'PR', 'GU', 'VI', 'MP', 'AS'])]
    
    # Create a union of all continental states
    continental_us = continental_states.unary_union
    
    # Clip the geometries
    clipped_gdf = gpd.clip(gdf, continental_us)
    
    return clipped_gdf

def save_data_individually(data_dict, output_dir):
    """
    Save each component of the data dictionary as a separate file.
    
    Args:
        data_dict (dict): Dictionary containing all data components
        output_dir (str): Directory where files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each component individually
    for key, value in data_dict.items():
        file_path = os.path.join(output_dir, f"{key}.pkl")
        print(f"Saving {key} to {file_path}...")
        
        try:
            with open(file_path, "wb") as f:
                pickle.dump(value, f)
            print(f"  Saved successfully.")
        except Exception as e:
            print(f"  Error saving {key}: {str(e)}")
    
    # Save GeoDataFrames as GeoJSON for visualization
    geo_components = {
        "clipped_delaunay_triangles": "clipped_delaunay_triangles.geojson",
        "clipped_delaunay_network": "clipped_delaunay_network.geojson",
    }
    
    for key, filename in geo_components.items():
        if key in data_dict and isinstance(data_dict[key], gpd.GeoDataFrame):
            file_path = os.path.join(output_dir, filename)
            print(f"Saving {key} as GeoJSON to {file_path}...")
            try:
                data_dict[key].to_file(file_path, driver="GeoJSON")
                print(f"  Saved successfully.")
            except Exception as e:
                print(f"  Error saving {key} as GeoJSON: {str(e)}")
    
    # Create an index file with all the keys
    index_path = os.path.join(output_dir, "data_index.txt")
    with open(index_path, "w") as f:
        f.write("Available data components:\n")
        for key in data_dict.keys():
            f.write(f"- {key}\n")
    
    print(f"Created index file at {index_path}")


if __name__ == "__main__":
    
    # Load all data
    data_zcta, population_df, regions_pop_df, zcta_business_gdf, states_gdf, df_other = load_socioeconomic_data(data_zcta_folder)
    transformer_info, sub_2_sub, G, substation_to_lines, transmission_lines = load_power_grid_data(data_risk_ass)
    subs_all_gdf = load_substation_data(data_risk_ass)
    
    random.seed(42)

    len_transformer_info = transformer_info.shape[0]

    # Test of the code
    # Select 5 random indices and loc the df
    random_indices = random.sample(range(len_transformer_info), 50)
    failed_transformers = transformer_info.loc[random_indices]
    failed_transformers.head()

    failed_transformers_dict = analyze_failed_transformers(failed_transformers, sub_2_sub, substation_to_lines)

    # Process the data
    sub_coordinates = prepare_substation_coordinates(subs_all_gdf)
    ehv_substation_data = enhance_substation_data(sub_2_sub, sub_coordinates)
    ehv_coordinates = {sub: data["coordinates"] for sub, data in ehv_substation_data.items()}

    output_dir = "delaunay_output"
    os.makedirs(output_dir, exist_ok=True)
    all_data_path = os.path.join(output_dir, "all_delaunay_data.pkl")

    # Initialize data dictionary
    all_data = {}
    
    # Check if Delaunay data already exists
    delaunay_pkl_path = os.path.join(output_dir, "clipped_delaunay_triangles.pkl")
    if os.path.exists(delaunay_pkl_path):
        # Load existing files
        with open(delaunay_pkl_path, "rb") as f:
            clipped_delaunay_triangles = pickle.load(f)
        
        with open(os.path.join(output_dir, "clipped_delaunay_network.pkl"), "rb") as f:
            clipped_delaunay_network = pickle.load(f)
            
        print("Loaded existing clipped Delaunay data from files")
    else:
        # Create new data
        print("Creating Delaunay triangulation...")
        delaunay_triangles = create_delaunay_triangles(ehv_coordinates)
        delaunay_network = create_delaunay_network(ehv_coordinates)
        
        # Clip to continental US
        print("Clipping to continental US...")
        clipped_delaunay_triangles = clip_to_continental_us(delaunay_triangles, states_gdf)
        clipped_delaunay_network = clip_to_continental_us(delaunay_network, states_gdf)
    
    # Store all data in the dictionary
    all_data = {
        # Original input data
        "zcta_business_gdf": zcta_business_gdf,
        "population_df": population_df,
        "business_other": df_other,
        "states_gdf": states_gdf,
        "transformer_info": transformer_info,
        "sub_2_sub": sub_2_sub,
        "substation_to_lines": substation_to_lines,
        "transmission_lines": transmission_lines,
        "subs_all_gdf": subs_all_gdf,
        
        # Processed data
        "failed_transformers": failed_transformers,
        "failed_transformers_dict": failed_transformers_dict,
        "sub_coordinates": sub_coordinates,
        "ehv_substation_data": ehv_substation_data,
        "ehv_coordinates": ehv_coordinates,
        
        # Delaunay triangulation results
        "clipped_delaunay_triangles": clipped_delaunay_triangles,
        "clipped_delaunay_network": clipped_delaunay_network
    }
    
    # Save everything to a single pickle file
    save_data_individually(all_data, output_dir)
    gc.collect()