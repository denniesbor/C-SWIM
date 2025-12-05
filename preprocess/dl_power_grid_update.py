"""Download and process US substation data.
Authors: Dennies Bor and E. Oughton
"""

import os
import time
import re
import requests
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon


from configs import setup_logger, get_data_dir

warnings.filterwarnings("ignore")

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/dl_substations.log")
data_path = DATA_LOC / "substation_locations"
os.makedirs(data_path, exist_ok=True)


def download_substations_full():
    """Download all US substations in one request."""
    logger.info("Attempting to download entire US at once...")

    overpass_url = "http://overpass-api.de/api/interpreter"

    query = """
    [out:json][timeout:1200];
    area["ISO3166-1"="US"]->.searchArea;
    (
      node["power"="substation"](area.searchArea);
      way["power"="substation"](area.searchArea);
      relation["power"="substation"](area.searchArea);
    );
    out body geom;
    """

    try:
        response = requests.post(overpass_url, data={"data": query}, timeout=1260)
        response.raise_for_status()
        data = response.json()

        gdf = convert_to_geodataframe(data)
        if gdf is not None and len(gdf) > 0:
            logger.info(f"Downloaded {len(gdf)} substations")
            gdf = process_substations(gdf)

            output_path = data_path / "us_substations_full.geojson"
            gdf.to_file(output_path, driver="GeoJSON")

            shp_path = data_path / "us_substations_full.shp"
            gdf.to_file(shp_path)

            logger.info(f"Saved to {output_path}")
            return gdf
        else:
            logger.error("No data received")
            return None

    except Exception as e:
        logger.error(f"Full download failed: {e}")
        return None


def download_substations_regions():
    """Download US substations by region."""
    logger.info("Downloading US substations by regions...")

    # Regions with overlap to ensure complete CONUS coverage
    regions = {
        "Northeast": (36.5, -82.5, 48.5, -65.5),  # ME to VA/PA
        "Southeast": (23.5, -91.5, 39.0, -74.5),  # FL to NC, LA to GA
        "Great_Lakes": (39.0, -93.0, 49.5, -73.0),  # OH to NY, MN to MI
        "Central": (34.0, -106.0, 49.5, -88.0),  # TX to ND, OK to MN
        "Southwest": (24.5, -125.5, 42.5, -93.0),  # CA to TX, AZ to CO
        "Northwest": (40.5, -125.5, 49.5, -101.0),  # WA to MT, OR to WY
        "Alaska": (51.0, -180.0, 72.0, -129.0),
        "Hawaii": (18.0, -161.0, 23.0, -154.0),
        "Central_Atlantic": (35.0, -82.0, 42.0, -71.0),  # NC to NJ
        "Four_Corners": (31.0, -115.0, 42.0, -102.0),  # UT/CO/AZ/NM area
    }

    all_substations = []
    failed_regions = []

    for region_name, bbox in regions.items():
        logger.info(f"Downloading {region_name} region...")
        gdf = download_region(bbox)

        if gdf is not None and len(gdf) > 0:
            gdf["region"] = region_name
            all_substations.append(gdf)
            logger.info(f"  Found {len(gdf)} substations")
        else:
            failed_regions.append(region_name)
            logger.warning(f"  Failed: {region_name}")

        time.sleep(3)

    if all_substations:
        combined_gdf = pd.concat(all_substations, ignore_index=True)

        combined_gdf = process_substations(combined_gdf)

        output_path = data_path / "us_substations_regions.geojson"
        combined_gdf.to_file(output_path, driver="GeoJSON")

        shp_path = data_path / "us_substations_regions.shp"
        combined_gdf.to_file(shp_path)

        logger.info(f"Saved {len(combined_gdf)} substations to {output_path}")
        if failed_regions:
            logger.warning(f"Failed regions: {failed_regions}")

        return combined_gdf
    else:
        logger.error("Failed to download any regions")
        return None


def convert_to_geodataframe(overpass_data):
    """Convert Overpass API response to GeoDataFrame."""
    if not overpass_data.get("elements"):
        return gpd.GeoDataFrame()

    features = []

    # essential tags to extract
    essential_tags = {
        "name",
        "operator",
        "voltage",
        "power",
        "substation",
        "capacity",
        "frequency",
        "ref",
        "network",
        "operator:type",
        "owner",
        "start_date",
        "addr:state",
        "addr:city",
        "addr:county",
    }

    for element in overpass_data["elements"]:
        # Only extract essential properties
        properties = {}
        tags = element.get("tags", {})

        # Only keep essential tags
        for tag in essential_tags:
            if tag in tags:
                properties[tag] = tags[tag]

        properties["osmid"] = element["id"]
        properties["element_type"] = element["type"]

        geometry = None

        if element["type"] == "node":
            geometry = Point(element["lon"], element["lat"])

        elif element["type"] == "way" and "geometry" in element:
            coords = [(point["lon"], point["lat"]) for point in element["geometry"]]

            if len(coords) > 3 and coords[0] == coords[-1]:
                geometry = Polygon(coords)
            else:
                geometry = LineString(coords)

        elif element["type"] == "relation":
            continue

        if geometry:
            features.append({"geometry": geometry, **properties})

    if not features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf.set_index("osmid", inplace=True)

    return gdf


def process_substations(gdf):
    """Clean and convert geometries to points."""
    # Remove duplicates based on osmid
    initial_count = len(gdf)
    gdf = gdf[~gdf.index.duplicated(keep="first")]
    logger.info(f"Removed {initial_count - len(gdf)} duplicates")

    gdf["original_geom_type"] = gdf.geometry.geom_type

    point_mask = gdf.geometry.geom_type == "Point"
    gdf.loc[~point_mask, "geometry"] = gdf.loc[~point_mask, "geometry"].apply(
        lambda geom: geom.representative_point()
    )

    if "voltage" in gdf.columns:

        voltage_series = gdf["voltage"].copy()


        max_voltages = pd.Series(index=gdf.index, dtype=float)

        numeric_mask = pd.to_numeric(voltage_series, errors="coerce").notna()
        max_voltages[numeric_mask] = pd.to_numeric(voltage_series[numeric_mask])


        string_mask = ~numeric_mask & voltage_series.notna()

        if string_mask.any():

            string_voltages = voltage_series[string_mask].astype(str)


            simple_pattern = r"^(\d+)$"
            simple_matches = string_voltages.str.extract(simple_pattern)
            simple_mask = simple_matches[0].notna()


            simple_indices = string_voltages[simple_mask].index
            max_voltages.loc[simple_indices] = pd.to_numeric(
                simple_matches.loc[simple_mask, 0]
            )

            complex_mask = string_voltages.str.contains(";", na=False)

            if complex_mask.any():

                complex_voltages = string_voltages[complex_mask]

                def extract_max_from_string(s):
                    numbers = re.findall(r"\d+", s)
                    return max(map(int, numbers)) if numbers else np.nan

                complex_results = complex_voltages.apply(extract_max_from_string)
                max_voltages.loc[complex_results.index] = complex_results

        # Assign results
        gdf["max_voltage"] = max_voltages
        gdf["voltage_kv"] = gdf["max_voltage"] / 1000

    essential_columns = [
        "geometry",
        # Core identifiers
        "osmid",
        "element_type",
        "original_geom_type",
        # Power system attributes
        "name",
        "power",
        "substation",
        "voltage",
        "voltage_kv",
        "max_voltage",
        "capacity",
        "frequency",
        # Operators/ownership
        "operator",
        "operator:type",
        "owner",
        # Location
        "addr:state",
        "addr:city",
        "addr:county",
        "ref",
        "network",
        "start_date",
    ]

    existing_columns = [col for col in essential_columns if col in gdf.columns]
    gdf = gdf[existing_columns]

    logger.info(f"Reduced to {len(gdf.columns)} essential columns")

    return gdf


def download_region(bbox, timeout=300):
    """Download substations for a region."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    min_lat, min_lon, max_lat, max_lon = bbox

    query = f"""
    [out:json][timeout:{timeout}];
    (
      node["power"="substation"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["power"="substation"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["power"="substation"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body geom;
    """

    try:
        response = requests.post(
            overpass_url, data={"data": query}, timeout=timeout + 30
        )
        response.raise_for_status()
        data = response.json()

        gdf = convert_to_geodataframe(data)

        if len(gdf) > 0:
            gdf = process_substations(gdf)

        return gdf
    except Exception as e:
        logger.error(f"Error downloading region: {e}")
        return None


if __name__ == "__main__":
    logger.info("Starting substations download...")

    # Try full download first
    full_result = download_substations_full()

    # Always do regional download as backup
    regional_result = download_substations_regions()

    if full_result is not None:
        logger.info(f"Full download: {len(full_result)} substations")
    else:
        logger.error("Full download: Failed")

    if regional_result is not None:
        logger.info(f"Regional download: {len(regional_result)} substations")
    else:
        logger.error("Regional download: Failed")

    logger.info(f"Data saved to: {data_path}")
