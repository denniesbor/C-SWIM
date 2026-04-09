"""
Download ONLY US substation RELATIONS (power=substation) from Overpass,
store them as POINTS using Overpass-provided 'center',
and ALIGN schema to match us_substations_full.geojson essential columns.

Saves:
- data/substation_locations_relations/us_substations_relations.geojson
- data/substation_locations_relations/us_substations_relations.shp   (optional)
"""

import os
import time
import re
import requests
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

from configs import setup_logger, get_data_dir

warnings.filterwarnings("ignore")

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/dl_substations_relations.log")

data_path = DATA_LOC / "substation_locations_relations"
os.makedirs(data_path, exist_ok=True)

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]


ESSENTIAL_COLUMNS = [
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


def compute_voltage_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute max_voltage and voltage_kv from 'voltage' strings."""
    if "voltage" not in gdf.columns:
        gdf["max_voltage"] = np.nan
        gdf["voltage_kv"] = np.nan
        return gdf

    voltage_series = gdf["voltage"].copy()
    max_voltages = pd.Series(index=gdf.index, dtype=float)

    numeric_mask = pd.to_numeric(voltage_series, errors="coerce").notna()
    max_voltages[numeric_mask] = pd.to_numeric(voltage_series[numeric_mask])

    string_mask = ~numeric_mask & voltage_series.notna()
    if string_mask.any():
        string_voltages = voltage_series[string_mask].astype(str)

        # simple integer
        simple_matches = string_voltages.str.extract(r"^(\d+)$")
        simple_mask = simple_matches[0].notna()
        if simple_mask.any():
            idx = string_voltages[simple_mask].index
            max_voltages.loc[idx] = pd.to_numeric(simple_matches.loc[simple_mask, 0])

        # complex strings like "500000;230000;161000"
        complex_mask = string_voltages.str.contains(";", na=False)
        if complex_mask.any():

            def extract_max(s):
                nums = re.findall(r"\d+", s)
                return max(map(int, nums)) if nums else np.nan

            res = string_voltages[complex_mask].apply(extract_max)
            max_voltages.loc[res.index] = res

    gdf["max_voltage"] = max_voltages
    gdf["voltage_kv"] = gdf["max_voltage"] / 1000
    return gdf


def align_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Force same schema/order as main substations file."""
    # ensure required columns exist
    for col in ESSENTIAL_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = np.nan

    # set known values
    gdf["element_type"] = "relation"
    gdf["original_geom_type"] = "RelationCenter"

    # compute voltage derived fields
    gdf = compute_voltage_fields(gdf)

    # keep exact column order
    gdf = gdf[ESSENTIAL_COLUMNS]

    return gdf


def download_us_substation_relations(timeout=1200, retries=3) -> gpd.GeoDataFrame:
    """Download US substation relations with center points."""
    query = f"""
    [out:json][timeout:{timeout}];
    area["ISO3166-1"="US"]->.searchArea;
    (
      relation["power"="substation"](area.searchArea);
    );
    out body center;
    """

    last_err = None
    data = None

    for url in OVERPASS_SERVERS:
        for k in range(retries):
            try:
                logger.info(f"Trying Overpass: {url} (attempt {k+1}/{retries})")
                r = requests.post(url, data={"data": query}, timeout=timeout + 60)
                if r.status_code == 200:
                    data = r.json()
                    break
                last_err = f"{url} -> HTTP {r.status_code}"
            except Exception as e:
                last_err = f"{url} -> {repr(e)}"

            time.sleep(3 * (k + 1))

        if data is not None:
            break

    if data is None or not data.get("elements"):
        raise RuntimeError(
            f"Failed to download relation substations. Last error: {last_err}"
        )

    feats = []
    for el in data["elements"]:
        if el.get("type") != "relation":
            continue

        if "center" not in el:
            continue

        tags = el.get("tags", {})
        geom = Point(el["center"]["lon"], el["center"]["lat"])

        feats.append(
            {
                "osmid": el["id"],
                "element_type": "relation",
                **tags,
                "geometry": geom,
            }
        )

    gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326")

    if len(gdf) == 0:
        return gdf

    gdf = gdf.drop_duplicates(subset="osmid", keep="first")

    # ALIGN to main schema
    gdf = align_schema(gdf)

    return gdf


if __name__ == "__main__":
    logger.info("Downloading US substation relations only...")

    gdf_rel = download_us_substation_relations()
    logger.info(f"Downloaded {len(gdf_rel)} relation substations")

    target = 14133615
    has_target = (gdf_rel["osmid"].astype("int64") == target).any()
    logger.info(f"Contains {target}: {has_target}")
    print(f"Contains {target}: {has_target}")

    out_geojson = data_path / "us_substations_relations.geojson"
    out_shp = data_path / "us_substations_relations.shp"

    gdf_rel.to_file(out_geojson, driver="GeoJSON")
    # NOTE: SHP will truncate/rename long fields; safe to keep if you need it.
    gdf_rel.to_file(out_shp)

    logger.info(f"Saved relations GeoJSON: {out_geojson}")
    logger.info(f"Saved relations SHP: {out_shp}")
    logger.info(f"Done. Output dir: {data_path}")
