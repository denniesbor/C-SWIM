"""
Economic data pipeline: processes raw census, BEA CAGDP2, NAICS, and FERC
region spatial data into ZCTA-level sectoral GDP and establishment counts,
plus the Voronoi polygons consumed downstream by the areal interpolation.
Author: Dennies Bor & Edward Oughton
"""

import os
import warnings
import pickle
import gc
from pathlib import Path
from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import Voronoi

warnings.filterwarnings("ignore")

from configs import setup_logger, get_data_dir, DATA_DIR
from econ.preprocess.fetch_bea_census import fetch_all, DEFAULT_YEAR

DATA_LOC = get_data_dir(econ=True)
raw_data_folder = DATA_LOC / "raw_econ_data"
processed_econ_dir = DATA_LOC / "processed_econ"
processed_voronoi_dir = DATA_LOC / "processed_voronoi"
logger = setup_logger(log_file="logs/p_econ_data.log")

# 2-digit NAICS -> 10-sector label. Government (NAICS 92) is handled
# separately because CBP does not cover it.
NAICS_TO_SECTOR = {
    11: "AGR",
    21: "MINING",
    22: "UTIL_CONST",
    23: "UTIL_CONST",
    31: "MANUF",
    32: "MANUF",
    33: "MANUF",
    42: "TRADE_TRANSP",
    44: "TRADE_TRANSP",
    48: "TRADE_TRANSP",
    51: "INFO",
    52: "FIRE",
    53: "FIRE",
    54: "PROF_OTHER",
    55: "PROF_OTHER",
    56: "PROF_OTHER",
    81: "PROF_OTHER",
    61: "EDUC_ENT",
    62: "EDUC_ENT",
    71: "EDUC_ENT",
    72: "EDUC_ENT",
}
SECTORS_CBP = sorted(set(NAICS_TO_SECTOR.values()))
ALL_SECTORS = SECTORS_CBP + ["G"]


def read_text_file(file_path):
    """Read a text/csv file tolerating encoding errors."""
    logger.debug(f"Reading {file_path}")
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return pd.read_csv(StringIO(content), low_memory=False)


def create_zcta_population_csv(data_loc: Path):
    """Process 2020 decennial population at the ZCTA level."""
    logger.info("Creating ZCTA population data")
    zcta_pop_20 = read_text_file(data_loc / "pop_2020_zcta.csv")
    zcta_pop_20.drop(0, inplace=True)
    zcta_pop_20.reset_index(drop=True, inplace=True)

    zcta_pop_20["zcta"] = zcta_pop_20["zcta"].astype(str).str.strip()
    zcta_pop_20 = zcta_pop_20[zcta_pop_20["zcta"] != ""]
    zcta_pop_20[["zcta", "pop20"]] = zcta_pop_20[["zcta", "pop20"]].astype(int)

    zcta_pop_20.columns = [c.upper() for c in zcta_pop_20.columns]
    zcta_pop_20.rename(columns={"STAB": "STABBR"}, inplace=True)

    zcta_pop_20.to_csv(
        processed_econ_dir / "2020_decennial_census_at_ZCTA_level.csv", index=False
    )
    logger.info(f"Created ZCTA population data: {len(zcta_pop_20):,} rows")
    return zcta_pop_20


def create_naics_establishments_data(data_loc: Path, year: int = DEFAULT_YEAR):
    """Process ZBP NAICS establishments and map ZIP codes to ZCTAs.

    The ZBP industry detail schema is stable across reference years, so only
    the filename depends on the vintage.
    """
    yy = f"{year % 100:02d}"
    logger.info(f"Processing NAICS establishments from zbp{yy}detail.txt")

    zcta_cbp_detailed = read_text_file(data_loc / f"zbp{yy}detail.txt")
    zcta_cbp_detailed.columns = [c.upper() for c in zcta_cbp_detailed.columns]

    # 2-digit subsector rows look like "22----". We keep those, then add back
    # an "UNCLFD" bucket for any establishments the "------" total row
    # reports beyond what the 2-digit rows sum to.
    subsector_sums = (
        zcta_cbp_detailed[zcta_cbp_detailed["NAICS"].str.contains(r"\d{2}----")]
        .groupby("ZIP")["EST"]
        .sum()
    )
    total_establishments = zcta_cbp_detailed[
        zcta_cbp_detailed["NAICS"] == "------"
    ].set_index("ZIP")["EST"]
    other_establishments = total_establishments.sub(
        subsector_sums, fill_value=0
    ).reset_index()
    other_establishments["NAICS"] = "UNCLFD"

    filtered_df = zcta_cbp_detailed[
        zcta_cbp_detailed["NAICS"].str.contains(r"\b\d{2}----", na=False, regex=True)
    ].copy()
    filtered_df["NAICS"] = (
        filtered_df["NAICS"].str.replace("-", "", regex=True).astype(int)
    )

    filtered_ = filtered_df[["ZIP", "NAICS", "EST", "STABBR"]]
    combined_df = pd.concat([filtered_, other_establishments[["ZIP", "NAICS", "EST"]]])
    combined_df = combined_df.sort_values(by="ZIP")

    # ZIP -> ZCTA via UDS crosswalk (frozen at 2021, the last release; ZCTAs
    # are only revised at the decennial so 2021 is fine through 2029).
    zcta_zip_df = pd.read_excel(data_loc / "ZIPCodetoZCTACrosswalk2021UDS.xlsx")
    zz_not_na = zcta_zip_df[~zcta_zip_df.ZCTA.isna()].copy()
    zz_not_na["ZCTA"] = zz_not_na["ZCTA"].astype(int)
    zz_not_na.rename(columns={"ZIP_CODE": "ZIP", "STATE": "STABBR"}, inplace=True)

    zz_dtld = combined_df.merge(
        zz_not_na[["STABBR", "ZIP", "PO_NAME", "ZCTA"]], on="ZIP", how="left"
    )
    zz_dtld.drop("STABBR_x", axis=1, inplace=True)
    zz_dtld.rename(columns={"STABBR_y": "STABBR"}, inplace=True)

    df_naics_zcta = (
        zz_dtld.groupby(["ZCTA", "NAICS", "STABBR"])["EST"].sum().reset_index()
    )
    logger.info(f"NAICS establishments: {len(df_naics_zcta):,} rows")
    return df_naics_zcta


def create_zcta_within_rto(data_loc: Path):
    """Spatially join ZCTA points to NERC transmission regions."""
    logger.info("Creating ZCTA within RTO mapping")
    rto_gdf = gpd.read_file(data_loc / "NERC Map" / "electricity_operators.shp")

    overlaps = gpd.sjoin(rto_gdf, rto_gdf, how="inner", predicate="intersects")
    overlaps = overlaps[overlaps["id_left"] != overlaps["id_right"]]

    nerc_gdf = rto_gdf.copy()
    for _, row in overlaps.iterrows():
        outer_geom = nerc_gdf.loc[nerc_gdf["id"] == row["id_left"], "geometry"].iloc[0]
        inner_geom = nerc_gdf.loc[nerc_gdf["id"] == row["id_right"], "geometry"].iloc[0]
        if inner_geom.within(outer_geom):
            nerc_gdf.loc[nerc_gdf["id"] == row["id_left"], "geometry"] = (
                outer_geom.difference(inner_geom)
            )

    nerc_gdf.to_crs(epsg=4326, inplace=True)

    states = gpd.read_file(data_loc / "tl_2022_us_state.zip").to_crs(epsg=4326)
    non_cont_fips = ["02", "15", "72", "66", "60", "69", "78"]
    states = states[~states.GEOID.isin(non_cont_fips)]

    nerc_gdf.rename(columns={"id": "REGION_ID"}, inplace=True)
    nerc_gdf.loc[nerc_gdf["REGION_ID"] == 23, "REGIONS"] = "ERCOT"

    utm = 32633
    nerc_proj = nerc_gdf.to_crs(epsg=utm)
    states_proj = states.to_crs(epsg=utm)

    nerc_buffered = nerc_proj.buffer(10)
    states_boundary = states_proj.geometry.unary_union
    states_boundary_gdf = gpd.GeoDataFrame(geometry=[states_boundary], crs=utm)

    aligned = gpd.overlay(
        gpd.GeoDataFrame(geometry=nerc_buffered, crs=utm),
        states_boundary_gdf,
        how="intersection",
    ).to_crs(nerc_gdf.crs)
    nerc_gdf.geometry = aligned.geometry

    zcta_gdf = gpd.read_file(data_loc / "tl_2020_us_zcta520.zip")
    zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"}, inplace=True)
    zcta_gdf["ZCTA"] = zcta_gdf["ZCTA"].astype(int)
    zcta_gdf["representative_point"] = zcta_gdf.geometry.representative_point()

    nerc_gdf = nerc_gdf.to_crs(zcta_gdf.crs)
    zcta_within_rto = gpd.sjoin(
        zcta_gdf.set_geometry("representative_point"),
        nerc_gdf,
        how="inner",
        predicate="within",
    )
    zcta_within_rto.columns = [c.upper() for c in zcta_within_rto.columns]
    logger.info(f"ZCTA-RTO mapping: {len(zcta_within_rto):,} rows")
    return zcta_within_rto


def create_zcta_to_county_mapping(data_loc: Path, year: int = DEFAULT_YEAR):
    """Build a ZCTA -> county FIPS lookup via spatial join (CONUS only).

    Uses TIGER county polygons for the given year and ZCTA 2020 polygons.
    ZCTAs are only redefined at the decennial so the 2020 polygons remain
    the correct choice through 2029.
    """
    logger.info(f"Creating ZCTA -> county FIPS mapping ({year})")

    zcta_gdf = gpd.read_file(data_loc / "tl_2020_us_zcta520.zip")
    zcta_gdf = zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"})
    zcta_gdf["ZCTA"] = zcta_gdf["ZCTA"].astype(str).str.zfill(5)

    county_gdf = gpd.read_file(data_loc / f"tl_{year}_us_county.zip")
    county_gdf["GeoFips"] = county_gdf["STATEFP"] + county_gdf["COUNTYFP"]
    non_conus = ["02", "15", "60", "66", "69", "72", "78"]
    county_gdf = county_gdf[~county_gdf["STATEFP"].isin(non_conus)]

    aea = "EPSG:5070"
    zcta_pts = zcta_gdf[["ZCTA", "geometry"]].to_crs(aea).copy()
    zcta_pts["geometry"] = zcta_pts.geometry.representative_point()
    county_proj = county_gdf[["GeoFips", "geometry"]].to_crs(aea)

    joined = gpd.sjoin(zcta_pts, county_proj, how="left", predicate="within")
    zcta_to_county = (
        joined[["ZCTA", "GeoFips"]]
        .dropna(subset=["GeoFips"])
        .drop_duplicates(subset="ZCTA")
        .reset_index(drop=True)
    )
    zcta_to_county["ZCTA"] = zcta_to_county["ZCTA"].astype(int)

    out_fp = processed_econ_dir / f"ZCTA_county_mapping_{year}.csv"
    zcta_to_county.to_csv(out_fp, index=False)
    logger.info(f"Mapped {len(zcta_to_county):,} CONUS ZCTAs to counties")
    return zcta_to_county


def create_zcta_county_anchored_gdp(
    df_naics_zcta: pd.DataFrame,
    zcta_to_county: pd.DataFrame,
    zcta_pop: pd.DataFrame,
    cagdp2: pd.DataFrame,
    year: int = DEFAULT_YEAR,
):
    """Allocate CAGDP2 county GDP to ZCTAs using establishment-share dasymetry.

    Primary share: establishment count within each (county, sector). Fallback
    share: POP20, used when a county has GDP in a sector but ZBP suppression
    leaves zero establishments. Government is always allocated by POP20 share
    because CBP does not cover NAICS 92.

    Returns long DataFrame (ZCTA, GeoFips, SECTOR, EST, GDP). GDP is daily
    $ millions per ZCTA per sector.
    """
    logger.info("Allocating CAGDP2 county GDP to ZCTAs")

    # Narrow ZBP rows to those with an integer NAICS that maps into our scheme.
    est = df_naics_zcta.copy()
    est = est[est["NAICS"].apply(lambda v: isinstance(v, (int, np.integer)))]
    est["SECTOR"] = est["NAICS"].map(NAICS_TO_SECTOR)
    est = est.dropna(subset=["SECTOR"])
    est["EST"] = pd.to_numeric(est["EST"], errors="coerce").fillna(0)

    zcta_sector_est = est.groupby(["ZCTA", "SECTOR"], as_index=False)["EST"].sum()

    # ZCTA master: every CONUS ZCTA with its county and POP20, regardless of
    # whether it has any establishments at all.
    zcta_master = zcta_to_county[["ZCTA", "GeoFips"]].merge(
        zcta_pop[["ZCTA", "POP20"]], on="ZCTA", how="left"
    )
    zcta_master["POP20"] = zcta_master["POP20"].fillna(0).astype(int)

    cagdp2_long = cagdp2.reset_index().melt(
        id_vars=["GeoFips", "GeoName"], var_name="SECTOR", value_name="COUNTY_GDP"
    )
    cagdp2_long["GeoFips"] = cagdp2_long["GeoFips"].astype(str).str.zfill(5)

    # Dense (ZCTA x CBP sector) grid so zero-establishment cells exist and
    # can receive POP20-weighted GDP when the primary share is undefined.
    grid = (
        zcta_master.assign(key=1)
        .merge(pd.DataFrame({"SECTOR": SECTORS_CBP, "key": 1}), on="key")
        .drop(columns="key")
        .merge(zcta_sector_est, on=["ZCTA", "SECTOR"], how="left")
    )
    grid["EST"] = grid["EST"].fillna(0)
    grid["EST_COUNTY"] = grid.groupby(["GeoFips", "SECTOR"])["EST"].transform("sum")
    grid["POP_COUNTY"] = grid.groupby(["GeoFips", "SECTOR"])["POP20"].transform("sum")

    est_share = np.where(grid["EST_COUNTY"] > 0, grid["EST"] / grid["EST_COUNTY"], 0.0)
    pop_share = np.where(
        grid["POP_COUNTY"] > 0, grid["POP20"] / grid["POP_COUNTY"], 0.0
    )
    use_fallback = grid["EST_COUNTY"] == 0
    grid["SHARE"] = np.where(use_fallback, pop_share, est_share)

    grid = grid.merge(
        cagdp2_long[["GeoFips", "SECTOR", "COUNTY_GDP"]],
        on=["GeoFips", "SECTOR"],
        how="left",
    )
    grid["GDP"] = grid["SHARE"] * grid["COUNTY_GDP"].fillna(0)
    fallback_gdp = grid.loc[use_fallback, "GDP"].sum()
    logger.info(f"POP20 fallback absorbed ${fallback_gdp:,.0f}M in suppressed cells")

    cbp_out = grid[["ZCTA", "GeoFips", "SECTOR", "EST", "GDP"]].copy()

    # Government: county G GDP split within county by POP20 share.
    g_long = cagdp2_long[cagdp2_long["SECTOR"] == "G"][["GeoFips", "COUNTY_GDP"]]
    g = zcta_master.merge(g_long, on="GeoFips", how="left")
    g["POP_COUNTY"] = g.groupby("GeoFips")["POP20"].transform("sum")
    g["SHARE"] = np.where(g["POP_COUNTY"] > 0, g["POP20"] / g["POP_COUNTY"], 0.0)
    g["GDP"] = g["SHARE"] * g["COUNTY_GDP"].fillna(0)
    g["SECTOR"] = "G"
    g["EST"] = 0
    g_out = g[["ZCTA", "GeoFips", "SECTOR", "EST", "GDP"]]

    result = pd.concat([cbp_out, g_out], ignore_index=True)
    result["EST"] = result["EST"].astype(int)

    # Convert annual $ millions to daily $ millions to match the downstream
    # convention preserved from the original pipeline.
    result["GDP"] = (result["GDP"] / 365.0).astype(float)

    out_fp = processed_econ_dir / f"ZCTA_county_anchored_GDP_{year}.csv"
    result.to_csv(out_fp, index=False)
    logger.info(f"Saved {len(result):,} ZCTA x sector rows to {out_fp}")
    return result


def load_socioeconomic_data(naics_est_gdp, population_df):
    """Pivot the long ZCTA x SECTOR table into the wide GeoDataFrame that
    downstream areal interpolation consumes.

    Returns (regions_pop_df, zcta_business_gdf, states_gdf) where
    zcta_business_gdf carries GDP_<sector>, EST_<sector>, and POP20 columns
    on ZCTA representative points.
    """
    logger.info("Assembling socioeconomic GeoDataFrame")

    # Regions + population side table (unchanged contract).
    regions = naics_est_gdp[["ZCTA", "GeoFips"]].drop_duplicates()
    regions_pop_df = regions.merge(population_df, on="ZCTA", how="left")

    # ZCTA polygons for the geometry side of the output GeoDataFrame.
    zcta_gdf = gpd.read_file(raw_data_folder / "tl_2020_us_zcta520.zip")
    zcta_gdf = zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"}).astype({"ZCTA": "Int64"})

    # Wide pivots: one column per sector for GDP and EST.
    gdp_wide = (
        naics_est_gdp.pivot_table(
            index="ZCTA",
            columns="SECTOR",
            values="GDP",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=ALL_SECTORS, fill_value=0.0)
        .add_prefix("GDP_")
        .reset_index()
    )
    est_wide = (
        naics_est_gdp.pivot_table(
            index="ZCTA",
            columns="SECTOR",
            values="EST",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=ALL_SECTORS, fill_value=0)
        .add_prefix("EST_")
        .reset_index()
    )

    zcta_wide = gdp_wide.merge(est_wide, on="ZCTA")
    zcta_business_gdf = zcta_gdf.merge(zcta_wide, on="ZCTA", how="inner")

    # Keep polygons as the active geometry. Tobler's masked_area_interpolate
    # is area-weighted, so points (zero area) silently produce zero allocations.
    zcta_business_gdf = zcta_business_gdf.to_crs(epsg=4326)

    zcta_business_gdf = zcta_business_gdf.merge(
        population_df[["ZCTA", "POP20"]], on="ZCTA", how="left"
    )
    zcta_business_gdf["POP20"] = zcta_business_gdf["POP20"].fillna(0).astype(np.uint32)
    zcta_business_gdf = zcta_business_gdf.drop_duplicates(subset="ZCTA", keep="first")

    gdp_cols = [f"GDP_{s}" for s in ALL_SECTORS]
    est_cols = [f"EST_{s}" for s in ALL_SECTORS]
    zcta_business_gdf[gdp_cols] = zcta_business_gdf[gdp_cols].astype(np.float32)
    zcta_business_gdf[est_cols] = zcta_business_gdf[est_cols].astype(np.uint32)

    states_gdf = gpd.read_file(raw_data_folder / "tl_2022_us_state.zip").to_crs(
        epsg=4326
    )

    logger.info("Socioeconomic data assembled")
    return regions_pop_df, zcta_business_gdf, states_gdf


def create_voronoi_polygons(
    sub_coordinates: Dict[str, Tuple[float, float]], states_gdf
) -> gpd.GeoDataFrame:
    """Create Voronoi polygons from substation coordinates, clipped to CONUS."""
    voronoi_file = processed_voronoi_dir / "voronoi_polygons_clipped.geojson"
    if voronoi_file.exists():
        logger.info(f"Loading existing Voronoi polygons from {voronoi_file}")
        return gpd.read_file(voronoi_file)

    logger.info("Creating Voronoi polygons")
    coords = list(sub_coordinates.values())
    vor = Voronoi(coords)
    sub_ids = list(sub_coordinates.keys())

    polygons = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if -1 not in region and region:
            polygons.append(Polygon([vor.vertices[i] for i in region]))
        else:
            polygons.append(None)

    valid = [
        {"sub_id": sid, "geometry": poly}
        for sid, poly in zip(sub_ids, polygons)
        if poly is not None
    ]
    voronoi_gdf = gpd.GeoDataFrame(valid, crs="EPSG:4326")

    conus_states = states_gdf[~states_gdf["STATEFP"].isin(["02", "15", "72"])]
    conus_boundary = conus_states.unary_union

    voronoi_gdf = voronoi_gdf[voronoi_gdf.is_valid & ~voronoi_gdf.is_empty]
    voronoi_gdf = gpd.overlay(
        voronoi_gdf,
        gpd.GeoDataFrame(geometry=[conus_boundary], crs="EPSG:4326"),
        how="intersection",
    )

    os.makedirs(processed_voronoi_dir, exist_ok=True)
    voronoi_gdf.to_file(voronoi_file, driver="GeoJSON")
    logger.info("Saved Voronoi polygons")
    return voronoi_gdf


if __name__ == "__main__":
    logger.info("Starting economic data pipeline")

    logger.info("Phase 1: upstream fetch")
    cagdp2, _, _ = fetch_all(year=DEFAULT_YEAR)

    logger.info("Phase 2: raw data processing")
    os.makedirs(processed_econ_dir, exist_ok=True)

    zcta_pop_20 = create_zcta_population_csv(raw_data_folder)
    df_naics_zcta = create_naics_establishments_data(raw_data_folder, year=DEFAULT_YEAR)
    zcta_within_rto = create_zcta_within_rto(raw_data_folder)
    zcta_to_county = create_zcta_to_county_mapping(raw_data_folder, year=DEFAULT_YEAR)

    logger.info("Phase 3: county-anchored GDP allocation")
    naics_est_gdp = create_zcta_county_anchored_gdp(
        df_naics_zcta=df_naics_zcta,
        zcta_to_county=zcta_to_county,
        zcta_pop=zcta_pop_20,
        cagdp2=cagdp2,
        year=DEFAULT_YEAR,
    )

    logger.info("Phase 4: analysis data preparation")
    regions_pop_df, zcta_business_gdf, states_gdf = load_socioeconomic_data(
        naics_est_gdp, zcta_pop_20
    )

    logger.info("Phase 5: spatial Voronoi polygons")
    df_substation = pd.read_csv(DATA_DIR / "admittance_matrix" / "substation_info.csv")
    ehv_coordinates = dict(
        zip(
            df_substation["name"],
            zip(df_substation["longitude"], df_substation["latitude"]),
        )
    )
    voronoi_gdf = create_voronoi_polygons(ehv_coordinates, states_gdf)

    logger.info("Phase 6: save processed pickle")
    processed_file = processed_econ_dir / "socioeconomic_data.pkl"
    with open(processed_file, "wb") as f:
        pickle.dump(
            (naics_est_gdp, zcta_pop_20, regions_pop_df, zcta_business_gdf, states_gdf),
            f,
        )
    logger.info(f"Saved {processed_file}")
