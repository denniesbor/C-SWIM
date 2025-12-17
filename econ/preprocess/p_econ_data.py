"""
Economic data pipeline that processes raw census, GDP, NAICS, and FERC regions spatial data.
Author: Dennies Bor & Edward Oughton
"""

import os
import warnings
import pickle
import gc
from pathlib import Path
from io import StringIO
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import geopandas as gpd

warnings.filterwarnings("ignore")

from configs import setup_logger, get_data_dir, DATA_DIR

DATA_LOC = get_data_dir(econ=True)
raw_data_folder = DATA_LOC / "raw_econ_data"
processed_econ_dir = DATA_LOC / "processed_econ"
processed_voronoi_dir = DATA_LOC / "processed_voronoi"
logger = setup_logger(log_file="logs/p_econ_data.log")


def read_text_file(file_path):
    """Reads large text/csv files with encoding error handling."""
    logger.debug(f"Reading text file: {file_path}")
    with open(file_path, encoding="utf-8", errors="ignore") as file:
        content = file.read()
    content_io = StringIO(content)
    return pd.read_csv(content_io, low_memory=False)


def create_zcta_population_csv(data_loc: Path):
    """Processes 2020 decennial census population data at ZCTA level."""
    logger.info("Creating ZCTA population data")
    zcta_pop_20 = read_text_file(data_loc / "pop_2020_zcta.csv")

    zcta_pop_20.drop(0, inplace=True)
    zcta_pop_20.reset_index(inplace=True, drop=True)

    zcta_pop_20["zcta"] = zcta_pop_20.zcta.str.strip()
    zcta_pop_20 = zcta_pop_20[zcta_pop_20["zcta"] != ""]
    zcta_pop_20[["zcta", "pop20"]] = zcta_pop_20[["zcta", "pop20"]].astype(int)

    zcta_pop_20.columns = [col.upper() for col in zcta_pop_20.columns]
    zcta_pop_20.rename(columns={"STAB": "STABBR"}, inplace=True)

    zcta_pop_20.to_csv(
        processed_econ_dir / "2020_decennial_census_at_ZCTA_level.csv", index=False
    )
    logger.info(f"Created ZCTA population data with {len(zcta_pop_20):,} records")
    return zcta_pop_20


def create_naics_establishments_data(data_loc: Path):
    """Processes NAICS establishment data and maps ZIP codes to ZCTAs."""
    logger.info("Processing NAICS establishment data")
    zcta_cbp_detailed = read_text_file(data_loc / "zbp21detail.txt")
    zcta_cbp_detailed.columns = [col.upper() for col in zcta_cbp_detailed.columns]

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
    ]
    filtered_df["NAICS"] = (
        filtered_df["NAICS"].str.replace("-", "", regex=True).astype(int)
    )

    filtered_ = filtered_df[["ZIP", "NAICS", "EST", "STABBR"]]
    combined_df = pd.concat([filtered_, other_establishments[["ZIP", "NAICS", "EST"]]])
    combined_df = combined_df.sort_values(by="ZIP")

    zcta_zip_df = pd.read_excel(data_loc / "ZIPCodetoZCTACrosswalk2021UDS.xlsx")
    zz_not_na = zcta_zip_df[~zcta_zip_df.ZCTA.isna()]
    zz_not_na.ZCTA = zz_not_na.ZCTA.astype(int)
    zz_not_na.rename(columns={"ZIP_CODE": "ZIP", "STATE": "STABBR"}, inplace=True)

    zz_dtld = combined_df.merge(
        zz_not_na[["STABBR", "ZIP", "PO_NAME", "ZCTA"]], on="ZIP", how="left"
    )
    zz_dtld.drop("STABBR_x", axis=1, inplace=True)
    zz_dtld.rename(columns={"STABBR_y": "STABBR"}, inplace=True)

    df_naics_zcta = (
        zz_dtld.groupby(["ZCTA", "NAICS", "STABBR"])["EST"].sum().reset_index()
    )

    logger.info(
        f"Created NAICS establishments data with {len(df_naics_zcta):,} records"
    )
    return df_naics_zcta


def create_state_gdp_employment_data(data_loc: Path):
    """Creates consolidated state-level GDP, employment, and population dataset."""
    logger.info("Creating state GDP and employment data")
    gdp_state = read_text_file(data_loc / "SAGDP" / "SAGDP1__ALL_AREAS_2017_2022.csv")

    state_2022 = gdp_state[
        gdp_state.Description == "Real GDP (millions of chained 2017 dollars) 1/"
    ]
    state_2022.loc[:, "STATE"] = state_2022.GeoFIPS.apply(lambda fips: int(fips[2:4]))

    us_states_details_naics = read_text_file(
        data_loc / "us_state_naics_detailedsizes_2020.txt"
    )

    state_empl = us_states_details_naics[
        (us_states_details_naics.ENTRSIZE == 1)
        & (us_states_details_naics.NAICSDSCR == "Total")
    ]
    state_empl_reset = state_empl.reset_index(drop=True)

    pop_data = pd.read_csv(data_loc / "pop_2020_state.csv")

    state_gdp_empl_pop = state_empl_reset.merge(
        state_2022[["2022", "STATE"]], on="STATE"
    ).merge(pop_data, on="STATE")
    state_gdp_empl_pop.rename(
        columns={"2022": "2022REALGDP", "POPULATION": "STPOP", "stab": "STABBR"},
        inplace=True,
    )
    state_gdp_empl_pop.drop("STATE", axis=1, inplace=True)

    del gdp_state, state_empl, state_empl_reset, state_2022
    gc.collect()

    logger.info(f"Created state data with {len(state_gdp_empl_pop):,} records")
    return state_gdp_empl_pop


def create_zcta_within_rto(data_loc: Path):
    """Spatially joins ZCTA polygons with NERC transmission regions."""
    logger.info("Creating ZCTA within RTO spatial mapping")
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

    states = gpd.read_file(data_loc / "tl_2022_us_state.zip")
    states.to_crs(epsg=4326, inplace=True)

    non_cont_fips_codes = ["02", "15", "72", "66", "60", "69", "78", "78"]
    states = states[~states.GEOID.isin(non_cont_fips_codes)]

    nerc_gdf.rename(columns={"id": "REGION_ID"}, inplace=True)
    nerc_gdf.loc[nerc_gdf["REGION_ID"] == 23, "REGIONS"] = "ERCOT"

    utm_epsg_code = 32633
    nerc_gdf_projected = nerc_gdf.to_crs(epsg=utm_epsg_code)
    states_projected = states.to_crs(epsg=utm_epsg_code)

    buffer_distance = 10
    nerc_gdf_buffered = nerc_gdf_projected.buffer(buffer_distance)

    states_boundary = states_projected.geometry.unary_union
    states_boundary_gdf = gpd.GeoDataFrame(
        geometry=[states_boundary], crs=utm_epsg_code
    )

    aligned_nerc = gpd.overlay(
        gpd.GeoDataFrame(geometry=nerc_gdf_buffered, crs=utm_epsg_code),
        states_boundary_gdf,
        how="intersection",
    )

    aligned_nerc = aligned_nerc.to_crs(nerc_gdf.crs)
    nerc_gdf.geometry = aligned_nerc.geometry

    zcta_gdf = gpd.read_file(data_loc / "tl_2020_us_zcta520.zip")
    zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"}, inplace=True)
    zcta_gdf.ZCTA = zcta_gdf.ZCTA.astype(int)

    zcta_gdf["representative_point"] = zcta_gdf.geometry.representative_point()

    nerc_gdf = nerc_gdf.to_crs(zcta_gdf.crs)
    zcta_within_rto = gpd.sjoin(
        zcta_gdf.set_geometry("representative_point"),
        nerc_gdf,
        how="inner",
        predicate="within",
    )

    zcta_within_rto.columns = [col.upper() for col in zcta_within_rto.columns]
    logger.info(f"Created ZCTA-RTO mapping with {len(zcta_within_rto):,} records")
    return zcta_within_rto


def create_naics_est_gdp2022_zcta_csv(
    data_loc: Path, df_naics_zcta: pd.DataFrame, zcta_within_rto: pd.DataFrame
):
    """Calculates ZCTA-level GDP by allocating state GDP based on establishment counts."""
    logger.info("Creating NAICS EST GDP dataset with GDP calculations")
    df_naics_regions = df_naics_zcta.merge(
        zcta_within_rto[["REGIONS", "ZCTA"]], on="ZCTA"
    )

    naics_wo_govt = df_naics_regions[df_naics_regions.NAICS != 99]

    GDP_DIR = data_loc / "SAGDP"
    state_gdp_files = [f for f in GDP_DIR.iterdir() if f.is_file()]
    STATES = df_naics_regions.STABBR.unique()
    pattern = r"^\d{2}$|^\d{2}-\d{2}$"

    state_data = {}

    for file_path in state_gdp_files:
        file_name = file_path.name
        if "_" in file_name:
            parts = file_name.split("_")
            if len(parts) >= 2:
                file_type, state_abbr = parts[0], parts[1]
                if state_abbr in STATES and file_type == "SAGDP9N":
                    df_gdp = pd.read_csv(file_path)
                    df_gdp["STABBR"] = state_abbr

                    filtered_gdp = df_gdp[
                        df_gdp["IndustryClassification"].str.match(pattern, na=False)
                    ]
                    if not filtered_gdp.empty:
                        filtered_gdp["IndustryClassification"] = (
                            filtered_gdp["IndustryClassification"]
                            .str.split("-")
                            .str[0]
                            .astype(int)
                        )
                        filtered_gdp.rename(
                            columns={
                                "IndustryClassification": "NAICS",
                                "2022": "CGDP2022",
                            },
                            inplace=True,
                        )

                        summary_row = {
                            "NAICS": "--",
                            "Description": "--",
                            "CGDP2022": filtered_gdp["CGDP2022"].astype(float).sum(),
                            "STABBR": state_abbr,
                        }

                        filtered_gdp = pd.concat(
                            [pd.DataFrame([summary_row]), filtered_gdp],
                            ignore_index=True,
                        )
                        state_data[state_abbr] = filtered_gdp[
                            ["NAICS", "Description", "CGDP2022", "STABBR"]
                        ]

    naics_wo_govt["EST"] = naics_wo_govt["EST"].apply(pd.to_numeric, errors="ignore")

    total_est = naics_wo_govt.groupby("STABBR")["EST"].sum()
    state_firms = total_est.to_dict()

    grouped_data = naics_wo_govt.groupby(["STABBR", "NAICS"])["EST"].sum()
    nested_dict_emp = {
        state: grouped_data[state].to_dict() for state in grouped_data.index.levels[0]
    }

    map_state_gdp = {
        state: dict(zip(state_data[state].NAICS, state_data[state].CGDP2022))
        for state in state_data.keys()
    }

    def calc_gdp(row):
        try:
            state, naics, est = row["STABBR"], row["NAICS"], row["EST"]
            total_est = nested_dict_emp[state][naics]

            if naics == "UNCLFD":
                naics = 92

            naics_gdp = map_state_gdp[state][naics]
            gdp_zcta_naic = round((float(est) / float(total_est)) * float(naics_gdp), 2)
        except (TypeError, KeyError) as e:
            return np.nan
        return gdp_zcta_naic

    naics_wo_govt["GDP2022"] = naics_wo_govt.apply(calc_gdp, axis=1)

    naics_wo_govt.to_csv(processed_econ_dir / "NAICS_EST_GDP2022_ZCTA.csv", index=False)
    logger.info(f"Created NAICS EST GDP data with {len(naics_wo_govt):,} records")
    return naics_wo_govt


def aggregate_economic_columns(df, economic_cols):
    """Aggregates detailed NAICS codes into broader industry sectors."""
    missing_cols = [col for col in economic_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    result_df = df.copy()
    logger.info(
        f"Aggregating economic columns into broader sectors.. {result_df.columns.tolist()}"
    )

    industry_groups = {
        "AGR": ["11"],
        "MINING": ["21"],
        "UTIL_CONST": ["22", "23"],
        "MANUF": ["31", "32", "33"],
        "TRADE_TRANSP": ["42", "44", "48"],
        "INFO": ["51"],
        "FIRE": ["52", "53"],
        "PROF_OTHER": ["54", "55", "56", "81"],
        "EDUC_ENT": ["61", "62", "71", "72"],
        "G": ["92", "UNCLFD"],
    }

    for prefix in ["GDP_", "EST_"]:
        for new_sector, naics_codes in industry_groups.items():
            new_col = f"{prefix}{new_sector}"
            result_df[new_col] = 0

            matching_cols = [
                f"{prefix}{code}"
                for code in naics_codes
                if f"{prefix}{code}" in economic_cols
            ]
            if matching_cols:
                result_df[new_col] = df[matching_cols].sum(axis=1)

    aggregated_cols = [
        f"{prefix}{sector}"
        for prefix in ["GDP_", "EST_"]
        for sector in industry_groups.keys()
    ]
    if "POP20" in df.columns:
        aggregated_cols.append("POP20")

    non_economic_cols = [
        col
        for col in df.columns
        if col not in economic_cols or col in ["geometry", "ZCTA", "GEOID20"]
    ]

    columns_to_keep = list(set(non_economic_cols + aggregated_cols))
    final_df = result_df[columns_to_keep].copy()

    return final_df, aggregated_cols


def load_socioeconomic_data(data_naics_zcta, population_df):
    """Transforms long-format NAICS data into wide-format GeoDataFrame."""
    logger.info("Processing socioeconomic data")

    regions = data_naics_zcta[["ZCTA", "REGIONS", "STABBR"]].drop_duplicates()
    regions_pop_df = regions.merge(population_df, on=["ZCTA", "STABBR"])

    zcta_gdf = gpd.read_file(raw_data_folder / "tl_2020_us_zcta520.zip")
    zcta_gdf = zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"}).astype({"ZCTA": "Int64"})
    zcta_business_gdf = zcta_gdf.merge(data_naics_zcta, on="ZCTA", how="inner")
    zcta_business_gdf["points"] = zcta_business_gdf.geometry.representative_point()
    zcta_business_gdf.set_geometry("points", inplace=True)
    zcta_business_gdf = zcta_business_gdf.to_crs(epsg=4326)
    zcta_business_gdf["GDP2022"] = zcta_business_gdf["GDP2022"] / 365.0

    keep_cols = ["ZCTA", "GEOID20", "NAICS", "EST", "GDP2022", "geometry"]
    gdf_min = zcta_business_gdf[keep_cols].copy()
    gdf_min["ZCTA"] = gdf_min["ZCTA"].astype(np.uint32)
    gdf_min["GEOID20"] = gdf_min["GEOID20"].astype(np.uint32)
    gdf_min["EST"] = gdf_min["EST"].astype(np.uint16)
    gdf_min["GDP2022"] = gdf_min["GDP2022"].astype(np.float32)

    gdp_wide = gdf_min.pivot_table(
        index=["ZCTA", "GEOID20", "geometry"],
        columns="NAICS",
        values="GDP2022",
        fill_value=0,
    ).reset_index()
    gdp_wide.columns = [
        f"GDP_{col}" if col not in ["ZCTA", "GEOID20", "geometry"] else col
        for col in gdp_wide.columns
    ]

    est_wide = gdf_min.pivot_table(
        index=["ZCTA", "GEOID20", "geometry"],
        columns="NAICS",
        values="EST",
        fill_value=0,
    ).reset_index()
    est_wide.columns = [
        f"EST_{col}" if col not in ["ZCTA", "GEOID20", "geometry"] else col
        for col in est_wide.columns
    ]

    zcta_wide_gdf = pd.merge(
        gdp_wide, est_wide.drop("geometry", axis=1), on=["ZCTA", "GEOID20"]
    )
    zcta_wide_gdf = gpd.GeoDataFrame(
        zcta_wide_gdf, geometry="geometry", crs="EPSG:4326"
    )
    zcta_wide_gdf = pd.merge(
        zcta_wide_gdf, regions_pop_df[["ZCTA", "POP20"]], on="ZCTA", how="left"
    )
    zcta_wide_gdf["POP20"] = zcta_wide_gdf["POP20"].fillna(0).astype(np.uint32)
    zcta_wide_gdf = zcta_wide_gdf.drop_duplicates(subset="ZCTA", keep="first")

    economic_cols = [
        col
        for col in zcta_wide_gdf.columns
        if col.startswith("GDP_") or col.startswith("EST_")
    ]
    zcta_wide_gdf, new_economic_cols = aggregate_economic_columns(
        zcta_wide_gdf, economic_cols
    )
    zcta_wide_gdf[new_economic_cols] = zcta_wide_gdf[new_economic_cols].astype(
        np.float32
    )

    other_cols = [c for c in zcta_business_gdf.columns if c not in keep_cols]
    df_other = zcta_business_gdf[["ZCTA"] + other_cols].copy()

    del zcta_business_gdf, gdf_min, gdp_wide, est_wide
    gc.collect()

    states_gdf = gpd.read_file(raw_data_folder / "tl_2022_us_state.zip").to_crs(
        epsg=4326
    )

    logger.info("Processed socioeconomic data")
    return regions_pop_df, zcta_wide_gdf, states_gdf, df_other


def create_voronoi_polygons(
    sub_coordinates: Dict[str, Tuple[float, float]], states_gdf
) -> gpd.GeoDataFrame:
    """Creates Voronoi polygons for substation coordinates and clips to continental US."""
    voronoi_file = processed_voronoi_dir / "voronoi_polygons_clipped.geojson"

    if voronoi_file.exists():
        logger.info(
            f"Voronoi polygons already exist at {voronoi_file}. Loading existing file."
        )
        voronoi_gdf = gpd.read_file(voronoi_file)
        return voronoi_gdf

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
        for sub_id, polygon in zip(sub_ids, polygons)
        if polygon is not None
    ]

    voronoi_gdf = gpd.GeoDataFrame(valid_data, crs="EPSG:4326")

    continental_us_states = states_gdf[~states_gdf["STATEFP"].isin(["02", "15", "72"])]
    continental_us_boundary = continental_us_states.unary_union

    voronoi_gdf = voronoi_gdf[voronoi_gdf.is_valid & ~voronoi_gdf.is_empty]
    voronoi_gdf = gpd.overlay(
        voronoi_gdf,
        gpd.GeoDataFrame(geometry=[continental_us_boundary], crs="EPSG:4326"),
        how="intersection",
    )

    os.makedirs(processed_voronoi_dir, exist_ok=True)
    voronoi_gdf.to_file(voronoi_file, driver="GeoJSON")
    logger.info("Saved Voronoi polygons")

    return voronoi_gdf


if __name__ == "__main__":
    logger.info("Starting consolidated economic data pipeline")

    logger.info("=" * 50)
    logger.info("PHASE 1: RAW DATA PROCESSING")
    logger.info("=" * 50)

    os.makedirs(processed_econ_dir, exist_ok=True)

    zcta_pop_20 = create_zcta_population_csv(raw_data_folder)
    state_gdp_empl_pop = create_state_gdp_employment_data(raw_data_folder)
    df_naics_zcta = create_naics_establishments_data(raw_data_folder)
    zcta_within_rto = create_zcta_within_rto(raw_data_folder)
    naics_est_gdp = create_naics_est_gdp2022_zcta_csv(
        raw_data_folder, df_naics_zcta, zcta_within_rto
    )

    logger.info("=" * 50)
    logger.info("PHASE 2: ANALYSIS DATA PREPARATION")
    logger.info("=" * 50)

    regions_pop_df, zcta_wide_gdf, states_gdf, df_other = load_socioeconomic_data(
        naics_est_gdp, zcta_pop_20
    )

    logger.info("=" * 50)
    logger.info("PHASE 3: SPATIAL DATA PROCESSING")
    logger.info("=" * 50)

    df_substation = pd.read_csv(DATA_DIR / "admittance_matrix" / "substation_info.csv")
    ehv_coordinates = dict(
        zip(
            df_substation["name"],
            zip(df_substation["longitude"], df_substation["latitude"]),
        )
    )

    voronoi_gdf = create_voronoi_polygons(ehv_coordinates, states_gdf)

    logger.info("=" * 50)
    logger.info("PHASE 4: SAVING PROCESSED DATA")
    logger.info("=" * 50)

    processed_file = processed_econ_dir / "socioeconomic_data.pkl"
    result = (
        naics_est_gdp,
        zcta_pop_20,
        regions_pop_df,
        zcta_wide_gdf,
        states_gdf,
        df_other,
    )
    with open(processed_file, "wb") as f:
        pickle.dump(result, f)

    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETED SUCCESSFULLY")
    logger.info(f"Saved processed data to: {processed_file}")
