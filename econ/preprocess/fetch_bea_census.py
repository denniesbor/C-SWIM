"""
BEA and Census data fetcher for the county-anchored economic pipeline.
Author: Dennies Bor & Edward Oughton

Pulls the upstream datasets required to allocate county-level GDP to ZCTAs
and to build the 10-sector A matrix:
    - BEA CAGDP2 county GDP by industry (Regional API)
    - BEA sector-level Use table (InputOutput API)
    - Census ZIP Code Business Patterns industry detail (flat file)
    - TIGER county polygons (flat file)

All fetchers are idempotent: cached files are reused unless overwrite=True.
BEA_API_KEY must be set in the environment (a .env file is supported).
"""

import os
import time
import zipfile

import requests
import pandas as pd
from dotenv import load_dotenv

from configs import setup_logger, get_data_dir

logger = setup_logger("BEA/Census Fetcher")

DATA_LOC = get_data_dir(econ=True)
raw_data_folder = DATA_LOC / "raw_econ_data"

BEA_BASE_URL = "https://apps.bea.gov/api/data/"

# Default vintage. Bump annually; BEA CAGDP2 and Census CBP lag by ~18 months.
DEFAULT_YEAR = 2023

# 10-sector aggregation aligned with the c-swim production technology scheme.
# Each label maps to CAGDP2 LineCodes that get summed. Composites
# (TRADE_TRANSP, PROF_OTHER, EDUC_ENT) combine multiple lines.
CAGDP2_SECTOR_LINECODES = {
    "AGR": [3],  # Agriculture, forestry, fishing (NAICS 11)
    "MINING": [6],  # Mining (NAICS 21)
    "UTIL_CONST": [10, 11],  # Utilities + Construction (NAICS 22, 23)
    "MANUF": [12],  # Manufacturing (NAICS 31-33)
    "TRADE_TRANSP": [34, 35, 36],  # Wholesale + Retail + Transport
    "INFO": [45],  # Information (NAICS 51)
    "FIRE": [50],  # Finance, Insurance, Real Estate (52, 53)
    "PROF_OTHER": [59, 82],  # Professional/business + Other services
    "EDUC_ENT": [68, 75],  # Education + Health + Arts + Accommodation
    "G": [83],  # Government (NAICS 92)
}

# Sector-level Use table (~15 industries x 22 columns including F010-F100,
# T001/T019 totals, and T018 gross output). p_technology.py aggregates this
# into the 10-sector A matrix and gross output vector.
USE_TABLE_ID = 258


def _bea_api_key():
    """Load the BEA API key from .env or the process environment."""
    load_dotenv()
    key = os.environ.get("BEA_API_KEY")
    if not key:
        raise RuntimeError(
            "BEA_API_KEY not set. Register a free key at "
            "https://apps.bea.gov/api/signup/ and add it to .env"
        )
    return key


def _bea_get(params, timeout=60):
    """Single GET against the BEA API with error unwrapping."""
    params = {"UserID": _bea_api_key(), "ResultFormat": "JSON", **params}
    resp = requests.get(BEA_BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    results = resp.json().get("BEAAPI", {}).get("Results", {})
    if isinstance(results, dict) and "Error" in results:
        raise RuntimeError(f"BEA API error: {results['Error']}")
    return results


def _download_file(url, out_fp, chunk=1024 * 1024, timeout=600):
    """Stream a large file to disk."""
    logger.info(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_fp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
    logger.info(f"Saved -> {out_fp}")


def fetch_cagdp2_county(year=DEFAULT_YEAR, overwrite=False):
    """Pull CAGDP2 county GDP by industry for the LineCodes feeding the
    10-sector aggregation. Returns wide DataFrame indexed by (GeoFips,
    GeoName), one column per sector, values in $ millions.
    """
    out_fp = raw_data_folder / f"cagdp2_{year}.parquet"
    if out_fp.exists() and not overwrite:
        logger.info(f"CAGDP2 cache hit: {out_fp}")
        return pd.read_parquet(out_fp)

    logger.info(f"Fetching CAGDP2 {year} from BEA Regional API")
    line_codes = sorted(
        {lc for codes in CAGDP2_SECTOR_LINECODES.values() for lc in codes}
    )

    frames = []
    for lc in line_codes:
        res = _bea_get(
            {
                "Method": "GetData",
                "DatasetName": "Regional",
                "TableName": "CAGDP2",
                "LineCode": str(lc),
                "GeoFips": "COUNTY",
                "Year": str(year),
            }
        )
        rows = res.get("Data", [])
        if not rows:
            logger.warning(f"CAGDP2 LineCode {lc} returned no rows")
            continue
        df = pd.DataFrame(rows)
        df["LineCode"] = lc
        frames.append(df)
        time.sleep(0.25)  # be polite to the BEA API

    raw = pd.concat(frames, ignore_index=True)
    raw["DataValue"] = pd.to_numeric(
        raw["DataValue"].astype(str).str.replace(",", ""), errors="coerce"
    )

    # Collapse LineCodes into sector labels so composites sum to a single column.
    lc_to_sector = {
        lc: sector for sector, codes in CAGDP2_SECTOR_LINECODES.items() for lc in codes
    }
    raw["SECTOR"] = raw["LineCode"].map(lc_to_sector)

    wide = (
        raw.groupby(["GeoFips", "GeoName", "SECTOR"], as_index=False)["DataValue"]
        .sum()
        .pivot(index=["GeoFips", "GeoName"], columns="SECTOR", values="DataValue")
        .reindex(columns=list(CAGDP2_SECTOR_LINECODES.keys()))
        .fillna(0.0)
    )

    # CAGDP2 is reported in $ thousands; rescale to $ millions.
    wide = wide / 1_000.0

    raw_data_folder.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_fp)
    logger.info(f"Saved CAGDP2 {year}: {len(wide):,} counties -> {out_fp}")
    return wide


def fetch_zbp_detail(year=DEFAULT_YEAR, overwrite=False):
    """Download and extract the Census ZBP industry detail file. Returns the
    path to the extracted txt.
    """
    yy = f"{year % 100:02d}"
    txt_fp = raw_data_folder / f"zbp{yy}detail.txt"
    if txt_fp.exists() and not overwrite:
        logger.info(f"ZBP detail cache hit: {txt_fp}")
        return txt_fp

    url = (
        f"https://www2.census.gov/programs-surveys/cbp/datasets/"
        f"{year}/zbp{yy}detail.zip"
    )
    zip_fp = raw_data_folder / f"zbp{yy}detail.zip"
    raw_data_folder.mkdir(parents=True, exist_ok=True)
    _download_file(url, zip_fp)

    with zipfile.ZipFile(zip_fp) as z:
        z.extractall(raw_data_folder)
    logger.info(f"Extracted ZBP detail -> {txt_fp}")
    return txt_fp


def fetch_tiger_county(year=DEFAULT_YEAR, overwrite=False):
    """Download the TIGER county polygons (kept zipped; geopandas reads in
    place). Returns the path to the zip.
    """
    zip_fp = raw_data_folder / f"tl_{year}_us_county.zip"
    if zip_fp.exists() and not overwrite:
        logger.info(f"TIGER county cache hit: {zip_fp}")
        return zip_fp

    url = (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/"
        f"tl_{year}_us_county.zip"
    )
    raw_data_folder.mkdir(parents=True, exist_ok=True)
    _download_file(url, zip_fp)
    return zip_fp


def fetch_bea_use_table(year=DEFAULT_YEAR, overwrite=False):
    """Fetch the BEA sector-level Use table and write use_tables.csv in the
    layout expected by p_technology.preprocess_use_table.

    The API returns one row per (RowCode, ColCode). We pivot to wide, fill
    suppressed cells with zero, and order rows/cols to match the historical
    hand-downloaded CSV so diffs across years stay clean.
    """
    out_dir = DATA_LOC / "supply_use_tables"
    out_fp = out_dir / "use_tables.csv"
    if out_fp.exists() and not overwrite:
        logger.info(f"Use table cache hit: {out_fp}")
        return out_fp

    logger.info(f"Fetching BEA Use table {year} (TableID {USE_TABLE_ID})")
    res = _bea_get(
        {
            "Method": "GetData",
            "DatasetName": "InputOutput",
            "TableID": str(USE_TABLE_ID),
            "Year": str(year),
        }
    )

    # InputOutput responses return a list of blocks, one per requested table.
    block = res if isinstance(res, dict) else res[0]
    rows = block.get("Data", [])
    if not rows:
        raise RuntimeError(f"BEA returned no data for Use table {year}")

    df = pd.DataFrame(rows)
    df["DataValue"] = pd.to_numeric(
        df["DataValue"].astype(str).str.replace(",", ""), errors="coerce"
    )

    wide = df.pivot_table(
        index="RowCode", columns="ColCode", values="DataValue", aggfunc="first"
    ).fillna(0.0)
    wide.index.name = "Commodities/Industries"

    # preprocess_use_table filters rows by regex so extras are tolerated, but
    # a stable order keeps the CSV diffable across years.
    industry_codes = [
        "11",
        "21",
        "22",
        "23",
        "31G",
        "42",
        "44RT",
        "48TW",
        "51",
        "FIRE",
        "PROF",
        "6",
        "7",
        "81",
        "G",
    ]
    aux_commodity_rows = ["Used", "Other"]
    va_rows = [
        "T005",
        "V001",
        "V003",
        "T00OTOP",
        "T00OSUB",
        "T00TOP",
        "T00SUB",
        "VABAS",
        "VAPRO",
        "T018",
    ]
    row_order = [
        r for r in industry_codes + aux_commodity_rows + va_rows if r in wide.index
    ]
    wide = wide.reindex(row_order)

    fd_cols = ["T001", "F010", "F020", "F030", "F040", "F100", "T019"]
    col_order = [c for c in industry_codes + fd_cols if c in wide.columns]
    wide = wide[col_order]

    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_fp)
    logger.info(f"Saved Use table {year}: {wide.shape} -> {out_fp}")
    return out_fp


def fetch_all(year=DEFAULT_YEAR, overwrite=False):
    """Run every fetcher in order. Safe to call on each pipeline start."""
    logger.info(f"Fetching upstream datasets for {year}")
    cagdp2 = fetch_cagdp2_county(year=year, overwrite=overwrite)
    zbp_fp = fetch_zbp_detail(year=year, overwrite=overwrite)
    county_fp = fetch_tiger_county(year=year, overwrite=overwrite)
    use_fp = fetch_bea_use_table(year=year, overwrite=overwrite)
    logger.info("Upstream fetch complete")
    return cagdp2, zbp_fp, county_fp, use_fp


if __name__ == "__main__":
    fetch_all()
