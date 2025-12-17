"""
Raster-based economic data interpolation using dasymetric mapping to redistribute ZCTA-level economic data to substation service areas.
Authors: Dennies and Oughton
"""

import os
import pickle
import gc
import threading

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box
from tobler.dasymetric import masked_area_interpolate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from econ.preprocess.p_econ_data import (
    load_socioeconomic_data,
    create_voronoi_polygons,
    create_zcta_population_csv,
    create_state_gdp_employment_data,
    create_naics_establishments_data,
    create_zcta_within_rto,
    create_naics_est_gdp2022_zcta_csv,
)
from configs import setup_logger, get_data_dir, DATA_DIR

os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "0")

DATA_LOC = get_data_dir(econ=True)
raw_data_folder = DATA_LOC / "raw_econ_data"
processed_econ_dir = DATA_LOC / "processed_econ"
land_mask_dir = DATA_LOC / "land_mask"

USE_COARSE = True
COARSE_SINGLE = land_mask_dir / "coarse" / "nlcd_coarse_mask.tif"
COARSE_TILES = land_mask_dir / "tiles"
COARSE_IS_BINARY = True

out_dir = land_mask_dir / "coarse_interpolation_tiles"
tile_rows, tile_cols = 8, 8
max_workers = 4
raster_open_limit = 3

logger = setup_logger(log_file="logs/raster_interpolation.log")

_z_sindex = None
_v_sindex = None

_rio_gate = threading.BoundedSemaphore(raster_open_limit)


def _gpkg_looks_complete(fp):
    try:
        if not fp.exists():
            return False
        if fp.stat().st_size < 32768:
            return False
        with open(fp, "rb") as f:
            return b"SQLite format 3\000" in f.read(16)
    except Exception:
        return False


def _choose_raster_for_grid():
    if USE_COARSE and COARSE_SINGLE.exists():
        return COARSE_SINGLE
    return land_mask_dir / "Annual_NLCD_LndCov_2023_CU_C1V0.tif"


def divide_raster_into_bbox(n_rows, n_cols):
    src_path = _choose_raster_for_grid()
    logger.info(f"Dividing raster into {n_rows}x{n_cols} grid: {src_path}")
    bboxes = []
    with rasterio.open(src_path) as src:
        H, W = src.height, src.width
        h_step, w_step = H // n_rows, W // n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                r0, c0 = i * h_step, j * w_step
                r1 = H if i == n_rows - 1 else (i + 1) * h_step
                c1 = W if j == n_cols - 1 else (j + 1) * w_step
                win = Window(c0, r0, c1 - c0, r1 - r0)
                trs = src.window_transform(win)
                bboxes.append(
                    {
                        "window": win,
                        "transform": trs,
                        "height": int(win.height),
                        "width": int(win.width),
                    }
                )
    logger.info(f"Created {len(bboxes)} bounding boxes")
    return src_path, bboxes


def _mask_from_coarse_single(coarse_path, bbox):
    with _rio_gate:
        with rasterio.open(coarse_path) as src:
            arr = src.read(1, window=bbox["window"])
            trs = bbox["transform"]
            crs = src.crs
            nodata = src.nodata if src.nodata is not None else 0

    if arr.size == 0:
        return None, None, None

    if COARSE_IS_BINARY:
        mask = (arr != 0).astype("uint8")
    else:
        mask = np.isin(arr, np.array([21, 22, 23, 24], dtype=arr.dtype)).astype("uint8")
        if mask.sum() == 0:
            mask = (arr != nodata).astype("uint8")

    return mask, trs, crs


def interpolate_chunk(bbox, idx, grid_raster_path):
    try:
        if USE_COARSE:
            coarse_tile_fp = COARSE_TILES / f"tile_{idx}.tif"
            if coarse_tile_fp.exists():
                vsipath = str(coarse_tile_fp)
                with rasterio.open(vsipath) as src:
                    trs = src.transform
                    crs = src.crs
                    arr = src.read(1)
                if arr.size == 0 or np.all(arr == 0):
                    logger.debug(f"Chunk {idx} coarse tile has no valid pixels")
                    return None
            else:
                mask, trs, crs = _mask_from_coarse_single(grid_raster_path, bbox)
                if mask is None or mask.sum() == 0:
                    logger.debug(f"Chunk {idx} has no valid pixels (coarse single)")
                    return None
                os.makedirs(out_dir, exist_ok=True)
                mask_fp = out_dir / f"mask_tile_{idx}.tif"
                if not mask_fp.exists():
                    with rasterio.open(
                        mask_fp,
                        "w",
                        driver="GTiff",
                        height=mask.shape[0],
                        width=mask.shape[1],
                        count=1,
                        dtype="uint8",
                        crs=crs,
                        transform=trs,
                        nodata=0,
                        tiled=True,
                        compress="LZW",
                    ) as dst:
                        dst.write(mask, 1)
                vsipath = str(mask_fp)
                del mask
        else:
            with _rio_gate:
                with rasterio.open(grid_raster_path) as src:
                    arr = src.read(1, window=bbox["window"])
                    trs = bbox["transform"]
                    crs = src.crs
                    nodata = src.nodata if src.nodata is not None else 0
            if arr.size == 0:
                logger.debug(f"Chunk {idx} empty raster window")
                return None
            mask = np.isin(arr, np.array([21, 22, 23, 24], dtype=arr.dtype)).astype(
                "uint8"
            )
            if mask.sum() == 0:
                mask = (arr != nodata).astype("uint8")
                if mask.sum() == 0:
                    logger.debug(f"Chunk {idx} has no valid pixels")
                    return None
            os.makedirs(out_dir, exist_ok=True)
            mask_fp = out_dir / f"mask_tile_{idx}.tif"
            if not mask_fp.exists():
                with rasterio.open(
                    mask_fp,
                    "w",
                    driver="GTiff",
                    height=mask.shape[0],
                    width=mask.shape[1],
                    count=1,
                    dtype="uint8",
                    crs=crs,
                    transform=trs,
                    nodata=0,
                    tiled=True,
                    compress="LZW",
                ) as dst:
                    dst.write(mask, 1)
            vsipath = str(mask_fp)
            del mask, arr

        x0, y1 = trs[2], trs[5]
        x1 = x0 + bbox["width"] * trs[0]
        y0 = y1 + bbox["height"] * trs[4]
        zone = box(x0, y0, x1, y1)

        z_idx = list(_z_sindex.query(zone)) if _z_sindex is not None else []
        v_idx = list(_v_sindex.query(zone)) if _v_sindex is not None else []

        zs = (zcta_aea.iloc[z_idx] if z_idx else zcta_aea).copy()
        vs = (voronoi_aea.iloc[v_idx] if v_idx else voronoi_aea).copy()

        zs = zs[zs.intersects(zone)]
        vs = vs[vs.intersects(zone)]
        if zs.empty or vs.empty:
            logger.debug(f"Chunk {idx} empty after clip (zs={len(zs)}, vs={len(vs)})")
            return None

        id_col = (
            "sub_id"
            if "sub_id" in vs.columns
            else ("name" if "name" in vs.columns else None)
        )
        if id_col is None:
            raise RuntimeError("Voronoi polygons missing both 'sub_id' and 'name'.")

        zs = zs.set_index("ZCTA", drop=False)
        vs = vs.set_index(id_col, drop=False)

        res = masked_area_interpolate(
            raster=vsipath,
            source_df=zs,
            target_df=vs,
            extensive_variables=economic_cols,
            intensive_variables=[],
            pixel_values=[1],
            allocate_total=True,
        )
        del zs, vs
        return res

    except Exception as e:
        logger.error(f"Error processing chunk {idx}: {str(e)}")
        return None


def load_processed_data():
    processed_file = land_mask_dir / "interpolation_data.pkl"
    logger.info("Processing interpolation data")

    processed_econ_file = processed_econ_dir / "socioeconomic_data.pkl"
    if processed_econ_file.exists():
        logger.info("Loading from consolidated economic data pipeline")
        with open(processed_econ_file, "rb") as f:
            (
                naics_est_gdp,
                zcta_pop_20,
                regions_pop_df,
                zcta_business_gdf,
                states_gdf,
                df_other,
            ) = pickle.load(f)
    else:
        logger.warning(f"Processed economic data not found at {processed_econ_file}")
        logger.info("Running consolidated economic data pipeline...")
        os.makedirs(processed_econ_dir, exist_ok=True)
        zcta_pop_20 = create_zcta_population_csv(raw_data_folder)
        _ = create_state_gdp_employment_data(raw_data_folder)
        df_naics_zcta = create_naics_establishments_data(raw_data_folder)
        zcta_within_rto = create_zcta_within_rto(raw_data_folder)
        naics_est_gdp = create_naics_est_gdp2022_zcta_csv(
            raw_data_folder, df_naics_zcta, zcta_within_rto
        )
        regions_pop_df, zcta_business_gdf, states_gdf, df_other = (
            load_socioeconomic_data(naics_est_gdp, zcta_pop_20)
        )
        with open(processed_econ_file, "wb") as f:
            pickle.dump(
                (
                    naics_est_gdp,
                    zcta_pop_20,
                    regions_pop_df,
                    zcta_business_gdf,
                    states_gdf,
                    df_other,
                ),
                f,
            )
        logger.info(f"Saved processed data to: {processed_econ_file}")

    grid_raster_path = _choose_raster_for_grid()
    with rasterio.open(grid_raster_path) as _src:
        raster_crs = _src.crs
        logger.info(f"Raster for grid: {grid_raster_path}")
        logger.info(f"Raster bounds: {_src.bounds}")

    df_substation = pd.read_csv(DATA_DIR / "admittance_matrix" / "substation_info.csv")
    if "name" not in df_substation.columns:
        raise RuntimeError("substation_info.csv missing 'name' column.")
    ehv_coordinates = dict(
        zip(
            df_substation["name"],
            zip(df_substation["longitude"], df_substation["latitude"]),
        )
    )

    voronoi_gdf = create_voronoi_polygons(ehv_coordinates, states_gdf)

    zcta_aea = zcta_business_gdf.to_crs(raster_crs)
    voronoi_aea = voronoi_gdf.to_crs(raster_crs)

    if "sub_id" in voronoi_aea.columns:
        voronoi_aea["sub_id"] = voronoi_aea["sub_id"].astype(str)
    elif "name" in voronoi_aea.columns:
        voronoi_aea["name"] = voronoi_aea["name"].astype(str)
    else:
        raise RuntimeError("Voronoi polygons missing both 'sub_id' and 'name'.")

    logger.info(f"ZCTA bounds: {zcta_aea.total_bounds}")
    logger.info(f"Voronoi bounds: {voronoi_aea.total_bounds}")

    group_keys = (
        "AGR",
        "MINING",
        "UTIL_CONST",
        "MANUF",
        "TRADE_TRANSP",
        "INFO",
        "FIRE",
        "PROF_OTHER",
        "EDUC_ENT",
        "G",
    )
    economic_cols = [f"GDP_{k}" for k in group_keys] + [f"EST_{k}" for k in group_keys]
    economic_cols = [c for c in economic_cols if c in zcta_aea.columns]
    if "POP20" in zcta_aea.columns:
        economic_cols.append("POP20")
    logger.info(f"econ cols: {len(economic_cols)} -> {economic_cols[:8]}...")

    global _z_sindex, _v_sindex
    _z_sindex = zcta_aea.sindex
    _v_sindex = voronoi_aea.sindex

    with open(processed_file, "wb") as f:
        pickle.dump((zcta_aea, voronoi_aea, economic_cols, str(grid_raster_path)), f)

    logger.info("Saved processed interpolation data")
    return zcta_aea, voronoi_aea, economic_cols, str(grid_raster_path)


def main(start_tile=0, end_tile=None, overwrite=False):
    logger.info(f"Starting raster interpolation from tile {start_tile}")

    global zcta_aea, voronoi_aea, economic_cols
    zcta_aea, voronoi_aea, economic_cols, grid_raster_path = load_processed_data()

    os.makedirs(out_dir, exist_ok=True)

    grid_path, bboxes = _ = (
        _choose_raster_for_grid(),
        divide_raster_into_bbox(tile_rows, tile_cols)[1],
    )
    if end_tile is None or end_tile > len(bboxes):
        end_tile = len(bboxes)

    logger.info(
        f"Processing tiles {start_tile} through {end_tile} (out of {len(bboxes)} total)"
    )

    if not economic_cols:
        logger.error("No economic columns found. Exiting.")
        return

    tile_range = list(range(start_tile, end_tile + 1))

    def _run_tile(idx):
        fp = out_dir / f"tile_{idx}.gpkg"
        if not overwrite and _gpkg_looks_complete(fp):
            logger.info(f"Tile {idx} already exists and looks valid; skipping")
            return str(fp)

        bb = bboxes[idx - 1]
        out = interpolate_chunk(bb, idx, grid_raster_path)
        if out is None:
            return None

        tmp_fp = out_dir / f"tile_{idx}_tmp.gpkg"
        out.to_file(tmp_fp, driver="GPKG", layer=f"tile_{idx}")
        os.replace(tmp_fp, fp)

        del out
        gc.collect()
        logger.info(f"Saved tile {idx} -> {fp}")
        return str(fp)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_tile, i) for i in tile_range]
        for _ in tqdm(as_completed(futs), total=len(futs), unit="tile"):
            pass

    logger.info("Raster interpolation complete")


if __name__ == "__main__":
    main(overwrite=False)
