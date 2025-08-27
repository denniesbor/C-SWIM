import os
import pickle
import argparse
import logging
import gc
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from rasterio.windows import Window
from rasterio.io import MemoryFile
from shapely.geometry import box, Polygon
from scipy.spatial import Voronoi
from tobler.dasymetric import masked_area_interpolate
from osgeo import gdal
from multiprocessing import cpu_count
from tqdm import tqdm

from preprocess.p_econ_data import load_socioeconomic_data, create_voronoi_polygons
from configs import setup_logger, get_data_dir, DENNIES_DATA_LOC

DATA_LOC = get_data_dir()
data_zcta_folder = DATA_LOC / "econ_data"
land_mask_dir = DATA_LOC / "land_mask"
nlcd_aea_path = land_mask_dir / "Annual_NLCD_LndCov_2023_CU_C1V0.tif"
out_dir = land_mask_dir / "tiles"
tile_rows, tile_cols = 8, 8
jobs = 1

logger = setup_logger(log_file="logs/raster_interpolation.log")

def divide_raster_into_bbox(path, n_rows, n_cols):
    logger.info(f"Dividing raster into {n_rows}x{n_cols} grid")
    bboxes = []
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        h_step, w_step = H // n_rows, W // n_cols
        
        for i in range(n_rows):
            for j in range(n_cols):
                r0, c0 = i*h_step, j*w_step
                r1 = H if i==n_rows-1 else (i+1)*h_step
                c1 = W if j==n_cols-1 else (j+1)*w_step
                win = Window(c0, r0, c1-c0, r1-r0)
                trs = src.window_transform(win)
                bboxes.append({"window":win, "transform":trs,
                               "height":int(win.height), "width":int(win.width)})
    
    logger.info(f"Created {len(bboxes)} bounding boxes")
    return bboxes

def mask_to_vsimem(mask_arr, transform, crs, idx):
    mem = MemoryFile()
    with mem.open(
        driver="GTiff",
        height=mask_arr.shape[0], width=mask_arr.shape[1],
        count=1, dtype="uint8",
        crs=crs, transform=transform, nodata=0
    ) as dst:
        dst.write(mask_arr, 1)
    vsipath = f"/vsimem/built_mask_{idx}.tif"
    gdal.FileFromMemBuffer(vsipath, mem.read())
    return vsipath

def interpolate_chunk(bbox, idx, pbar=None):
    try:
        if pbar:
            pbar.set_description(f"Reading raster data for tile {idx}")
        
        with rasterio.open(nlcd_aea_path) as src:
            arr = src.read(1, window=bbox["window"])
            trs = bbox["transform"]
            crs = src.crs
        
        if arr.size == 0 or np.all(arr == 0):
            logger.warning(f"Chunk {idx} has no valid data, skipping")
            return None
        
        if pbar:
            pbar.set_description(f"Creating mask for tile {idx}")
            
        mask = np.isin(arr, [21,22,23,24]).astype('uint8')
        del arr
        
        vsipath = mask_to_vsimem(mask, trs, crs, idx)
        del mask
        
        if pbar:
            pbar.set_description(f"Filtering spatial data for tile {idx}")
        
        x0,y1 = trs[2], trs[5]
        x1 = x0 + bbox['width']*trs[0]
        y0 = y1 + bbox['height']*trs[4]
        zone = box(x0,y0,x1,y1)
        
        zs = zcta_5070[zcta_5070.intersects(zone)].copy()
        vs = voronoi_5070[voronoi_5070.intersects(zone)].copy()
        
        if zs.empty or vs.empty:
            logger.warning(f"Chunk {idx} has no data, skipping")
            gdal.Unlink(vsipath)
            return None
            
        zs = zs.set_index('ZCTA', drop=False)
        vs = vs.set_index('sub_id', drop=False)
        
        if pbar:
            pbar.set_description(f"Running interpolation for tile {idx}")
        
        res = masked_area_interpolate(
            raster=vsipath,
            source_df=zs,
            target_df=vs,
            extensive_variables=economic_cols,
            intensive_variables=[],
            pixel_values=[1],
            allocate_total=True,
        )
        
        gdal.Unlink(vsipath)
        del zs, vs
        return res
        
    except Exception as e:
        logger.error(f"Error processing chunk {idx}: {str(e)}")
        return None

def load_processed_data():
    processed_file = land_mask_dir / "interpolation_data.pkl"
    
    # if processed_file.exists():
    #     logger.info("Loading existing processed interpolation data")
    #     with open(processed_file, 'rb') as f:
    #         return pickle.load(f)
    
    logger.info("Processing interpolation data")
    
    data_zcta, population_df, regions_pop_df, zcta_business_gdf, states_gdf, df_other = load_socioeconomic_data(data_zcta_folder)

    df_substation = pd.read_csv(DENNIES_DATA_LOC / "admittance_matrix" / "substation_info.csv")
    ehv_coordinates = dict(zip(df_substation['name'],
                            zip(df_substation['longitude'], df_substation['latitude'])))
    
    voronoi_gdf = create_voronoi_polygons(ehv_coordinates, states_gdf)
    
    zcta_5070 = zcta_business_gdf.to_crs(epsg=5070)
    voronoi_5070 = voronoi_gdf.to_crs(epsg=5070)
    
    economic_cols = [c for c in zcta_5070.columns if c.startswith("GDP_") or c.startswith("EST_")]
    if "POP20" in zcta_5070.columns:
        economic_cols.append("POP20")
    
    result = (zcta_5070, voronoi_5070, economic_cols)
    
    os.makedirs(land_mask_dir, exist_ok=True)
    with open(processed_file, 'wb') as f:
        pickle.dump(result, f)
    
    logger.info("Saved processed interpolation data")
    return result

def main(start_tile=33, end_tile=64):
    logger.info(f"Starting raster interpolation from tile {start_tile}")
    
    global zcta_5070, voronoi_5070, economic_cols
    zcta_5070, voronoi_5070, economic_cols = load_processed_data()
    
    os.makedirs(out_dir, exist_ok=True)
    
    bboxes = divide_raster_into_bbox(nlcd_aea_path, tile_rows, tile_cols)
    
    if end_tile is None or end_tile > len(bboxes):
        end_tile = len(bboxes)
    
    logger.info(f"Processing tiles {start_tile} through {end_tile} (out of {len(bboxes)} total)")
    
    tile_range = range(start_tile, end_tile + 1)
    
    with tqdm(tile_range, desc="Processing tiles", unit="tile") as pbar:
        for idx in pbar:
            bb = bboxes[idx-1]
            fp = out_dir / f"tile_{idx}.gpkg"
            
            # if fp.exists():
            #     pbar.set_description(f"Tile {idx} exists, skipping")
            #     continue
            
            pbar.set_description(f"Processing tile {idx}")
            out = interpolate_chunk(bb, idx, pbar)
            
            if out is None:
                pbar.set_description(f"Tile {idx} empty, skipping")
                continue
            
            pbar.set_description(f"Saving tile {idx}")
            out.to_file(fp, driver="GPKG")
            
            del out
            gc.collect()

if __name__ == "__main__":
    main()