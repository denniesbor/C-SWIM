import os
import pickle
import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from scipy.spatial import Voronoi
from tobler.dasymetric import masked_area_interpolate
from osgeo import gdal
from multiprocessing import cpu_count
import logging
from tqdm import tqdm
import argparse

# === Configure logging ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Configuration ===
nlcd_aea_path        = "Annual_NLCD_LndCov_2023_CU_C1V0.tif"  # AEA NLCD (EPSG:5070)
out_dir              = "tiles"                              # where to save each tile
tile_rows, tile_cols = 2, 4                                # 2×4 grid for two machines
jobs                 = 1                                    # Single-threaded to avoid memory issues

# Make output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# === Load pre-processed data from pickles ===
def load_pickle(name, folder="delaunay_output"):
    path = os.path.join(folder, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    logger.debug(f"Loading pickle: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.debug(f"Pickle loaded: {name} - {type(data)}")
    return data

# === Build or load Voronoi polygons ===
def build_or_load_voronoi(ehv_coordinates, states_gdf):
    logger.info("Building or loading Voronoi polygons")
    if os.path.exists("voronoi_polygons_clipped.geojson"):
        logger.info("Loading existing Voronoi polygons from file")
        voronoi_gdf = gpd.read_file("voronoi_polygons_clipped.geojson")
        logger.debug(f"Loaded {len(voronoi_gdf)} Voronoi polygons")
        return voronoi_gdf
    else:
        logger.info("Building new Voronoi polygons")
        coords = list(ehv_coordinates.values())
        logger.debug(f"Creating Voronoi diagram with {len(coords)} points")
        vor = Voronoi(coords)
        sub_ids = list(ehv_coordinates.keys())
        polys = []
        
        logger.debug("Constructing Voronoi polygons")
        for pid, region_idx in tqdm(enumerate(vor.point_region), total=len(vor.point_region), desc="Creating polygons"):
            region = vor.regions[region_idx]
            if region and -1 not in region:
                polys.append(Polygon([vor.vertices[i] for i in region]))
            else:
                polys.append(None)
                
        logger.debug("Creating GeoDataFrame")
        vor_df = gpd.GeoDataFrame(
            {"sub_id": sub_ids, "geometry": polys},
            crs="EPSG:4326"
        ).dropna(subset=["geometry"])
        
        logger.debug("Clipping with US boundary")
        boundary = states_gdf[~states_gdf['STATEFP'].isin(['02','15','72'])].unary_union
        voronoi_gdf = gpd.overlay(
            vor_df, gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326"), how="intersection"
        )
        
        logger.info(f"Saving {len(voronoi_gdf)} Voronoi polygons to file")
        voronoi_gdf.to_file("voronoi_polygons_clipped.geojson", driver="GeoJSON")
        return voronoi_gdf

# === Raster tiling ===
def divide_raster_into_bbox(path, n_rows, n_cols):
    logger.info(f"Dividing raster into {n_rows}x{n_cols} grid")
    bboxes = []
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        logger.debug(f"Raster dimensions: {W}x{H}")
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
                logger.debug(f"Created bbox {len(bboxes)}: {c0},{r0} -> {c1},{r1}")
    
    logger.info(f"Created {len(bboxes)} bounding boxes")
    return bboxes

# === In-memory mask writer ===
def mask_to_vsimem(mask_arr, transform, crs, idx):
    logger.debug(f"Creating in-memory mask for tile {idx}")
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
    logger.debug(f"Created virtual mask at {vsipath}")
    return vsipath

# === Chunk interpolation ===
def interpolate_chunk(bbox, idx):
    try:
        logger.info(f"Interpolating chunk {idx}")
        with rasterio.open(nlcd_aea_path) as src:
            logger.debug(f"Reading chunk {idx} from raster")
            arr = src.read(1, window=bbox["window"])
            trs = bbox["transform"]
            crs = src.crs
        
        # Check data validity before proceeding
        if arr.size == 0 or np.all(arr == 0):
            logger.warning(f"Chunk {idx} has no valid data, skipping")
            return None
            
        logger.debug(f"Creating development mask for chunk {idx}")
        mask = np.isin(arr, [21,22,23,24]).astype('uint8')
        logger.debug(f"Mask stats - sum: {np.sum(mask)}, shape: {mask.shape}")
        
        # Free memory
        del arr
        
        vsipath = mask_to_vsimem(mask, trs, crs, idx)
        
        # Free memory
        del mask
        
        x0,y1 = trs[2], trs[5]
        x1 = x0 + bbox['width']*trs[0]
        y0 = y1 + bbox['height']*trs[4]
        zone = box(x0,y0,x1,y1)
        
        logger.debug(f"Filtering source data for chunk {idx}")
        zs = zcta_5070[zcta_5070.intersects(zone)].copy()
        vs = voronoi_5070[voronoi_5070.intersects(zone)].copy()
        
        if zs.empty or vs.empty:
            logger.warning(f"Chunk {idx} has no data, skipping")
            gdal.Unlink(vsipath)
            return None
            
        logger.debug(f"Chunk {idx} has {len(zs)} ZCTAs and {len(vs)} Voronoi polygons")
        zs = zs.set_index('ZCTA', drop=False)
        vs = vs.set_index('sub_id', drop=False)
        
        logger.info(f"Running masked area interpolation for chunk {idx}")
        res = masked_area_interpolate(
            raster=vsipath,
            source_df=zs,
            target_df=vs,
            extensive_variables=economic_cols,
            intensive_variables=[],
            pixel_values=[1],
            allocate_total=True,
            n_jobs=1  # Single process to avoid memory multiplication
        )
        
        logger.debug(f"Interpolation complete for chunk {idx}, cleaning up")
        gdal.Unlink(vsipath)
        
        # Free memory explicitly
        del zs
        del vs
        
        return res
    except Exception as e:
        logger.error(f"Error processing chunk {idx}: {str(e)}")
        return None

def main(start_tile=1, end_tile=None):
    logger.info(f"Starting processing from tile {start_tile}")
    
    # Load pickle data
    logger.info("Loading pickled data")
    zcta_business_gdf = load_pickle("zcta_business_gdf")
    states_gdf = load_pickle("states_gdf")
    ehv_coordinates = load_pickle("ehv_coordinates")
    
    # Build or load Voronoi polygons
    voronoi_gdf = build_or_load_voronoi(ehv_coordinates, states_gdf)
    
    # Reproject into AEA CRS
    logger.info("Reprojecting data to EPSG:5070")
    global zcta_5070, voronoi_5070
    zcta_5070 = zcta_business_gdf.to_crs(epsg=5070)
    voronoi_5070 = voronoi_gdf.to_crs(epsg=5070)
    
    # Economic columns list
    logger.info("Identifying economic columns")
    global economic_cols
    economic_cols = [
        c for c in zcta_5070.columns
        if c.startswith("GDP_") or c.startswith("EST_")
    ]
    if "POP20" in zcta_5070.columns:
        economic_cols.append("POP20")
    logger.debug(f"Economic columns: {economic_cols}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Process tiles
    bboxes = divide_raster_into_bbox(nlcd_aea_path, tile_rows, tile_cols)
    
    # Determine end tile if not provided
    if end_tile is None or end_tile > len(bboxes):
        end_tile = len(bboxes)
    
    logger.info(f"Will process tiles {start_tile} through {end_tile} (out of {len(bboxes)} total)")
    
    # Process only the tiles in the specified range
    for idx in range(start_tile, end_tile + 1):
        bb = bboxes[idx-1]  # Convert to 0-based index
        logger.info(f"Processing tile {idx}/{len(bboxes)}")
        fp = os.path.join(out_dir, f"tile_{idx}.gpkg")
        
        if os.path.exists(fp):
            logger.warning(f"Tile {idx} already exists, skipping")
            continue
            
        out = interpolate_chunk(bb, idx)
        
        if out is None:
            logger.warning(f"Tile {idx} is empty, skipping")
            continue
            
        logger.info(f"Saving tile {idx} to {fp}")
        out.to_file(fp, driver="GPKG")
        logger.info(f"Tile {idx} saved successfully")
        
        # Explicitly free memory
        del out
        import gc
        gc.collect()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process NLCD data tiles for areal interpolation')
    parser.add_argument('--start', type=int, default=1, help='Start tile index (default: 1)')
    parser.add_argument('--end', type=int, help='End tile index (default: process all tiles)')
    args = parser.parse_args()
    
    # Run the main function with the specified tile range
    main(args.start, args.end)