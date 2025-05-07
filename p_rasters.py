# nightlights_processor.py
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import numpy as np
import os
import pickle

def load_pickle_data(data_path):
    """Load pickled file."""
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        pickle_file = pickle.load(f)

    return  pickle_file

def create_conus_clipped_raster(raster_path, clip_path, states_gdf):
    """
    Clip the global nightlights raster to the continental US.
    
    Args:
        raster_path (Path): Path to the global nightlights raster
        clip_path (Path): Path where the clipped raster will be saved
        states_gdf (GeoDataFrame): GeoDataFrame containing state boundaries
    
    Returns:
        bool: True if successful, False if file already exists
    """
    if os.path.exists(clip_path):
        print(f"Using existing clipped nightlights raster: {clip_path}")
        return False
    
    print("Clipping global nightlights to CONUS...")
    
    # Get CONUS bounding box from states_gdf
    conus = states_gdf.unary_union
    minx, miny, maxx, maxy = conus.bounds
    
    # Ensure bounding box stays within valid longitude/latitude ranges
    minx = max(minx - 0.1, -180)
    miny = max(miny - 0.1, -90)
    maxx = min(maxx + 0.1, 180)
    maxy = min(maxy + 0.1, 90)
    
    print(f"CONUS bounds with buffer: {minx}, {miny}, {maxx}, {maxy}")
    bbox = box(minx, miny, maxx, maxy)
    
    with rasterio.open(raster_path) as src:
        # Check that the window is valid
        window = src.window(*bbox.bounds)
        window = window.round_offsets().round_lengths()
        
        # Verify window is within bounds
        if (window.row_off < 0 or window.col_off < 0 or 
            window.row_off + window.height > src.height or 
            window.col_off + window.width > src.width):
            print("Warning: Window extends beyond raster bounds, adjusting...")
            # Adjust window to fit within raster bounds
            window = Window(
                max(0, window.col_off),
                max(0, window.row_off),
                min(window.width, src.width - max(0, window.col_off)),
                min(window.height, src.height - max(0, window.row_off))
            )
        
        print(f"Window: col_off={window.col_off}, row_off={window.row_off}, width={window.width}, height={window.height}")
        
        # Update metadata for the destination file
        profile = src.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=rasterio.windows.transform(window, src.transform),
            compress="lzw",
            nodata=0
        )
        
        # Read data for this window
        data = src.read(1, window=window)
        
        # Write to output file
        with rasterio.open(clip_path, "w", **profile) as dst:
            dst.write(data, 1)
        
        print(f"Created clipped CONUS nightlights raster: {clip_path}")
        return True

def create_binary_mask(clip_path, binary_mask_path):
    """
    Create a binary mask from the clipped nightlights raster.
    
    Args:
        clip_path (Path): Path to the clipped nightlights raster
        binary_mask_path (Path): Path where the binary mask will be saved
    
    Returns:
        bool: True if successful, False if file already exists
    """
    if os.path.exists(binary_mask_path):
        print(f"Using existing binary mask: {binary_mask_path}")
        return False
    
    print("Creating binary mask from clipped raster...")
    
    with rasterio.open(clip_path) as src:
        # Check bounds to verify it's properly clipped
        print(f"Clipped raster bounds: {src.bounds}")
        print(f"Clipped raster shape: {src.shape}")
        
        # Copy the profile for the binary mask
        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            nodata=0,
            compress="lzw"
        )
        
        # Create the binary mask (process in blocks to handle large rasters)
        with rasterio.open(binary_mask_path, "w", **profile) as dst:
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window)
                mask = (data > 0).astype(np.uint8)
                dst.write(mask, 1, window=window)
        
        print(f"Created binary mask: {binary_mask_path}")
        return True

def update_data_dictionary(all_data, clip_path, binary_mask_path, output_path):
    """
    Update the data dictionary with nightlights paths and save it.
    
    Args:
        all_data (dict): The data dictionary to update
        clip_path (Path): Path to the clipped nightlights raster
        binary_mask_path (Path): Path to the binary mask
        output_path (Path): Path where the updated dictionary will be saved
    """
    # Update the dictionary with the nightlights paths
    all_data["nightlights_raster_path"] = str(clip_path)
    all_data["nightlights_binary_path"] = str(binary_mask_path)
    
    # Save the updated dictionary
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Updated data dictionary saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    output_dir = "delaunay_output"
    states_gdf_path = os.path.join(output_dir, "states_gdf.pkl")
    
    raster_path = Path("VNL_npp_2023_global_vcmslcfg_v2_c202402081600.average_masked.dat.tif")
    clip_path = Path("viirs2023_CONUS_proper.tif")
    binary_mask_path = Path("viirs2023_CONUS_binary.tif")
    updated_data_path = os.path.join(output_dir, "all_delaunay_data_with_nightlights.pkl")
    
    # Load data
    states_gdf = load_pickle_data(states_gdf_path)
    
    # Process nightlights data
    create_conus_clipped_raster(raster_path, clip_path, states_gdf)
    create_binary_mask(clip_path, binary_mask_path)
    
    print("Nightlights processing complete!")