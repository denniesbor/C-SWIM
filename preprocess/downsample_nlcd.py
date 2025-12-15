"""
Downsample NLCD (categorical) and optionally reclass to a binary mask; write a single coarse file and/or tile outputs.
Authors: Dennies and Oughton
"""

import os
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.transform import Affine

from configs import setup_logger, get_data_dir

DATA_LOC = get_data_dir()
land_mask_dir = DATA_LOC / "land_mask"
in_raster = land_mask_dir / "Annual_NLCD_LndCov_2023_CU_C1V0.tif"

coarse_dir = land_mask_dir / "coarse"
tiles_dir = land_mask_dir / "tiles_coarse"
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(tiles_dir, exist_ok=True)

scale_factor = 4
make_binary = True
target_classes = [21, 22, 23, 24]
tile_rows, tile_cols = 8, 8
skip_existing = True

logger = setup_logger(log_file="logs/downsample_nlcd.log")


def _new_shape_res(src, scale):
    h_new = math.ceil(src.height / scale)
    w_new = math.ceil(src.width / scale)
    xres = src.transform.a * scale
    yres = -src.transform.e * scale
    return h_new, w_new, (xres, yres)


def downsample_to_single(
    in_path: Path, out_path: Path, scale=2, as_binary=False, classes=None
):
    if skip_existing and out_path.exists():
        logger.info(f"Exists, skipping: {out_path}")
        return out_path

    with rasterio.open(in_path) as src:
        h_new, w_new, (xres, yres) = _new_shape_res(src, scale)
        transform = Affine(xres, 0.0, src.bounds.left, 0.0, -yres, src.bounds.top)

        profile = src.profile.copy()
        profile.update(
            height=h_new,
            width=w_new,
            transform=transform,
            compress="LZW",
            tiled=True,
            predictor=2 if profile.get("dtype") in ("float32", "float64") else 1,
            BIGTIFF="IF_SAFER",
            nodata=src.nodata if src.nodata is not None else 0,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                data = src.read(
                    b,
                    out_shape=(h_new, w_new),
                    resampling=Resampling.mode,
                )
                if as_binary:
                    if classes is None:
                        raise ValueError("classes must be provided for binary mask.")
                    data = np.isin(data, classes).astype("uint8")
                    dst.write(data, b)
                else:
                    dst.write(data, b)

    logger.info(f"Wrote: {out_path}")
    return out_path


def divide_into_windows(src, n_rows, n_cols):
    H, W = src.height, src.width
    h_step, w_step = H // n_rows, W // n_cols
    wins = []
    for i in range(n_rows):
        for j in range(n_cols):
            r0, c0 = i * h_step, j * w_step
            r1 = H if i == n_rows - 1 else (i + 1) * h_step
            c1 = W if j == n_cols - 1 else (j + 1) * w_step
            win = Window(c0, r0, c1 - c0, r1 - r0)
            trs = src.window_transform(win)
            wins.append((win, trs, int(win.height), int(win.width), i, j))
    return wins


def write_tiles_from_single(single_path: Path, out_dir: Path):
    with rasterio.open(single_path) as src:
        wins = divide_into_windows(src, tile_rows, tile_cols)
        for k, (win, trs, h, w, i, j) in enumerate(wins, 1):
            fp = out_dir / f"tile_{k}.tif"
            if skip_existing and fp.exists():
                continue
            profile = src.profile.copy()
            profile.update(height=h, width=w, transform=trs)
            with rasterio.open(fp, "w", **profile) as dst:
                for b in range(1, src.count + 1):
                    arr = src.read(b, window=win)
                    dst.write(arr, b)
            logger.info(f"Saved tile {k} -> {fp}")


def main():
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)

    single_out = coarse_dir / (
        "nlcd_coarse_mask.tif" if make_binary else "nlcd_coarse_mode.tif"
    )
    downsample_to_single(
        in_raster,
        single_out,
        scale=scale_factor,
        as_binary=make_binary,
        classes=target_classes if make_binary else None,
    )

    write_tiles_from_single(single_out, tiles_dir)
    logger.info("Downsampling and tiling complete")


if __name__ == "__main__":
    main()