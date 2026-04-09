"""Visualization utilities for map setup and data processing."""

import pickle

import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from scipy.interpolate import griddata

from configs import setup_logger, get_data_dir

logger = setup_logger("plot-utils")
DATA_LOC = get_data_dir(econ=True)


def setup_map(ax, spatial_extent=[-120, -75, 25, 50]):
    """Setup map features and styling."""
    ax.set_extent(spatial_extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="darkgrey")
    ax.add_feature(cfeature.LAKES, alpha=0.5, linewidth=0.5, edgecolor="grey")
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
    return ax


def linestring_to_array(geom):
    """Convert LineString geometry to coordinate array."""
    if isinstance(geom, LineString):
        return np.array(geom.coords)
    else:
        return np.array([])


SUBSTATION_MAPPING = {
    "transmission": "transmission",
    "subtransmission": "transmission",
    "distribution": "distribution",
    "minor_distribution": "distribution",
    "industrial": "distribution",
    "transformer": "distribution",
    "generation": "generation",
    "generator": "generation",
    "collector": "generation",
    "switching": "switching",
    "switchyard": "switching",
    "converter": "switching",
    "compensation": "switching",
    "transition": "switching",
    "capacitor": "switching",
    None: "unknown",
    "yes": "unknown",
    "TM": "unknown",
    "traction": "unknown",
}


def process_substations(substations_gdf):
    """Process substation data and create categorized GeoDataFrame."""
    substations_gdf["SS_TYPE_CATEGORY"] = substations_gdf["SS_TYPE"].map(
        SUBSTATION_MAPPING
    )

    df = substations_gdf[
        [
            "SS_ID",
            "SS_NAME",
            "SS_OPERATOR",
            "SS_VOLTAGE",
            "SS_TYPE",
            "REGION_ID",
            "REGION",
            "lat",
            "lon",
        ]
    ].copy()

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["lat", "lon"]),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    return gdf, substations_gdf


def get_conus_polygon():
    """Retrieve the polygon representing the continental United States."""
    logger.info("Retrieving CONUS polygon...")

    try:
        import cartopy.io.shapereader as shpreader

        shapename = "admin_1_states_provinces_lakes"
        us_states = shpreader.natural_earth(
            resolution="110m", category="cultural", name=shapename
        )

        conus_states = []
        for state in shpreader.Reader(us_states).records():
            if state.attributes["admin"] == "United States of America":
                if state.attributes["name"] not in ["Alaska", "Hawaii"]:
                    conus_states.append(state.geometry)

        conus_polygon = unary_union(conus_states)
        logger.info("CONUS polygon retrieved.")
        return conus_polygon
    except Exception as e:
        logger.error(f"Failed to retrieve CONUS polygon: {e}")
        return None


def generate_grid_and_mask(
    e_fields,
    mt_coordinates,
    resolution=(500, 1000),
    filename="grid.pkl",
):
    """Generate grid and mask for interpolating E-field data."""
    if mt_coordinates.shape[0] != e_fields.shape[0]:
        logger.warning("Number of points and values don't match")

    logger.info("Generating grid and mask...")
    lon_min, lon_max = np.min(mt_coordinates[:, 1]), np.max(mt_coordinates[:, 1])
    lat_min, lat_max = np.min(mt_coordinates[:, 0]), np.max(mt_coordinates[:, 0])

    grid_x, grid_y = np.mgrid[
        lon_min : lon_max : complex(0, resolution[0]),
        lat_min : lat_max : complex(0, resolution[1]),
    ]

    conus_polygon = get_conus_polygon()

    if conus_polygon is not None:
        mask = np.array(
            [
                conus_polygon.contains(Point(x, y))
                for x, y in zip(grid_x.ravel(), grid_y.ravel())
            ]
        ).reshape(grid_x.shape)
    else:
        mask = (
            (grid_x >= lon_min)
            & (grid_x <= lon_max)
            & (grid_y >= lat_min)
            & (grid_y <= lat_max)
        )

    grid_z = griddata(
        mt_coordinates[:, [1, 0]], e_fields, (grid_x, grid_y), method="linear"
    )

    grid_z = np.ma.array(grid_z, mask=~mask)

    with open(filename, "wb") as f:
        pickle.dump((grid_x, grid_y, grid_z, e_fields), f)
        logger.info("Grid and mask saved to file.")


def extract_line_coordinates(
    df, geometry_col="geometry", source_crs=None, target_crs="EPSG:4326", filename=None
):
    """Extract line coordinates from DataFrame with geometry column."""
    logger.info("Extracting line coordinates...")
    line_coordinates = []
    valid_indices = []

    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=geometry_col)

    if df.crs is None:
        if source_crs is None:
            raise ValueError("GeoDataFrame has no CRS and source_crs is not provided")
        df = df.set_crs(source_crs, allow_override=True)

    if df.crs.to_string() != target_crs:
        df = df.to_crs(target_crs)

    for idx, geometry in enumerate(df[geometry_col]):
        if isinstance(geometry, LineString):
            coords = np.array(geometry.coords)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                line_coordinates.append(coords[:, :2])
                valid_indices.append(idx)
            else:
                logger.error(
                    f"Skipping linestring at index {idx} with unexpected shape: {coords.shape}"
                )
        else:
            logger.error(f"Invalid LineString at index {idx}: {geometry}")

    logger.info("Line coordinates extracted.")

    if filename:
        with open(filename, "wb") as f:
            pickle.dump((line_coordinates, valid_indices), f)
            logger.info("Line coordinates saved to file.")

    return line_coordinates, valid_indices


def add_ferc_regions(ax, zorder=2):
    """Add grid region boundaries to the map."""
    nerc_shp_path = (
        DATA_LOC / "raw_econ_data" / "NERC Map" / "electricity_operators.shp"
    )
    grid_regions_gdf = gpd.read_file(nerc_shp_path)

    for idx, region in grid_regions_gdf.iterrows():
        ax.add_geometries(
            [region.geometry],
            crs=ccrs.PlateCarree(),
            edgecolor="grey",
            facecolor="none",
            linewidth=0.8,
            alpha=1,
            zorder=zorder,
        )

    return ax