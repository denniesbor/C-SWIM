"""
Author: Dennies Bor
Role: Regional regression validation comparing alpha-beta GIC predictions
      against physics-based simulations and measured GIC data.
      Computes regional mean GIC on a spatial grid and plots scatter
      comparisons for Gannon and return period scenarios.
Inputs:
    - data/regression/substations_with_gic_uncertainty.geojson
    - data/regression/substations_with_gic_uncertainty_scaled.geojson
    - data/gnd_gic_processed/gnd_gic_aggregated.nc
    - NERC and TVA GIC measurement data
Outputs:
    - figures/regression_validation.png/.pdf
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from shapely.geometry import box

from validation.utils import (
    read_gnd_gic,
    load_trafo_gic_data,
    load_nerc_gic_monitors,
    load_tva_gic_metadata,
    load_or_create_nerc_gic_dataset,
    load_or_create_tva_gic_dataset,
    get_filtered_site_ids,
)
from configs import (
    DATA_DIR as DENNIES_DATA_LOC,
    LUCY_DATA_LOC,
    SWERVE_DIR,
    nerc_gic,
    tva_gic,
    cache_file,
    VAL_FIGURES_DIR as figures_dir,
    DEFAULT_SITE_IDS,
    TVA_NAME_MAP,
    setup_logger,
    setup_matplotlib,
)

setup_matplotlib()

logger = setup_logger(log_file="logs/val_regress_vs_sim.log")

GANNON_INPUT_FILE = DENNIES_DATA_LOC / "regression" / "substations_with_gic_uncertainty.geojson"
SCALED_INPUT_FILE = DENNIES_DATA_LOC / "regression" / "substations_with_gic_uncertainty_scaled.geojson"
NC_PATH           = DENNIES_DATA_LOC / "gnd_gic_processed" / "gnd_gic_aggregated.nc"
CELL_SIZE_M       = 200_000
SCENARIOS         = [100, 200, 250]

NERC_SITE_IDS = [
    "10052", "10076", "10099", "10238", "10255", "10587", "10618", "10619", "10622",
    "10063", "10077", "10079", "10107", "10112", "10113", "10114", "10115", "10402",
    "10428", "10438", "10659", "10660", "10693", "10181", "10182", "10184", "10185",
    "10186", "10187", "10195", "10197", "10200", "10201", "10203", "10204", "10207",
    "10208", "10212", "10220", "10249", "10250", "50100", "50127", "50112", "50131",
    "50132", "50103", "50104", "50109", "50115", "50116", "50117", "50118", "50119",
    "50120", "50122",
]


def absmax_per_device(ds, source):
    """Extract peak absolute GIC per device across the storm window."""
    gic_vars = [k for k, v in ds.data_vars.items() if {"time", "device"}.issubset(v.dims) and np.issubdtype(v.dtype, np.number)]
    vname    = "gic" if "gic" in ds.data_vars else ("GIC" if "GIC" in ds.data_vars else gic_vars[0])
    s        = np.abs(ds[vname].astype(float)).max(dim="time").to_pandas()
    df       = s.rename("gic_absmax").reset_index()
    dev_index = ds["device"].to_pandas()
    for key in ["latitude", "longitude", "type", "_installation_type", "_connection", "_minimum_value_in_measurement_range"]:
        if key in ds:
            meta     = ds[key].to_pandas()
            df[key]  = meta.reindex(dev_index).reindex(df["device"]).to_numpy()
    df["source"] = source
    return df


def load_gic_data(gannon_file, scaled_file, nc_path, scenarios):
    """Load alpha-beta predictions, scaled predictions, and simulation results."""
    sub_key = "name"

    gan_gdf = gpd.read_file(gannon_file)[[sub_key, "geometry", "mean_prediction"]]
    gan_gdf[sub_key] = gan_gdf[sub_key].astype(str)
    gan_gdf = gan_gdf.set_geometry("geometry").to_crs(3857)

    scaled_data = {}
    for yr in scenarios:
        col_name     = f"gic_{yr}yr_mean_prediction"
        sca_gdf      = gpd.read_file(scaled_file)[[sub_key, "geometry", col_name]].rename(columns={col_name: f"scaled_{yr}yr"})
        sca_gdf[sub_key] = sca_gdf[sub_key].astype(str)
        scaled_data[yr]  = sca_gdf.set_geometry("geometry").to_crs(3857)

    da      = xr.load_dataarray(nc_path)
    sim_gan = (
        da.sel(stat="mean").sel(scenario="GIC_gannon")
        .to_pandas().rename("sim_gannon").reset_index()
        .rename(columns={"substation": sub_key})
    )
    sim_gan[sub_key] = sim_gan[sub_key].astype(str)
    sim_gan = sim_gan.loc[np.abs(sim_gan["sim_gannon"]) <= 225].dropna(subset=["sim_gannon"])

    sim_data = {"gannon": sim_gan}
    for yr in scenarios:
        sim_df = (
            da.sel(stat="mean").sel(scenario=f"GIC_{yr}")
            .to_pandas().rename(f"sim_{yr}yr").reset_index()
            .rename(columns={"substation": sub_key})
        )
        sim_df[sub_key] = sim_df[sub_key].astype(str)
        sim_data[yr]    = sim_df.loc[np.abs(sim_df[f"sim_{yr}yr"]) <= 225].dropna(subset=[f"sim_{yr}yr"])

    return gan_gdf, scaled_data, sim_data


def make_grid(gdf_list, cell):
    """Create a regular spatial grid covering all input GeoDataFrames."""
    bxs  = [g.total_bounds for g in gdf_list]
    minx = min(b[0] for b in bxs)
    miny = min(b[1] for b in bxs)
    maxx = max(b[2] for b in bxs)
    maxy = max(b[3] for b in bxs)
    xs   = np.arange(minx, maxx + cell, cell)
    ys   = np.arange(miny, maxy + cell, cell)

    polys, idx = [], []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            polys.append(box(xs[i], ys[j], xs[i + 1], ys[j + 1]))
            idx.append((i, j))

    grid = gpd.GeoDataFrame(
        {"cell_i": [i for i, _ in idx], "cell_j": [j for _, j in idx]},
        geometry=polys, crs=3857,
    )
    grid["region"] = grid.apply(lambda r: f"cell_{int(r.cell_i)}_{int(r.cell_j)}", axis=1)
    return grid


def process_regional_comparison(gan_gdf, scaled_data, sim_data, scenarios, cell_size_m):
    """Aggregate GIC predictions and simulations to regional grid cells."""
    sub_key     = "name"
    all_gdfs    = [gan_gdf] + list(scaled_data.values())
    grid        = make_grid(all_gdfs, cell_size_m)
    grid["y_mean"] = grid.geometry.centroid.y
    ord_regions = grid.sort_values("y_mean", ascending=False)[["region", "y_mean"]]

    def regional_agg(tagged_gdf, lhs_col, rhs_col):
        tagged_gdf["lhs_abs"] = np.abs(tagged_gdf[lhs_col])
        tagged_gdf["rhs_abs"] = np.abs(tagged_gdf[rhs_col])
        return (
            tagged_gdf.dropna(subset=["lhs_abs", "rhs_abs"])
            .groupby("region")
            .agg(lhs_mean=("lhs_abs", "mean"), rhs_mean=("rhs_abs", "mean"), n=(sub_key, "size"))
            .reset_index()
            .merge(ord_regions, on="region", how="left")
            .sort_values("y_mean", ascending=False)
        )

    gan_tag = gpd.sjoin(gan_gdf, grid[["region", "geometry"]], how="inner", predicate="within")
    gan_tag = gan_tag.merge(sim_data["gannon"][[sub_key, "sim_gannon"]], on=sub_key, how="left")
    results = {"gannon": regional_agg(gan_tag, "mean_prediction", "sim_gannon")}

    for yr in scenarios:
        sca_tag = gpd.sjoin(scaled_data[yr], grid[["region", "geometry"]], how="inner", predicate="within")
        sca_tag = sca_tag.merge(sim_data[yr][[sub_key, f"sim_{yr}yr"]], on=sub_key, how="left")
        results[yr] = regional_agg(sca_tag, f"scaled_{yr}yr", f"sim_{yr}yr")

    return results, grid


def plot_regression_validation(regional_results, xm, ym, cell_size_m, figures_dir):
    """Create 2x2 scatter plot comparing regional GIC means across scenarios."""
    regional_gan = regional_results["gannon"]
    regional_100 = regional_results[100]
    regional_200 = regional_results[200]
    regional_250 = regional_results[250]

    x1, y1 = regional_gan["lhs_mean"].to_numpy(), regional_gan["rhs_mean"].to_numpy()
    x2, y2 = regional_100["lhs_mean"].to_numpy(), regional_100["rhs_mean"].to_numpy()
    x3, y3 = regional_200["lhs_mean"].to_numpy(), regional_200["rhs_mean"].to_numpy()
    x4, y4 = regional_250["lhs_mean"].to_numpy(), regional_250["rhs_mean"].to_numpy()

    all_vals = np.concatenate([x1, x2, x3, x4, y1, y2, y3, y4])
    lim_max  = max(150, np.nanmax(all_vals))
    ref_x    = np.array([0.0, lim_max])

    pairs  = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    titles = [
        "(a) Gannon storm simulation vs Alpha–Beta GIC means",
        "(b) 1-in-100-year simulation vs scaled GIC means",
        "(c) 1-in-200-year simulation vs scaled GIC means",
        "(d) 1-in-250-year simulation vs scaled GIC means",
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, (ax, (x, y), title) in enumerate(zip(axes, pairs, titles)):
        r    = np.corrcoef(x, y)[0, 1] if np.isfinite(x).sum() > 1 else np.nan
        rmse = np.sqrt(np.nanmean((x - y) ** 2)) if x.size > 0 else np.nan

        ax.scatter(x, y, s=48, alpha=0.55, facecolors="C0", edgecolors="#333", linewidths=0.7)
        ax.plot(ref_x, ref_x, color="0.5", lw=1.0, ls=":")

        rm, rmse_m = np.nan, np.nan
        if i == 0:
            maskm = np.isfinite(xm) & np.isfinite(ym)
            if maskm.sum() > 0:
                ax.scatter(xm[maskm], ym[maskm], s=64, alpha=0.9, facecolors="none", edgecolors="crimson", linewidths=1.2, marker="D")
                rm     = np.corrcoef(xm[maskm], ym[maskm])[0, 1]
                rmse_m = np.sqrt(np.mean((xm[maskm] - ym[maskm]) ** 2))

        ax.set_xlim(0.0, lim_max)
        ax.set_ylim(0.0, lim_max)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="both", linestyle=":", linewidth=0.6, alpha=0.85)

        txt = f"r={r:.1f}, RMSE={rmse:.1f}"
        if np.isfinite(rm) and np.isfinite(rmse_m):
            txt += f"\nMeas vs Sim: r={rm:.1f}, RMSE={rmse_m:.1f}"
        ax.text(0.6, 0.4, txt, transform=ax.transAxes, va="bottom", ha="left", fontsize=9)
        ax.set_title(title, fontsize=11, loc="left", pad=2)

    axes[0].set_ylabel("Simulation GIC abs [A]")
    axes[2].set_ylabel("Simulation GIC abs [A]")
    axes[2].set_xlabel("GIC abs [A] (Alpha–Beta derived / Scaled)")
    axes[3].set_xlabel("GIC abs [A] (Alpha–Beta derived / Scaled)")

    axes[0].text(
        0.0, 1.02,
        f"Regional means on a {cell_size_m // 1000} km grid. Crimson diamonds: measured (TVA & NERC) vs simulation (Gannon)",
        transform=axes[0].transAxes, fontsize=9, ha="left", va="bottom",
    )

    plt.tight_layout(h_pad=0.8, w_pad=0.8)
    for ext in ["png", "pdf"]:
        fig.savefig(figures_dir / f"regression_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved regression validation figure to {figures_dir}")


def main():
    """Run regional regression validation pipeline."""
    logger.info("Loading GIC measurement data...")
    trafo_gic_gdf        = load_trafo_gic_data(DENNIES_DATA_LOC)
    gdf_monitors_nerc    = load_nerc_gic_monitors(nerc_gic)
    tva_gic_meas_path    = tva_gic / "GIC-measured"
    tva_gic_meas_metadat = load_tva_gic_metadata(tva_gic_meas_path)
    site_ids             = get_filtered_site_ids(SWERVE_DIR, DEFAULT_SITE_IDS)

    ds_gic_nerc = load_or_create_nerc_gic_dataset(nerc_gic, LUCY_DATA_LOC, gdf_monitors_nerc)
    ds_gic_tva  = load_or_create_tva_gic_dataset(tva_gic_meas_path, LUCY_DATA_LOC, TVA_NAME_MAP, tva_gic_meas_metadat)

    df_nerc = absmax_per_device(ds_gic_nerc, "NERC")
    df_nerc["device"] = df_nerc["device"].astype(str)
    df_nerc = df_nerc[df_nerc["device"].isin(NERC_SITE_IDS)]

    df_tva = absmax_per_device(ds_gic_tva, "TVA")
    df_all = pd.concat([df_nerc, df_tva], ignore_index=True)

    logger.info("Loading alpha-beta and simulation GIC data...")
    gan_gdf, scaled_data, sim_data = load_gic_data(GANNON_INPUT_FILE, SCALED_INPUT_FILE, NC_PATH, SCENARIOS)

    logger.info("Computing regional comparisons...")
    regional_results, grid = process_regional_comparison(gan_gdf, scaled_data, sim_data, SCENARIOS, CELL_SIZE_M)

    meas_gdf = gpd.GeoDataFrame(
        df_all.dropna(subset=["latitude", "longitude"]).copy(),
        geometry=gpd.points_from_xy(df_all["longitude"], df_all["latitude"]),
        crs=4326,
    ).to_crs(3857)

    meas_tag      = gpd.sjoin(meas_gdf[["device", "gic_absmax", "geometry"]], grid[["region", "geometry"]], how="inner", predicate="within")
    meas_regional = meas_tag.groupby("region").agg(meas_absmax_mean=("gic_absmax", "mean")).reset_index()
    reg_merge     = regional_results["gannon"].merge(meas_regional, on="region", how="inner")
    xm            = reg_merge["meas_absmax_mean"].to_numpy()
    ym            = reg_merge["rhs_mean"].to_numpy()

    logger.info("Generating regression validation figure...")
    plot_regression_validation(regional_results, xm, ym, CELL_SIZE_M, figures_dir)


if __name__ == "__main__":
    main()