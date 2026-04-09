"""
Author: Dennies Bor
Role: SECS magnetic field interpolation validation.
      Compares SECS-predicted horizontal B-field components against
      measured magnetometer data from NERC and TVA networks.
Inputs:
    - NERC magnetometer data
    - TVA magnetometer data
    - SECS-predicted B-field from Gannon storm (ds_gannon.nc)
Outputs:
    - figures/nerc_secs_mag_comparison.png/.pdf
    - figures/tva_secs_mag_comparison.png/.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from utils import (
    load_nerc_magnetometer_data,
    load_tva_magnetometer_data,
    load_simulated_data,
    get_close_mag_sites,
    get_pred_metrics,
)
from configs import (
    VAL_FIGURES_DIR as figures_dir,
    DEFAULT_START_TIME,
    DEFAULT_END_TIME,
    setup_logger,
    setup_matplotlib,
)

logger = setup_logger(log_file="logs/val_bfield.log")

setup_matplotlib()

def prepare_data_for_plotting(ds_mag_data, simulated_ds, threshold=30, n_sites=6):
    """Match magnetometer stations to nearest MT sites and slice to overlapping time window."""
    ds_mag_close = get_close_mag_sites(ds_mag_data, simulated_ds, threshold=threshold)

    start_time = max(ds_mag_close.time.min().values, simulated_ds.time.min().values)
    end_time = min(ds_mag_close.time.max().values, simulated_ds.time.max().values)

    if start_time >= end_time:
        raise ValueError(
            f"No time overlap. Magnetometer: {ds_mag_close.time.min().values} to "
            f"{ds_mag_close.time.max().values}, Simulation: {simulated_ds.time.min().values} "
            f"to {simulated_ds.time.max().values}"
        )

    logger.info(f"Using overlapping time range: {start_time} to {end_time}")

    ds_mag_sliced = ds_mag_close.sel(time=slice(start_time, end_time))
    ds_mt_sliced = simulated_ds.sel(time=slice(start_time, end_time))
    times = pd.to_datetime(ds_mag_sliced.time.values)

    try:
        mt_names = simulated_ds.name.values
    except AttributeError:
        mt_names = simulated_ds.device.values

    valid = ~np.isnan(ds_mag_sliced.nearest_mt_site.values)
    mag_stations = ds_mag_sliced.device.values[valid][:n_sites]
    mt_indices = ds_mag_sliced.nearest_mt_site.values[valid].astype(int)[:n_sites]
    n_sites = len(mag_stations)

    return (
        ds_mag_sliced,
        ds_mt_sliced,
        times,
        mt_names,
        mag_stations,
        mt_indices,
        n_sites,
        ds_mag_close,
    )


def _configure_axes_appearance(
    ax_x_main, ax_y_main, ax_x_res, ax_y_res, times, site_idx, n_sites
):
    """Configure spine visibility, tick labels, and x-axis formatting."""
    for ax in [ax_x_main, ax_y_main]:
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)

    for ax in [ax_x_res, ax_y_res]:
        ax.spines["top"].set_visible(False)

    for ax in [ax_x_main, ax_y_main, ax_x_res, ax_y_res]:
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    ax_x_res.sharex(ax_x_main)
    ax_y_res.sharex(ax_y_main)

    if site_idx == n_sites - 1:
        tick_positions = np.linspace(0, len(times) - 1, 5, dtype=int)
        tick_times = [times[i] for i in tick_positions]
        for ax in [ax_x_res, ax_y_res]:
            ax.set_xlabel("Time", fontsize=9)
            ax.set_xticks(tick_times)
            ax.set_xticklabels([t.strftime("%H:%M\n%m/%d") for t in tick_times])
    else:
        ax_x_res.tick_params(labelbottom=False)
        ax_y_res.tick_params(labelbottom=False)


def _plot_single_site_comparison(
    site_idx,
    mag_station,
    mt_idx,
    axes,
    times,
    ds_mag_sliced,
    ds_mt_sliced,
    ds_mag_close,
    simulated_ds,
    mt_names,
    colors,
    labels,
    n_sites,
):
    """Plot measured vs predicted B-field and residuals for one magnetometer station."""
    ax_x_main, ax_y_main = axes[0]
    ax_x_res, ax_y_res = axes[1]

    mag_lat = ds_mag_close.latitude.sel(device=mag_station).values
    mag_lon = ds_mag_close.longitude.sel(device=mag_station).values
    mt_lat = simulated_ds.latitude.isel(site=mt_idx).values
    mt_lon = simulated_ds.longitude.isel(site=mt_idx).values
    mt_name = mt_names[mt_idx]
    dist = ds_mag_close.nearest_distance.sel(device=mag_station).values

    mag_Bx = ds_mag_sliced.Bx.sel(device=mag_station).values
    mag_By = ds_mag_sliced.By.sel(device=mag_station).values
    mt_Bx = ds_mt_sliced.B_pred.isel(site=mt_idx).sel(bcomp="Bx").values
    mt_By = ds_mt_sliced.B_pred.isel(site=mt_idx).sel(bcomp="By").values

    pe_x, corr_x = get_pred_metrics(mag_Bx, mt_Bx)
    pe_y, corr_y = get_pred_metrics(mag_By, mt_By)
    res_x = mt_Bx - mag_Bx
    res_y = mt_By - mag_By

    ax_x_main.plot(
        times,
        mag_Bx,
        color=colors[0],
        linewidth=0.8,
        label=labels[0] if site_idx == 0 else "",
    )
    ax_x_main.plot(
        times,
        mt_Bx,
        color=colors[1],
        linewidth=0.6,
        linestyle="--",
        label=labels[1] if site_idx == 0 else "",
    )
    ax_x_main.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax_x_main.set_ylabel(r"$\Delta B_x$ (nT)", fontsize=9)

    ax_y_main.plot(times, mag_By, color=colors[0], linewidth=0.8)
    ax_y_main.plot(times, mt_By, color=colors[1], linewidth=0.6, linestyle="--")
    ax_y_main.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax_y_main.set_ylabel(r"$\Delta B_y$ (nT)", fontsize=9)

    ax_x_res.plot(
        times,
        res_x,
        color=colors[2],
        linewidth=0.5,
        label=labels[2] if site_idx == 0 else "",
    )
    ax_x_res.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax_x_res.set_ylabel("Res (nT)", fontsize=8)

    ax_y_res.plot(times, res_y, color=colors[2], linewidth=0.5)
    ax_y_res.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax_y_res.set_ylabel("Res (nT)", fontsize=8)

    ax_x_res.text(
        0.8, 0.1, f"PE: {pe_x:.2f}", transform=ax_x_main.transAxes, fontsize=9
    )
    ax_x_res.text(
        0.8, -0.2, f"Corr: {corr_x:.2f}", transform=ax_x_main.transAxes, fontsize=9
    )
    ax_y_res.text(
        0.8, 0.1, f"PE: {pe_y:.2f}", transform=ax_y_main.transAxes, fontsize=9
    )
    ax_y_res.text(
        0.8, -0.2, f"Corr: {corr_y:.2f}", transform=ax_y_main.transAxes, fontsize=9
    )

    ax_y_main.text(
        1.02,
        1.01,
        f"({chr(97 + site_idx)})",
        transform=ax_y_main.transAxes,
        fontsize=9,
        weight="bold",
    )
    ax_y_main.text(
        1.02, 0.7, f"MT: {mt_name}", transform=ax_y_main.transAxes, fontsize=9
    )
    ax_y_main.text(
        1.02, 0.45, f"MAG: {mag_station}", transform=ax_y_main.transAxes, fontsize=9
    )
    ax_y_main.text(
        1.02,
        0.2,
        f"Sep. dist: {dist:.1f} km",
        transform=ax_y_main.transAxes,
        fontsize=9,
    )

    _configure_axes_appearance(
        ax_x_main, ax_y_main, ax_x_res, ax_y_res, times, site_idx, n_sites
    )


def create_magnetic_field_comparison_plot(
    ds_mag_data, simulated_ds, operator_name, threshold=30, n_sites=6, figsize=(8.5, 12)
):
    """Create multi-site B-field comparison plot with residuals for one operator."""
    (
        ds_mag_sliced,
        ds_mt_sliced,
        times,
        mt_names,
        mag_stations,
        mt_indices,
        n_sites,
        ds_mag_close,
    ) = prepare_data_for_plotting(ds_mag_data, simulated_ds, threshold, n_sites)

    height_ratios = []
    for i in range(n_sites):
        height_ratios.extend([1, 1])
        if i < n_sites - 1:
            height_ratios.append(0.5)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        len(height_ratios),
        2,
        figure=fig,
        height_ratios=height_ratios,
        wspace=0.25,
        hspace=0.01,
    )
    colors = ["#B2182B", "#2166AC", "#762A83"]
    labels = ["Measured", "Predicted", "Residuals"]
    axes = []
    row_idx = 0

    for site_idx in range(n_sites):
        ax_x_main = fig.add_subplot(gs[row_idx, 0])
        ax_y_main = fig.add_subplot(gs[row_idx, 1])
        ax_x_res = fig.add_subplot(gs[row_idx + 1, 0])
        ax_y_res = fig.add_subplot(gs[row_idx + 1, 1])
        row_idx += 2

        site_axes = [[ax_x_main, ax_y_main], [ax_x_res, ax_y_res]]
        axes.append(site_axes)

        _plot_single_site_comparison(
            site_idx,
            mag_stations[site_idx],
            int(mt_indices[site_idx]),
            site_axes,
            times,
            ds_mag_sliced,
            ds_mt_sliced,
            ds_mag_close,
            simulated_ds,
            mt_names,
            colors,
            labels,
            n_sites,
        )

        if site_idx < n_sites - 1:
            row_idx += 1

    legend_elements = [
        Line2D([0], [0], color=colors[0], linewidth=0.8, label=labels[0]),
        Line2D(
            [0], [0], color=colors[1], linewidth=0.6, linestyle="--", label=labels[1]
        ),
        Line2D([0], [0], color=colors[2], linewidth=0.5, label=labels[2]),
    ]
    axes[0][0][1].legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(1.15, -0.3),
        fontsize=9,
        frameon=False,
    )

    output_file = f"{operator_name.lower()}_secs_mag_comparison"
    for ext in ["png", "pdf"]:
        fig.savefig(figures_dir / f"{output_file}.{ext}", dpi=300, bbox_inches="tight")

    return fig


def main():
    """Run SECS B-field validation for NERC and TVA magnetometer networks."""
    logger.info("Loading NERC magnetometer data...")
    ds_mag_nerc = load_nerc_magnetometer_data().resample(time="min").first()

    logger.info("Loading TVA magnetometer data...")
    ds_tva_mag = load_tva_magnetometer_data().resample(time="min").first()

    ds_mag_nerc = ds_mag_nerc.sel(time=slice(DEFAULT_START_TIME, DEFAULT_END_TIME))
    ds_tva_mag = ds_tva_mag.sel(time=slice(DEFAULT_START_TIME, DEFAULT_END_TIME))

    logger.info("Loading simulated data...")
    simulated_ds = load_simulated_data().sel(
        time=slice(DEFAULT_START_TIME, DEFAULT_END_TIME)
    )

    logger.info("Creating NERC comparison plot...")
    fig_nerc = create_magnetic_field_comparison_plot(
        ds_mag_nerc, simulated_ds, "NERC", threshold=30
    )
    plt.close(fig_nerc)

    logger.info("Creating TVA comparison plot...")
    fig_tva = create_magnetic_field_comparison_plot(
        ds_tva_mag, simulated_ds, "TVA", threshold=50
    )
    plt.close(fig_tva)

    return fig_nerc, fig_tva


if __name__ == "__main__":
    main()
