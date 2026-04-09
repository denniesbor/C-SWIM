"""
Author: Dennies Bor
Role: Time-domain GIC validation analysis.
      Compares simulated vs measured GIC signals at TVA and NERC monitoring sites.
Inputs:
    - Transformer GIC simulation results (winding_gic_rand_0.csv)
    - NERC GIC monitor data
    - TVA GIC measurement data
    - Simulated GIC cache (partial_pveubuntu.npz)
Outputs:
    - figures/gic_comparison_sim_nerc.png/.pdf
    - figures/gic_comparison_sim_tva.png/.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm

from utils import (
    haversine_dist,
    read_gnd_gic,
    get_pred_metrics,
    load_trafo_gic_data,
    load_nerc_gic_monitors,
    load_tva_gic_metadata,
    load_or_create_nerc_gic_dataset,
    load_or_create_tva_gic_dataset,
    get_filtered_site_ids,
    prepare_validation_data,
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
    DEFAULT_START_TIME,
    DEFAULT_END_TIME,
    setup_matplotlib,
)

setup_matplotlib()

C_SWIM_TVA_SUBS = [
    164559261,  # East Point
    158794208,  # Johnsonville
    108137138,  # Bull Run
    106759357,  # Gleason
    158794210,  # Johnsonville
    158794217,  # Johnsonville
    708931360,  # Montgomery
    1210590700,  # Pinhook
    157842592,  # Paradise
    287491782,  # Weakley
    108718796,  # Sullivan
    164138694,  # Rutherford
]


def load_data():
    """Load simulation and measurement datasets for both NERC and TVA operators."""
    trafo_gic_gdf = load_trafo_gic_data(DENNIES_DATA_LOC)
    gdf_monitors_nerc = load_nerc_gic_monitors(nerc_gic)
    tva_gic_meas_path = tva_gic / "GIC-measured"
    tva_gic_meas_metadat = load_tva_gic_metadata(tva_gic_meas_path)
    site_ids = get_filtered_site_ids(SWERVE_DIR, DEFAULT_SITE_IDS)

    ds_gic_nerc = load_or_create_nerc_gic_dataset(
        nerc_gic, LUCY_DATA_LOC, gdf_monitors_nerc
    )
    ds_gic_tva = load_or_create_tva_gic_dataset(
        tva_gic_meas_path, LUCY_DATA_LOC, TVA_NAME_MAP, tva_gic_meas_metadat
    )

    (
        data_array,
        peak_times,
        median_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    ) = read_gnd_gic(cache_file)
    substation_names = substation_names.astype(np.int64)

    ground_truth = (
        data_array,
        peak_times,
        median_values,
        mean_values,
        uncertainty_arr,
        substation_names,
    )

    nerc_data = prepare_validation_data(
        trafo_gic_gdf,
        ds_gic_nerc,
        site_ids,
        ground_truth,
        start_time=DEFAULT_START_TIME,
        end_time=DEFAULT_END_TIME,
        threshold=15,
        savgol_window=5,
        nerc=True,
    )
    tva_data = prepare_validation_data(
        trafo_gic_gdf,
        ds_gic_tva,
        site_ids,
        ground_truth,
        start_time=DEFAULT_START_TIME,
        end_time=DEFAULT_END_TIME,
        threshold=1,
        savgol_window=10,
        nerc=False,
    )

    return nerc_data, tva_data


def filter_tva_indices(tva_data, subs):
    """Return indices ordered to match the defined substation list."""
    sub_to_idx = {
        tva_data["valid_substations"][idx]: idx for idx in tva_data["selected_indices"]
    }
    return [sub_to_idx[sub] for sub in subs if sub in sub_to_idx]


def create_gic_validation_plot(
    selected_indices,
    valid_match_ids,
    valid_substations,
    trafo_unique,
    filtered_measured_data,
    filtered_simulated_data,
    ds,
    peak_times_trimmed,
    operator_name,
    figures_dir,
    figsize=None,
    fontsize_main=8,
    fontsize_info=9,
):
    """Plot simulated vs measured GIC time series for each matched substation."""
    n_selected = len(selected_indices)

    if figsize is None:
        figsize = (8, 12) if operator_name == "TVA" else (8, 8)

    fig, axes = plt.subplots(nrows=n_selected, ncols=1, figsize=figsize, sharex=True)
    if n_selected == 1:
        axes = [axes]

    for j, idx in enumerate(tqdm(selected_indices)):
        current_ax = axes[j]
        close_site_list = valid_match_ids[idx]
        mag_station_close = valid_substations[idx]
        sub_lat_close, sub_lon_close = trafo_unique[
            trafo_unique.sub_id == mag_station_close
        ][["latitude", "longitude"]].values[0]

        median_sim = filtered_simulated_data[idx]

        current_ax.plot(
            peak_times_trimmed, median_sim, color="black", linestyle="--", linewidth=0.8
        )
        current_ax.axhline(0.0, color="gray", linewidth=0.5)
        current_ax.set_ylabel("GIC (A)", fontsize=fontsize_main)
        current_ax.spines["top"].set_visible(False)
        current_ax.spines["right"].set_visible(False)

        all_pe_values, all_corr_values = [], []
        for close_site in close_site_list:
            pe, pr_corr = get_pred_metrics(
                filtered_measured_data[idx][close_site], median_sim
            )
            if not (np.isnan(pe) or np.isnan(pr_corr)):
                all_pe_values.append(pe)
                all_corr_values.append(pr_corr)

        avg_pe = np.mean(all_pe_values) if all_pe_values else np.nan
        avg_corr = np.mean(all_corr_values) if all_corr_values else np.nan

        measured_lats = ds.latitude.sel(device=close_site_list).values
        measured_lons = ds.longitude.sel(device=close_site_list).values
        avg_distance = np.mean(
            [
                haversine_dist(sub_lat_close, sub_lon_close, lat, lon)
                for lat, lon in zip(measured_lats, measured_lons)
            ]
        )

        text_x, text_y, dy = 1.02, 0.95, 0.15
        current_ax.text(
            text_x,
            text_y,
            f"({chr(97 + j)})",
            transform=current_ax.transAxes,
            fontsize=fontsize_info,
            weight="bold",
        )
        current_ax.text(
            text_x,
            text_y - dy,
            f"OSM ID: {mag_station_close}",
            transform=current_ax.transAxes,
            fontsize=fontsize_info,
        )
        current_ax.text(
            text_x,
            text_y - dy * 2,
            f"Monitor ID: {close_site_list[0]}",
            transform=current_ax.transAxes,
            fontsize=fontsize_info,
        )
        current_ax.text(
            text_x,
            text_y - dy * 3,
            f"PE: {avg_pe:.2f}" if not np.isnan(avg_pe) else "PE: N/A",
            transform=current_ax.transAxes,
            fontsize=fontsize_info,
        )
        current_ax.text(
            text_x,
            text_y - dy * 4,
            f"Corr: {avg_corr:.2f}" if not np.isnan(avg_corr) else "Corr: N/A",
            transform=current_ax.transAxes,
            fontsize=fontsize_info,
        )
        current_ax.text(
            text_x,
            text_y - dy * 5,
            f"Sep. dist: {avg_distance:.2f} km",
            transform=current_ax.transAxes,
            fontsize=fontsize_info,
        )

        palette = sns.color_palette("colorblind", len(close_site_list))
        for i_cl, close_site in enumerate(close_site_list):
            current_ax.plot(
                peak_times_trimmed,
                filtered_measured_data[idx][close_site],
                color=palette[i_cl],
                linewidth=1,
            )

        if j == 0:
            legend_elements = [
                Line2D(
                    [0, 0.3], [0, 0], linestyle="--", color="black", label="Simulated"
                ),
            ] + [
                Line2D(
                    [0, 0.3],
                    [0, 0],
                    linestyle="-",
                    color=palette[i_cl],
                    label=operator_name,
                )
                for i_cl in range(len(close_site_list))
            ]
            current_ax.legend(
                handles=legend_elements,
                loc="lower left",
                ncol=1,
                frameon=False,
                bbox_to_anchor=(0, -0.01),
                fontsize=fontsize_main,
            )

        if j == n_selected - 1:
            current_ax.set_xlim(peak_times_trimmed[0], peak_times_trimmed[-1])
            tick_positions = [
                peak_times_trimmed[0],
                peak_times_trimmed[len(peak_times_trimmed) // 4],
                peak_times_trimmed[len(peak_times_trimmed) // 2],
                peak_times_trimmed[3 * len(peak_times_trimmed) // 4],
                peak_times_trimmed[-1],
            ]
            current_ax.set_xticks(tick_positions)
            current_ax.set_xticklabels(
                [pd.to_datetime(t).strftime("%H:%M\n%m/%d") for t in tick_positions],
                fontsize=fontsize_main,
            )
            current_ax.set_xlabel("Time", fontsize=fontsize_main)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08 if operator_name == "NERC" else 0.05)

    for ext in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"gic_comparison_sim_{operator_name.lower()}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def main():
    """Run TVA and NERC GIC time-domain validation and save figures."""
    nerc_data, tva_data = load_data()

    print(f"Found {len(nerc_data['selected_indices'])} valid NERC sites")
    # Uncomment to generate NERC figure
    # create_gic_validation_plot(
    #     nerc_data["selected_indices"], nerc_data["valid_match_ids"],
    #     nerc_data["valid_substations"], nerc_data["trafo_unique"],
    #     nerc_data["filtered_measured_data"], nerc_data["filtered_simulated_data"],
    #     nerc_data["ds_operator"], nerc_data["peak_times_trimmed"], "NERC", figures_dir,
    # )

    filtered_indices = filter_tva_indices(tva_data, C_SWIM_TVA_SUBS)
    print(f"Plotting {len(filtered_indices)} TVA sites")

    create_gic_validation_plot(
        filtered_indices,
        tva_data["valid_match_ids"],
        tva_data["valid_substations"],
        tva_data["trafo_unique"],
        tva_data["filtered_measured_data"],
        tva_data["filtered_simulated_data"],
        tva_data["ds_operator"],
        tva_data["peak_times_trimmed"],
        "TVA",
        figures_dir,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
