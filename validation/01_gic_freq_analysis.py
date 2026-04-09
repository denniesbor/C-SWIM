"""
Author: Dennies Bor
Role: Frequency-domain GIC validation analysis.
      Computes coherence and Welch PSD for measured vs simulated GIC signals.
Inputs:
    - Transformer GIC simulation results (winding_gic_rand_0.csv)
    - NERC GIC monitor data
    - TVA GIC measurement data
    - Simulated GIC cache (partial_pveubuntu.npz)
Outputs:
    - figures/gic_coherence_welch_nerc.png/.pdf
    - figures/gic_coherence_welch_tva.png/.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
import seaborn as sns
from tqdm import tqdm

from validation.utils import (
    haversine_dist,
    get_pred_metrics,
    compute_coherence_and_psd,
    load_trafo_gic_data,
    load_nerc_gic_monitors,
    load_tva_gic_metadata,
    load_or_create_nerc_gic_dataset,
    load_or_create_tva_gic_dataset,
    get_filtered_site_ids,
    prepare_validation_data,
    read_gnd_gic,
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
    108137138,  # Bull Run
    158794210,  # Johnsonville
    287491782,  # Weakley
    108718796,  # Sullivan
    1210590700,  # Pinhook
    708931360,  # Montgomery
    164138694,  # Rutherford
]


def load_data():
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
        threshold=3.0,
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
        threshold=0.2,
        savgol_window=10,
        nerc=False,
    )

    return nerc_data, tva_data


def filter_tva_indices(tva_data, subs):
    sub_to_idx = {
        tva_data["valid_substations"][idx]: idx for idx in tva_data["selected_indices"]
    }
    return [sub_to_idx[sub] for sub in subs if sub in sub_to_idx]


def plot_coherence_psd(
    selected_indices,
    valid_match_ids,
    valid_substations,
    trafo_unique,
    filtered_measured_data,
    filtered_simulated_data,
    ds,
    operator_name,
    figures_dir,
    fs=1 / 60,
    nperseg=256,
    noverlap=128,
    window="hann",
):
    n_selected = len(selected_indices)
    fig, axes = plt.subplots(
        nrows=n_selected,
        ncols=2,
        figsize=(8.5, min(12, 2.5 * n_selected)),
        sharex="col",
    )
    if n_selected == 1:
        axes = np.array([axes])

    axes[0, 0].set_title("Coherence", loc="left", fontsize=12, fontweight="bold")
    axes[0, 1].set_title(
        "Power Spectral Density", loc="left", fontsize=12, fontweight="bold"
    )

    for j, idx in enumerate(tqdm(selected_indices)):
        ax_coh = axes[j, 0]
        ax_psd = axes[j, 1]

        close_site_list = valid_match_ids[idx]
        mag_station_close = valid_substations[idx]
        sub_lat_close, sub_lon_close = trafo_unique[
            trafo_unique.sub_id == mag_station_close
        ][["latitude", "longitude"]].values[0]

        median_sim = filtered_simulated_data[idx]
        palette = sns.color_palette("colorblind", len(close_site_list))

        all_scale_factors = []

        for i_cl, close_site in enumerate(close_site_list):
            measured = filtered_measured_data[idx][close_site]

            f_coh, Cxy, f_welch, S_meas, S_sim = compute_coherence_and_psd(
                measured,
                median_sim,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window,
            )

            ax_coh.plot(f_coh, Cxy, color=palette[i_cl], linewidth=0.8)

            int_meas = integrate.trapezoid(S_meas, f_welch)
            int_sim = integrate.trapezoid(S_sim, f_welch)
            sf = int_sim / int_meas if int_meas != 0 else np.nan
            if np.isfinite(sf):
                all_scale_factors.append(sf)

            ax_psd.semilogy(
                f_welch, S_meas, color=palette[i_cl], linestyle="-", linewidth=0.8
            )
            ax_psd.semilogy(
                f_welch, S_sim, color="black", linestyle="--", linewidth=0.8
            )

        measured_lats = ds.latitude.sel(device=close_site_list).values
        measured_lons = ds.longitude.sel(device=close_site_list).values
        avg_distance = np.mean(
            [
                haversine_dist(sub_lat_close, sub_lon_close, lat, lon)
                for lat, lon in zip(measured_lats, measured_lons)
            ]
        )
        avg_sf = np.mean(all_scale_factors) if all_scale_factors else np.nan

        text_x, text_y, dy = 1.02, 0.85, 0.18
        ax_psd.text(
            text_x,
            text_y,
            f"({chr(97 + j)})",
            transform=ax_psd.transAxes,
            fontsize=10,
            weight="bold",
        )
        ax_psd.text(
            text_x,
            text_y - dy,
            f"OSM ID: {mag_station_close}",
            transform=ax_psd.transAxes,
            fontsize=9,
        )
        ax_psd.text(
            text_x,
            text_y - dy * 2,
            f"Monitor ID: {close_site_list[0]}",
            transform=ax_psd.transAxes,
            fontsize=9,
        )
        ax_psd.text(
            text_x,
            text_y - dy * 3,
            f"Scale: {avg_sf:.2f}" if np.isfinite(avg_sf) else "Scale: N/A",
            transform=ax_psd.transAxes,
            fontsize=9,
        )
        ax_psd.text(
            text_x,
            text_y - dy * 4,
            f"Sep. dist: {avg_distance:.2f} km",
            transform=ax_psd.transAxes,
            fontsize=9,
        )

        ax_coh.set_ylabel("Coherence", fontsize=9)
        ax_coh.axhline(0.0, color="gray", linewidth=0.5)
        ax_psd.set_ylabel("PSD (A²/Hz)", fontsize=9)

        for ax in [ax_coh, ax_psd]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)

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
            ax_psd.legend(
                handles=legend_elements,
                loc="lower left",
                ncol=1,
                frameon=False,
                bbox_to_anchor=(0, -0.01),
                fontsize=8,
            )

        if j == n_selected - 1:
            ax_coh.set_xlabel("Frequency (Hz)", fontsize=9)
            ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    for ext in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"gic_coherence_welch_{operator_name.lower()}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    nerc_data, tva_data = load_data()

    plot_coherence_psd(
        nerc_data["selected_indices"],
        nerc_data["valid_match_ids"],
        nerc_data["valid_substations"],
        nerc_data["trafo_unique"],
        nerc_data["filtered_measured_data"],
        nerc_data["filtered_simulated_data"],
        nerc_data["ds_operator"],
        "NERC",
        figures_dir,
    )

    filtered_indices_tva = filter_tva_indices(tva_data, C_SWIM_TVA_SUBS)
    print(f"Plotting {len(filtered_indices_tva)} TVA sites")

    plot_coherence_psd(
        filtered_indices_tva,
        tva_data["valid_match_ids"],
        tva_data["valid_substations"],
        tva_data["trafo_unique"],
        tva_data["filtered_measured_data"],
        tva_data["filtered_simulated_data"],
        tva_data["ds_operator"],
        "TVA",
        figures_dir,
    )


if __name__ == "__main__":
    main()
