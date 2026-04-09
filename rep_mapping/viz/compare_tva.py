"""
Plot Modeled vs measured ground GIC for TVA Gannon storm.
Author: Dennies
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from rep_mapping.rep_config import (
    TVA_DIR,
    DEVICES_NC,
    FIGURES_DIR,
    setup_logger,
    setup_matplotlib,
)

logger = setup_logger(log_file="logs/plot_tva_gic.log")

setup_matplotlib()

FONTSIZE_MAIN = 11
FONTSIZE_INFO = 10
C_MEASURED = "#E69F00"  # Wong orange — colorblind safe


def load_data():
    ground_gic_df = pd.read_parquet(TVA_DIR / "ground_gic_ts.parquet")
    substations_df = pd.read_parquet(TVA_DIR / "substations_df.parquet")
    matched_devices = pd.read_parquet(TVA_DIR / "matched_devices.parquet")
    time_axis = np.load(TVA_DIR / "time_axis.npy", allow_pickle=True)
    ds_measured = xr.open_dataset(DEVICES_NC)
    return ground_gic_df, substations_df, matched_devices, time_axis, ds_measured


def match_substations_to_devices(matched_devices, ground_gic_df):
    monitored_osmids = set(matched_devices["osmid"].astype(float).unique())
    device_to_subs = {}
    for osmid in monitored_osmids:
        sub_id = str(float(osmid))
        if sub_id not in ground_gic_df.index:
            continue
        dev = matched_devices[matched_devices["osmid"].astype(float) == osmid][
            "device"
        ].values
        if len(dev) > 0:
            device_to_subs.setdefault(dev[0], []).append(sub_id)
    return device_to_subs


def create_gic_validation_plot(
    device_to_subs,
    ground_gic_df,
    time_axis,
    ds_measured,
    figsize=(8, 11),
):
    gic_var = list(ds_measured.data_vars)[0]

    meas_t_min = pd.to_datetime(ds_measured.time.values).min()
    meas_t_max = pd.to_datetime(ds_measured.time.values).max()
    mod_t_min = pd.to_datetime(time_axis[0])
    mod_t_max = pd.to_datetime(time_axis[-1])

    t0 = max(meas_t_min, mod_t_min).to_datetime64()
    t1 = min(meas_t_max, mod_t_max).to_datetime64()

    logger.info(f"Modeled:  {mod_t_min} to {mod_t_max}")
    logger.info(f"Measured:  {meas_t_min} to {meas_t_max}")
    logger.info(f"Overlap:   {pd.Timestamp(t0)} to {pd.Timestamp(t1)}")

    tmask = (time_axis >= t0) & (time_axis <= t1)
    time_plot = time_axis[tmask]
    logger.info(f"Overlapping timesteps: {tmask.sum()}")

    valid_devices = []
    for d in device_to_subs:
        if d not in ds_measured.device.values:
            continue
        meas_t = ds_measured.sel(device=d).time.values
        if ((meas_t >= t0) & (meas_t <= t1)).sum() > 10:
            valid_devices.append(d)

    n = len(valid_devices)
    if n == 0:
        logger.warning("No valid devices to plot")
        return

    logger.info(f"Plotting {n} devices")

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for j, device in enumerate(tqdm(valid_devices, desc="Plotting")):
        ax = axes[j]

        Modeled = np.zeros(tmask.sum())
        for sub_id in device_to_subs[device]:
            if sub_id in ground_gic_df.index:
                Modeled += ground_gic_df.loc[sub_id].values[tmask]

        meas = ds_measured.sel(device=device)
        meas_t = meas.time.values
        meas_v = meas[gic_var].values.astype(float)
        meas_mask = (meas_t >= t0) & (meas_t <= t1)
        meas_t_plot = pd.to_datetime(meas_t[meas_mask])
        meas_v_plot = meas_v[meas_mask]

        if len(meas_t_plot) > 10:
            mod_interp = np.interp(
                meas_t_plot.astype(np.int64),
                pd.to_datetime(time_plot).astype(np.int64),
                Modeled,
            )
            finite = np.isfinite(mod_interp) & np.isfinite(meas_v_plot)
            r = (
                np.corrcoef(mod_interp[finite], meas_v_plot[finite])[0, 1]
                if finite.sum() > 5
                else np.nan
            )
        else:
            r = np.nan

        if not np.isnan(r) and r < 0:
            Modeled = -Modeled
            r = abs(r)

        ax.plot(time_plot, Modeled, color="black", linestyle="--", linewidth=0.8)
        ax.plot(meas_t_plot, meas_v_plot, color=C_MEASURED, linewidth=1.0)
        ax.axhline(0, color="gray", linewidth=0.4)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", labelsize=10)
        ax.set_ylabel("GIC (A)", fontsize=FONTSIZE_MAIN)

        text_x = 1.02
        text_y = 0.95
        dy = 0.18
        ax.text(
            text_x,
            text_y,
            f"({chr(97+j)})",
            transform=ax.transAxes,
            fontsize=FONTSIZE_INFO,
        )
        ax.text(
            text_x, text_y - dy, device, transform=ax.transAxes, fontsize=FONTSIZE_INFO
        )
        ax.text(
            text_x,
            text_y - dy * 2,
            f"$r$ = {r:.2f}" if not np.isnan(r) else "$r$ = N/A",
            transform=ax.transAxes,
            fontsize=FONTSIZE_INFO,
        )

        if j == 0:
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        linestyle="--",
                        color="black",
                        label="Modeled",
                        linewidth=0.8,
                    ),
                    Line2D(
                        [0],
                        [0],
                        linestyle="-",
                        color=C_MEASURED,
                        label="Measured",
                        linewidth=1.0,
                    ),
                ],
                loc="upper left",
                frameon=False,
                fontsize=FONTSIZE_MAIN,
                bbox_to_anchor=(0, 1.0),
            )

        if j == n - 1:
            ax.set_xlim(time_plot[0], time_plot[-1])
            tick_pos = [
                time_plot[0],
                time_plot[len(time_plot) // 4],
                time_plot[len(time_plot) // 2],
                time_plot[3 * len(time_plot) // 4],
                time_plot[-1],
            ]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(
                [pd.Timestamp(t).strftime("%H:%M\n%m/%d") for t in tick_pos],
                fontsize=10,
            )
            ax.set_xlabel("Time (UTC)", fontsize=FONTSIZE_MAIN)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "tva_gic_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "tva_gic_validation.pdf", dpi=300, bbox_inches="tight")
    logger.info(f"Saved to {FIGURES_DIR}")
    plt.close()


def main():
    ground_gic_df, substations_df, matched_devices, time_axis, ds_measured = load_data()

    device_to_subs = match_substations_to_devices(matched_devices, ground_gic_df)

    logger.info(f"Devices with Modeled subs: {len(device_to_subs)}")
    for d, subs in device_to_subs.items():
        logger.info(f"  {d}: {len(subs)} substations")

    create_gic_validation_plot(
        device_to_subs,
        ground_gic_df,
        time_axis,
        ds_measured,
    )


if __name__ == "__main__":
    main()
