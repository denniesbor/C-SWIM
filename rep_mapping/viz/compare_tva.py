"""
Plot Modeled vs measured ground GIC for TVA Gannon storm.
Author: Dennies
"""

import json
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
FONTSIZE_INFO = 9
C_MEASURED = "#E69F00"
C_MODELED = "black"


def load_data():
    matched_devices = pd.read_parquet(TVA_DIR / "matched_devices.parquet")
    time_axis = np.load(TVA_DIR / "time_axis.npy", allow_pickle=True)
    ds_measured = xr.open_dataset(DEVICES_NC)
    mc_ts = np.load(TVA_DIR / "ground_gic_mc_ts.npy")
    with open(TVA_DIR / "ground_gic_mc_ts_subs.json") as f:
        mon_subs = json.load(f)
    mon_subs_idx = {s: i for i, s in enumerate(mon_subs)}
    return matched_devices, time_axis, ds_measured, mc_ts, mon_subs_idx


def match_substations_to_devices(matched_devices, mon_subs_idx):
    device_to_subs = {}
    for _, row in matched_devices.iterrows():
        sub_id = str(float(row["osmid"]))
        if sub_id not in mon_subs_idx:
            continue
        dev = row["device"]
        device_to_subs.setdefault(dev, []).append(sub_id)
    return device_to_subs


def create_gic_validation_plot(
    device_to_subs,
    mc_ts,
    mon_subs_idx,
    time_axis,
    ds_measured,
    figsize=(8, 11),
):
    gic_var = list(ds_measured.data_vars)[0]
    n_runs, n_mon, n_times = mc_ts.shape

    meas_t_min = pd.to_datetime(ds_measured.time.values).min()
    meas_t_max = pd.to_datetime(ds_measured.time.values).max()
    mod_t_min = pd.to_datetime(time_axis[0])
    mod_t_max = pd.to_datetime(time_axis[-1])

    t0 = max(meas_t_min, mod_t_min).to_datetime64()
    t1 = min(meas_t_max, mod_t_max).to_datetime64()

    logger.info("Modeled:  %s to %s", mod_t_min, mod_t_max)
    logger.info("Measured: %s to %s", meas_t_min, meas_t_max)
    logger.info("Overlap:  %s to %s", pd.Timestamp(t0), pd.Timestamp(t1))

    tmask = (time_axis >= t0) & (time_axis <= t1)
    time_plot = time_axis[tmask]
    logger.info("Overlapping timesteps: %d", tmask.sum())

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

    logger.info("Plotting %d devices", n)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for j, device in enumerate(tqdm(valid_devices, desc="Plotting")):
        ax = axes[j]

        ts_runs = np.zeros((n_runs, n_times), dtype=np.float32)
        for sub_id in device_to_subs[device]:
            m_i = mon_subs_idx.get(sub_id)
            if m_i is not None:
                ts_runs += mc_ts[:, m_i, :]

        ts_trim = ts_runs[:, tmask]

        meas = ds_measured.sel(device=device)
        meas_t = meas.time.values
        meas_v = meas[gic_var].values.astype(float)
        meas_mask = (meas_t >= t0) & (meas_t <= t1)
        meas_t_plot = pd.to_datetime(meas_t[meas_mask])
        meas_v_plot = meas_v[meas_mask]

        ts_med_pre = np.median(ts_trim, axis=0)
        r = np.nan
        if len(meas_t_plot) > 10:
            mod_interp = np.interp(
                meas_t_plot.astype(np.int64),
                pd.to_datetime(time_plot).astype(np.int64),
                ts_med_pre,
            )
            finite = np.isfinite(mod_interp) & np.isfinite(meas_v_plot)
            if finite.sum() > 5:
                r = np.corrcoef(mod_interp[finite], meas_v_plot[finite])[0, 1]

        if not np.isnan(r) and r < 0:
            ts_trim = -ts_trim
            r = abs(r)

        ts_med = np.median(ts_trim, axis=0)

        ax.plot(time_plot, ts_med, color=C_MODELED, linestyle="--", linewidth=0.8)
        ax.plot(meas_t_plot, meas_v_plot, color=C_MEASURED, linewidth=1.0)
        ax.axhline(0, color="gray", linewidth=0.4)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", labelsize=FONTSIZE_INFO)

        text_x = 1.02
        text_y = 0.95
        dy = 0.25
        ax.text(text_x, text_y, f"({chr(97+j)})", transform=ax.transAxes,
                fontsize=FONTSIZE_INFO)
        ax.text(text_x, text_y - dy, device, transform=ax.transAxes,
                fontsize=FONTSIZE_INFO)
        ax.text(
            text_x, text_y - dy * 2,
            f"$r$ = {r:.2f}" if not np.isnan(r) else "$r$ = N/A",
            transform=ax.transAxes, fontsize=FONTSIZE_INFO,
        )

        if j == 0:
            ax.legend(
                handles=[
                    Line2D([0], [0], linestyle="--", color=C_MODELED,
                           label="Modeled median", linewidth=0.8),
                    Line2D([0], [0], linestyle="-", color=C_MEASURED,
                           label="Measured", linewidth=1.0),
                ],
                loc="upper left",
                frameon=False,
                fontsize=FONTSIZE_INFO,
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
                fontsize=FONTSIZE_INFO,
            )
            ax.set_xlabel("Time (UTC)", fontsize=FONTSIZE_MAIN)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    fig.supylabel("GIC (A)", fontsize=FONTSIZE_MAIN, x=-0.01)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "tva_gic_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "tva_gic_validation.pdf", bbox_inches="tight")
    logger.info("Saved to %s", FIGURES_DIR)
    plt.close()


def main():
    matched_devices, time_axis, ds_measured, mc_ts, mon_subs_idx = load_data()
    device_to_subs = match_substations_to_devices(matched_devices, mon_subs_idx)

    logger.info("Devices with modeled substations: %d", len(device_to_subs))
    for d, subs in device_to_subs.items():
        logger.info("  %s: %d substations", d, len(subs))

    create_gic_validation_plot(
        device_to_subs,
        mc_ts,
        mon_subs_idx,
        time_axis,
        ds_measured,
    )


if __name__ == "__main__":
    main()
