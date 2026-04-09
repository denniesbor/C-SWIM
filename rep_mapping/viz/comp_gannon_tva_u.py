import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from rep_mapping.rep_config import (
    UIUC_DIR,
    TVA_DIR,
    DEVICES_NC,
    FIGURES_DIR,
    setup_matplotlib,
)

setup_matplotlib()

# Wong colorblind-safe palette
C_UIUC = "#0072B2"  # blue
C_TVA = "#E69F00"  # orange
C_MEASURED = "black"

uiuc_gic = pd.read_parquet(UIUC_DIR / "ground_gic_ts_gannon.parquet")
tva_gic = pd.read_parquet(TVA_DIR / "ground_gic_ts.parquet")
time_axis = np.load(TVA_DIR / "time_axis.npy", allow_pickle=True)
ds = xr.open_dataset(DEVICES_NC)
gic_var = list(ds.data_vars)[0]

# Time overlap
t0 = max(pd.to_datetime(time_axis[0]), pd.to_datetime(ds.time.values[0]))
t1 = min(pd.to_datetime(time_axis[-1]), pd.to_datetime(ds.time.values[-1]))
tmask = (pd.to_datetime(time_axis) >= t0) & (pd.to_datetime(time_axis) <= t1)
time_plot = time_axis[tmask]

# Data
uiuc_vals = uiuc_gic.loc[93].values[tmask].copy()
tva_vals = tva_gic.loc["106782533.0"].values[tmask].copy()

meas = ds.sel(device="Johnsonville")
meas_t = pd.to_datetime(meas.time.values)
meas_mask = (meas_t >= t0) & (meas_t <= t1)
meas_t_plot = meas_t[meas_mask]
meas_v = meas[gic_var].values[meas_mask].astype(float)

# Pearson r — polarity correct if negative
mod_interp_uiuc = np.interp(
    meas_t_plot.astype(np.int64), pd.to_datetime(time_plot).astype(np.int64), uiuc_vals
)
mod_interp_tva = np.interp(
    meas_t_plot.astype(np.int64), pd.to_datetime(time_plot).astype(np.int64), tva_vals
)

finite = np.isfinite(meas_v)
r_uiuc = np.corrcoef(mod_interp_uiuc[finite], meas_v[finite])[0, 1]
r_tva = np.corrcoef(mod_interp_tva[finite], meas_v[finite])[0, 1]

if r_uiuc < 0:
    uiuc_vals = -uiuc_vals
    r_uiuc = abs(r_uiuc)
if r_tva < 0:
    tva_vals = -tva_vals
    r_tva = abs(r_tva)

# Peak GIC distributions
uiuc_max = uiuc_gic.abs().max(axis=1).values
tva_max = tva_gic.abs().max(axis=1).values

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# (a) CDF
ax = axes[0]
ax.plot(
    np.sort(uiuc_max),
    np.linspace(0, 1, len(uiuc_max)),
    color=C_UIUC,
    linewidth=1.5,
    label=f"UIUC150 synthetic (n={len(uiuc_max)})",
)
ax.plot(
    np.sort(tva_max),
    np.linspace(0, 1, len(tva_max)),
    color=C_TVA,
    linewidth=1.5,
    linestyle="--",
    label=f"TVA OSM+HIFLD (n={len(tva_max)})",
)
ax.axvline(
    np.percentile(uiuc_max, 95), color=C_UIUC, linewidth=0.8, linestyle=":", alpha=0.7
)
ax.axvline(
    np.percentile(tva_max, 95), color=C_TVA, linewidth=0.8, linestyle=":", alpha=0.7
)
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.set_xlabel("Peak ground GIC (A)")
ax.set_ylabel("CDF")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="in")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
ax.text(
    0.0,
    1.04,
    "(a) Peak ground GIC distribution — Gannon storm",
    transform=ax.transAxes,
    fontsize=11,
    va="bottom",
)

# (b) Johnsonville time series
ax = axes[1]
ax.plot(
    time_plot,
    uiuc_vals,
    color=C_UIUC,
    linewidth=0.9,
    linestyle="--",
    label=f"UIUC150 sub 93 (0.6 km, $r$={r_uiuc:.2f})",
)
ax.plot(
    time_plot,
    tva_vals,
    color=C_TVA,
    linewidth=0.9,
    linestyle="-.",
    label=f"TVA OSM sub (0.7 km, $r$={r_tva:.2f})",
)
ax.plot(meas_t_plot, meas_v, color=C_MEASURED, linewidth=1.0, label="Measured")
ax.axhline(0, color="gray", linewidth=0.4)

tick_pos = [
    time_plot[0],
    time_plot[len(time_plot) // 4],
    time_plot[len(time_plot) // 2],
    time_plot[3 * len(time_plot) // 4],
    time_plot[-1],
]
ax.set_xticks(tick_pos)
ax.set_xticklabels(
    [pd.Timestamp(t).strftime("%H:%M\n%m/%d") for t in tick_pos], fontsize=9
)
ax.set_xlim(time_plot[0], time_plot[-1])
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Ground GIC (A)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="in")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(
    0.0,
    1.04,
    "(b) Johnsonville — Modeled vs measured",
    transform=ax.transAxes,
    fontsize=11,
    va="bottom",
)

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES_DIR / "gannon_cdf_johnsonville.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "gannon_cdf_johnsonville.pdf", dpi=300, bbox_inches="tight")
plt.close()
