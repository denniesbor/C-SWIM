import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from configs import setup_logger, get_data_dir

DATA_LOC = Path(get_data_dir())
SRC_DIR = DATA_LOC / "gnd_gic"
OUT_DIR = DATA_LOC / "gnd_gic_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger(log_file=str(OUT_DIR / "aggregate_gic_nc.log"))
logger.info(f"Reading CSVs from {SRC_DIR}")

csv_files = sorted(SRC_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSVs found in {SRC_DIR}")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    if "Substation" not in df.columns:
        continue
    df["Substation"] = df["Substation"].astype(str)
    dfs.append(df)

all_subs = sorted(set().union(*[d["Substation"] for d in dfs]))
all_scen = sorted(set().union(*[(set(d.columns) - {"Substation"}) for d in dfs]))

nF, nS, nC = len(dfs), len(all_subs), len(all_scen)
sub_index = {s: i for i, s in enumerate(all_subs)}
col_index = {c: i for i, c in enumerate(all_scen)}

# Build 3D cube: (files, substations, scenarios)
cube = np.full((nF, nS, nC), np.nan, dtype=np.float64)
for f_idx, df in enumerate(dfs):
    dfw = df.set_index("Substation").reindex(all_subs)
    for col in all_scen:
        if col not in dfw.columns:
            dfw[col] = np.nan
    cube[f_idx, :, :] = dfw[all_scen].to_numpy(dtype=np.float64)


def q(a, qval):
    return np.nanpercentile(a, qval, axis=0)


# Compute statistics across all files
count = np.sum(~np.isnan(cube), axis=0)
mean = np.nanmean(cube, axis=0)
median = np.nanmedian(cube, axis=0)
std = np.nanstd(cube, axis=0, ddof=1)
vmin = np.nanmin(cube, axis=0)
vmax = np.nanmax(cube, axis=0)
p05, p25, p75, p95 = q(cube, 5), q(cube, 25), q(cube, 75), q(cube, 95)

stats_stack = np.stack(
    [count, mean, median, std, vmin, p05, p25, p75, p95, vmax], axis=2
)
stat_names = [
    "count",
    "mean",
    "median",
    "std",
    "min",
    "p05",
    "p25",
    "p75",
    "p95",
    "max",
]

coords = {"substation": all_subs, "scenario": all_scen, "stat": stat_names}
da = xr.DataArray(
    stats_stack, dims=("substation", "scenario", "stat"), coords=coords, name="gic_stat"
)

nc_out = OUT_DIR / "gnd_gic_aggregated.nc"
da.to_netcdf(nc_out)
logger.info(f"Saved NetCDF: {nc_out}")
print("Done.")