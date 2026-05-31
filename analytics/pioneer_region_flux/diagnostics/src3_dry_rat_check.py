"""
Diagnostics for src #3 (element 50526) injection issue:

  A. dryFlagElement time series — how often is the source cell dry, especially
     during the pulse (days 3-7)?  Dry cells are skipped at schism_step.F90:7938.

  B. bigv = area × column water depth at the source cell each timestep, and
     rat = Q*dt/bigv (with dt=45 s) — how does the injected-volume ratio
     evolve over time?  rat >> 1 would mean we're trying to dump more water
     than the cell can hold in one step → severe under-application of msource.

Outputs:
  src3_dry_rat_check.png      — 4-panel time series
  src3_dry_rat_check.csv      — time, dry-flag, bigv, rat, Q3
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry


CFG = active_run()
RUN = CFG.run_dir
OUT_PNG = CFG.out_dir / "src3_dry_rat_check.png"
OUT_CSV = CFG.out_dir / "src3_dry_rat_check.csv"
START = CFG.start
PERIOD_DAYS = CFG.period_days
SRC3_ELEM_1IDX = 50526
DT_MODEL = 45.0   # seconds, model timestep


def main():
    src_idx = SRC3_ELEM_1IDX - 1
    print("=== Loading hgrid for src#3 area ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    area_src3 = float(hgrid["areas"][src_idx])
    cx, cy = hgrid["centroids"][src_idx]
    print(f"  src#3 elem {SRC3_ELEM_1IDX}: area = {area_src3:.1f} m²,  "
          f"centroid ({cx:.4f}, {cy:.4f})")
    nc = hgrid["elements"][src_idx][0]
    print(f"  cell type: {nc}-vertex ({'triangle' if nc==3 else 'quad'})")

    # ---- dryFlagElement across stacks ----
    print("\n=== Loading dryFlagElement across stacks ===")
    paths_out2d = geometry.discover_scribed_stacks(RUN / "outputs", "out2d")
    dry_flags = []
    times = []
    for p in paths_out2d[:PERIOD_DAYS + 1]:
        ds = xr.open_dataset(p, engine="h5netcdf")
        dry = ds["dryFlagElement"].isel(nSCHISM_hgrid_face=src_idx).values
        t = ds["time"].values
        keep = (t - START) <= np.timedelta64(PERIOD_DAYS, "D")
        dry_flags.append(dry[keep]); times.append(t[keep])
        ds.close()
    dry_flag = np.concatenate(dry_flags)
    times = np.concatenate(times)
    t_sec = (times - times[0]) / np.timedelta64(1, "s")
    print(f"  loaded {len(dry_flag)} timesteps")
    print(f"  fraction DRY (all 14 d) : {(dry_flag>0).mean()*100:.2f} %")
    pulse_mask = (t_sec >= 3*86400) & (t_sec <= 7*86400)
    print(f"  fraction DRY (pulse 3-7d): {(dry_flag[pulse_mask]>0).mean()*100:.2f} %")
    print(f"  unique dry-flag values seen: {sorted(set(dry_flag.tolist()))}")

    # ---- ENV_layer_ht for src#3 across stacks ----
    print("\n=== Loading ENV_layer_ht for src#3 across stacks ===")
    paths_cmb = geometry.discover_cmb_stacks(RUN / "outputs")
    lh_series = []
    for p in paths_cmb[:PERIOD_DAYS + 1]:
        ds = xr.open_dataset(p, engine="h5netcdf")
        lh = ds["ENV_layer_ht"].isel(nSCHISM_hgrid_face=src_idx).values   # (T, L)
        t = ds["time"].values
        keep = (t - START) <= np.timedelta64(PERIOD_DAYS, "D")
        lh_series.append(lh[keep])
        ds.close()
    lh = np.concatenate(lh_series, axis=0)
    lh = np.where(np.isfinite(lh) & (lh > 0), lh, 0.0)
    column_depth = lh.sum(axis=1)   # (T,) total water depth
    bigv = area_src3 * column_depth   # (T,) m³
    print(f"  column depth range (14d): [{column_depth.min():.3f}, "
          f"{column_depth.max():.3f}] m,  mean = {column_depth.mean():.3f} m")
    print(f"  bigv range: [{bigv.min():.1f}, {bigv.max():.1f}] m³,  "
          f"mean = {bigv.mean():.1f} m³")

    # ---- vsource for src#3 over the same time axis ----
    vs = np.loadtxt(RUN / "vsource.th")
    t_vs = vs[:, 0]
    Q3 = vs[:, 3]
    Q3_on = np.interp(t_sec, t_vs, Q3)
    rat = np.where(bigv > 0, Q3_on * DT_MODEL / bigv, np.nan)
    print(f"\n=== rat = Q*dt/bigv (dt={DT_MODEL}s) ===")
    print(f"  rat full 14d : mean={np.nanmean(rat):.3f}, "
          f"max={np.nanmax(rat):.3f}")
    print(f"  rat pre-pulse 0-3d   : mean={np.nanmean(rat[t_sec<3*86400]):.4f}, "
          f"max={np.nanmax(rat[t_sec<3*86400]):.4f}")
    print(f"  rat pulse 3-7d       : mean={np.nanmean(rat[pulse_mask]):.3f}, "
          f"max={np.nanmax(rat[pulse_mask]):.3f}")
    print(f"  rat post-pulse 7-14d : mean={np.nanmean(rat[t_sec>7*86400]):.4f}, "
          f"max={np.nanmax(rat[t_sec>7*86400]):.4f}")

    print(f"\n=== Apparent mass-loss factor 1/(1+rat) under no-volume-growth assumption ===")
    print(f"  pulse-window mean 1/(1+rat) = {(1/(1+np.nanmean(rat[pulse_mask]))):.3f}  "
          f"(if cell volume doesn't track source, this is the fraction of "
          f"mass actually retained per timestep)")

    # ---- Plot ----
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axA, axB, axC, axD = axes

    # A) dry flag
    axA.fill_between(times, 0, dry_flag, step="post", color="#dc2626", alpha=0.7,
                     label="dryFlagElement (src #3 elem 50526)")
    axA.set_ylabel("dry flag (0=wet)")
    axA.set_ylim(-0.1, 1.5)
    axA.set_title("(a) src#3 dry-flag — when ==1, msource is SKIPPED per Fortran",
                  fontweight="bold", loc="left", fontsize=10)
    axA.grid(True, ls=":", alpha=0.4)
    axA.legend(loc="upper right", fontsize=9)
    axA.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.10, label="pulse window")

    # B) column depth & bigv
    axB.plot(times, column_depth, color="#0f766e", lw=1.4,
             label="Σ layer_ht (water column depth at src#3)")
    axB.set_ylabel("Column depth (m)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="upper left", fontsize=9)
    axB.set_title(f"(b) Water column depth at src#3  (cell area = {area_src3:.1f} m²)",
                  fontweight="bold", loc="left", fontsize=10)
    axB2 = axB.twinx()
    axB2.plot(times, bigv, color="#f59e0b", lw=1.0, alpha=0.6,
              label="bigv = area × column_depth")
    axB2.set_ylabel("bigv (m³)")
    axB2.legend(loc="upper right", fontsize=9)
    axB.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.10)

    # C) Q + rat
    axC.plot(times, Q3_on, color="#b91c1c", lw=1.8,
             label="vsource src#3 (m³/s)")
    axC.set_ylabel("Q (m³/s)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="upper left", fontsize=9)
    axC.set_title("(c) vsource Q at src #3  and  rat = Q×dt/bigv",
                  fontweight="bold", loc="left", fontsize=10)
    axC2 = axC.twinx()
    axC2.plot(times, rat, color="#1d4ed8", lw=1.2, alpha=0.8,
              label="rat = Q×45 / bigv (per-timestep volume fraction)")
    axC2.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6,
                 label="rat = 1.0  (cell-volume-equivalent / step)")
    axC2.set_ylabel("rat (dimensionless)")
    axC2.legend(loc="upper right", fontsize=9)
    axC.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.10)

    # D) Apparent mass retention fraction per step under no-grow assumption
    retain = 1.0 / (1.0 + rat)
    axD.plot(times, retain, color="#7c2d12", lw=1.6,
             label="1 / (1 + rat)  —  mass-injection efficiency per step")
    axD.axhline(1.0, color="#999", lw=0.5)
    axD.set_ylim(-0.05, 1.10)
    axD.set_ylabel("Retention efficiency")
    axD.set_xlabel("Date (model time)")
    axD.grid(True, ls=":", alpha=0.4)
    axD.legend(loc="lower left", fontsize=9)
    axD.set_title("(d) Hypothetical mass-injection efficiency 1/(1+rat) per timestep\n"
                  "(only relevant if cell volume doesn't grow with source — needs continuity check)",
                  fontweight="bold", loc="left", fontsize=10)
    axD.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.10)
    axD.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle("src #3 dry-flag and rat=Q*dt/bigv diagnostics (P18_flood)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")

    # CSV dump
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "dryFlagElement", "column_depth_m", "bigv_m3",
                    "Q3_m3ps", "rat_unitless", "retention_1_over_1plusrat"])
        for i in range(len(times)):
            w.writerow([str(times[i]), int(dry_flag[i]), column_depth[i],
                        bigv[i], Q3_on[i], rat[i], retain[i]])
    print(f"  wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
