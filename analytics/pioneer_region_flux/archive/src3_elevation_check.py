"""
Check elevation η at src #3 (element 50526) during the pulse.

If the source coupling is mass-conservative, η should rise at src #3 during
the high-Q pulse (so the cell volume grows to absorb the added water), and
then either plateau (steady state drainage = source input) or relax back.

If η stays clamped near tidal baseline during the pulse, the cell volume
is NOT growing to accommodate the source water — the continuity equation
isn't responding to vsource — and the source mass is being silently lost.

Also plots:
  - elevation at src #3 vs reference cell (nearby non-source cell)
  - elevation rise = Q / area implied IF no drainage (theoretical upper bound)
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
from core import geometry


RUN = Path("s:/Matt_Working/schism/P18_flood")
OUT_PNG = Path(__file__).parent / "src3_elevation_check.png"
START = np.datetime64("2021-04-01")
PERIOD_DAYS = 14
SRC3_ELEM_1IDX = 50526


def main():
    src_idx = SRC3_ELEM_1IDX - 1
    print("=== Loading hgrid ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    area_src3 = float(hgrid["areas"][src_idx])
    nc_src3, *src3_nodes = hgrid["elements"][src_idx]
    src3_nodes = [int(n) for n in src3_nodes[:int(nc_src3)]]
    cx, cy = hgrid["centroids"][src_idx]
    print(f"  src#3 elem {SRC3_ELEM_1IDX}: area={area_src3:.1f} m², "
          f"centroid=({cx:.4f}, {cy:.4f})")
    print(f"  src#3 vertex nodes (1-idx): {[n+1 for n in src3_nodes]}")

    # Pick a few nearby reference cells for comparison
    # - one in the channel further upstream
    # - one in the channel further downstream
    # - one off to the side (out of channel)
    cx_all = hgrid["centroids"][:, 0]
    cy_all = hgrid["centroids"][:, 1]
    # Reference: an element ~500 m east, roughly same latitude (in channel)
    east_mask = (cx_all > cx + 0.005) & (cx_all < cx + 0.015) & \
                (np.abs(cy_all - cy) < 0.0015)
    if east_mask.any():
        ref_east = int(np.where(east_mask)[0][0])
    else:
        ref_east = src_idx
    nc_e, *e_nodes = hgrid["elements"][ref_east]
    e_nodes = [int(n) for n in e_nodes[:int(nc_e)]]
    print(f"  east reference elem {ref_east+1} at ({cx_all[ref_east]:.4f}, "
          f"{cy_all[ref_east]:.4f})")

    # All nodes we need
    nodes_needed = sorted(set(src3_nodes) | set(e_nodes))
    node_pos = {n: i for i, n in enumerate(nodes_needed)}
    src3_idx_local = [node_pos[n] for n in src3_nodes]
    east_idx_local = [node_pos[n] for n in e_nodes]

    # ---- elevation from out2d ----
    print("\n=== Reading elevation across out2d stacks ===")
    paths = geometry.discover_scribed_stacks(RUN / "outputs", "out2d")
    elev_list = []
    times = []
    for p in paths[:PERIOD_DAYS + 1]:
        ds = xr.open_dataset(p, engine="h5netcdf")
        elev = ds["elevation"].isel(nSCHISM_hgrid_node=nodes_needed).values
        t = ds["time"].values
        keep = (t - START) <= np.timedelta64(PERIOD_DAYS, "D")
        elev_list.append(elev[keep]); times.append(t[keep])
        ds.close()
    elev = np.concatenate(elev_list, axis=0)
    times = np.concatenate(times)
    t_sec = (times - times[0]) / np.timedelta64(1, "s")
    elev = np.where(np.isfinite(elev) & (np.abs(elev) <= 1e30), elev, np.nan)

    eta_src3 = elev[:, src3_idx_local].mean(axis=1)
    eta_east = elev[:, east_idx_local].mean(axis=1)
    print(f"  eta_src3 range: [{np.nanmin(eta_src3):.3f}, {np.nanmax(eta_src3):.3f}] m, "
          f"mean={np.nanmean(eta_src3):.3f}")
    print(f"  eta_east range: [{np.nanmin(eta_east):.3f}, {np.nanmax(eta_east):.3f}] m, "
          f"mean={np.nanmean(eta_east):.3f}")

    # ---- vsource for src #3 over the same time axis ----
    vs = np.loadtxt(RUN / "vsource.th")
    t_vs = vs[:, 0]
    Q3 = vs[:, 3]
    Q3_on = np.interp(t_sec, t_vs, Q3)

    # ---- expected η-rise rates ----
    # If NO drainage, dη/dt = Q / area → η would rise linearly
    # Compute theoretical η_no_drain(t) = ∫ Q/area dt   (huge — just an upper bound)
    eta_no_drain = np.concatenate(
        [[0], np.cumsum(0.5*(Q3_on[1:]/area_src3 + Q3_on[:-1]/area_src3)*np.diff(t_sec))]
    )

    # ---- Pulse-window stats ----
    pulse_mask = (t_sec >= 3*86400) & (t_sec <= 7*86400)
    pre_mask   = t_sec < 3*86400
    post_mask  = t_sec > 7*86400
    delta_pulse_src3 = np.nanmean(eta_src3[pulse_mask]) - np.nanmean(eta_src3[pre_mask])
    delta_pulse_east = np.nanmean(eta_east[pulse_mask]) - np.nanmean(eta_east[pre_mask])
    print()
    print(f"=== Mean-η shift between pre-pulse and pulse windows ===")
    print(f"  src#3      : {delta_pulse_src3:+.4f} m")
    print(f"  east-ref   : {delta_pulse_east:+.4f} m")
    print(f"  difference : {delta_pulse_src3 - delta_pulse_east:+.4f} m  "
          f"(if continuity is responding to vsource, src#3 should be HIGHER)")

    # If no drainage, theoretical rise over the 4-day pulse:
    Q_pulse_mean = float(np.mean(Q3_on[pulse_mask]))
    eta_rise_no_drain_per_step = Q_pulse_mean * 45.0 / area_src3   # m per 45-s step
    eta_rise_no_drain_per_day  = Q_pulse_mean * 86400.0 / area_src3
    print()
    print(f"=== Theoretical η-rise upper bound (no drainage) ===")
    print(f"  per timestep (45 s): {eta_rise_no_drain_per_step:.2f} m")
    print(f"  per day            : {eta_rise_no_drain_per_day/1000:.2f} km   "
          f"(obviously physically impossible — drainage must dominate)")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axA, axB, axC = axes

    axA.plot(times, eta_src3, color="#dc2626", lw=1.5,
             label=f"η at src #3 (elem {SRC3_ELEM_1IDX})")
    axA.plot(times, eta_east, color="#1d4ed8", lw=1.5, alpha=0.8,
             label=f"η at east ref (elem {ref_east+1}, ~500 m downstream)")
    axA.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.12, label="pulse window")
    axA.set_ylabel("η (m above MSL)")
    axA.grid(True, ls=":", alpha=0.4)
    axA.legend(loc="upper left", fontsize=10)
    axA.set_title("(a) Surface elevation at src #3 vs nearby downstream cell",
                  fontweight="bold", loc="left", fontsize=11)

    # Difference: src3 - east_ref
    diff = eta_src3 - eta_east
    axB.plot(times, diff, color="#7c2d12", lw=1.5,
             label="η_src3 - η_east_ref  (head driving drainage out of src cell)")
    axB.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.12)
    axB.axhline(0, color="#999", lw=0.5)
    axB.set_ylabel("Δη (m)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="upper left", fontsize=10)
    axB.set_title("(b) Δη = head between src #3 and the downstream reference",
                  fontweight="bold", loc="left", fontsize=11)

    # Q overlay
    axC.plot(times, Q3_on, color="#b91c1c", lw=1.8,
             label="vsource Q at src #3 (m³/s)")
    axC.axvspan(START + np.timedelta64(3, "D"), START + np.timedelta64(7, "D"),
                color="grey", alpha=0.12)
    axC.set_ylabel("Q (m³/s)")
    axC.set_xlabel("Date (model time)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="upper left", fontsize=10)
    axC.set_title("(c) vsource Q at src #3 for reference",
                  fontweight="bold", loc="left", fontsize=11)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle("Source-cell elevation response during the 100 m³/s pulse",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
