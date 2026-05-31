"""
Raw cumulative volume comparison — no transformations.

    line A — cumulative volume into src 3 from vsource.th
             = integral of Q_src3(t) dt   in m^3

    line B — cumulative volume in flux.out column 1
             (= flow from region 1 to region 0, per SCHISM docs)
             = integral of VOL_col1(t) dt  in m^3

    line C — cumulative volume in flux.out column 2
             (= flow from region 2 to region 1, per SCHISM docs)
             = integral of VOL_col2(t) dt  in m^3

All native sign conventions kept; same 14-day window; both in cubic metres.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core.boundary_fluxes import parse_flux_out
from core.point_sources import parse_source_sink, parse_vsource, parse_msource


def cumtrap(rate, t_sec):
    if len(t_sec) < 2:
        return np.zeros_like(rate)
    dt = np.diff(t_sec)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    out_path = Path(__file__).parent / "vol_src3_vs_region1.png"
    START = np.datetime64("2021-04-01")
    PERIOD_DAYS = 14

    # ---- src 3 cumulative inflow volume + TRC_tr1 mass ----
    src_elems = parse_source_sink(run / "source_sink.in")
    n_src = len(src_elems)
    t_vs, Q = parse_vsource(run / "vsource.th")
    t_ms, conc = parse_msource(run / "msource.th", n_src)
    period_s = PERIOD_DAYS * 86400.0
    mvs = t_vs <= period_s
    t_vs = t_vs[mvs]
    Q3 = Q[mvs, 2]                                      # m^3/s at src 3
    c3_tr1 = np.interp(t_vs, t_ms, conc["TRC_tr1"][:, 2])   # c at src 3
    cum_src3_vol  = cumtrap(Q3, t_vs)                       # m^3
    cum_src3_tr1  = cumtrap(Q3 * c3_tr1, t_vs)              # m^3 · c
    src_times = START + (t_vs * 1e9).astype("timedelta64[ns]")
    print(f"src 3 final cumulative volume: {cum_src3_vol[-1]:.3e} m^3")
    print(f"src 3 final cumulative TRC_tr1: {cum_src3_tr1[-1]:.3e} m^3·c")

    # ---- flux.out cumulative VOL + TRC_tr1 for columns 1 and 2 ----
    print("\nReading flux.out...")
    da = parse_flux_out(outputs / "flux.out", n_regions=9)
    t_flux_d = da["time_days"].values
    keep = t_flux_d <= PERIOD_DAYS
    t_flux_d = t_flux_d[keep]
    flux_times = START + (t_flux_d * 86400 * 1e9).astype("timedelta64[ns]")
    t_flux_s = t_flux_d * 86400.0

    vol_col1 = da.sel(tracer="VOL", region=1).values[keep]     # m^3/s
    vol_col2 = da.sel(tracer="VOL", region=2).values[keep]
    tr1_col1 = da.sel(tracer="TRC_tr1", region=1).values[keep] # m^3/s · c
    tr1_col2 = da.sel(tracer="TRC_tr1", region=2).values[keep]
    cum_vol_col1 = cumtrap(vol_col1, t_flux_s)
    cum_vol_col2 = cumtrap(vol_col2, t_flux_s)
    cum_tr1_col1 = cumtrap(tr1_col1, t_flux_s)
    cum_tr1_col2 = cumtrap(tr1_col2, t_flux_s)
    print(f"  flux.out col 1 VOL cum:     {cum_vol_col1[-1]:+.3e} m^3")
    print(f"  flux.out col 2 VOL cum:     {cum_vol_col2[-1]:+.3e} m^3")
    print(f"  flux.out col 1 TRC_tr1 cum: {cum_tr1_col1[-1]:+.3e} m^3·c")
    print(f"  flux.out col 2 TRC_tr1 cum: {cum_tr1_col2[-1]:+.3e} m^3·c")

    # ---- plot ----
    print(f"\nWriting plot to {out_path}")
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel a — volume
    axA.plot(src_times, cum_src3_vol / 1e6, color="#dc2626", lw=2.2,
             label="src 3 input volume  ∫ Q dt")
    axA.plot(flux_times, cum_vol_col1 / 1e6, color="#1d4ed8", lw=1.6,
             label="flux.out col 1  (flow region 1 → 0)")
    axA.plot(flux_times, cum_vol_col2 / 1e6, color="#f59e0b", lw=1.6,
             label="flux.out col 2  (flow region 2 → 1)")
    axA.axhline(0, color="#999", lw=0.4)
    axA.set_ylabel("Cumulative volume (10⁶ m³)")
    axA.set_title("(a) Cumulative volume",
                  fontweight="bold", loc="left")
    axA.legend(loc="best", fontsize=10)
    axA.grid(True, ls=":", alpha=0.4)

    # Panel b — TRC_tr1
    axB.plot(src_times, cum_src3_tr1 / 1e6, color="#dc2626", lw=2.2,
             label="src 3 input TRC_tr1  ∫ Q·c dt")
    axB.plot(flux_times, cum_tr1_col1 / 1e6, color="#1d4ed8", lw=1.6,
             label="flux.out col 1 TRC_tr1")
    axB.plot(flux_times, cum_tr1_col2 / 1e6, color="#f59e0b", lw=1.6,
             label="flux.out col 2 TRC_tr1")
    axB.axhline(0, color="#999", lw=0.4)
    axB.set_ylabel("Cumulative TRC_tr1 mass (10⁶ m³·c)")
    axB.set_xlabel("Date (model time)")
    axB.set_title("(b) Cumulative TRC_tr1 mass",
                  fontweight="bold", loc="left")
    axB.legend(loc="best", fontsize=10)
    axB.grid(True, ls=":", alpha=0.4)
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle("Raw cumulative — src 3 input vs flux.out columns 1 and 2\n"
                 "no sign flips, no transformations; native SCHISM convention "
                 "(positive = flow from higher to lower region)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print()
    print("Summary (14-day totals, raw):")
    print(f"  src 3 input volume:           {cum_src3_vol[-1]/1e6:+.3f} (10^6 m^3)")
    print(f"  flux.out col 1 VOL:           {cum_vol_col1[-1]/1e6:+.3f} (10^6 m^3)")
    print(f"  flux.out col 2 VOL:           {cum_vol_col2[-1]/1e6:+.3f} (10^6 m^3)")
    print(f"  src 3 input TRC_tr1:          {cum_src3_tr1[-1]/1e6:+.3f} (10^6 m^3.c)")
    print(f"  flux.out col 1 TRC_tr1:       {cum_tr1_col1[-1]/1e6:+.3f} (10^6 m^3.c)")
    print(f"  flux.out col 2 TRC_tr1:       {cum_tr1_col2[-1]/1e6:+.3f} (10^6 m^3.c)")


if __name__ == "__main__":
    main()
