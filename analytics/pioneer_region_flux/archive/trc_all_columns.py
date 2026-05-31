"""
For each conservative tracer (TRC_tr1, TRC_tr2, TRC_tr3):
    - cumulative input at src 3 (one line, from vsource × msource)
    - cumulative signed flux through EACH of the 9 flux.out columns
      (one line per column, native sign convention from the docs:
       column N = flow from region N to region N-1, +ve = high -> low)

Plot the 9 + 1 lines together so we can visually see which column most
closely matches the src 3 input.

Output: trc_all_columns.png — 3 panels (one per tracer), 14-day window.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core.boundary_fluxes import parse_flux_out
from core.point_sources import (
    parse_source_sink, parse_vsource, parse_msource,
)

TRACERS = ["TRC_tr1", "TRC_tr2", "TRC_tr3"]
SCALE = 1.0e6
START = np.datetime64("2021-04-01")
PERIOD_DAYS = 14
SRC_IDX_0 = 2  # src 3


def cumtrap(rate, t_sec):
    """Trapezoidal cumulative integration; returns array same length as rate."""
    if len(t_sec) < 2:
        return np.zeros_like(rate)
    dt = np.diff(t_sec)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    out_path = Path(__file__).parent / "trc_all_columns.png"

    # --- src 3 cumulative input per tracer ---
    print("Reading vsource.th + msource.th...")
    src_elems = parse_source_sink(run / "source_sink.in")
    n_src = len(src_elems)
    t_vs, Q = parse_vsource(run / "vsource.th")
    t_ms, conc = parse_msource(run / "msource.th", n_src)

    period_s = PERIOD_DAYS * 86400.0
    mvs = t_vs <= period_s
    t_vs = t_vs[mvs]
    Q = Q[mvs]
    src_times = START + (t_vs * 1e9).astype("timedelta64[ns]")

    Q3 = Q[:, SRC_IDX_0]
    src_cum = {}
    for tr in TRACERS:
        c_raw = conc[tr][:, SRC_IDX_0]
        c3 = np.interp(t_vs, t_ms, c_raw)
        rate = Q3 * c3                                 # m^3/s · c
        src_cum[tr] = cumtrap(rate, t_vs)              # m^3 · c
        print(f"  src 3 cum 14d {tr}: {src_cum[tr][-1]/SCALE:+.4f} (x10^6 m^3·c)")

    # --- flux.out cumulative per tracer per column ---
    print("\nReading flux.out...")
    da = parse_flux_out(outputs / "flux.out", n_regions=9)
    t_flux_d = da["time_days"].values
    keep = t_flux_d <= PERIOD_DAYS
    t_flux_d = t_flux_d[keep]
    flux_times = START + (t_flux_d * 86400 * 1e9).astype("timedelta64[ns]")
    t_flux_s = t_flux_d * 86400.0

    col_cum = {}     # col_cum[tr][col_idx] = cum series
    for tr in TRACERS:
        col_cum[tr] = {}
        for col in range(1, 10):
            raw = da.sel(tracer=tr, region=col).values[keep]
            col_cum[tr][col] = cumtrap(raw, t_flux_s)
            print(f"  flux.out col {col} {tr} cum 14d: "
                  f"{col_cum[tr][col][-1]/SCALE:+.4f}")

    # --- plot 3 panels, one per tracer ---
    print(f"\nWriting plot to {out_path}")
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    tab10 = cm.get_cmap("tab10", 10)

    for ax, tr in zip(axes, TRACERS):
        # src 3 line, thick black
        ax.plot(src_times, src_cum[tr] / SCALE, color="black", lw=3.0,
                label="src 3 input (Pioneer R. @ Dumbleton)", zorder=20)
        # each flux.out column
        for col in range(1, 10):
            ax.plot(flux_times, col_cum[tr][col] / SCALE,
                    color=tab10((col - 1) % 10), lw=1.4,
                    label=f"col {col} (flow {col}->{col-1})")
        ax.axhline(0, color="#999", lw=0.4)
        ax.set_ylabel(f"{tr}\ncumulative (10⁶ m³·c)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(f"{tr} — cumulative at src 3 (black) vs each flux.out column",
                     fontweight="bold", loc="left", fontsize=10)

    axes[0].legend(loc="upper left", fontsize=8, ncols=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].set_xlabel("Date (model time)")

    fig.suptitle("Conservative TRC tracers — src 3 input vs all 9 flux.out columns\n"
                 "Cumulative signed values; flux.out sign convention = positive flow "
                 "from region N to region N-1",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
