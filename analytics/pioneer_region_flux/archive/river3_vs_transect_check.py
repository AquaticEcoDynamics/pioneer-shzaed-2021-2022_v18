"""
Cross-check: src 3 (Pioneer River at Dumbleton) N input vs the immediately
downstream (1, 2) interface in flux.out row 2.

If mass is conserved and no major in-CV transformations occur between the
source and the transect, the cumulative per-tracer N moving downstream
through the (1,2) interface should track the cumulative N injected at src 3.

Produces a two-panel stacked-area figure:
    panel a — cumulative per-tracer N from src 3 (mmol -> tonnes N)
    panel b — cumulative per-tracer N flowing downstream through flux.out row 2

Both axes in tonnes N, over 14 days of P18.
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
from core.point_sources import (
    parse_source_sink, parse_vsource, parse_msource,
)
from core.tracer_groups import REDFIELD_N_OVER_C


TRACERS = ["NIT_amm", "NIT_nit", "OGM_don", "OGM_pon", "PHY_mixed"]
N_PER_MOL = {
    "NIT_amm":   1.0,
    "NIT_nit":   1.0,
    "OGM_don":   1.0,
    "OGM_pon":   1.0,
    "PHY_mixed": REDFIELD_N_OVER_C,
}
COLORS = {
    "NIT_amm":   "#0ea5e9",
    "NIT_nit":   "#1d4ed8",
    "OGM_don":   "#16a34a",
    "OGM_pon":   "#15803d",
    "PHY_mixed": "#a855f7",
}

MMOL_TO_TONNES = 14.0067 / 1.0e9


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    plot_path = Path(__file__).parent / "river3_vs_transect_check.png"

    START = np.datetime64("2021-04-01")
    PERIOD_DAYS = 14
    SRC_IDX_0 = 2  # src 3 (1-indexed) -> index 2 (0-indexed)

    # ---- Source 3: cumulative per-tracer N over time ----
    print("Reading vsource.th + msource.th...")
    src_elems = parse_source_sink(run / "source_sink.in")
    n_src = len(src_elems)
    t_vs, Q = parse_vsource(run / "vsource.th")
    t_ms, conc = parse_msource(run / "msource.th", n_src)

    # Trim to 14 days
    period_s = PERIOD_DAYS * 86400.0
    mvs = t_vs <= period_s
    t_vs = t_vs[mvs]
    Q = Q[mvs]
    # msource often has slightly different time stamps — interp onto vsource grid
    src_times = START + (t_vs * 1e9).astype("timedelta64[ns]")

    Q3 = Q[:, SRC_IDX_0]   # m^3/s
    src_cum = {}
    print(f"  src 3 Q stats over 14d: mean={Q3.mean():.3f}, max={Q3.max():.3f} m^3/s")
    for tr in TRACERS:
        c_raw = conc[tr][:, SRC_IDX_0]
        # interp onto vsource time axis
        c3 = np.interp(t_vs, t_ms, c_raw)
        rate_mmolN_per_s = Q3 * c3 * N_PER_MOL[tr]
        # trapezoidal integral over t_vs (s) -> cumulative mmol N
        dt_s = np.diff(t_vs)
        incr = 0.5 * (rate_mmolN_per_s[1:] + rate_mmolN_per_s[:-1]) * dt_s
        cum_mmolN = np.concatenate([[0.0], np.cumsum(incr)])
        src_cum[tr] = cum_mmolN * MMOL_TO_TONNES  # tonnes N
        print(f"  src 3 cum 14d {tr:10s}: {src_cum[tr][-1]:+.4f} t N")

    # ---- Transect: per-tracer cumulative through flux.out rows 2, 3, 4 ----
    # SCHISM convention: column N = "flow from region N to region N-1"
    #   positive raw = flow N -> N-1  (high-to-low region number)
    # We plot each panel "downstream-positive" by NEGATING raw values
    # (so a positive cumulative means net N moving in the N-1 -> N direction,
    # which is the bay-ward direction for these consecutively-numbered
    # transects following the river plume).
    print("\nReading flux.out...")
    da = parse_flux_out(outputs / "flux.out", n_regions=9)
    t_flux_d = da["time_days"].values
    keep = t_flux_d <= PERIOD_DAYS
    t_flux_d = t_flux_d[keep]
    flux_times = START + (t_flux_d * 86400 * 1e9).astype("timedelta64[ns]")
    DOWNSTREAM_SIGN = -1

    trans_cum = {row: {} for row in (2, 3, 4)}
    for row in (2, 3, 4):
        for tr in TRACERS:
            raw = da.sel(tracer=tr, region=row).values[keep]
            rate_mmolN_per_s = DOWNSTREAM_SIGN * raw * N_PER_MOL[tr]
            dt_s = np.diff(t_flux_d) * 86400.0
            incr = 0.5 * (rate_mmolN_per_s[1:] + rate_mmolN_per_s[:-1]) * dt_s
            cum_mmolN = np.concatenate([[0.0], np.cumsum(incr)])
            trans_cum[row][tr] = cum_mmolN * MMOL_TO_TONNES
            print(f"  row {row} (flow {row}->{row-1}) cum 14d {tr:10s}: "
                  f"{trans_cum[row][tr][-1]:+.4f} t N")

    src_total = sum(src_cum[t][-1] for t in TRACERS)
    print(f"\n  Total N (5 pools), src 3 input:                {src_total:+.4f} t")
    for row in (2, 3, 4):
        tra_total = sum(trans_cum[row][t][-1] for t in TRACERS)
        pct = tra_total / src_total * 100 if src_total else 0
        print(f"  Total N at row {row} (downstream-positive): "
              f"{tra_total:+.4f} t   ({pct:+.1f}% of src 3)")

    # ---- Plot ----
    print(f"\nWriting plot to {plot_path}")
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    axA, axB, axC, axD = axes

    # Panel a: src 3 stacked area
    stack_a = [src_cum[t] for t in TRACERS]
    axA.stackplot(src_times, stack_a, labels=TRACERS,
                  colors=[COLORS[t] for t in TRACERS], alpha=0.85)
    axA.axhline(0, color="#999", lw=0.5)
    axA.set_ylabel("Cumulative N\n(tonnes N)")
    axA.set_title("(a) src 3 — Pioneer River at Dumbleton: cumulative N input by pool",
                  fontweight="bold", loc="left")
    axA.legend(loc="upper left", fontsize=9, ncols=5)
    axA.grid(True, ls=":", alpha=0.4)

    row_titles = {
        2: "(b) flux.out row 2 = (1, 2) interface  [downstream-positive, flow 1 -> 2]",
        3: "(c) flux.out row 3 = (2, 3) interface  [downstream-positive, flow 2 -> 3]",
        4: "(d) flux.out row 4 = (3, 4) interface  [downstream-positive, flow 3 -> 4]",
    }
    for ax, row in zip((axB, axC, axD), (2, 3, 4)):
        stack = [trans_cum[row][t] for t in TRACERS]
        ax.stackplot(flux_times, stack, labels=TRACERS,
                     colors=[COLORS[t] for t in TRACERS], alpha=0.85)
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_ylabel("Cumulative N\n(tonnes N)")
        ax.set_title(row_titles[row], fontweight="bold", loc="left")
        ax.legend(loc="upper left", fontsize=9, ncols=5)
        ax.grid(True, ls=":", alpha=0.4)
    axD.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axD.set_xlabel("Date (model time)")

    fig.suptitle("Cross-check: src 3 N input vs consecutive flux.out transects (rows 2, 3, 4)\n"
                 "If conserved with no in-CV processing, panel b/c/d should track panel a",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
