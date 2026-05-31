"""
Cross-check using the conservative TRC_trX tracers.

Same comparison as river3_vs_transect_check.py, but with the conservative
tracers (TRC_tr1, TRC_tr2, TRC_tr3) which have no internal sources or
sinks — only advection and diffusion act on them. For a sufficiently long
window the cumulative downstream flux through the (1, 2) transect should
equal the cumulative input from src 3 (within transit-time / storage
adjustments).

Panels:
    a — cumulative per-tracer input from src 3 (TRC_tr1, TRC_tr2, TRC_tr3)
    b — cumulative per-tracer downstream transport through flux.out row 2

Units: 10^6 (m^3 × concentration_units), since TRC concentrations are
dimensionless markers in this build. Magnitudes should match between panels.
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

TRACERS = ["TRC_tr1", "TRC_tr2", "TRC_tr3"]
COLORS = {
    "TRC_tr1": "#0ea5e9",
    "TRC_tr2": "#16a34a",
    "TRC_tr3": "#dc2626",
}
SCALE = 1.0e6   # report in 10^6 m^3·c units


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    plot_path = Path(__file__).parent / "river3_vs_transect_trc.png"

    START = np.datetime64("2021-04-01")
    PERIOD_DAYS = 14
    SRC_IDX_0 = 2   # src 3

    # ---- src 3 ----
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
    print(f"  src 3 Q stats over 14d: mean={Q3.mean():.3f}, max={Q3.max():.3f} m^3/s")
    print("  src 3 TRC concentrations (first row, then mean):")
    for tr in TRACERS:
        c_raw = conc[tr][:, SRC_IDX_0]
        print(f"    {tr}: c[0]={c_raw[0]}, c.mean()={c_raw.mean():.4f}")
        c3 = np.interp(t_vs, t_ms, c_raw)
        rate = Q3 * c3                                  # m^3/s · c
        dt_s = np.diff(t_vs)
        incr = 0.5 * (rate[1:] + rate[:-1]) * dt_s     # m^3 · c per step
        src_cum[tr] = np.concatenate([[0.0], np.cumsum(incr)])
        print(f"      cum 14d at src 3: {src_cum[tr][-1]/SCALE:+.4f} (×10^6 m^3·c)")

    # ---- Transect: flux.out row 2 ----
    print("\nReading flux.out...")
    da = parse_flux_out(outputs / "flux.out", n_regions=9)
    t_flux_d = da["time_days"].values
    keep = t_flux_d <= PERIOD_DAYS
    t_flux_d = t_flux_d[keep]
    flux_times = START + (t_flux_d * 86400 * 1e9).astype("timedelta64[ns]")

    # flux.out row 3 = (2, 3) interface = Pioneer Mouth proper.
    # Per the SCHISM convention I worked out (ftmp > 0 = high->low region),
    # at row 3: +ve = flow 3->2 = INTO CV. Downstream river-to-sea flow is
    # 2->3 = -ve raw -> downstream-positive needs a sign flip.
    TRANSECT_ROW = 3
    DOWNSTREAM_SIGN = -1

    trans_cum = {}
    for tr in TRACERS:
        raw = da.sel(tracer=tr, region=TRANSECT_ROW).values[keep]
        rate = DOWNSTREAM_SIGN * raw
        dt_s = np.diff(t_flux_d) * 86400.0
        incr = 0.5 * (rate[1:] + rate[:-1]) * dt_s
        trans_cum[tr] = np.concatenate([[0.0], np.cumsum(incr)])
        print(f"  transect (row {TRANSECT_ROW}) cum 14d {tr}: "
              f"{trans_cum[tr][-1]/SCALE:+.4f} (×10^6 m^3·c)")

    # Ratios
    print()
    for tr in TRACERS:
        s = src_cum[tr][-1]
        t = trans_cum[tr][-1]
        ratio = t / s * 100 if abs(s) > 1e-12 else float("nan")
        print(f"  {tr}: src3={s/SCALE:+.4f},  transect={t/SCALE:+.4f},  ratio={ratio:6.1f}%")

    # ---- Plot ----
    print(f"\nWriting plot to {plot_path}")
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    stack_a = [src_cum[t] / SCALE for t in TRACERS]
    axA.stackplot(src_times, stack_a, labels=TRACERS,
                  colors=[COLORS[t] for t in TRACERS], alpha=0.85)
    axA.axhline(0, color="#999", lw=0.5)
    axA.set_ylabel("Cumulative input\n(10⁶ m³·c)")
    axA.set_title("(a) src 3 — Pioneer River at Dumbleton: cumulative conservative-tracer input",
                  fontweight="bold", loc="left")
    axA.legend(loc="upper left", fontsize=9)
    axA.grid(True, ls=":", alpha=0.4)

    stack_b = [trans_cum[t] / SCALE for t in TRACERS]
    axB.stackplot(flux_times, stack_b, labels=TRACERS,
                  colors=[COLORS[t] for t in TRACERS], alpha=0.85)
    axB.axhline(0, color="#999", lw=0.5)
    axB.set_ylabel("Cumulative downstream\ntransport (10⁶ m³·c)")
    axB.set_title("(b) flux.out row 3 = (2, 3) interface = Pioneer Mouth: cumulative conservative-tracer transport",
                  fontweight="bold", loc="left")
    axB.legend(loc="upper left", fontsize=9)
    axB.grid(True, ls=":", alpha=0.4)
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axB.set_xlabel("Date (model time)")

    fig.suptitle("Conservative tracer cross-check: src 3 input vs (1,2) transect\n"
                 "For a perfectly conservative tracer with no in-CV storage delay, "
                 "panel b should equal panel a",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
