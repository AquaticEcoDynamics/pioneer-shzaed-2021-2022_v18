"""
Loop through every mass/heat variable (T, S, + 17 AED state vars) and compare:
    cumulative input at src 3 from vsource × msource
    cumulative export through flux.out col 2 (flow region 2 -> region 1)

Both over the same 14-day window, no sign flips, native units.

For each variable, compute:
    src_3_cum     = integral over 14 days of  Q_src3 × c_var(t)        (m^3 · unit)
    col2_cum      = integral over 14 days of  raw_col2_var(t)          (m^3/s · unit -> m^3·unit)
    ratio_pct     = |col2_cum / src_3_cum| × 100   (only if src has signal)

Prints a tabular summary and saves a horizontal bar chart so the misalignment
across all variables is immediately visible.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core.boundary_fluxes import parse_flux_out, DEFAULT_TRACER_ORDER
from core.point_sources import (
    parse_source_sink, parse_vsource, parse_msource,
    DEFAULT_AED_TRACER_ORDER,
)


def cumtrap(rate, t_sec):
    if len(t_sec) < 2:
        return np.zeros_like(rate)
    dt = np.diff(t_sec)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    out_path = Path(__file__).parent / "all_vars_src3_vs_col2.png"
    csv_path = Path(__file__).parent / "all_vars_src3_vs_col2.csv"

    PERIOD_DAYS = 14
    SRC_IDX_0 = 2

    # AED state-var order in msource (excluding T and S):
    aed_order = DEFAULT_AED_TRACER_ORDER     # 17 names
    # flux.out variable order (excluding VOL):
    flux_var_order = DEFAULT_TRACER_ORDER    # 19 names (T, S, then 17 AED)
    print(f"AED order ({len(aed_order)}):       {aed_order}")
    print(f"flux.out order ({len(flux_var_order)}): {flux_var_order}")

    # ---- vsource + msource ----
    print("\nReading vsource.th + msource.th...")
    src_elems = parse_source_sink(run / "source_sink.in")
    n_src = len(src_elems)
    t_vs, Q = parse_vsource(run / "vsource.th")
    t_ms, conc_aed = parse_msource(run / "msource.th", n_src)
    # T and S also live in msource columns 2..15 and 16..29
    # parse_msource only returns AED tracers. Let's parse those two ourselves:
    raw = np.loadtxt(run / "msource.th")
    # cols: 0 = time; 1..n_src = T; n_src+1..2*n_src = S; then AED
    T_all = raw[:, 1:1+n_src]
    S_all = raw[:, 1+n_src:1+2*n_src]
    # consolidate into a single dict for clarity
    msource = {"Heat": T_all, "Salinity": S_all}
    for nm, arr in conc_aed.items():
        msource[nm] = arr

    period_s = PERIOD_DAYS * 86400.0
    mvs = t_vs <= period_s
    t_vs = t_vs[mvs]
    Q3 = Q[mvs, SRC_IDX_0]
    print(f"  src 3 mean Q over 14d: {Q3.mean():.3f} m^3/s")

    # ---- flux.out ----
    print("\nReading flux.out...")
    da = parse_flux_out(outputs / "flux.out", n_regions=9)
    t_flux_d = da["time_days"].values
    keep = t_flux_d <= PERIOD_DAYS
    t_flux_d = t_flux_d[keep]
    t_flux_s = t_flux_d * 86400.0

    # ---- per-variable cumulative ----
    results = []
    print()
    print(f"{'variable':<14}{'src3 cum':>16}{'col2 cum':>16}{'|col2/src3|':>14}{'units note':>18}")
    print("-" * 78)
    for v in flux_var_order:
        # src 3 input
        if v in msource:
            c_raw = msource[v][:, SRC_IDX_0]
            c3 = np.interp(t_vs, t_ms, c_raw)
            src_cum = cumtrap(Q3 * c3, t_vs)
            src_total = src_cum[-1]
        else:
            src_total = np.nan
        # flux.out col 2
        raw_c2 = da.sel(tracer=v, region=2).values[keep]
        col2_cum = cumtrap(raw_c2, t_flux_s)
        col2_total = col2_cum[-1]
        ratio = abs(col2_total / src_total * 100) if (src_total and abs(src_total) > 1e-12) else float("nan")
        unit_note = ""
        if v == "Heat":
            unit_note = "(C · m^3)"
        elif v == "Salinity":
            unit_note = "(psu · m^3)"
        else:
            unit_note = "(mmol/m^3 · m^3)"
        print(f"{v:<14}{src_total:>16.3e}{col2_total:>16.3e}{ratio:>13.2f}%{unit_note:>18}")
        results.append((v, src_total, col2_total, ratio))

    # ---- save CSV ----
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "src3_cum_14d", "col2_cum_14d", "abs_ratio_pct"])
        for v, s, c, r in results:
            w.writerow([v, f"{s:.6e}", f"{c:.6e}", f"{r:.3f}"])
    print(f"\nWrote {csv_path}")

    # ---- bar chart ----
    print(f"Writing plot to {out_path}")
    names = [r[0] for r in results]
    ratios = [r[3] if not np.isnan(r[3]) else 0 for r in results]
    has_src = [not np.isnan(r[3]) for r in results]
    colors = ["#16a34a" if hs else "#9ca3af" for hs in has_src]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, ratios, color=colors, alpha=0.85)
    for y, (v, s, c, r) in enumerate(results):
        if np.isnan(r):
            txt = "no src signal"
        else:
            txt = f"{r:.2f}%  (src={s:.2e}, col2={c:.2e})"
        ax.text(max(ratios)*0.02, y, txt, va="center", fontsize=8, color="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("|col2 cumulative / src 3 cumulative| × 100  (%)")
    ax.axvline(100, color="red", lw=1.0, ls="--", label="100% (perfect conservation)")
    ax.invert_yaxis()
    ax.set_title("Per-variable: cumulative export at flux.out col 2 vs cumulative input at src 3\n"
                 "14-day window, no transformations. Grey = src 3 doesn't carry this variable.",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
