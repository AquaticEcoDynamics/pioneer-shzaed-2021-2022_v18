"""
Plot raw flux.out POS / NEG / NET cumulative across the (2<->3) gate
(region-3 column) for VOL, Salinity, GEN_1, TRC_tr1, TRC_tr3.

Each panel:
    POS = the cumulative of the "ftmp >= 0" row from flux.out
    NEG = the cumulative of the "ftmp <  0" row
    NET = POS + NEG
Bottom-left annotation: final value of NET (sign + magnitude).

For VOL the NET row is also written explicitly by SCHISM (block row 0);
for tracers we construct NET = POS + NEG from the two written rows.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_config import active_run
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CFG = active_run()
RUN = CFG.run_dir
OUT = CFG.out_dir / "conservation_check"
START = CFG.start
PERIOD_DAYS = CFG.period_days
N_REGIONS = CFG.n_regions
REGION = 3   # region-3 column = the (2<->3) gate

TRACER_ORDER = [
    "Heat", "Salinity", "GEN_1",
    "NCS_ss1", "OXY_oxy", "NIT_amm", "NIT_nit",
    "PHS_frp", "PHS_frp_ads",
    "OGM_doc", "OGM_poc", "OGM_don", "OGM_pon", "OGM_dop", "OGM_pop",
    "PHY_mixed",
    "TRC_tr1", "TRC_tr2", "TRC_tr3", "TRC_age",
]
N_VARS = len(TRACER_ORDER)

SHOW = [
    # (name, units for cumulative, units for instantaneous)
    ("VOL",      "m^3"),
    ("Salinity", "psu * m^3"),
    ("GEN_1",    "mmol/m^3 * m^3"),
    ("TRC_tr1",  "mmol/m^3 * m^3"),
    ("TRC_tr3",  "mmol/m^3 * m^3"),
]


def fix_fortran(t):
    try:
        float(t); return t
    except ValueError:
        return re.sub(r'([+-]?\d*\.\d+)([+-]\d+)$', r'\1E\2', t)


def parse_blocks(path, n_regions, n_vars, period_days):
    block_len = 3 + 2 * n_vars
    rows = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) < 1 + n_regions:
                continue
            try:
                t = float(fix_fortran(toks[0]))
                vals = [float(fix_fortran(x)) for x in toks[1:1+n_regions]]
                rows.append([t] + vals)
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float)
    n_total = arr.shape[0]
    n_blocks = n_total // block_len
    arr = arr[: n_blocks * block_len]
    print(f"  Loaded {n_blocks} timesteps from {path.name}")
    times = arr[::block_len, 0]
    block = arr[:, 1:].reshape(n_blocks, block_len, n_regions)
    keep = times <= period_days
    return times[keep], block[keep]


def get_pos_neg_net(block, var, R):
    """Return (pos, neg, net) at region column R for `var`."""
    if var == "VOL":
        return block[:, 1, R], block[:, 2, R], block[:, 0, R]
    k = TRACER_ORDER.index(var)
    pos = block[:, 3 + 2 * k,     R]
    neg = block[:, 3 + 2 * k + 1, R]
    return pos, neg, pos + neg


def cumtrap(rate, t_sec):
    out = np.zeros_like(rate)
    if len(t_sec) >= 2:
        out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t_sec))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    times, block = parse_blocks(RUN / "outputs/flux.out", N_REGIONS, N_VARS,
                                PERIOD_DAYS)
    t_sec = times * 86400.0
    flux_times = START + (times * 86400 * 1e9).astype("timedelta64[ns]")
    R = REGION - 1

    n = len(SHOW)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.7 * n), sharex=True)
    if n == 1: axes = [axes]

    for ax, (var, u_cum) in zip(axes, SHOW):
        pos, neg, net = get_pos_neg_net(block, var, R)
        cum_pos = cumtrap(pos, t_sec)
        cum_neg = cumtrap(neg, t_sec)
        cum_net = cumtrap(net, t_sec)

        # Implied upwind concentration (only meaningful for tracer vars)
        cum_vol_pos = cumtrap(block[:, 1, R], t_sec)[-1]
        cum_vol_neg = cumtrap(block[:, 2, R], t_sec)[-1]
        if var != "VOL":
            c_pos = cum_pos[-1] / cum_vol_pos if cum_vol_pos != 0 else float("nan")
            c_neg = cum_neg[-1] / cum_vol_neg if cum_vol_neg != 0 else float("nan")
            up_note = f"  |  implied upwind  c_POS={c_pos:+.3g}, c_NEG={c_neg:+.3g}"
        else:
            up_note = ""

        ax.plot(flux_times, cum_pos, color="#dc2626", lw=1.5, label="cum POS  (ftmp>=0)")
        ax.plot(flux_times, cum_neg, color="#1d4ed8", lw=1.5, label="cum NEG  (ftmp<0)")
        ax.plot(flux_times, cum_net, color="black",    lw=2.0, label="cum NET  (POS+NEG)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(f"{var}\n({u_cum})", fontsize=9)
        ax.set_title(f"{var}  —  cumulative POS/NEG/NET at (2<->3) gate{up_note}",
                      fontweight="bold", fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        # Bottom-left final value annotation
        annot = (
            f"FINAL ({PERIOD_DAYS}d):\n"
            f"  cum POS = {cum_pos[-1]:+.3e}\n"
            f"  cum NEG = {cum_neg[-1]:+.3e}\n"
            f"  cum NET = {cum_net[-1]:+.3e}"
        )
        ax.text(0.005, 0.05, annot, transform=ax.transAxes,
                 ha="left", va="bottom",
                 fontsize=8, family="monospace",
                 bbox=dict(facecolor="white", edgecolor="#999",
                           alpha=0.9, pad=4))

        print(f"\n  --- {var} ---")
        print(f"    cum POS end : {cum_pos[-1]:+.3e}")
        print(f"    cum NEG end : {cum_neg[-1]:+.3e}")
        print(f"    cum NET end : {cum_net[-1]:+.3e}")
        if var != "VOL":
            print(f"    implied c_POS={c_pos:+.3g}, c_NEG={c_neg:+.3g}")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date")

    fig.suptitle(
        f"flux.out (2<->3) gate — POS/NEG/NET cumulative (P18_flood, {PERIOD_DAYS}d)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = OUT / "flux_pos_neg_breakdown.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
