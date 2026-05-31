"""
Plot raw flux.out columns — no interpretation, no sign-flipping,
no source bookkeeping. Just what SCHISM wrote.

For VOL the file contains three rows per timestep: NET, POS (>=0), NEG (<=0).
For each tracer it contains two rows: POS, NEG. We construct NET = POS + NEG.

Plotted per variable:
    LEFT  — instantaneous POS, NEG, NET, |GROSS|=POS-NEG  (raw values, no time avg)
    RIGHT — cumulative of the same series (time-integrated)

All series are taken from region-3 column of flux.out (the (2<->3) transect
in this simplified fluxflag.prop; columns 1 and 2 are zero everywhere).

Variables shown: VOL, Salinity, GEN_1, TRC_tr1.
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


RUN     = Path("s:/Matt_Working/schism/P18_flood")
OUT_DIR = Path(__file__).parent / "conservation_check"
START   = np.datetime64("2021-04-01")
PERIOD_DAYS = 21
N_REGIONS = 3
REGION    = 3   # column index (1-based) we care about

TRACER_ORDER = [
    "Heat", "Salinity", "GEN_1",
    "NCS_ss1", "OXY_oxy", "NIT_amm", "NIT_nit",
    "PHS_frp", "PHS_frp_ads",
    "OGM_doc", "OGM_poc", "OGM_don", "OGM_pon", "OGM_dop", "OGM_pop",
    "PHY_mixed",
    "TRC_tr1", "TRC_tr2", "TRC_tr3", "TRC_age",
]
N_VARS = len(TRACER_ORDER)

# Order to display:
SHOW_VARS = ["VOL", "Salinity", "GEN_1", "TRC_tr1"]
UNITS = {
    "VOL":      ("m^3/s",          "m^3"),
    "Salinity": ("psu * m^3/s",    "psu * m^3"),
    "GEN_1":    ("mmol/m^3 * m^3/s", "mmol/m^3 * m^3"),
    "TRC_tr1":  ("mmol/m^3 * m^3/s", "mmol/m^3 * m^3"),
}


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
    print(f"  Loaded {n_blocks} timesteps ({n_blocks * block_len} rows) from "
          f"{path.name}")
    times = arr[::block_len, 0]
    block = arr[:, 1:].reshape(n_blocks, block_len, n_regions)
    keep = times <= period_days
    return times[keep], block[keep]


def cumtrap(rate, t_sec):
    out = np.zeros_like(rate)
    if len(t_sec) >= 2:
        out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t_sec))
    return out


def get_pos_neg_net(block, var, region_col_idx):
    """Return POS, NEG, NET arrays for the given variable, taken from
    flux.out's region-3 column.

    For VOL: pos = block[:, 1, R], neg = block[:, 2, R], net = block[:, 0, R].
    For tracers: pos = block[:, 3+2k, R], neg = block[:, 4+2k, R],
                 net = pos + neg  (tracers have no explicit NET row).
    """
    R = region_col_idx
    if var == "VOL":
        return block[:, 1, R], block[:, 2, R], block[:, 0, R]
    k = TRACER_ORDER.index(var)
    pos = block[:, 3 + 2 * k,     R]
    neg = block[:, 3 + 2 * k + 1, R]
    net = pos + neg
    return pos, neg, net


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flux_path = RUN / "outputs/flux.out"
    print(f"Parsing {flux_path} ...")
    times_d, block = parse_blocks(flux_path, N_REGIONS, N_VARS, PERIOD_DAYS)
    t_sec  = times_d * 86400.0
    t_dt   = START + (times_d * 86400 * 1e9).astype("timedelta64[ns]")
    R = REGION - 1   # 0-indexed column

    n = len(SHOW_VARS)
    fig, axes = plt.subplots(n, 2, figsize=(15, 3.2 * n),
                              sharex=True)

    for i, var in enumerate(SHOW_VARS):
        u_rate, u_cum = UNITS[var]
        pos, neg, net = get_pos_neg_net(block, var, R)
        gross = pos - neg   # POS + |NEG|  (always >= 0)

        # Cumulative
        cum_pos = cumtrap(pos, t_sec)
        cum_neg = cumtrap(neg, t_sec)
        cum_net = cumtrap(net, t_sec)
        cum_gros = cumtrap(gross, t_sec)

        # Print quick summary
        print(f"\n  --- {var} ---")
        print(f"    instantaneous range  POS: {pos.min():+.3e} .. {pos.max():+.3e}")
        print(f"    instantaneous range  NEG: {neg.min():+.3e} .. {neg.max():+.3e}")
        print(f"    instantaneous range  NET: {net.min():+.3e} .. {net.max():+.3e}")
        print(f"    cum (end)  POS: {cum_pos[-1]:+.3e}")
        print(f"    cum (end)  NEG: {cum_neg[-1]:+.3e}")
        print(f"    cum (end)  NET: {cum_net[-1]:+.3e}")
        print(f"    cum (end)  GROSS: {cum_gros[-1]:+.3e}")

        # --- left panel: instantaneous ---
        axL = axes[i, 0]
        axL.plot(t_dt, pos,   color="#dc2626", lw=0.6, label="POS  (>0 row)")
        axL.plot(t_dt, neg,   color="#1d4ed8", lw=0.6, label="NEG  (<0 row)")
        axL.plot(t_dt, net,   color="black",   lw=0.9, label="NET  = POS+NEG", alpha=0.85)
        axL.axhline(0, color="grey", lw=0.5)
        axL.set_ylabel(f"{var}\n({u_rate})", fontsize=9)
        axL.set_title(f"{var} — instantaneous (region 3, raw values)",
                       fontweight="bold", fontsize=10, loc="left")
        axL.legend(loc="upper right", fontsize=8)
        axL.grid(True, alpha=0.3)

        # --- right panel: cumulative ---
        axR = axes[i, 1]
        axR.plot(t_dt, cum_pos,  color="#dc2626", lw=1.5, label="cum POS")
        axR.plot(t_dt, cum_neg,  color="#1d4ed8", lw=1.5, label="cum NEG")
        axR.plot(t_dt, cum_net,  color="black",   lw=2.0, label="cum NET")
        axR.plot(t_dt, cum_gros, color="#7c3aed", lw=1.2, ls="--",
                  label="cum GROSS (=POS-NEG)")
        axR.axhline(0, color="grey", lw=0.5)
        axR.set_ylabel(f"cumulative\n({u_cum})", fontsize=9)
        axR.set_title(f"{var} — cumulative (region 3)",
                       fontweight="bold", fontsize=10, loc="left")
        axR.legend(loc="best", fontsize=8)
        axR.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1, 0].set_xlabel("Date")
    axes[-1, 1].set_xlabel("Date")

    fig.suptitle(f"Raw flux.out region-3 column — POS, NEG, NET, GROSS  "
                  f"(P18_flood, {PERIOD_DAYS}d)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = OUT_DIR / "flux_raw_pos_neg_net.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
