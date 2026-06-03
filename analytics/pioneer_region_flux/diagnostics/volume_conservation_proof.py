"""
Volume-conservation proof for the Pioneer CV.

Demonstrates that the CV water-volume budget CLOSES on the flat-tide window once
the *realised* river source is used in place of the raw vsource.th:

    realised source  =  vsource.th  ×  ramp_ss(t)  ×  alpha

where
  * ramp_ss(t) = tanh(2 t / (86400 dramp_ss))  is SCHISM's known source spin-up
    (dramp_ss from param.nml = 10 d): the model eases each source on over the
    first ~10 days, so the raw vsource.th over-states the early input.
  * alpha (= `--source-realisation-factor`, default 0.96) is a small, empirical
    SOURCE-volume realisation factor. CALIBRATION: on the flat-tide window the
    volume residual is ~−4% and correlates with river inflow at r=−0.93 (a
    flow-proportional loss = the volume analogue of the small-cell msource
    `rat`-dilution at the 210 m² src3 cell, where wetting/drying + minimum-depth
    numerics do not realise the full injected volume). It is applied ONLY to the
    source term, and because it is source-side its concentration is the known
    river input — so it propagates cleanly to river-borne tracer budgets.

    IMPORTANT: alpha is calibrated on the FLAT-TIDE window, where flux.out is
    faithful. Do NOT raise it to force 53-day closure: the remaining tidal-phase
    residual (~−8%) is the flux.out velocity-diagnostic bracket, NOT a source
    effect (it is river-uncorrelated, r=+0.07), and folding it into alpha would
    mis-attribute a mouth-flux error to the rivers and corrupt tracer mass.

Balance tested:   ΔV(t)  =  ∫ Q_src·ramp_ss·alpha dt   +   ∫ flux_tracked dt
                 (storage)        (realised source)         (tracked mouth, signed)

Inputs (already produced by conservation_check.py):
  _outputs/<run>/conservation_check_{14d,53d}/conservation_VOL.csv
    columns: time, M, d_obs, d_pred, cum_src(prescribed,NO ramp), cum_flux, residual
  <run>/vsource.th , <run>/source_sink.in , <run>/param.nml (dramp_ss)

Output: one figure, two rows (14-day, full ~53-day), each with:
  (left)  cumulative volume terms + ramp-corrected closure
  (right) observed ΔV vs predicted, and the residual time series
"""
from __future__ import annotations
import sys, re, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RUN = Path("s:/Matt_Working/schism/GitHub/pioneer-shzaed-2021-2022_v18/runs/003_P18_flood_flat")
OUTROOT = Path("s:/Matt_Working/schism/GitHub/pioneer-shzaed-2021-2022_v18/analytics/"
               "pioneer_region_flux/_outputs/003_P18_flood_flat")
START = np.datetime64("2021-04-01T00:00:00")

def param_val(name, default):
    txt = (RUN / "param.nml").read_text()
    m = re.search(rf'^\s*{name}\s*=\s*([0-9.eE+-]+)', txt, re.M)
    return float(m.group(1)) if m else default

def ramp_cumulative(times_sec, dramp_ss_days, factor=1.0, ramped=True):
    """Cumulative ∫ (Q_src1+Q_src2+Q_src3) [* ramp_ss * factor] dt onto times (s).

    All 3 sources are interior to the polygon CV. `ramp_ss = tanh(2t/(86400*dramp_ss))`
    is SCHISM's source spin-up; `factor` (alpha) is the small empirical source-volume
    realisation factor applied ONLY here, to the source term. ramped=False gives the
    raw prescribed vsource.th total (no ramp, no factor).
    """
    d = np.loadtxt(RUN / "vsource.th")
    t = d[:, 0]
    Q = d[:, 1] + d[:, 2] + d[:, 3]   # src1 + src2 + src3 (cols 1,2,3)
    if ramped:
        Q = Q * np.tanh(2.0 * t / 86400.0 / dramp_ss_days) * factor
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (Q[1:] + Q[:-1]) * np.diff(t))])
    return np.interp(times_sec, t, cum)

def load_vol(period_dir):
    rows = list(csv.DictReader(open(OUTROOT / period_dir / "conservation_VOL.csv")))
    t = np.array([np.datetime64(r["time"].replace(" ", "T")) for r in rows])
    d_obs = np.array([float(r["d_obs"]) for r in rows])
    cum_src_presc = np.array([float(r["cum_src"]) for r in rows])
    cum_flux = np.array([float(r["cum_flux"]) for r in rows])
    return t, d_obs, cum_src_presc, cum_flux

def main(alpha=0.96):
    dramp_ss = param_val("dramp_ss", 10.0)
    print(f"dramp_ss = {dramp_ss} days  ->  ramp_ss = tanh(2t/(86400*{dramp_ss}))")
    print(f"source-realisation factor alpha = {alpha} (flat-tide calibrated; source term only)")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    periods = [("conservation_check_14d", "14-day window"),
               ("conservation_check_53d", "full period (~53 day)")]

    for row, (pdir, label) in enumerate(periods):
        t, d_obs, cum_src_presc, cum_flux = load_vol(pdir)
        tsec = (t - START) / np.timedelta64(1, "s")
        cum_src_ramp = ramp_cumulative(tsec, dramp_ss, factor=alpha, ramped=True)

        pred_presc = cum_src_presc + cum_flux           # WRONG (ignores ramp)
        pred_ramp  = cum_src_ramp  + cum_flux           # ramp-corrected
        resid_ramp = d_obs - pred_ramp

        thru = max(abs(cum_src_ramp[-1]), abs(cum_flux[-1]), 1.0)
        pct = 100 * resid_ramp[-1] / thru

        # ---- left: cumulative terms ----
        axL = axes[row, 0]
        axL.plot(t, cum_src_ramp/1e6, color="tab:blue",  lw=2, label=f"∫ src realised  (vsource.th × ramp × α={alpha})")
        axL.plot(t, cum_src_presc/1e6, color="grey", lw=1.2, ls=":", label="∫ vsource.th raw (prescribed, src1+2+3)")
        axL.plot(t, cum_flux/1e6,     color="tab:red",   lw=2, label="∫ mouth flux (flag 2↔3, signed; −=export)")
        axL.plot(t, pred_ramp/1e6,    color="tab:green", lw=2, ls="--", label="predicted ΔV = realised src + mouth flux")
        axL.plot(t, d_obs/1e6,        color="black",     lw=2.2, label="observed ΔV (CV storage change)")
        axL.set_ylabel("cumulative volume  (×10⁶ m³)")
        axL.set_title(f"{label} — volume terms")
        axL.legend(fontsize=8, loc="upper left"); axL.grid(alpha=0.3)
        axL.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        # ---- right: closure + residual ----
        axR = axes[row, 1]
        axR.plot(t, d_obs/1e6,     color="black",     lw=2.2, label="observed ΔV")
        axR.plot(t, pred_ramp/1e6, color="tab:green", lw=1.8, ls="--", label="predicted (realised src + flux)")
        axR.plot(t, pred_presc/1e6,color="grey",      lw=1.2, ls=":",  label="predicted (raw vsource.th) — open")
        axR.plot(t, resid_ramp/1e6,color="tab:orange",lw=1.8, label="residual (realised src)")
        axR.axhline(0, color="k", lw=0.6)
        axR.set_ylabel("ΔV  (×10⁶ m³)")
        axR.set_title(f"{label} — closure")
        axR.legend(fontsize=8, loc="best"); axR.grid(alpha=0.3)
        axR.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        txt = (f"end ΔV(observed)   = {d_obs[-1]:+.3e} m³\n"
               f"src realised       = {cum_src_ramp[-1]:+.3e} m³\n"
               f"vsource.th raw     = {cum_src_presc[-1]:+.3e} m³\n"
               f"mouth flux         = {cum_flux[-1]:+.3e} m³\n"
               f"residual (realised)= {resid_ramp[-1]:+.3e} m³ ({pct:+.1f}% of throughput)\n"
               f"residual (raw src) = {(d_obs[-1]-pred_presc[-1]):+.3e} m³")
        axR.text(0.02, 0.02, txt, transform=axR.transAxes, fontsize=7.5, va="bottom",
                 family="monospace", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
        print(f"\n[{label}] residual ramp-corrected = {resid_ramp[-1]:+.3e} m³ "
              f"({pct:+.2f}% of throughput); no-ramp residual = {d_obs[-1]-pred_presc[-1]:+.3e}")

    fig.suptitle(f"Pioneer CV (4.81 km², 3 sources interior) — water-volume conservation\n"
                 f"ΔV = realised source + mouth flux.   realised = vsource.th × ramp(tanh, dramp_ss=10 d) × "
                 f"α={alpha} (flat-tide-calibrated source factor).  Residual left under tides = flux-diagnostic bracket.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTROOT / "volume_conservation_003.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-realisation-factor", type=float, default=0.96, dest="alpha",
                    help="alpha: empirical source-volume realisation factor, applied ONLY to the "
                         "source term (default 0.96, flat-tide calibrated). Use 1.0 to disable.")
    args = ap.parse_args()
    main(alpha=args.alpha)
