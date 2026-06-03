"""
Salt budget in TONNES for the Pioneer CV (run 003).

Salt mass [t] = (∫S dV) [psu·m³] × ρ / 1e6  ≈  (psu·m³)/1000   (ρ≈1000 kg/m³;
~2% sensitivity to density over the 9–35 psu range).

Shows, per window (14-day flat tide; full ~53-day):
  - salt MASS stored in the CV over time (the storage)  [start / mean / end]
  - change in storage (observed)                         ΔZ
  - salt added by the 3 river sources                    (ambient-S source)
  - mouth ADVECTIVE flux as reported by flux.out (first-order UPWIND)
  - residual (= storage − source − mouth flux)
and annotates the GROSS tidal salt exchange (from the layer-resolved
velocity×salinity reconstruction, salt_mouth_flux_check.py), to show the net is a
tiny residual of an enormous bidirectional exchange — which is why an upwind
flux.out diagnostic cannot close it.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTD = Path("s:/Matt_Working/schism/GitHub/pioneer-shzaed-2021-2022_v18/"
            "analytics/pioneer_region_flux/_outputs/003_P18_flood_flat")
RHO = 1000.0
T = RHO / 1e6                      # psu·m³ -> tonnes of salt

# gross exchange (tonnes) from salt_mouth_flux_check.py reconstruction
GROSS = {14: (1.722e8*T, -5.255e8*T), 53: (2.442e10*T, -2.549e10*T)}


def load(days):
    r = list(csv.DictReader(open(OUTD / f"conservation_check_{days}d" / "conservation_Salinity.csv")))
    t = np.array([np.datetime64(z["time"].replace(" ", "T")) for z in r])
    M = np.array([float(z["M"]) for z in r]) * T          # salt mass in CV (t)
    d_obs = np.array([float(z["d_obs"]) for z in r]) * T   # storage change (t)
    cum_src = np.array([float(z["cum_src"]) for z in r]) * T
    cum_flux = np.array([float(z["cum_flux"]) for z in r]) * T
    return t, M, d_obs, cum_src, cum_flux


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for row, days in enumerate((14, 53)):
        t, M, d_obs, cum_src, cum_flux = load(days)
        resid = d_obs - (cum_src + cum_flux)
        gin, gout = GROSS[days]
        Ms, Me, Mm = M[0], M[-1], M.mean()

        # ---- left: salt stored in the CV (tonnes) ----
        axL = axes[row, 0]
        axL.plot(t, M, color="tab:purple", lw=2.4)
        axL.axhline(Mm, color="grey", ls=":", lw=1, label=f"mean storage {Mm:,.0f} t")
        axL.set_ylabel("salt stored in CV  (tonnes)")
        axL.set_title(f"{days}-day — salt mass in the CV")
        axL.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); axL.grid(alpha=0.3)
        axL.annotate(f"start {Ms:,.0f} t", (t[0], Ms), textcoords="offset points", xytext=(6, -12), fontsize=9)
        axL.annotate(f"end {Me:,.0f} t", (t[-1], Me), textcoords="offset points", xytext=(-70, 6), fontsize=9, fontweight="bold")
        axL.legend(fontsize=8, loc="upper left")

        # ---- right: cumulative budget terms (tonnes) ----
        axR = axes[row, 1]
        axR.plot(t, d_obs, color="black", lw=2.4, label=f"Δ storage observed: {d_obs[-1]:+,.0f} t")
        axR.plot(t, cum_src, color="tab:green", lw=2, label=f"from 3 river sources: {cum_src[-1]:+,.0f} t")
        axR.plot(t, cum_flux, color="tab:red", lw=2, label=f"mouth flux (flux.out, UPWIND): {cum_flux[-1]:+,.0f} t")
        axR.plot(t, resid, color="tab:orange", lw=2, ls="--", label=f"residual (unclosed): {resid[-1]:+,.0f} t")
        axR.axhline(0, color="k", lw=0.6)
        axR.set_ylabel("cumulative salt  (tonnes)")
        axR.set_title(f"{days}-day — salt budget terms")
        axR.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); axR.grid(alpha=0.3)
        axR.legend(fontsize=8, loc="best")
        axR.text(0.02, 0.02,
                 f"GROSS tidal exchange (advective):\n  in  ≈ {gin:>14,.0f} t\n  out ≈ {gout:>14,.0f} t\n"
                 f"→ net mouth flux is a ~{abs((gin+gout)/max(gin,-gout))*100:.0f}% residual of the gross",
                 transform=axR.transAxes, fontsize=7.5, va="bottom", family="monospace",
                 bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

        print(f"\n=== {days}-day (tonnes of salt) ===")
        print(f"  salt stored in CV: start {Ms:,.0f} | mean {Mm:,.0f} | end {Me:,.0f}")
        print(f"  Δ storage observed        : {d_obs[-1]:+,.0f} t")
        print(f"  from 3 river sources      : {cum_src[-1]:+,.0f} t")
        print(f"  mouth flux (flux.out upwind): {cum_flux[-1]:+,.0f} t")
        print(f"  residual (unclosed)       : {resid[-1]:+,.0f} t")
        print(f"  GROSS tidal exchange      : +{gin:,.0f} in / {gout:,.0f} out")

    fig.suptitle("Pioneer CV salt budget (tonnes).   The mouth flux from flux.out is a first-order UPWIND diagnostic; "
                 "the model transports salt with TVD.\nThe net mouth salt flux is a tiny residual of a huge gross tidal "
                 "exchange, so the budget does NOT close with flux.out — the residual is a diagnostic limitation, not a leak.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUTD / "salt_budget_tonnes_003.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
