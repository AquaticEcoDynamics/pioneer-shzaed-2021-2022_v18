"""
HYPOTHETICAL end-member salt budget for the Pioneer CV (run 003).

The real flux.out salt flux is first-order UPWIND (donor-cell salinity per layer)
and does NOT close the CV salt budget. This script asks a different question:

    If we keep the flux.out VOLUME flux through the mouth (which DID close the
    volume budget), but assign END-MEMBER salinities to it -
        inflowing water  (flux.out VOL > 0)  carries OCEAN salinity
        outflowing water (flux.out VOL < 0)  carries ESTUARY salinity
    - plus the three river sources' own salt load, does the budget then close
    to the observed change in CV salt storage?

This is the classic estuarine box-model mechanism: if the gross tidal exchange
is large and the two end-members differ, the box can IMPORT salt even while the
NET volume flux is export (saltier water in at depth/flood, fresher water out).

End-members:
  ocean_salinity   : constant, OCEAN_S below (marine end-member, ~35 psu)
  estuary_salinity : the CV's OWN mean salinity S_cv(t) = M_salt/M_vol at each
                     instant (rises 9 -> 35 psu over the run). A well-mixed box
                     exports water at its own concentration, so this is the
                     physically correct outflow end-member, not a fixed number.

Outputs, per window (14-day flat tide; full ~53-day), in TONNES of salt:
  - cumulative INCOMING  salt  (+ flux.out VOL x ocean_salinity)
  - cumulative OUTGOING  salt  (- flux.out VOL x estuary_salinity)
  - cumulative river SOURCE salt (src1+2+3, from conservation_Salinity.csv)
  - cumulative HYPOTHETICAL TOTAL (= source + incoming + outgoing)
  - the OBSERVED change in CV salt storage (the target the total should match)
  - for reference, the real UPWIND flux.out salt (cum_flux) that does NOT close
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "nutrient_budget"))
from run_config import active_run
from core.boundary_fluxes import parse_flux_out

CFG = active_run()
RUN = CFG.run_dir
START = CFG.start
OUTD = CFG.out_dir                       # _outputs/003_P18_flood_flat
TONNE = 1e-3                             # psu*m^3 -> tonnes of salt (rho~1000)

# --- the one knob: ocean (inflow) end-member salinity [psu] -----------------
OCEAN_S = 35.0


def cumtrap(rate, t_sec):
    if len(t_sec) < 2:
        return np.zeros_like(rate)
    incr = 0.5 * (rate[1:] + rate[:-1]) * np.diff(t_sec)
    return np.concatenate([[0.0], np.cumsum(incr)])


def main():
    # flux.out: region-3 VOLUME flux (m^3/s, + = INTO CV) over the whole run
    flux_da = parse_flux_out(RUN / "outputs" / "flux.out", n_regions=None)
    t_d = flux_da["time_days"].values
    vol = flux_da.sel(tracer="VOL", region=3).values        # m^3/s, +into CV

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for row, days in enumerate((14, 53)):
        keep = t_d <= days
        t_s = t_d[keep] * 86400.0
        Qv = vol[keep]
        Qin = np.where(Qv > 0, Qv, 0.0)        # inflow  (m^3/s)
        Qout = np.where(Qv < 0, Qv, 0.0)       # outflow (m^3/s, negative)
        t_flux = START + (t_s * 1e9).astype("timedelta64[ns]")

        # CV storage + source salt + CV-mean salinity, from the budget CSVs
        cdir = OUTD / f"conservation_check_{days}d"
        sal = pd.read_csv(cdir / "conservation_Salinity.csv")
        volc = pd.read_csv(cdir / "conservation_VOL.csv")
        t_cv_s = (pd.to_datetime(sal["time"]).values - START) / np.timedelta64(1, "s")
        S_cv = sal["M"].values / volc["M"].values             # CV mean salinity (psu)
        S_cv_f = np.interp(t_s, t_cv_s, S_cv)                  # onto flux time base
        cum_src = np.interp(t_s, t_cv_s, sal["cum_src"].values)
        d_obs = np.interp(t_s, t_cv_s, sal["d_obs"].values)
        cum_upwind = np.interp(t_s, t_cv_s, sal["cum_flux"].values)

        # hypothetical end-member salt loads (psu*m^3 -> tonnes)
        cum_in = cumtrap(Qin * OCEAN_S, t_s) * TONNE          # + incoming (ocean S)
        cum_out = cumtrap(Qout * S_cv_f, t_s) * TONNE         # - outgoing (estuary S)
        cum_src_t = cum_src * TONNE
        cum_total = cum_src_t + cum_in + cum_out              # hypothetical total
        d_obs_t = d_obs * TONNE
        cum_upwind_t = cum_upwind * TONNE

        # ---- left: cumulative hypothetical salt loads (tonnes) ----
        axL = axes[row, 0]
        axL.plot(t_flux, cum_in, color="tab:blue", lw=2,
                 label=f"+ incoming  (ocean {OCEAN_S:g} psu): {cum_in[-1]:+,.0f} t")
        axL.plot(t_flux, cum_out, color="tab:red", lw=2,
                 label=f"- outgoing  (estuary S(t)): {cum_out[-1]:+,.0f} t")
        axL.plot(t_flux, cum_src_t, color="tab:green", lw=1.6,
                 label=f"river sources (1+2+3): {cum_src_t[-1]:+,.0f} t")
        axL.plot(t_flux, cum_total, color="black", lw=2.6,
                 label=f"HYPOTHETICAL total: {cum_total[-1]:+,.0f} t")
        axL.plot(t_flux, d_obs_t, color="tab:purple", lw=2.2, ls="--",
                 label=f"OBSERVED Δstorage: {d_obs_t[-1]:+,.0f} t")
        axL.plot(t_flux, cum_upwind_t, color="grey", lw=1.2, ls=":",
                 label=f"(real upwind flux.out: {cum_upwind_t[-1]:+,.0f} t)")
        axL.axhline(0, color="k", lw=0.6)
        axL.set_ylabel("cumulative salt  (tonnes)")
        axL.set_title(f"{days}-day - hypothetical end-member salt budget", fontweight="bold")
        axL.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); axL.grid(alpha=0.3)
        axL.legend(fontsize=8, loc="best")

        # ---- right: the inputs (volume flux + end-member salinities) ----
        axR = axes[row, 1]
        axR.plot(t_flux, Qv, color="tab:gray", lw=0.8, label="flux.out VOL (m³/s, +in)")
        axR.axhline(0, color="k", lw=0.6); axR.set_ylabel("mouth volume flux (m³/s)")
        axR.grid(alpha=0.3)
        ax2 = axR.twinx()
        ax2.plot(t_flux, S_cv_f, color="tab:red", lw=2, label="estuary S(t) (CV mean)")
        ax2.axhline(OCEAN_S, color="tab:blue", lw=2, ls="--", label=f"ocean S = {OCEAN_S:g}")
        ax2.set_ylabel("salinity (psu)"); ax2.set_ylim(0, 38)
        axR.set_title(f"{days}-day - inputs to the box model", fontweight="bold")
        axR.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        l1, la1 = axR.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
        axR.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper left")

        # ---- console summary ----
        miss = d_obs_t[-1] - cum_total[-1]
        print(f"\n=== {days}-day box model (tonnes) ===")
        print(f"  + incoming (ocean {OCEAN_S:g} psu)   : {cum_in[-1]:+,.0f}")
        print(f"  - outgoing (estuary S(t))      : {cum_out[-1]:+,.0f}")
        print(f"  river sources                  : {cum_src_t[-1]:+,.0f}")
        print(f"  HYPOTHETICAL total             : {cum_total[-1]:+,.0f}")
        print(f"  OBSERVED Δstorage              : {d_obs_t[-1]:+,.0f}")
        print(f"  miss (obs - hypothetical)      : {miss:+,.0f}  "
              f"({100*miss/d_obs_t[-1] if d_obs_t[-1] else float('nan'):+.1f}% of obs)")
        print(f"  [ref] real upwind flux.out     : {cum_upwind_t[-1]:+,.0f}")
        print(f"  gross volume in/out            : {cumtrap(Qin,t_s)[-1]:+.3e} / {cumtrap(Qout,t_s)[-1]:+.3e} m³")

    fig.suptitle(
        f"Pioneer CV - HYPOTHETICAL end-member salt budget.  Inflow @ ocean {OCEAN_S:g} psu, "
        f"outflow @ estuary's own S(t), plus river sources.\n"
        "Tests whether end-member box accounting (vs flux.out's upwind salt) can import the salt "
        "needed to match the observed CV salinification.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTD / "salt_box_hypothetical_003.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
