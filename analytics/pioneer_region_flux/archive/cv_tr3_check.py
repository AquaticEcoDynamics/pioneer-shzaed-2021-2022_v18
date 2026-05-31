"""
CV mass-balance check for TRC_tr3 on the pioneer_estuary control volume.

For tr3:
  - No point sources (msource = 0 for all 14 srcs)
  - Ocean BC = 1.0 mmol/m³
  - No internal reactions / sinks (conservative tracer)

If flux.out row 3 (Pioneer Mouth: region 3 ↔ region 2) is the ONLY relevant
boundary of the CV (per pioneer_estuary.yaml), then:

    M_CV(t)  =  M_CV(0)  +  ∫₀ᵗ (flux.out row 3, tracer=TRC_tr3) ds

i.e., the volume-integrated tr3 mass in the CV at any time equals its
initial value plus the cumulative net mass advected through the Pioneer
Mouth.

Plots:
  (a) M_CV(t) actual (from scribed TRC_tr3 × lh × area)
      vs    M_CV(0) + cum row-3 flux (predicted from flux.out)
  (b) residual (actual − predicted)
  (c) V_CV(t) and the instantaneous mouth flux rate

Any persistent residual implies tr3 mass enters/leaves the CV via paths
that flux.out doesn't see — the (-1 ↔ region) and unflagged interfaces
identified in the fluxflag audit.
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core import geometry
from core.boundary_fluxes import parse_flux_out

RUN = Path("s:/Matt_Working/schism/P18_flood")
OUT_PNG = Path(__file__).parent / "cv_tr3_check.png"
START = np.datetime64("2021-04-01")
PERIOD_DAYS = 14


def build_cv_averager(hgrid, cv_elements):
    elements = hgrid["elements"]
    cv_nodes_set = set()
    cv_elem_node_ids = []
    for ei in cv_elements:
        nc, *ids = elements[ei]
        nodes_e = [int(n) for n in ids[:int(nc)]]
        cv_nodes_set.update(nodes_e)
        cv_elem_node_ids.append((nc, nodes_e))
    cv_nodes_union = np.array(sorted(cv_nodes_set), dtype=int)
    pos = {n: i for i, n in enumerate(cv_nodes_union.tolist())}
    rows, cols, data = [], [], []
    for k, (nc, nodes_e) in enumerate(cv_elem_node_ids):
        for n in nodes_e:
            rows.append(k); cols.append(pos[n]); data.append(1.0 / nc)
    A_avg = csr_matrix(
        (np.asarray(data, np.float32),
         (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
        shape=(len(cv_elements), len(cv_nodes_union)), dtype=np.float32,
    )
    return cv_nodes_union, A_avg


def main():
    print("=== Loading CV geometry ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    cv_path = Path(__file__).parent / "configs/cv_pioneer_estuary_elements.txt"
    cv_ids = [int(l.split('#',1)[0].strip())
              for l in open(cv_path)
              if l.strip() and not l.strip().startswith('#')]
    cv_elements = np.array(cv_ids, dtype=int) - 1
    cv_area = hgrid["areas"][cv_elements].astype(np.float64)
    n_cv = len(cv_elements)
    cv_nodes_union, A_avg = build_cv_averager(hgrid, cv_elements)
    print(f"  {n_cv} CV elements, area = {cv_area.sum()/1e6:.2f} km²")

    # ------------------------------------------------------------------
    # CV V(t)  + M_CV_tr3(t)
    # ------------------------------------------------------------------
    print("\n=== ENV_layer_ht for CV → V(t) and lh_cv ===")
    ds_cmb = geometry.open_cmb_concat(RUN / "outputs", n_stacks=PERIOD_DAYS + 1)
    t_cmb = ds_cmb["time"].values
    _, uc = np.unique(t_cmb, return_index=True)
    if len(uc) != len(t_cmb):
        ds_cmb = ds_cmb.isel(time=np.sort(uc))
    keep_c = (ds_cmb["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    ds_cmb = ds_cmb.isel(time=keep_c)
    lh = ds_cmb["ENV_layer_ht"]
    elem_dim = next(d for d in lh.dims if "face" in d.lower())
    layer_dim_c = next(d for d in lh.dims if "layer" in d.lower() or "vgrid" in d.lower())
    lh_cv = lh.isel({elem_dim: cv_elements}).transpose(
        "time", layer_dim_c, elem_dim).values
    lh_cv = np.where(np.isfinite(lh_cv) & (lh_cv > 0), lh_cv, 0.0).astype(np.float64)
    times = ds_cmb["time"].values
    t_sec = (times - times[0]) / np.timedelta64(1, "s")
    V_cv = (lh_cv * cv_area[None, None, :]).sum(axis=(1, 2))
    print(f"  V_CV range: [{V_cv.min()/1e6:.2f}, {V_cv.max()/1e6:.2f}] M m³")

    print("\n=== Scribed TRC_tr3 → CV-integrated mass ===")
    da_tr3 = geometry.open_scribed_concat(RUN / "outputs", "TRC_tr3",
                                          n_stacks=PERIOD_DAYS + 1)
    tt = da_tr3["time"].values
    _, uq = np.unique(tt, return_index=True)
    if len(uq) != len(tt):
        da_tr3 = da_tr3.isel(time=np.sort(uq))
    keep_s = (da_tr3["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    da_tr3 = da_tr3.isel(time=keep_s)
    node_dim = next(d for d in da_tr3.dims if "node" in d.lower())
    layer_dim_s = next((d for d in da_tr3.dims
                       if "vgrid" in d.lower() or "layer" in d.lower()), None)
    da_cv = da_tr3.isel({node_dim: cv_nodes_union.tolist()}).compute()
    arr = da_cv.values
    dims = list(da_cv.dims)
    t_ax = dims.index(next(d for d in dims if d.lower() == "time"))
    n_ax = dims.index(node_dim)
    l_ax = dims.index(layer_dim_s)
    arr_p = np.transpose(arr, (t_ax, l_ax, n_ax))
    arr_p = np.where(np.isfinite(arr_p) & (np.abs(arr_p) <= 1e30), arr_p, 0.0)
    T_s, L_s, N_s = arr_p.shape
    arr_flat = arr_p.reshape(T_s * L_s, N_s).astype(np.float32)
    elem_flat = arr_flat @ A_avg.T
    elem_arr = elem_flat.reshape(T_s, L_s, n_cv).astype(np.float64)
    T_use = min(T_s, lh_cv.shape[0])
    M_cv_tr3 = (elem_arr[:T_use] * lh_cv[:T_use]
                * cv_area[None, None, :]).sum(axis=(1, 2))
    times = times[:T_use]
    t_sec = t_sec[:T_use]
    V_cv = V_cv[:T_use]
    print(f"  M_CV_tr3 range: [{M_cv_tr3.min()/1e6:.3f}, {M_cv_tr3.max()/1e6:.3f}] M mmol")
    print(f"  M_CV_tr3 (initial)  = {M_cv_tr3[0]/1e6:.3f} M mmol")
    print(f"  M_CV_tr3 (final 14d) = {M_cv_tr3[-1]/1e6:.3f} M mmol")
    print(f"  ΔM_CV_tr3 = {(M_cv_tr3[-1]-M_cv_tr3[0])/1e6:.3f} M mmol")

    # ------------------------------------------------------------------
    # flux.out row 3 for tr3 (sign = positive into CV per yaml)
    # ------------------------------------------------------------------
    print("\n=== Parsing flux.out → tr3 row 3 (Pioneer Mouth) ===")
    flux_da = parse_flux_out(RUN / "outputs" / "flux.out", n_regions=9)
    rate_mouth = flux_da.sel(tracer="TRC_tr3", region=3).values   # mmol/s, +ve INTO CV
    t_flux_d = flux_da["time_days"].values
    keep_fx = t_flux_d <= PERIOD_DAYS
    rate_mouth = rate_mouth[keep_fx]
    t_flux_s = t_flux_d[keep_fx] * 86400.0
    flux_times = START + (t_flux_d[keep_fx] * 86400 * 1e9).astype("timedelta64[ns]")
    # Cumulative net mass into CV via mouth (positive = inflow into CV)
    cum_mouth = np.concatenate(
        [[0], np.cumsum(0.5 * (rate_mouth[1:] + rate_mouth[:-1]) * np.diff(t_flux_s))]
    )
    print(f"  cum tr3 mouth flux over 14d (net INTO CV) = {cum_mouth[-1]/1e6:.3f} M mmol")
    print(f"  mean rate = {rate_mouth.mean():.2f} mmol/s")
    print(f"  rate range: [{rate_mouth.min():.2f}, {rate_mouth.max():.2f}] mmol/s")

    # Re-grid cum_mouth onto t_sec for direct comparison with M_CV
    cum_mouth_aligned = np.interp(t_sec, t_flux_s, cum_mouth)
    predicted_M = M_cv_tr3[0] + cum_mouth_aligned
    residual = M_cv_tr3 - predicted_M

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("=== CV MASS BALANCE FOR TRC_tr3 (pioneer_estuary CV, 14 d) ===")
    print("=" * 70)
    print(f"M_CV(0)                                  : {M_cv_tr3[0]/1e6:>10.3f} M mmol")
    print(f"M_CV(14d)                                : {M_cv_tr3[-1]/1e6:>10.3f} M mmol")
    print(f"Observed ΔM_CV  = M_CV(14d) − M_CV(0)    : "
          f"{(M_cv_tr3[-1]-M_cv_tr3[0])/1e6:>10.3f} M mmol")
    print(f"Predicted ΔM_CV = cum mouth flux (row 3) : "
          f"{cum_mouth[-1]/1e6:>10.3f} M mmol")
    print(f"Residual = observed − predicted          : "
          f"{residual[-1]/1e6:>10.3f} M mmol  "
          f"({residual[-1]/((M_cv_tr3[-1]-M_cv_tr3[0])+1e-30)*100:+.1f}% of observed)")
    print()
    print("Interpretation:")
    print("  - If residual ≈ 0: flux.out row 3 captures all CV boundary exchange.")
    print("  - If residual > 0: mass entered CV via paths flux.out doesn't see")
    print("                     (the unflagged-region interfaces from earlier audit).")
    print("  - If residual < 0: mass left CV via untracked paths.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    axA, axB, axC = axes

    axA.plot(times, M_cv_tr3 / 1e6, color="#1d4ed8", lw=2.0,
             label="M_CV_tr3  observed  (∫ tr3 dV over CV)")
    axA.plot(times, predicted_M / 1e6, color="#dc2626", lw=2.0,
             label="Predicted = M_CV(0) + ∫ flux.out row 3")
    axA.fill_between(times, predicted_M / 1e6, M_cv_tr3 / 1e6,
                     where=M_cv_tr3 > predicted_M, color="#16a34a", alpha=0.15,
                     label="observed > predicted (untracked inflow)")
    axA.fill_between(times, predicted_M / 1e6, M_cv_tr3 / 1e6,
                     where=M_cv_tr3 < predicted_M, color="#f59e0b", alpha=0.15,
                     label="observed < predicted (untracked outflow)")
    axA.set_ylabel("Mass (M mmol)")
    axA.grid(True, ls=":", alpha=0.4)
    axA.legend(loc="best", fontsize=10)
    axA.set_title("(a) CV TRC_tr3 mass: observed vs predicted from flux.out mouth row",
                  fontweight="bold", loc="left", fontsize=11)
    axA.axhline(0, color="#999", lw=0.4)

    axB.plot(times, residual / 1e6, color="#7c2d12", lw=2.0,
             label="Residual = observed − predicted")
    axB.set_ylabel("Residual (M mmol)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="best", fontsize=10)
    axB.set_title("(b) Mass-balance residual: M_CV(t) − [M_CV(0) + cum mouth flux]",
                  fontweight="bold", loc="left", fontsize=11)
    axB.axhline(0, color="#999", lw=0.4)

    # dM/dt vs mouth flux rate
    dM_dt = np.gradient(M_cv_tr3, t_sec)  # mmol/s
    axC.plot(times, dM_dt, color="#1d4ed8", lw=1.4, alpha=0.7,
             label="dM_CV/dt  (observed)")
    axC.plot(flux_times, rate_mouth, color="#dc2626", lw=1.2, alpha=0.7,
             label="flux.out row 3 rate  (tr3 mass into CV)")
    axC.set_ylabel("Rate (mmol/s)")
    axC.set_xlabel("Date (model time)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="best", fontsize=10)
    axC.set_title("(c) Instantaneous rates: dM_CV/dt vs mouth-row tr3 flux",
                  fontweight="bold", loc="left", fontsize=11)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axC.axhline(0, color="#999", lw=0.4)

    fig.suptitle("TRC_tr3 CV mass balance (P18_flood, pioneer_estuary CV)  "
                 "— tests whether flux.out row 3 captures the full CV-boundary flux",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
