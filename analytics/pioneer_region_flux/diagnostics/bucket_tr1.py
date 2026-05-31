"""
Independent bucket-model cross-check for TRC_tr1 in the pioneer_estuary CV.

TRC_tr1 is a conservative tracer (no AED reactions). At sources:
  - src #3 (Dumbleton): c = 1.0 mmol/m³ at all times
  - src #1, #2:         c = 0
Ocean BC (AED_14.th):    c = 0
So mass input rate to the CV is simply Q_src3(t) × 1.0 [mmol/s].

Well-mixed bucket mass balance (with plug-flow outflow):
    dM/dt   = Q_src3 × 1.0       —  Q_bdy_out × C(t)
    dV/dt   = Q_src_total       +  Q_bdy_net
    Q_bdy_net = dV/dt - Q_src_total       (volume residual)
    Q_bdy_out = max(0, -Q_bdy_net)        (only counts when net leaves CV)
    C(t)    = M(t) / V(t)

This is an UPPER BOUND on tr1 retention (plug-flow ignores re-entry of
tidally-exchanged water that already carries CV-mean concentration). If
SCHISM's CV-mean C is *lower* than the bucket, mass is being lost beyond
what advection can explain — that's the smoking gun for a model issue.

Outputs:
    bucket_tr1.png  — 4-panel comparison
    bucket_tr1.csv  — timeseries of all bucket internals
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
from run_config import active_run
from core import geometry
from core.boundary_fluxes import parse_flux_out


CFG = active_run()
RUN = CFG.run_dir
OUT_PNG = CFG.out_dir / "bucket_tr1.png"
OUT_CSV = CFG.out_dir / "bucket_tr1.csv"
START = CFG.start
PERIOD_DAYS = CFG.period_days
def load_cv_and_averager():
    """Replicates the CV setup from per_variable_plots.py."""
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    elements = hgrid["elements"]
    cv_path = CFG.cv_elements
    cv_ids = []
    with open(cv_path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                cv_ids.append(int(line))
    cv_elements = np.array(cv_ids, dtype=int) - 1
    cv_area = hgrid["areas"][cv_elements].astype(np.float64)
    cv_nodes_set = set()
    cv_elem_node_ids = []
    for ei in cv_elements:
        nc, *ids = elements[ei]
        nodes_e = [int(n) for n in ids[:int(nc)]]
        cv_nodes_set.update(nodes_e)
        cv_elem_node_ids.append((nc, nodes_e))
    cv_nodes_union = np.array(sorted(cv_nodes_set), dtype=int)
    node_pos = {n: i for i, n in enumerate(cv_nodes_union.tolist())}
    rows, cols, data = [], [], []
    for k, (nc, nodes_e) in enumerate(cv_elem_node_ids):
        for n in nodes_e:
            rows.append(k); cols.append(node_pos[n]); data.append(1.0 / nc)
    A_avg = csr_matrix(
        (np.asarray(data, np.float32),
         (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
        shape=(len(cv_elements), len(cv_nodes_union)), dtype=np.float32,
    )
    return hgrid, cv_elements, cv_area, cv_nodes_union, A_avg


def main():
    print("=== Loading CV geometry + averaging matrix ===")
    hgrid, cv_elements, cv_area, cv_nodes_union, A_avg = load_cv_and_averager()
    n_cv = len(cv_elements)
    print(f"  {n_cv} CV elements, area = {cv_area.sum()/1e6:.2f} km²")

    # ------------------------------------------------------------------
    # 1) CV volume V(t) from ENV_layer_ht
    # ------------------------------------------------------------------
    print("\n=== Loading ENV_layer_ht for V(t) ===")
    ds_cmb = geometry.open_cmb_concat(RUN / "outputs", n_stacks=PERIOD_DAYS + 1)
    t_cmb = ds_cmb["time"].values
    _, uc = np.unique(t_cmb, return_index=True)
    if len(uc) != len(t_cmb):
        ds_cmb = ds_cmb.isel(time=np.sort(uc))
    keep_c = (ds_cmb["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    ds_cmb = ds_cmb.isel(time=keep_c)
    lh = ds_cmb["ENV_layer_ht"]
    elem_dim = next(d for d in lh.dims if "face" in d.lower())
    layer_dim = next(d for d in lh.dims if "layer" in d.lower() or "vgrid" in d.lower())
    lh_cv = lh.isel({elem_dim: cv_elements}).transpose("time", layer_dim, elem_dim).values
    lh_cv = np.where(np.abs(lh_cv) <= 1e30, lh_cv, 0.0).astype(np.float64)
    V_t = (lh_cv * cv_area[None, None, :]).sum(axis=(1, 2))    # m³
    times = ds_cmb["time"].values
    t_sec = (times - times[0]) / np.timedelta64(1, "s")
    print(f"  V(t): n={len(V_t)}, range = [{V_t.min()/1e6:.2f}, {V_t.max()/1e6:.2f}] M m³")

    # ------------------------------------------------------------------
    # 2) Source Q + tr1 concentrations on the V(t) time axis
    # ------------------------------------------------------------------
    print("\n=== Loading vsource.th / msource.th for srcs 1, 2, 3 ===")
    vs = np.loadtxt(RUN / "vsource.th")
    ms = np.loadtxt(RUN / "msource.th")
    t_vs = vs[:, 0]
    Q_all = vs[:, 1:]                 # (n_t, n_src)
    N_SRC = 14
    # TRC_tr1 is AED tracer k=14
    def msrc_col(k, si): return 1 + 2 * N_SRC + (k - 1) * N_SRC + si
    t_ms = ms[:, 0]
    c_tr1_src3 = ms[:, msrc_col(14, 2)]
    print(f"  src #3 c_tr1 mean = {c_tr1_src3.mean():.3f}  (expected 1.0)")

    # Interpolate Q (srcs 1, 2, 3) and c_tr1_src3 onto V(t)'s axis
    Q1 = np.interp(t_sec, t_vs, Q_all[:, 0])
    Q2 = np.interp(t_sec, t_vs, Q_all[:, 1])
    Q3 = np.interp(t_sec, t_vs, Q_all[:, 2])
    c3 = np.interp(t_sec, t_ms, c_tr1_src3)
    Q_src_total = Q1 + Q2 + Q3        # m³/s
    F_in_src = Q3 * c3                # mmol/s   (srcs 1, 2 contribute 0)
    print(f"  Q_src_total mean = {Q_src_total.mean():.2f} m^3/s, "
          f"peak = {Q_src_total.max():.2f} m^3/s")
    print(f"  F_in (tr1) mean = {F_in_src.mean():.2f} mmol/s, "
          f"peak = {F_in_src.max():.2f} mmol/s")
    print(f"  pulse window total tr1 in = {np.trapezoid(F_in_src[(t_sec>=3*86400)&(t_sec<=7*86400)], t_sec[(t_sec>=3*86400)&(t_sec<=7*86400)])/1e6:.2f} M mmol")

    # ------------------------------------------------------------------
    # 3) Net boundary flow from volume residual: Q_bdy_net = dV/dt - ΣQ_src
    # ------------------------------------------------------------------
    dV_dt = np.gradient(V_t, t_sec)          # m³/s, centred differences
    Q_bdy_net = dV_dt - Q_src_total           # >0: net inflow from ocean
    Q_bdy_out = np.maximum(0.0, -Q_bdy_net)   # >0: net outflow to ocean
    print(f"\n  Q_bdy_net mean = {Q_bdy_net.mean():.2f} m³/s "
          f"(residual; ideal = small)")
    print(f"  Q_bdy_out mean = {Q_bdy_out.mean():.2f} m³/s "
          f"(magnitude of net outflow events)")
    print(f"  Q_bdy_net range = [{Q_bdy_net.min():.0f}, {Q_bdy_net.max():.0f}] m³/s "
          f"(tidal swing)")

    # ------------------------------------------------------------------
    # 4) Integrate bucket: dM/dt = F_in - Q_bdy_out × C   (explicit Euler)
    # ------------------------------------------------------------------
    M_bucket = np.zeros_like(V_t)
    C_bucket = np.zeros_like(V_t)
    M = 0.0
    for i in range(len(t_sec) - 1):
        C = M / V_t[i] if V_t[i] > 0 else 0.0
        dMdt = F_in_src[i] - Q_bdy_out[i] * C
        dt = t_sec[i+1] - t_sec[i]
        M = max(0.0, M + dMdt * dt)
        M_bucket[i+1] = M
        C_bucket[i+1] = M / V_t[i+1] if V_t[i+1] > 0 else 0.0
    print(f"\n  Bucket final mass = {M_bucket[-1]/1e6:.2f} M mmol, "
          f"peak = {M_bucket.max()/1e6:.2f} M mmol")
    print(f"  Bucket final C    = {C_bucket[-1]:.3f} mmol/m³, "
          f"peak = {C_bucket.max():.3f} mmol/m³")

    # ------------------------------------------------------------------
    # 5) SCHISM CV-mean tr1: ∫ tr1 dV / V(t)
    # ------------------------------------------------------------------
    print("\n=== Computing SCHISM CV-mean TRC_tr1 ===")
    da_tr1 = geometry.open_scribed_concat(RUN / "outputs", "TRC_tr1",
                                          n_stacks=PERIOD_DAYS + 1)
    tt = da_tr1["time"].values
    _, uq = np.unique(tt, return_index=True)
    if len(uq) != len(tt):
        da_tr1 = da_tr1.isel(time=np.sort(uq))
    keep_s = (da_tr1["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    da_tr1 = da_tr1.isel(time=keep_s)
    node_dim = next(d for d in da_tr1.dims if "node" in d.lower())
    layer_dim_s = next((d for d in da_tr1.dims
                        if "vgrid" in d.lower() or "layer" in d.lower()), None)
    da_cv = da_tr1.isel({node_dim: cv_nodes_union.tolist()}).compute()
    arr = da_cv.values
    dims = list(da_cv.dims)
    t_ax = dims.index(next(d for d in dims if d.lower() == "time"))
    n_ax = dims.index(node_dim)
    l_ax = dims.index(layer_dim_s)
    arr_p = np.transpose(arr, (t_ax, l_ax, n_ax))            # (T, L, N)
    arr_p = np.where(np.abs(arr_p) <= 1e30, arr_p, 0.0)
    T_s, L_s, N_s = arr_p.shape
    arr_flat = arr_p.reshape(T_s * L_s, N_s).astype(np.float32)
    elem_flat = arr_flat @ A_avg.T
    elem_arr = elem_flat.reshape(T_s, L_s, n_cv).astype(np.float64)
    T_use = min(T_s, lh_cv.shape[0])
    M_schism = (elem_arr[:T_use] * lh_cv[:T_use]
                * cv_area[None, None, :]).sum(axis=(1, 2))    # mmol
    C_schism = M_schism / V_t[:T_use]                          # mmol/m³
    print(f"  SCHISM final mass = {M_schism[-1]/1e6:.2f} M mmol, "
          f"peak = {M_schism.max()/1e6:.2f} M mmol")
    print(f"  SCHISM final C    = {C_schism[-1]:.3f} mmol/m³, "
          f"peak = {C_schism.max():.3f} mmol/m³")

    # ------------------------------------------------------------------
    # 6) Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    axA, axB, axC, axD = axes

    # Panel A: V(t), dV/dt, and Q_src_total + Q_bdy_net
    axA.plot(times, V_t / 1e6, color="#0f766e", lw=1.8, label="V(t)  CV water volume")
    axA.set_ylabel("V (M m³)")
    axA.grid(True, ls=":", alpha=0.4)
    axA.set_title("(a) CV water volume and source/boundary flows",
                  fontweight="bold", loc="left", fontsize=10)
    axA.legend(loc="upper left", fontsize=9)
    axA2 = axA.twinx()
    axA2.plot(times, Q_src_total, color="#dc2626", lw=1.2, alpha=0.8,
              label="ΣQ_src (vsource)")
    axA2.plot(times, Q_bdy_net,   color="#1d4ed8", lw=0.8, alpha=0.6,
              label="Q_bdy_net = dV/dt − ΣQ_src")
    axA2.set_ylabel("Flow (m³/s)")
    axA2.legend(loc="upper right", fontsize=9)
    axA2.axhline(0, color="#999", lw=0.4)

    # Panel B: mass time series
    axB.plot(times, M_bucket / 1e6, color="#b91c1c", lw=2.0,
             label="Bucket  M(t)  (plug-flow outflow)")
    axB.plot(times[:T_use], M_schism / 1e6, color="#1d4ed8", lw=2.0,
             label="SCHISM  ∫ tr1 dV  (volume-integrated)")
    axB.set_ylabel("Mass (M mmol)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="best", fontsize=10)
    axB.set_title("(b) Volume-integrated TRC_tr1 mass in CV: bucket vs SCHISM",
                  fontweight="bold", loc="left", fontsize=10)
    axB.axhline(0, color="#999", lw=0.4)

    # Panel C: CV-mean concentration time series
    axC.plot(times, C_bucket, color="#b91c1c", lw=2.0,
             label="Bucket  C_cv = M/V")
    axC.plot(times[:T_use], C_schism, color="#1d4ed8", lw=2.0,
             label="SCHISM  C_cv = ∫ tr1 dV / V")
    axC.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.7,
                label="src #3 inflow c = 1.0")
    axC.set_ylabel("C_cv (mmol/m³)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="best", fontsize=10)
    axC.set_title("(c) CV-mean TRC_tr1 concentration: bucket vs SCHISM",
                  fontweight="bold", loc="left", fontsize=10)
    axC.axhline(0, color="#999", lw=0.4)

    # Bucket cumulative IN / OUT  (computed up-front so we can print summary)
    F_out_bucket = Q_bdy_out * C_bucket          # mmol/s
    cum_in   = np.concatenate([[0], np.cumsum(0.5*(F_in_src[1:]+F_in_src[:-1])*np.diff(t_sec))])
    cum_out  = np.concatenate([[0], np.cumsum(0.5*(F_out_bucket[1:]+F_out_bucket[:-1])*np.diff(t_sec))])

    # ------------------------------------------------------------------
    # 6b) SCHISM cumulative tr1 export at Pioneer Mouth (flux.out row 3)
    # ------------------------------------------------------------------
    print("\n=== Parsing flux.out for SCHISM cumulative export ===")
    flux_da = parse_flux_out(RUN / "outputs" / "flux.out", n_regions=9)
    # row 3 = flow region-3 -> region-2 (bay -> CV). Positive => INTO CV;
    # negative => OUT of CV. Export = -row3.  Units: mmol/s.
    raw_row3 = flux_da.sel(tracer="TRC_tr1", region=3).values
    t_flux_d = flux_da["time_days"].values
    keep_fx = t_flux_d <= PERIOD_DAYS
    t_flux_s = t_flux_d[keep_fx] * 86400.0
    raw_row3 = raw_row3[keep_fx]
    flux_times = START + (t_flux_d[keep_fx] * 86400 * 1e9).astype("timedelta64[ns]")
    # Cumulative export at the mouth (sign-flipped so positive = export)
    cum_out_schism = np.concatenate(
        [[0], np.cumsum(0.5 * (-raw_row3[1:] - raw_row3[:-1]) * np.diff(t_flux_s))]
    )
    print(f"  SCHISM cum tr1 export at Pioneer Mouth (row 3, sign-flipped):")
    print(f"    final = {cum_out_schism[-1]/1e6:.3f} M mmol")
    print(f"  Bucket cum tr1 export:")
    print(f"    final = {cum_out[-1]/1e6:.3f} M mmol")
    print(f"  Total IN (src #3):")
    print(f"    final = {cum_in[-1]/1e6:.3f} M mmol")

    # Panel D: cumulative tr1 inputs and outputs (bucket + SCHISM)
    axD.plot(times, cum_in / 1e6,  color="black",   lw=2.0,
             label="cum tr1 IN  (src #3 ∫ Q × 1.0 dt)")
    axD.plot(times, cum_out / 1e6, color="#b91c1c", lw=1.8,
             label="cum tr1 OUT  bucket  (∫ Q_bdy_out × C_bucket dt)")
    axD.plot(flux_times, cum_out_schism / 1e6, color="#1d4ed8", lw=1.8,
             label="cum tr1 OUT  SCHISM  (−∫ flux.out row 3 dt, Pioneer Mouth)")
    axD.plot(times, (cum_in - cum_out) / 1e6, color="#b91c1c", lw=1.0, ls="--",
             alpha=0.6, label="bucket residual = IN − OUT_bucket")
    # SCHISM residual: align cum_out_schism onto src-time axis via interp
    cum_out_schism_on_src = np.interp(t_sec, t_flux_s, cum_out_schism)
    axD.plot(times, (cum_in - cum_out_schism_on_src) / 1e6,
             color="#1d4ed8", lw=1.0, ls="--", alpha=0.6,
             label="SCHISM residual = IN − OUT_SCHISM "
                   "(what mass should be in CV if no losses)")
    axD.set_ylabel("Cumulative tr1 (M mmol)")
    axD.set_xlabel("Date (model time)")
    axD.grid(True, ls=":", alpha=0.4)
    axD.legend(loc="upper left", fontsize=9)
    axD.set_title("(d) Cumulative tr1 fluxes:  IN (src #3) vs OUT (bucket vs SCHISM)",
                  fontweight="bold", loc="left", fontsize=10)
    axD.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axD.axhline(0, color="#999", lw=0.4)

    fig.suptitle("TRC_tr1 bucket-model cross-check vs SCHISM "
                 "(P18_flood, pioneer_estuary CV)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")

    # ------------------------------------------------------------------
    # 7) CSV dump
    # ------------------------------------------------------------------
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "V_m3", "Q_src_total_m3ps", "Q_bdy_net_m3ps",
                    "Q_bdy_out_m3ps", "F_in_src_mmolps",
                    "M_bucket_mmol", "C_bucket_mmolpm3",
                    "M_schism_mmol", "C_schism_mmolpm3"])
        for i in range(len(V_t)):
            row = [
                str(times[i]), V_t[i], Q_src_total[i], Q_bdy_net[i],
                Q_bdy_out[i], F_in_src[i],
                M_bucket[i], C_bucket[i],
                M_schism[i] if i < T_use else "",
                C_schism[i] if i < T_use else "",
            ]
            w.writerow(row)
    print(f"  wrote {OUT_CSV}")

    # ------------------------------------------------------------------
    # 8) Diagnostic summary
    # ------------------------------------------------------------------
    print("\n=== Diagnostic summary ===")
    print(f"Total tr1 input over 14 d       : {cum_in[-1]/1e6:.2f} M mmol")
    print(f"Bucket outflux (plug-flow)      : {cum_out[-1]/1e6:.2f} M mmol")
    print(f"SCHISM outflux (Pioneer Mouth)  : {cum_out_schism[-1]/1e6:.2f} M mmol")
    print(f"Bucket retained mass (day 14)   : {M_bucket[-1]/1e6:.2f} M mmol")
    print(f"SCHISM retained mass (day 14)   : {M_schism[-1]/1e6:.2f} M mmol")
    print(f"SCHISM apparent missing mass    : "
          f"{(cum_in[-1] - cum_out_schism_on_src[-1] - M_schism[-1])/1e6:.2f} M mmol  "
          f"(= IN  −  OUT_SCHISM  −  M_SCHISM)")
    if M_bucket[-1] > 0:
        print(f"SCHISM / bucket retention ratio : {M_schism[-1] / M_bucket[-1]:.2f}")


if __name__ == "__main__":
    main()
