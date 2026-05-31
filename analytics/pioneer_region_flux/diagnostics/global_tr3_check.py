"""
Global tr3 mass-conservation check (control for tr1 mass-loss diagnosis).

TRC_tr3 has ocean BC = 1.0 (per AED_16.th in per_variable_plots audit) and
msource = 0 everywhere. So its mass enters the domain only across the open
ocean boundary via tidal inflow.

For a CONSERVATIVE tracer with c_ocean = 1.0 and no other source/sink,
the asymptotic ceiling on domain-integral mass is:
    M_max = V_domain(t) × 1.0 mmol/m³ = V_domain in m³ as mmol units
i.e., once every cell has been replaced by tidal exchange, every cell
should hold c = 1.0 and the global integral equals the global water volume
(numerically, since 1 mmol/m³ × m³ = 1 mmol).

What the script asks:
  1. Does global tr3 mass MONOTONICALLY RISE over time?  (= conservative
     transport across the ocean boundary)
  2. Does it stay BELOW V_domain × 1.0?                  (= sane ceiling)
  3. Does it ever DROP for no reason?                    (= transport leak)

Compared to tr1, this isolates whether the mass-loss seen for tr1 lives in
the point-source (msource) injection path or in general transport.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry

CFG = active_run()
RUN = CFG.run_dir
OUT_PNG = CFG.out_dir / "global_tr3_check.png"
START = CFG.start
PERIOD_DAYS = CFG.period_days
TRACER = "TRC_tr3"


def main():
    print("=== Loading hgrid ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    elements = hgrid["elements"]
    areas_all = hgrid["areas"].astype(np.float64)
    n_elem = hgrid["n_elements"]
    print(f"  n_elements={n_elem}, total domain area = {areas_all.sum()/1e6:.2f} km²")

    nc_arr = np.zeros(n_elem, dtype=np.int8)
    elem_node_idx = np.zeros((n_elem, 4), dtype=np.int32)
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc_arr[ei] = int(nc)
        for k, n in enumerate(ids[:int(nc)]):
            elem_node_idx[ei, k] = int(n)
    elem_node_w = np.zeros((n_elem, 4), dtype=np.float32)
    for k in range(4):
        elem_node_w[:, k] = np.where(nc_arr > k, 1.0 / nc_arr, 0.0)

    print(f"\n=== Streaming {TRACER} stacks + ENV_layer_ht across whole domain ===")
    n_stacks = PERIOD_DAYS + 1
    paths_tr = geometry.discover_scribed_stacks(RUN / "outputs", TRACER)[:n_stacks]
    paths_cmb = geometry.discover_cmb_stacks(RUN / "outputs")[:n_stacks]

    M_global, V_global, times = [], [], []
    for path_tr, path_cm in zip(paths_tr, paths_cmb):
        ds_t = xr.open_dataset(path_tr, engine="h5netcdf")
        ds_c = xr.open_dataset(path_cm, engine="h5netcdf")
        da_t = ds_t[TRACER]
        da_l = ds_c["ENV_layer_ht"]
        keep = (da_t["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        da_t = da_t.isel(time=keep)
        keep_l = (da_l["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        da_l = da_l.isel(time=keep_l)
        T = min(da_t.sizes["time"], da_l.sizes["time"])
        if T == 0:
            ds_t.close(); ds_c.close(); continue

        node_dim = next(d for d in da_t.dims if "node" in d.lower())
        layer_dim_s = next(d for d in da_t.dims
                           if "vgrid" in d.lower() or "layer" in d.lower())
        layer_dim_c = next(d for d in da_l.dims
                           if "vgrid" in d.lower() or "layer" in d.lower())
        face_dim_c = next(d for d in da_l.dims if "face" in d.lower())

        arr_t = da_t.isel(time=slice(0, T)).transpose(
            "time", layer_dim_s, node_dim).values
        arr_l = da_l.isel(time=slice(0, T)).transpose(
            "time", layer_dim_c, face_dim_c).values
        arr_t = np.where(np.isfinite(arr_t) & (np.abs(arr_t) <= 1e30), arr_t, 0.0)
        arr_l = np.where(np.isfinite(arr_l) & (arr_l > 0), arr_l, 0.0)

        chunk = 16
        m_t = np.zeros(T); v_t = np.zeros(T)
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            sl = slice(t0, t1)
            gathered = arr_t[sl, :, elem_node_idx]               # (Tc, L, E, 4)
            elem_vals = (gathered * elem_node_w).sum(axis=-1)    # (Tc, L, E)
            cell_vol = arr_l[sl] * areas_all[None, None, :]      # (Tc, L, E)
            m_t[sl] = (elem_vals * cell_vol).sum(axis=(1, 2))
            v_t[sl] = cell_vol.sum(axis=(1, 2))
        M_global.append(m_t)
        V_global.append(v_t)
        times.append(da_t["time"].values[:T])
        print(f"  {path_tr.name}: T={T}, M peak={m_t.max()/1e6:.3f} M mmol, "
              f"V max={v_t.max()/1e6:.0f} M m³")
        ds_t.close(); ds_c.close()

    M_global = np.concatenate(M_global)
    V_global = np.concatenate(V_global)
    times    = np.concatenate(times)
    t_sec    = (times - times[0]) / np.timedelta64(1, "s")
    C_global = M_global / np.where(V_global > 0, V_global, np.nan)

    # The "asymptotic ceiling" if every cell were filled to c=1.0
    # (1 mmol/m³ × V_m³ = V mmol; so M_ceiling[t] = V_global[t])
    M_ceiling = V_global.copy()
    fill_frac = M_global / M_ceiling

    print()
    print("=" * 70)
    print(f"=== GLOBAL MASS CONSERVATION CHECK FOR {TRACER} ===")
    print("=" * 70)
    print(f"Ocean BC c_{TRACER:>3} = 1.0 (from AED_16.th)")
    print()
    print(f"Domain water volume range : "
          f"[{V_global.min()/1e6:.0f}, {V_global.max()/1e6:.0f}] M m³  "
          f"(= asymptotic mass ceiling)")
    print(f"Initial tr3 mass (day 0)  : {M_global[0]/1e6:8.3f} M mmol  "
          f"({fill_frac[0]*100:6.2f}% of ceiling)")
    print(f"Mid tr3 mass (day 7)      : {M_global[len(M_global)//2]/1e6:8.3f} M mmol  "
          f"({fill_frac[len(M_global)//2]*100:6.2f}% of ceiling)")
    print(f"Final tr3 mass (day 14)   : {M_global[-1]/1e6:8.3f} M mmol  "
          f"({fill_frac[-1]*100:6.2f}% of ceiling)")
    print(f"Peak tr3 mass             : {M_global.max()/1e6:8.3f} M mmol  "
          f"(at {times[M_global.argmax()]})")
    print()
    # Monotonicity diagnostic — daily-averaged trend
    daily_means = []
    for d in range(PERIOD_DAYS):
        m = (t_sec >= d * 86400.0) & (t_sec < (d + 1) * 86400.0)
        if m.any():
            daily_means.append(M_global[m].mean())
    daily_means = np.array(daily_means)
    print(f"Daily-mean tr3 mass (M mmol): {np.round(daily_means / 1e6, 3)}")
    print(f"Daily delta: {np.round(np.diff(daily_means) / 1e6, 3)}")
    print()
    print("Interpretation:")
    print("  - If daily mass increases monotonically -> conservative.")
    print("  - If daily mass plateaus near ceiling   -> conservative and saturated.")
    print("  - If daily mass DROPS at any point      -> transport is leaking.")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axA, axB, axC = axes

    axA.plot(times, M_ceiling / 1e6, color="black", lw=1.6, ls="--",
             label="Ceiling = V_domain × 1.0 mmol/m³")
    axA.plot(times, M_global / 1e6, color="#1d4ed8", lw=2.0,
             label=f"GLOBAL {TRACER} mass")
    axA.set_ylabel("Mass (M mmol)")
    axA.grid(True, ls=":", alpha=0.4)
    axA.legend(loc="best", fontsize=10)
    axA.set_title(f"(a) Global {TRACER} mass vs domain-volume ceiling "
                  f"(ocean BC c=1.0)",
                  fontweight="bold", loc="left", fontsize=11)
    axA.axhline(0, color="#999", lw=0.4)

    axB.plot(times, fill_frac * 100, color="#0f766e", lw=2.0,
             label=f"% of ceiling  =  M / (V × c_ocean)")
    axB.set_ylabel("Fill fraction (%)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="best", fontsize=10)
    axB.set_title("(b) Fill fraction — how full of tracer the domain is",
                  fontweight="bold", loc="left", fontsize=11)

    axC.plot(times, C_global, color="#7c2d12", lw=2.0,
             label="Domain-mean C")
    axC.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6,
                label="Ocean BC c = 1.0")
    axC.set_ylabel("C (mmol/m³)")
    axC.set_xlabel("Date (model time)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="best", fontsize=10)
    axC.set_title(f"(c) Domain-mean {TRACER} concentration",
                  fontweight="bold", loc="left", fontsize=11)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle(f"{TRACER} global mass conservation (P18_flood)  "
                 f"— transport-only control vs tr1 source-driven",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
