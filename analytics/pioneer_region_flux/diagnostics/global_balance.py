"""
Close the overall mass balance by inferring the ocean-BC contribution.

For the WHOLE domain (no CV restriction), conservation requires:
    DELTA_M_global(t)  =  cum_msource_input(t)  +  cum_ocean_BC(t)

The first two terms we have:
    DELTA_M_global    : volume-integrate the scribed tracer over EVERY
                         node in the mesh (using zCoordinates for dz,
                         element->node area sharing for A).
    cum_msource_input : sum over the 14 vsource srcs of int(Q * c_msource) dt,
                         using the same -9999 sentinel filter as the
                         fixed per_variable_plots.

The remainder is the net cumulative flux across the open ocean boundary.

Variables: VOL, Salinity, GEN_1, TRC_tr1, TRC_tr3.

Output: PNG with one row per variable showing
   - DELTA_M_global(t) observed
   - cum_msource_input(t)
   - implied cum_ocean_BC(t) = DELTA_M_global - cum_msource_input

(For VOL, the msource term is the vsource volume itself, and there is no
"c_ocean" — the ocean-BC contribution is just net water exchange.)
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry
from core.point_sources import parse_source_sink, parse_vsource


CFG = active_run()
RUN = CFG.run_dir
OUT_DIR = CFG.out_dir / "conservation_check"
START = CFG.start
PERIOD_DAYS = CFG.period_days
N_SRC = 14
N_GEN = 1

VARS = [
    # (name, scribed_filename_root, has_ocean_BC_input?, ocean_BC_concentration)
    ("VOL",      None,        True,  None),
    ("Salinity", "salinity",  True,  35.0),
    ("GEN_1",    "GEN_1",     False, 0.0),
    ("TRC_tr1",  "TRC_tr1",   False, 0.0),
    ("TRC_tr3",  "TRC_tr3",   True,  1.0),
]


def msource_col_slice(var):
    if var == "VOL":     return None
    if var == "Heat":    return slice(1, 1 + N_SRC)
    if var == "Salinity":return slice(1 + N_SRC, 1 + 2 * N_SRC)
    if var == "GEN_1":   return slice(1 + 2*N_SRC, 1 + 2*N_SRC + N_SRC)
    aed_k = {"NCS_ss1":1,"OXY_oxy":2,"NIT_amm":3,"NIT_nit":4,"PHS_frp":5,
             "PHS_frp_ads":6,"OGM_doc":7,"OGM_poc":8,"OGM_don":9,"OGM_pon":10,
             "OGM_dop":11,"OGM_pop":12,"PHY_mixed":13,"TRC_tr1":14,
             "TRC_tr2":15,"TRC_tr3":16,"TRC_age":17}
    k = aed_k[var]
    base = 1 + 2*N_SRC + N_GEN*N_SRC + (k-1)*N_SRC
    return slice(base, base + N_SRC)


def cumtrap(rate, t_sec):
    out = np.zeros_like(rate)
    if len(t_sec) >= 2:
        out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t_sec))
    return out


def load_zcoord_dz(outputs_dir, n_stacks):
    paths = geometry.discover_scribed_stacks(outputs_dir, "zCoordinates")[:n_stacks]
    dz_list, t_list = [], []
    for p in paths:
        ds = xr.open_dataset(p, engine="h5netcdf")
        z = ds["zCoordinates"]
        dims = list(z.dims)
        zarr = z.values
        ax_t = next(i for i, d in enumerate(dims) if d.lower() == "time")
        ax_l = next(i for i, d in enumerate(dims)
                    if "vgrid" in d.lower() or "layer" in d.lower())
        ax_n = next(i for i, d in enumerate(dims) if "node" in d.lower())
        zarr = np.transpose(zarr, (ax_t, ax_l, ax_n))
        sent = ~(np.isfinite(zarr) & (np.abs(zarr) <= 1e6))
        zarr_nan = np.where(sent, np.nan, zarr)
        dz = np.diff(zarr_nan, axis=1)
        dz = np.where(np.isfinite(dz), dz, 0.0)
        dz_list.append(dz)
        t_list.append(ds["time"].values)
        ds.close()
    return np.concatenate(t_list), np.concatenate(dz_list, axis=0).astype(np.float32)


def integrate_global_mass(da_s, layer_dim, node_dim, dz_full, A_node):
    """Integrate c * dz * A over the entire domain."""
    dims = list(da_s.dims)
    arr_raw = da_s.values
    ax_t = next(i for i, d in enumerate(dims) if d.lower() == "time")
    ax_l = next(i for i, d in enumerate(dims) if d == layer_dim)
    ax_n = next(i for i, d in enumerate(dims) if d == node_dim)
    arr = np.transpose(arr_raw, (ax_t, ax_l, ax_n)).astype(np.float32)
    arr = np.where(np.isfinite(arr) & (np.abs(arr) <= 1e6), arr, 0.0)
    T = min(arr.shape[0], dz_full.shape[0])
    L = arr.shape[1]
    L_dz = dz_full.shape[1]
    if L == L_dz + 1:
        c_mid = 0.5 * (arr[:T, :-1, :] + arr[:T, 1:, :])
    elif L == L_dz:
        c_mid = arr[:T]
    else:
        raise RuntimeError(f"shape mismatch: L={L}, L_dz={L_dz}")
    return (c_mid * dz_full[:T] * A_node[None, None, :]).sum(axis=(1, 2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hgrid ...")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    elements = hgrid["elements"]
    n_nodes = hgrid["n_nodes"]
    A_node = np.zeros(n_nodes, dtype=np.float64)
    for ei in range(hgrid["n_elements"]):
        nc, *ids = elements[ei]; nc = int(nc)
        share = hgrid["areas"][ei] / nc
        for n in ids[:nc]:
            A_node[int(n)] += share
    A_node = A_node.astype(np.float32)
    print(f"  total area = {A_node.sum():,.0f} m^2 = {A_node.sum()/1e6:.1f} km^2")

    print("Loading zCoordinates for dz ...")
    t_z, dz_full = load_zcoord_dz(RUN / "outputs", n_stacks=PERIOD_DAYS + 1)
    keep_z = (t_z - START) <= np.timedelta64(PERIOD_DAYS, "D")
    t_z = t_z[keep_z]
    dz_full = dz_full[keep_z]
    print(f"  T={dz_full.shape[0]}, n_layer-1={dz_full.shape[1]}, "
          f"n_node={dz_full.shape[2]}")

    # Domain water volume
    V_global = (dz_full * A_node[None, None, :]).sum(axis=(1, 2))
    print(f"  V_global range: [{V_global.min():.3e}, {V_global.max():.3e}] m^3")

    # vsource sum (no concentration)
    src_elems = parse_source_sink(RUN / "source_sink.in")
    t_vs, Q = parse_vsource(RUN / "vsource.th")
    mvs = t_vs <= PERIOD_DAYS * 86400.0
    t_vs = t_vs[mvs]; Q = Q[mvs]
    Qsum = Q.sum(axis=1)
    cum_vsource = cumtrap(Qsum, t_vs)
    print(f"  cum vsource (all 14 sources) over {PERIOD_DAYS}d: "
          f"{cum_vsource[-1]:+.3e} m^3")

    # msource (raw table; -9999 -> 0 filter)
    raw_ms = np.loadtxt(RUN / "msource.th")
    t_ms = raw_ms[:, 0]

    # ---- compute everything for each variable ----
    results = {}
    for var, scribed, _has_bc, _c_bc in VARS:
        print(f"\n=== {var} ===")
        # msource cumulative input
        if var == "VOL":
            cum_src = cum_vsource.copy()
            cum_src_label = "cum_vsource (m^3)"
        else:
            sl = msource_col_slice(var)
            ms_var = raw_ms[:, sl].copy()
            ms_var = np.where(ms_var <= -1e3, 0.0, ms_var)
            rate = np.zeros_like(t_vs)
            for s in range(N_SRC):
                c_i = np.interp(t_vs, t_ms, ms_var[:, s])
                rate = rate + Q[:, s] * c_i
            cum_src = cumtrap(rate, t_vs)
            cum_src_label = f"cum msource ({var})"
        print(f"  {cum_src_label} end = {cum_src[-1]:+.3e}")

        # Global mass M(t)
        if var == "VOL":
            M_global = V_global.copy()
            t_M = t_z
        else:
            try:
                da_s = geometry.open_scribed_concat(RUN / "outputs", scribed,
                                                     n_stacks=PERIOD_DAYS + 1)
                tt = da_s["time"].values
                _, uniq = np.unique(tt, return_index=True)
                if len(uniq) != len(tt):
                    da_s = da_s.isel(time=np.sort(uniq))
                keep_s = (da_s["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
                da_s = da_s.isel(time=keep_s)
                node_dim = next(d for d in da_s.dims if "node" in d.lower())
                layer_dim = next((d for d in da_s.dims
                                  if "vgrid" in d.lower() or "layer" in d.lower()), None)
                M_global = integrate_global_mass(da_s, layer_dim, node_dim,
                                                  dz_full, A_node)
                t_M = da_s["time"].values[: M_global.shape[0]]
            except Exception as e:
                print(f"  WARN: failed to load {scribed}: {e}")
                continue
        print(f"  M_global(0)   = {M_global[0]:+.3e}")
        print(f"  M_global(end) = {M_global[-1]:+.3e}")
        print(f"  DELTA M_global   = {M_global[-1]-M_global[0]:+.3e}")

        # Align time bases (use t_M as the reference)
        t_M_s = (t_M - START).astype("timedelta64[s]").astype(float)
        cum_src_on_M = np.interp(t_M_s, t_vs, cum_src)
        d_M = M_global - M_global[0]
        cum_ocean_BC = d_M - cum_src_on_M
        print(f"  IMPLIED cum_ocean_BC end = {cum_ocean_BC[-1]:+.3e}")

        results[var] = {
            "t": t_M,
            "M_global": M_global,
            "d_M": d_M,
            "cum_src": cum_src_on_M,
            "cum_ocean_BC": cum_ocean_BC,
        }

    # ---- plot ----
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n), sharex=True)
    if n == 1: axes = [axes]
    UNITS = {
        "VOL":      "m^3",
        "Salinity": "psu * m^3",
        "GEN_1":    "mmol/m^3 * m^3",
        "TRC_tr1":  "mmol/m^3 * m^3",
        "TRC_tr3":  "mmol/m^3 * m^3",
    }
    for ax, (var, r) in zip(axes, results.items()):
        u = UNITS.get(var, "")
        ax.plot(r["t"], r["d_M"],         color="black",   lw=2.0,
                 label=f"DELTA M_global ({var}, observed)")
        ax.plot(r["t"], r["cum_src"],     color="#16a34a", lw=1.5, ls="--",
                 label="cum source input (vsource * msource)")
        ax.plot(r["t"], r["cum_ocean_BC"], color="#dc2626", lw=1.5, ls="-",
                 label="implied cum ocean BC = DELTA_M_global - cum_src")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(f"{var}\n({u})", fontsize=9)
        ax.set_title(
            f"{var} — global balance  "
            f"DELTA_M={r['d_M'][-1]:+.2e}  src={r['cum_src'][-1]:+.2e}  "
            f"ocean_BC={r['cum_ocean_BC'][-1]:+.2e}",
            fontweight="bold", loc="left", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date")
    fig.suptitle("Global mass balance — closure of ocean BC term  (P18_flood, 21d)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = OUT_DIR / "global_balance.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_png}")

    # ---- summary table ----
    print("\n===== SUMMARY (end-of-period totals) =====")
    print(f"{'Variable':>10s}  {'DELTA M_global':>14s}  {'cum_src':>14s}  {'cum_ocean_BC':>14s}")
    for var, r in results.items():
        print(f"{var:>10s}  {r['d_M'][-1]:+14.3e}  {r['cum_src'][-1]:+14.3e}  {r['cum_ocean_BC'][-1]:+14.3e}")


if __name__ == "__main__":
    main()
