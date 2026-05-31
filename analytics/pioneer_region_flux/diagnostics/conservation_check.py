"""
Conservation check across the (2<->3) mouth transect, with residual panels.

For a CV bounded on its seaward side by the (2<->3) transect and on its
landward side by dry land, conservation requires:

    M_CV(t) - M_CV(0)  ==  cum_src_into_CV(t)  +  cum_flux_R3(t)

where the flux.out region-3 column stores flux from HI(3) to LO(2),
so positive = INTO CV.

For each variable (VOL, Salinity, GEN_1, TRC_tr1) the script produces a
2-row plot:
    top    : observed dM_CV(t), predicted dM_CV(t)=src+flux,
             cum_src_into_CV(t), cum_flux(t)
    bottom : residual(t) = obs - predicted

A non-zero residual is mass appearing/disappearing inside the CV that
can't be accounted for by sources or by the (2<->3) gate transport.

CV definition: flood-fill from src#3 (elem 50526) through edge-adjacency,
blocked by any cell flagged 3. This is the wet/dry topology approach
agreed with: the (2<->3) strip is the gate, and everything upstream
(including the flag=-1 cells beyond the strip) is the CV.
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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry
from core.boundary_fluxes import parse_flux_out
from core.point_sources import parse_source_sink, parse_vsource


CFG = active_run()
RUN = CFG.run_dir
OUT_DIR = CFG.out_dir / "conservation_check"
START = CFG.start
PERIOD_DAYS = CFG.period_days
N_SRC = 14
N_GEN = 1   # GEN_1 only

# msource.th column layout (0-indexed). Cols 0=time, 1..14=T, 15..28=S,
# 29..42=GEN_1, 43..end=AED tracers in registration order.
AED_TRACER_K = {
    "NCS_ss1": 1, "OXY_oxy": 2, "NIT_amm": 3, "NIT_nit": 4,
    "PHS_frp": 5, "PHS_frp_ads": 6, "OGM_doc": 7, "OGM_poc": 8,
    "OGM_don": 9, "OGM_pon": 10, "OGM_dop": 11, "OGM_pop": 12,
    "PHY_mixed": 13, "TRC_tr1": 14, "TRC_tr2": 15, "TRC_tr3": 16, "TRC_age": 17,
}


def msource_col_slice(var):
    """Return the column slice in msource.th for `var`, or None for VOL."""
    if var == "VOL":
        return None
    if var == "Heat":
        return slice(1, 1 + N_SRC)
    if var == "Salinity":
        return slice(1 + N_SRC, 1 + 2 * N_SRC)
    if var == "GEN_1":
        return slice(1 + 2 * N_SRC, 1 + 2 * N_SRC + N_SRC)
    if var in AED_TRACER_K:
        k = AED_TRACER_K[var]   # 1..17
        base = 1 + 2 * N_SRC + N_GEN * N_SRC + (k - 1) * N_SRC
        return slice(base, base + N_SRC)
    raise KeyError(var)


def cumtrap(rate, t_sec):
    if len(t_sec) < 2:
        return np.zeros_like(rate)
    dt = np.diff(t_sec)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])


def load_zcoord_dz(outputs_dir, n_stacks):
    """zCoordinates -> dz   (T, L-1, n_node)."""
    paths = geometry.discover_scribed_stacks(outputs_dir, "zCoordinates")
    paths = paths[:n_stacks]
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
        sentinel = ~(np.isfinite(zarr) & (np.abs(zarr) <= 1e6))
        zarr_nan = np.where(sentinel, np.nan, zarr)
        dz = np.diff(zarr_nan, axis=1)
        dz = np.where(np.isfinite(dz), dz, 0.0)
        dz_list.append(dz)
        t_list.append(ds["time"].values)
        ds.close()
    dz_full = np.concatenate(dz_list, axis=0).astype(np.float32)
    t_full = np.concatenate(t_list)
    return t_full, dz_full


def integrate_cv_mass(da_s, layer_dim, node_dim, dz_full, cv_nodes,
                       cv_area_per_node):
    """Integrate c * dz * A over the CV node set.

    Returns mass_t (T,) in (units_of_da_s * m^3)."""
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
    arr_cv = arr[:T, :, cv_nodes]
    dz_cv = dz_full[:T, :, cv_nodes]
    if L == L_dz + 1:
        c_mid = 0.5 * (arr_cv[:, :-1, :] + arr_cv[:, 1:, :])
    elif L == L_dz:
        c_mid = arr_cv
    else:
        raise RuntimeError(f"shape mismatch: tracer L={L}, dz L={L_dz}")
    mass_t = (c_mid * dz_cv * cv_area_per_node[None, None, :]).sum(axis=(1, 2))
    return mass_t


def build_cv_floodfill(hgrid, flag, seed_elem, block_flag=3):
    """Return cv_elements (1D array of elem idx, 0-based)."""
    elements = hgrid["elements"]
    edge_owners = defaultdict(list)
    for ei in range(hgrid["n_elements"]):
        nc, *ids = elements[ei]; nc = int(nc)
        for k in range(nc):
            a, b = int(ids[k]), int(ids[(k + 1) % nc])
            edge_owners[(min(a, b), max(a, b))].append(ei)
    elem_neigh = defaultdict(list)
    for key, owners in edge_owners.items():
        if len(owners) == 2:
            e0, e1 = owners
            elem_neigh[e0].append(e1)
            elem_neigh[e1].append(e0)
    seen = {seed_elem}; queue = [seed_elem]
    while queue:
        e = queue.pop()
        for nb in elem_neigh[e]:
            if nb in seen: continue
            if int(flag[nb]) == block_flag: continue
            seen.add(nb); queue.append(nb)
    return np.array(sorted(seen), dtype=int)


def compute_balance_one_var(
    var, *, scribed_name, flux_da, t_flux_s, keep_flux, flux_times,
    outputs, src_elems, cv_set, cv_nodes, cv_area_per_node, dz_full, t_z,
    raw_ms, t_vs, Q,
):
    """Compute observed and predicted ΔM_CV for one variable.

    Returns a dict with t_cv, d_obs, d_pred, residual, cum_src, cum_flux.
    """
    print(f"\n=== {var} ===")
    # ---- src contribution (cum_src) ----
    inside_idx = [s for s, e in enumerate(src_elems) if int(e) in cv_set]
    if var == "VOL":
        Qsum = Q[:, inside_idx].sum(axis=1) if inside_idx else np.zeros_like(t_vs)
        cum_src = cumtrap(Qsum, t_vs)
    else:
        sl = msource_col_slice(var)
        ms_var = raw_ms[:, sl]
        t_ms = raw_ms[:, 0]
        if inside_idx:
            rate = np.zeros_like(t_vs)
            for s in inside_idx:
                c_i = np.interp(t_vs, t_ms, ms_var[:, s])
                rate = rate + Q[:, s] * c_i
            cum_src = cumtrap(rate, t_vs)
        else:
            cum_src = np.zeros_like(t_vs)

    # ---- flux.out ----
    if var == "VOL":
        flux_tracer = "VOL"
    else:
        flux_tracer = var
    if flux_tracer not in flux_da.coords["tracer"].values:
        print(f"  WARN: {flux_tracer} not in flux.out — skipping")
        return None
    raw = flux_da.sel(tracer=flux_tracer, region=3).values[keep_flux]
    cum_flux = cumtrap(raw, t_flux_s)
    print(f"  cum src (end)   = {cum_src[-1]:+.3e}")
    print(f"  cum flux (end)  = {cum_flux[-1]:+.3e}")

    # ---- observed ΔM_CV ----
    if var == "VOL":
        # Volume = ∫dz dA  (no concentration)
        dz_cv = dz_full[:, :, cv_nodes]
        mass_t = (dz_cv * cv_area_per_node[None, None, :]).sum(axis=(1, 2))
        t_cv = t_z[: mass_t.shape[0]]
    else:
        try:
            da_s = geometry.open_scribed_concat(outputs, scribed_name,
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
            mass_t = integrate_cv_mass(da_s, layer_dim, node_dim, dz_full,
                                        cv_nodes, cv_area_per_node)
            t_cv = da_s["time"].values[: mass_t.shape[0]]
        except Exception as e:
            print(f"  WARN: failed to load {scribed_name}: {e}")
            return None

    print(f"  M(0)  = {mass_t[0]:+.3e}")
    print(f"  M(end)= {mass_t[-1]:+.3e}")
    print(f"  ΔM    = {mass_t[-1] - mass_t[0]:+.3e}")

    # ---- align time bases ----
    t_cv_s = (t_cv - START).astype("timedelta64[s]").astype(float)
    cum_flux_on_cv = np.interp(t_cv_s, t_flux_s, cum_flux)
    cum_src_on_cv = np.interp(t_cv_s, t_vs, cum_src)
    d_obs = mass_t - mass_t[0]
    d_pred = cum_src_on_cv + cum_flux_on_cv
    residual = d_obs - d_pred

    return {
        "t": t_cv,
        "M": mass_t,
        "d_obs": d_obs,
        "d_pred": d_pred,
        "cum_src": cum_src_on_cv,
        "cum_flux": cum_flux_on_cv,
        "residual": residual,
    }


def make_two_var_plot(results, var_a, var_b, unit_a, unit_b, title_a, title_b,
                       out_png):
    """4-row figure: [a_balance, a_residual, b_balance, b_residual]."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    for i, (var, unit, title) in enumerate(((var_a, unit_a, title_a),
                                              (var_b, unit_b, title_b))):
        r = results.get(var)
        if r is None:
            for ax in axes[2*i:2*i+2]:
                ax.text(0.5, 0.5, f"{var}: data unavailable",
                         transform=ax.transAxes, ha="center", va="center")
            continue
        ax_bal = axes[2*i]
        ax_res = axes[2*i + 1]
        ax_bal.plot(r["t"], r["d_obs"], color="black", lw=2.2,
                     label=f"dM_CV  ({var}, observed)")
        ax_bal.plot(r["t"], r["d_pred"], color="#dc2626", lw=1.5, ls="--",
                     label="PRED:  src + flux")
        ax_bal.plot(r["t"], r["cum_src"], color="#16a34a", lw=1.2, ls=":",
                     label="src into CV (cum)")
        ax_bal.plot(r["t"], r["cum_flux"], color="#9333ea", lw=1.2, ls=":",
                     label="flux.out region3 (cum)")
        ax_bal.axhline(0, color="grey", lw=0.5)
        ax_bal.set_ylabel(f"{var}  ({unit})")
        ax_bal.set_title(title, fontweight="bold", loc="left")
        ax_bal.legend(loc="best", fontsize=8, ncols=2)
        ax_bal.grid(True, alpha=0.3)

        ax_res.plot(r["t"], r["residual"], color="#b91c1c", lw=1.8,
                     label="residual = obs - pred")
        ax_res.axhline(0, color="grey", lw=0.5)
        ax_res.set_ylabel(f"residual ({unit})")
        ax_res.set_title(f"{var} residual: M_CV gained/lost outside the src + (2<->3) gate accounting",
                          fontweight="bold", loc="left", fontsize=10)
        ax_res.legend(loc="best", fontsize=8)
        ax_res.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = RUN / "outputs"

    print("Loading hgrid + fluxflag...")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]

    # CV flood-fill from src#3
    src3_elem = 50526 - 1
    cv_elements = build_cv_floodfill(hgrid, flag, src3_elem, block_flag=3)
    cv_set = set(int(e) for e in cv_elements)
    print(f"  CV (flood-fill, blocked by flag=3): {len(cv_elements)} elements")

    # CV node-area share
    share = {}
    elements = hgrid["elements"]
    for ei in cv_elements:
        nc, *ids = elements[ei]; nc = int(nc)
        a_share = hgrid["areas"][ei] / nc
        for n in ids[:nc]:
            n = int(n)
            share[n] = share.get(n, 0.0) + a_share
    cv_nodes = np.array(sorted(share.keys()), dtype=int)
    cv_area_per_node = np.array([share[n] for n in cv_nodes],
                                 dtype=np.float32)
    print(f"  CV: {len(cv_nodes)} nodes, area = {cv_area_per_node.sum():,.0f} m^2")

    # sources
    src_elems = parse_source_sink(RUN / "source_sink.in")
    t_vs, Q = parse_vsource(RUN / "vsource.th")
    mvs = t_vs <= PERIOD_DAYS * 86400.0
    t_vs = t_vs[mvs]; Q = Q[mvs]
    raw_ms = np.loadtxt(RUN / "msource.th")

    inside_idx = [s for s, e in enumerate(src_elems) if int(e) in cv_set]
    print(f"\nSources INSIDE CV: {len(inside_idx)} of {len(src_elems)}: "
          f"{[s+1 for s in inside_idx]}")

    # flux.out
    print("\nParsing flux.out...")
    flux_da = parse_flux_out(outputs / "flux.out", n_regions=None)
    t_d = flux_da["time_days"].values
    keep_flux = t_d <= PERIOD_DAYS
    t_flux_s = t_d[keep_flux] * 86400.0
    flux_times = START + (t_d[keep_flux] * 86400 * 1e9).astype("timedelta64[ns]")

    # zCoord-derived dz
    print("\nLoading zCoordinates for dz...")
    t_z, dz_full = load_zcoord_dz(outputs, n_stacks=PERIOD_DAYS + 1)
    keep_z = (t_z - START) <= np.timedelta64(PERIOD_DAYS, "D")
    t_z = t_z[keep_z]
    dz_full = dz_full[keep_z]
    print(f"  T={dz_full.shape[0]}, n_layer-1={dz_full.shape[1]}, "
          f"n_node={dz_full.shape[2]}")

    # ---- compute balance for each variable ----
    results = {}
    for var, scribed in (
        ("VOL",      None),
        ("Salinity", "salinity"),
        ("GEN_1",    "GEN_1"),
        ("TRC_tr1",  "TRC_tr1"),
    ):
        results[var] = compute_balance_one_var(
            var,
            scribed_name=scribed,
            flux_da=flux_da, t_flux_s=t_flux_s, keep_flux=keep_flux,
            flux_times=flux_times,
            outputs=outputs,
            src_elems=src_elems, cv_set=cv_set,
            cv_nodes=cv_nodes, cv_area_per_node=cv_area_per_node,
            dz_full=dz_full, t_z=t_z,
            raw_ms=raw_ms, t_vs=t_vs, Q=Q,
        )

    # ---- plot VOL + Salinity ----
    make_two_var_plot(
        results, "VOL", "Salinity",
        unit_a="m^3", unit_b="psu * m^3",
        title_a="(a) VOL conservation  (CV = flood-fill from src#3 blocked by flag=3)",
        title_b="(c) Salinity conservation",
        out_png=OUT_DIR / "conservation_VOL_Salinity.png",
    )

    # ---- plot GEN_1 + TRC_tr1 ----
    make_two_var_plot(
        results, "GEN_1", "TRC_tr1",
        unit_a="mmol/m^3 * m^3 (concentration*volume)",
        unit_b="mmol/m^3 * m^3",
        title_a="(a) GEN_1 conservation  (SCHISM-native tracer, src#3=1, others=0)",
        title_b="(c) TRC_tr1 conservation  (AED tracer, src#3=1, others=0)",
        out_png=OUT_DIR / "conservation_GEN1_TRC_tr1.png",
    )

    # ---- summary table ----
    print("\n===== SUMMARY =====")
    for var in ("VOL", "Salinity", "GEN_1", "TRC_tr1"):
        r = results.get(var)
        if r is None: continue
        print(f"{var:>10s}: dM_obs={r['d_obs'][-1]:+.3e}  "
              f"cum_src={r['cum_src'][-1]:+.3e}  "
              f"cum_flux={r['cum_flux'][-1]:+.3e}  "
              f"residual={r['residual'][-1]:+.3e}  "
              f"  (residual/cum_src = "
              f"{r['residual'][-1] / r['cum_src'][-1] * 100 if r['cum_src'][-1] else float('nan'):+.1f}%)")

    # ---- csv ----
    for var in ("VOL", "Salinity", "GEN_1", "TRC_tr1"):
        r = results.get(var)
        if r is None: continue
        df = pd.DataFrame({
            "time":     r["t"],
            "M":        r["M"],
            "d_obs":    r["d_obs"],
            "d_pred":   r["d_pred"],
            "cum_src":  r["cum_src"],
            "cum_flux": r["cum_flux"],
            "residual": r["residual"],
        })
        out_csv = OUT_DIR / f"conservation_{var}.csv"
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
