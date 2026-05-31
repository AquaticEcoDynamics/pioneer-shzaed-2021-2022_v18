"""
Decisive comparison: TRC_tr1 vs GEN_1 conservation in P18_flood.

Both tracers are set up identically at src #3 in msource.th:
  - TRC_tr1: AED conservative tracer, c=1.0 at src#3 always
  - GEN_1:   SCHISM-native conservative tracer, c=1.0 at src#3 always

Both are simulated in the same P18_flood run with depress_clutch=.TRUE.
(AED reactions disabled, tracers transported only by SCHISM advection).

depress_clutch=.TRUE. zeroes out ENV_layer_ht in the AED cmb file, so this
script computes layer thicknesses from zCoordinates (node-centered, always
written by SCHISM scribe) and integrates mass on the node-control-volume:

    M = Σ_node Σ_layer  c(n, k_mid) × Δz(n, k) × A_node(n)

where A_node = Σ_{elem containing node} A_elem / n_corners_elem  (each
element distributes its area equally to its corner nodes; Σ A_node = Σ A_elem).

This avoids needing ENV_layer_ht and is more consistent with the
node-centred scribed tracer fields.

Two checks per tracer:
  A. GLOBAL mass conservation  : integrate over the entire domain.
     Compare to cumulative msource input (Q3 × c × dt = Q3 × dt over 14 d).
  B. flux.out col 3 (Pioneer Mouth) — for completeness.

If GEN_1 conserves but TRC_tr1 doesn't, the bug is in AED's source coupling.
If both lose mass identically, the bug is in SCHISM's source/sink at high rat.
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
from core.boundary_fluxes import parse_flux_out

CFG = active_run()
RUN = CFG.run_dir
OUT_PNG = CFG.out_dir / "tr1_vs_gen1_compare.png"
START = CFG.start
PERIOD_DAYS = CFG.period_days
TRACERS = ["TRC_tr1", "GEN_1"]


def compute_node_areas(hgrid):
    """A_node[n] = Σ_{elem containing n} A_elem / n_corners_elem."""
    n_nodes = hgrid["n_nodes"]
    areas = hgrid["areas"]
    elements = hgrid["elements"]
    A_node = np.zeros(n_nodes, dtype=np.float64)
    for ei in range(hgrid["n_elements"]):
        nc, *ids = elements[ei]
        a_share = areas[ei] / int(nc)
        for n in ids[:int(nc)]:
            A_node[int(n)] += a_share
    return A_node


def stream_global_mass(tracer_name, A_node):
    """For each stack, compute global mass time series using zCoordinates."""
    n_stacks = PERIOD_DAYS + 1
    paths_tr = geometry.discover_scribed_stacks(RUN / "outputs", tracer_name)[:n_stacks]
    paths_z  = geometry.discover_scribed_stacks(RUN / "outputs", "zCoordinates")[:n_stacks]
    M_list, V_list, times_list = [], [], []
    for p_tr, p_z in zip(paths_tr, paths_z):
        ds_t = xr.open_dataset(p_tr, engine="h5netcdf")
        ds_z = xr.open_dataset(p_z, engine="h5netcdf")
        da_t = ds_t[tracer_name]      # (time, node, layer)
        da_z = ds_z["zCoordinates"]   # (time, node, layer)
        keep = (da_t["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        T = int(keep.sum())
        if T == 0:
            ds_t.close(); ds_z.close(); continue
        da_t = da_t.isel(time=keep); da_z = da_z.isel(time=keep)

        arr_t = da_t.transpose("time", "nSCHISM_vgrid_layers", "nSCHISM_hgrid_node").values
        arr_z = da_z.transpose("time", "nSCHISM_vgrid_layers", "nSCHISM_hgrid_node").values
        # Mask
        arr_t = np.where(np.isfinite(arr_t) & (np.abs(arr_t) <= 1e30), arr_t, 0.0)
        arr_z = np.where(np.isfinite(arr_z) & (np.abs(arr_z) <= 1e30), arr_z, 0.0)
        T_n, L, N = arr_t.shape

        # layer_thickness between layer k and k+1: dz[t, k, n] = z[t, k+1, n] - z[t, k, n]
        dz = np.diff(arr_z, axis=1)   # (T, L-1, N)
        dz = np.where(dz > 0, dz, 0.0)
        # mid-layer concentration: average of c at layer k and k+1
        c_mid = 0.5 * (arr_t[:, :-1, :] + arr_t[:, 1:, :])   # (T, L-1, N)

        # node mass per timestep:  M[t] = Σ_layer Σ_node c_mid × dz × A_node
        # Compute in chunks to be memory-friendly
        chunk = 16
        m_t = np.zeros(T_n)
        v_t = np.zeros(T_n)
        for t0 in range(0, T_n, chunk):
            t1 = min(T_n, t0 + chunk)
            sl = slice(t0, t1)
            cell_vol = dz[sl] * A_node[None, None, :]                # (Tc, L-1, N)
            m_t[sl] = (c_mid[sl] * cell_vol).sum(axis=(1, 2))
            v_t[sl] = cell_vol.sum(axis=(1, 2))
        M_list.append(m_t); V_list.append(v_t)
        times_list.append(da_t["time"].values[:T_n])
        ds_t.close(); ds_z.close()
        print(f"    {tracer_name} stack {p_tr.name}: T={T_n}, "
              f"M peak so far = {np.concatenate(M_list).max()/1e6:.3f} M mmol, "
              f"V peak so far = {np.concatenate(V_list).max()/1e6:.1f} M m³")
    return (np.concatenate(times_list),
            np.concatenate(M_list),
            np.concatenate(V_list))


def main():
    print("=== Loading hgrid + computing A_node ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    A_node = compute_node_areas(hgrid)
    print(f"  Σ A_node = {A_node.sum()/1e6:.2f} km²   (should equal "
          f"Σ A_elem = {hgrid['areas'].sum()/1e6:.2f} km²)")

    # Source input
    vs = np.loadtxt(RUN / "vsource.th")
    t_vs = vs[:, 0]
    Q3 = vs[:, 3]

    # flux.out for mouth flux (for both tracers)
    print("\n=== Parsing flux.out ===")
    flux_da = parse_flux_out(RUN / "outputs" / "flux.out", n_regions=9)
    t_d = flux_da["time_days"].values
    keep_fx = t_d <= PERIOD_DAYS
    t_fs = t_d[keep_fx] * 86400.0
    flux_times = START + (t_d[keep_fx] * 86400 * 1e9).astype("timedelta64[ns]")
    cum_fx = {}
    for tr in TRACERS:
        if tr in flux_da.coords["tracer"].values:
            rate = flux_da.sel(tracer=tr, region=3).values[keep_fx]
            cum = np.zeros_like(rate)
            cum[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t_fs))
            cum_fx[tr] = cum
            print(f"  {tr}: cum (col 3, INTO CV) = {cum[-1]/1e6:+.4f} M mmol")
        else:
            cum_fx[tr] = None
            print(f"  WARN: tracer '{tr}' not in flux.out")

    # Global integrals
    results = {}
    for tr in TRACERS:
        print(f"\n=== {tr} ===")
        times_g, M_global, V_global = stream_global_mass(tr, A_node)
        t_sec_g = (times_g - times_g[0]) / np.timedelta64(1, "s")
        Q3_g = np.interp(t_sec_g, t_vs, Q3)
        F_in = Q3_g * 1.0
        cum_in = np.zeros_like(F_in)
        cum_in[1:] = np.cumsum(0.5 * (F_in[1:] + F_in[:-1]) * np.diff(t_sec_g))
        results[tr] = {
            "times_g": times_g, "M_global": M_global, "V_global": V_global,
            "cum_in": cum_in,
            "cum_fx_mouth": cum_fx[tr],
        }

    # ---- Plot 2×2: rows = mass / fill fraction; cols = TRC_tr1, GEN_1 ----
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    colors = {"TRC_tr1": "#dc2626", "GEN_1": "#1d4ed8"}
    for j, tr in enumerate(TRACERS):
        r = results[tr]
        c = colors[tr]

        ax = axes[0, j]
        ax.plot(r["times_g"], r["cum_in"]/1e6, color="black", lw=2.0,
                label=f"Cum {tr} INPUT (src #3)")
        ax.plot(r["times_g"], r["M_global"]/1e6, color=c, lw=2.0,
                label=f"GLOBAL {tr} mass")
        ax.set_ylabel("Mass (M mmol)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=10)
        ax.axhline(0, color="#999", lw=0.4)
        ax.set_title(f"({chr(ord('a')+j)}) {tr} — GLOBAL mass vs cumulative input",
                     fontweight="bold", loc="left", fontsize=11)

        ax = axes[1, j]
        retention = np.where(r["cum_in"] > 0, r["M_global"]/r["cum_in"]*100, 0)
        ax.plot(r["times_g"], retention, color=c, lw=2.0,
                label=f"{tr} retention = M_global / cum_in")
        ax.axhline(100, color="black", lw=0.6, ls="--", alpha=0.6,
                   label="100% (perfect conservation)")
        ax.set_ylabel("Retention (% of input)")
        ax.set_ylim(0, 120)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=10)
        ax.set_title(f"({chr(ord('c')+j)}) {tr} — retention fraction over time",
                     fontweight="bold", loc="left", fontsize=11)

        ax = axes[2, j]
        if r["cum_fx_mouth"] is not None:
            ax.plot(flux_times, r["cum_fx_mouth"]/1e6, color=c, lw=2.0,
                    label=f"flux.out col 3 cum {tr} (+ve = INTO CV)")
        else:
            ax.text(0.5, 0.5, f"{tr} not in flux.out",
                    transform=ax.transAxes, ha="center", va="center",
                    color="grey", fontsize=12)
        ax.set_ylabel("Cum mouth flux (M mmol)")
        ax.set_xlabel("Date")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=10)
        ax.axhline(0, color="#999", lw=0.4)
        ax.set_title(f"({chr(ord('e')+j)}) {tr} — cum flux through (2↔3) mouth",
                     fontweight="bold", loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle("TRC_tr1 vs GEN_1 conservation — same msource setup, different tracer pipelines (depress_clutch=.TRUE.)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")

    # ---- Summary ----
    print()
    print("=" * 80)
    print("=== SUMMARY ===")
    print("=" * 80)
    for tr in TRACERS:
        r = results[tr]
        cin = r["cum_in"][-1]
        Mg_peak = r["M_global"].max()
        Mg_end = r["M_global"][-1]
        print(f"\n{tr}:")
        print(f"  Cum input over 14d        : {cin/1e6:>10.3f} M mmol")
        print(f"  Peak GLOBAL mass          : {Mg_peak/1e6:>10.3f} M mmol "
              f"({Mg_peak/cin*100:6.1f}% of input)")
        print(f"  Final GLOBAL mass (14d)   : {Mg_end/1e6:>10.3f} M mmol "
              f"({Mg_end/cin*100:6.1f}% of input)")
        if r["cum_fx_mouth"] is not None:
            print(f"  Cum mouth flux (col 3) net: {r['cum_fx_mouth'][-1]/1e6:>+10.3f} M mmol")

    # Are TR1 and GEN_1 identical?
    a = results["TRC_tr1"]["M_global"]
    b = results["GEN_1"]["M_global"]
    if len(a) == len(b):
        max_abs_diff = np.abs(a - b).max()
        rel_diff = max_abs_diff / max(np.abs(a).max(), 1e-30)
        print(f"\nmax|M_TR1 - M_GEN1| = {max_abs_diff:.3e} mmol  "
              f"(relative to peak: {rel_diff*100:.2f}%)")
        if rel_diff < 0.001:
            print("  -> TRC_tr1 and GEN_1 are BIT-IDENTICAL in global mass.")
            print("     This means: AED's conservative-tracer transport behaves "
                  "identically to SCHISM-native GEN transport.")
        else:
            print(f"  -> TRC_tr1 and GEN_1 DIVERGE — investigate further.")


if __name__ == "__main__":
    main()
