"""
Independent edge-flux reconstruction for the (2<->3) mouth edges, using
horizontalVelX/Y at nodes + ENV_layer_ht at faces. Cross-check against
flux.out col 3.

For each of the 14 (2<->3) edges in the mesh:
    edge endpoints  : nodes a, b
    adjacent elems  : e_lo (flag=2), e_hi (flag=3)
    edge length L   : great-circle approximation (lat/lon)
    edge tangent t  : (b - a)/|b - a|
    edge normal  n  : rotate t by +90°. Convention: n points from
                      LOW-flag side (flag=2) to HIGH-flag side (flag=3).
                      This matches the Fortran's "max(flag)" convention
                      where positive vnn = flow into the high-flag region.

Per timestep, per layer:
    u_edge, v_edge = average of u, v at the two endpoint nodes
    layer_ht_edge   = average of ENV_layer_ht at the two adjacent elements
    Q_layer  = (u_edge*nx + v_edge*ny) * L * layer_ht_edge          [m³/s]
    The sign of Q is INVERTED to match flux.out's convention
    (positive = flow from HIGH to LOW = "INTO low-flag region"),
    by negating the velocity-normal because our normal points low->high.

Then accumulate volume and tracer (TR1, TR3) flux per edge, then sum
over edges, then integrate over 14 days.

If our reconstruction == flux.out col 3, the Fortran/parser are correct
and the mass-balance shortfall lies elsewhere. If reconstruction is
much larger than col 3, then SCHISM is under-reporting and we have a
model-side issue.
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry
from core.boundary_fluxes import parse_flux_out


CFG = active_run()
RUN = CFG.run_dir
OUT_PNG = CFG.out_dir / "edge_flux_reconstruction.png"
START = CFG.start
PERIOD_DAYS = CFG.period_days
M_PER_DEG_LAT = 111_320.0


def find_23_edges(hgrid, flag):
    """Find all (2<->3) edges. Return list of dicts with edge geometry."""
    n_elem = hgrid["n_elements"]
    elements = hgrid["elements"]
    edge2elem = defaultdict(list)
    edge2nodes = {}
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k+1)%nc]
            key = (min(a, b), max(a, b))
            edge2elem[key].append(ei)
            edge2nodes[key] = (a, b)
    out = []
    for key, elems in edge2elem.items():
        if len(elems) != 2: continue
        e0, e1 = elems
        f0, f1 = flag[e0], flag[e1]
        if {f0, f1} == {2, 3}:
            e_lo = e0 if f0 == 2 else e1
            e_hi = e0 if f0 == 3 else e1
            a, b = edge2nodes[key]
            out.append({
                "node_a": a, "node_b": b,
                "elem_lo": e_lo, "elem_hi": e_hi,
                "key": key,
            })
    return out


def main():
    print("=== Loading hgrid + fluxflag ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    x, y = hgrid["x"], hgrid["y"]
    elements = hgrid["elements"]
    y_mid = float(np.mean(y))
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(y_mid))

    edges = find_23_edges(hgrid, flag)
    print(f"  {len(edges)} (2<->3) edges found")

    # Compute geometry for each edge: length L (m), normal (nx, ny) pointing low->high
    edge_geom = []
    nodes_needed = set()
    elems_needed = set()
    for e in edges:
        a, b = e["node_a"], e["node_b"]
        nodes_needed.add(a); nodes_needed.add(b)
        elems_needed.add(e["elem_lo"]); elems_needed.add(e["elem_hi"])
        # edge in metric coords
        dx = (x[b] - x[a]) * m_per_deg_lon
        dy = (y[b] - y[a]) * M_PER_DEG_LAT
        L = math.sqrt(dx*dx + dy*dy)
        # tangent then rotate +90° to get normal: t=(dx,dy)/L  ;  n=(-dy,dx)/L
        if L < 1e-9:
            nx, ny = 0.0, 0.0
        else:
            nx = -dy / L
            ny = dx / L
        # We want normal to point from elem_lo (flag=2) to elem_hi (flag=3).
        # Check by looking at element centroids: vector from lo-centroid to hi-centroid.
        cx_lo, cy_lo = hgrid["centroids"][e["elem_lo"]]
        cx_hi, cy_hi = hgrid["centroids"][e["elem_hi"]]
        v_lo_to_hi_x = (cx_hi - cx_lo) * m_per_deg_lon
        v_lo_to_hi_y = (cy_hi - cy_lo) * M_PER_DEG_LAT
        # If dot(n, v_lo_to_hi) < 0, flip normal
        if nx * v_lo_to_hi_x + ny * v_lo_to_hi_y < 0:
            nx, ny = -nx, -ny
        edge_geom.append({**e, "L": L, "nx": nx, "ny": ny})

    print(f"  edge lengths (m): "
          f"min={min(g['L'] for g in edge_geom):.1f}, "
          f"max={max(g['L'] for g in edge_geom):.1f}, "
          f"mean={np.mean([g['L'] for g in edge_geom]):.1f}, "
          f"total={sum(g['L'] for g in edge_geom):.1f}")

    nodes_arr = np.array(sorted(nodes_needed), dtype=int)
    node_pos = {n: i for i, n in enumerate(nodes_arr.tolist())}
    elems_arr = np.array(sorted(elems_needed), dtype=int)
    elem_pos = {e: i for i, e in enumerate(elems_arr.tolist())}

    # For upwind tracer concentration we need element-mean tracer for each
    # adjacent element. Build a per-element vertex list for the elements involved.
    elem_vertex_nodes = {}  # elem_id -> [node_ids]
    tracer_nodes_needed = set()
    for ei in elems_arr.tolist():
        nc, *ids = elements[ei]
        nc = int(nc)
        nlist = [int(n) for n in ids[:nc]]
        elem_vertex_nodes[ei] = nlist
        tracer_nodes_needed.update(nlist)
    tnodes_arr = np.array(sorted(tracer_nodes_needed), dtype=int)
    tnode_pos = {n: i for i, n in enumerate(tnodes_arr.tolist())}

    print(f"  unique velocity nodes needed: {len(nodes_arr)}")
    print(f"  unique elements needed: {len(elems_arr)}")
    print(f"  unique tracer-mean nodes needed: {len(tnodes_arr)}")

    # ------------------------------------------------------------------
    # Stream through stacks: load needed subsets, compute per-timestep fluxes
    # ------------------------------------------------------------------
    n_stacks = PERIOD_DAYS + 1
    paths_velX = geometry.discover_scribed_stacks(RUN / "outputs", "horizontalVelX")[:n_stacks]
    paths_velY = geometry.discover_scribed_stacks(RUN / "outputs", "horizontalVelY")[:n_stacks]
    paths_tr1  = geometry.discover_scribed_stacks(RUN / "outputs", "TRC_tr1")[:n_stacks]
    paths_tr3  = geometry.discover_scribed_stacks(RUN / "outputs", "TRC_tr3")[:n_stacks]
    paths_cmb  = geometry.discover_cmb_stacks(RUN / "outputs")[:n_stacks]

    Q_total_t  = []   # m^3/s, net flow region2->region3 (positive=OUT of CV, low->high)
    M_tr1_t    = []   # mmol/s, sign matches Q
    M_tr3_t    = []
    Q_pos_t    = []   # m^3/s, gross positive
    Q_neg_t    = []
    M_tr3_pos_t = []
    M_tr3_neg_t = []
    times_all  = []

    for vX_p, vY_p, tr1_p, tr3_p, cm_p in zip(paths_velX, paths_velY,
                                              paths_tr1, paths_tr3, paths_cmb):
        ds_vX = xr.open_dataset(vX_p, engine="h5netcdf")
        ds_vY = xr.open_dataset(vY_p, engine="h5netcdf")
        ds_t1 = xr.open_dataset(tr1_p, engine="h5netcdf")
        ds_t3 = xr.open_dataset(tr3_p, engine="h5netcdf")
        ds_cm = xr.open_dataset(cm_p, engine="h5netcdf")

        for ds in (ds_vX, ds_vY, ds_t1, ds_t3, ds_cm):
            pass  # could trim by time, but stacks are small enough

        node_dim = next(d for d in ds_vX["horizontalVelX"].dims
                         if "node" in d.lower())
        layer_dim_s = next(d for d in ds_vX["horizontalVelX"].dims
                            if "vgrid" in d.lower() or "layer" in d.lower())
        face_dim   = next(d for d in ds_cm["ENV_layer_ht"].dims
                            if "face" in d.lower())
        layer_dim_c = next(d for d in ds_cm["ENV_layer_ht"].dims
                            if "vgrid" in d.lower() or "layer" in d.lower())

        # Subset to needed nodes/elems
        vX = ds_vX["horizontalVelX"].isel({node_dim: nodes_arr.tolist()}).transpose(
            "time", layer_dim_s, node_dim).values
        vY = ds_vY["horizontalVelY"].isel({node_dim: nodes_arr.tolist()}).transpose(
            "time", layer_dim_s, node_dim).values
        tr1 = ds_t1["TRC_tr1"].isel({node_dim: tnodes_arr.tolist()}).transpose(
            "time", layer_dim_s, node_dim).values
        tr3 = ds_t3["TRC_tr3"].isel({node_dim: tnodes_arr.tolist()}).transpose(
            "time", layer_dim_s, node_dim).values
        lh = ds_cm["ENV_layer_ht"].isel({face_dim: elems_arr.tolist()}).transpose(
            "time", layer_dim_c, face_dim).values

        # mask fill values
        vX = np.where(np.isfinite(vX) & (np.abs(vX) <= 1e30), vX, 0.0)
        vY = np.where(np.isfinite(vY) & (np.abs(vY) <= 1e30), vY, 0.0)
        tr1 = np.where(np.isfinite(tr1) & (np.abs(tr1) <= 1e30), tr1, 0.0)
        tr3 = np.where(np.isfinite(tr3) & (np.abs(tr3) <= 1e30), tr3, 0.0)
        lh = np.where(np.isfinite(lh) & (lh > 0), lh, 0.0)

        T = vX.shape[0]
        L_ax = vX.shape[1]   # 6 layers
        Q_net  = np.zeros(T)
        Q_pos  = np.zeros(T)
        Q_neg  = np.zeros(T)
        Mt1    = np.zeros(T)
        Mt3    = np.zeros(T)
        Mt3_p  = np.zeros(T)
        Mt3_n  = np.zeros(T)

        # Per-edge accumulator
        for g in edge_geom:
            a_pos = node_pos[g["node_a"]]
            b_pos = node_pos[g["node_b"]]
            elem_lo_pos = elem_pos[g["elem_lo"]]
            elem_hi_pos = elem_pos[g["elem_hi"]]
            L  = g["L"]
            nx = g["nx"]; ny = g["ny"]
            # Element-mean concentration of TR1, TR3 for upwind
            verts_lo = elem_vertex_nodes[g["elem_lo"]]
            verts_hi = elem_vertex_nodes[g["elem_hi"]]
            lo_idx = np.array([tnode_pos[n] for n in verts_lo])
            hi_idx = np.array([tnode_pos[n] for n in verts_hi])
            c_t1_lo = tr1[:, :, lo_idx].mean(axis=2)   # (T, L)
            c_t1_hi = tr1[:, :, hi_idx].mean(axis=2)
            c_t3_lo = tr3[:, :, lo_idx].mean(axis=2)
            c_t3_hi = tr3[:, :, hi_idx].mean(axis=2)

            # Velocity at edge: average of endpoint nodes
            u_e = 0.5 * (vX[:, :, a_pos] + vX[:, :, b_pos])    # (T, L)
            v_e = 0.5 * (vY[:, :, a_pos] + vY[:, :, b_pos])
            # Velocity component along normal (from lo to hi)
            vn = u_e * nx + v_e * ny   # m/s
            # Layer thickness at edge: average of the two adjacent element lh
            lh_lo = lh[:, :, elem_lo_pos]
            lh_hi = lh[:, :, elem_hi_pos]
            lh_e = 0.5 * (lh_lo + lh_hi)
            # Per-layer volumetric flux: vn * L * lh
            q = vn * L * lh_e     # m^3/s; positive = lo->hi (CV out -> bay)
            # Sum over layers for this edge
            q_per_t = q.sum(axis=1)
            Q_net  += q_per_t

            # Gross positive (q > 0 = OUT of CV) — but per-layer sign matters
            q_p = np.where(q > 0, q, 0).sum(axis=1)
            q_n = np.where(q < 0, q, 0).sum(axis=1)
            Q_pos += q_p
            Q_neg += q_n

            # Tracer flux upwind: q>0 picks up c_lo (low side), q<0 picks up c_hi
            ft1_per_layer = np.where(q > 0, q * c_t1_lo, q * c_t1_hi)
            ft3_per_layer = np.where(q > 0, q * c_t3_lo, q * c_t3_hi)
            ft1_t = ft1_per_layer.sum(axis=1)
            ft3_t = ft3_per_layer.sum(axis=1)
            Mt1 += ft1_t
            Mt3 += ft3_t
            # Gross pos / neg of tr3
            Mt3_p += np.where(ft3_per_layer > 0, ft3_per_layer, 0).sum(axis=1)
            Mt3_n += np.where(ft3_per_layer < 0, ft3_per_layer, 0).sum(axis=1)

        # Now flip sign so positive = flow HIGH -> LOW = INTO low-flag region
        # = INTO CV (matching flux.out's convention)
        Q_total_t.append(-Q_net)
        Q_pos_t.append(-Q_neg)   # what was negative (hi->lo) becomes positive into CV
        Q_neg_t.append(-Q_pos)
        M_tr1_t.append(-Mt1)
        M_tr3_t.append(-Mt3)
        M_tr3_pos_t.append(-Mt3_n)
        M_tr3_neg_t.append(-Mt3_p)
        times_all.append(ds_vX["time"].values[:T])

        ds_vX.close(); ds_vY.close(); ds_t1.close(); ds_t3.close(); ds_cm.close()
        print(f"  stack {vX_p.name}: T={T}, "
              f"Q_net mean={Q_net.mean():+8.2f}, "
              f"Mt3 mean={Mt3.mean():+8.2f}")

    times = np.concatenate(times_all)
    Q  = np.concatenate(Q_total_t)
    Mt1 = np.concatenate(M_tr1_t)
    Mt3 = np.concatenate(M_tr3_t)
    Q_p = np.concatenate(Q_pos_t)
    Q_n = np.concatenate(Q_neg_t)
    Mt3p = np.concatenate(M_tr3_pos_t)
    Mt3n = np.concatenate(M_tr3_neg_t)
    t_sec = (times - times[0]) / np.timedelta64(1, "s")
    keep = (times - START) <= np.timedelta64(PERIOD_DAYS, "D")
    Q   = Q[keep];  Mt1 = Mt1[keep]; Mt3 = Mt3[keep]
    Q_p = Q_p[keep]; Q_n = Q_n[keep]
    Mt3p = Mt3p[keep]; Mt3n = Mt3n[keep]
    times = times[keep]; t_sec = t_sec[keep]

    def cumtrap(rate, t):
        out = np.zeros_like(rate)
        out[1:] = np.cumsum(0.5*(rate[1:]+rate[:-1])*np.diff(t))
        return out

    cum_Q    = cumtrap(Q, t_sec)
    cum_Mt1  = cumtrap(Mt1, t_sec)
    cum_Mt3  = cumtrap(Mt3, t_sec)
    cum_Qp   = cumtrap(Q_p, t_sec)
    cum_Qn   = cumtrap(Q_n, t_sec)
    cum_Mt3p = cumtrap(Mt3p, t_sec)
    cum_Mt3n = cumtrap(Mt3n, t_sec)

    # ------------------------------------------------------------------
    # flux.out reference values
    # ------------------------------------------------------------------
    print("\n=== Loading flux.out for reference ===")
    flux_da = parse_flux_out(RUN / "outputs" / "flux.out", n_regions=9)
    t_d = flux_da["time_days"].values
    keep_fx = t_d <= PERIOD_DAYS
    t_fs = t_d[keep_fx] * 86400.0
    rate_VOL  = flux_da.sel(tracer="VOL",     region=3).values[keep_fx]
    rate_tr1  = flux_da.sel(tracer="TRC_tr1", region=3).values[keep_fx]
    rate_tr3  = flux_da.sel(tracer="TRC_tr3", region=3).values[keep_fx]
    flux_times = START + (t_d[keep_fx] * 86400 * 1e9).astype("timedelta64[ns]")
    cum_fx_VOL = cumtrap(rate_VOL, t_fs)
    cum_fx_tr1 = cumtrap(rate_tr1, t_fs)
    cum_fx_tr3 = cumtrap(rate_tr3, t_fs)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"=== Cross-check: edge-flux reconstruction vs flux.out (14d, 14 edges) ===")
    print("=" * 72)
    print(f"  ALL values: positive = INTO the CV (low-flag region 2)")
    print()
    print(f"VOLUME (m^3 cum)")
    print(f"  Reconstruction net : {cum_Q[-1]:+12.3e}   pos={cum_Qp[-1]:+12.3e}  neg={cum_Qn[-1]:+12.3e}")
    print(f"  flux.out col 3 net : {cum_fx_VOL[-1]:+12.3e}")
    print(f"  ratio recon/flux   : {cum_Q[-1]/cum_fx_VOL[-1]:.3f}")
    print()
    print(f"TRC_tr1 (mmol cum)")
    print(f"  Reconstruction net : {cum_Mt1[-1]:+12.3e}")
    print(f"  flux.out col 3 net : {cum_fx_tr1[-1]:+12.3e}")
    print()
    print(f"TRC_tr3 (mmol cum)")
    print(f"  Reconstruction net : {cum_Mt3[-1]:+12.3e}   pos={cum_Mt3p[-1]:+12.3e}  neg={cum_Mt3n[-1]:+12.3e}")
    print(f"  flux.out col 3 net : {cum_fx_tr3[-1]:+12.3e}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axA, axB, axC = axes
    axA.plot(times, cum_Q / 1e6, color="#1d4ed8", lw=2.0,
             label="Recon  cum  net  (sum of 14 edges, into CV)")
    axA.plot(flux_times, cum_fx_VOL / 1e6, color="#dc2626", lw=2.0, ls="--",
             label="flux.out col 3 cum net")
    axA.set_ylabel("VOL cumulative (M m³)")
    axA.grid(True, ls=":", alpha=0.4)
    axA.legend(loc="best", fontsize=10)
    axA.set_title("(a) VOLUME cum across (2↔3) mouth edges",
                  fontweight="bold", loc="left", fontsize=11)
    axA.axhline(0, color="#999", lw=0.4)

    axB.plot(times, cum_Mt1 / 1e6, color="#1d4ed8", lw=2.0,
             label="Recon  cum  net  (into CV)")
    axB.plot(flux_times, cum_fx_tr1 / 1e6, color="#dc2626", lw=2.0, ls="--",
             label="flux.out col 3 cum net")
    axB.set_ylabel("TRC_tr1 cum (M mmol)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="best", fontsize=10)
    axB.set_title("(b) TRC_tr1 cum across (2↔3) mouth edges",
                  fontweight="bold", loc="left", fontsize=11)
    axB.axhline(0, color="#999", lw=0.4)

    axC.plot(times, cum_Mt3 / 1e6, color="#1d4ed8", lw=2.0,
             label="Recon  cum  net  (into CV)")
    axC.plot(flux_times, cum_fx_tr3 / 1e6, color="#dc2626", lw=2.0, ls="--",
             label="flux.out col 3 cum net")
    axC.set_ylabel("TRC_tr3 cum (M mmol)")
    axC.set_xlabel("Date (model time)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="best", fontsize=10)
    axC.set_title("(c) TRC_tr3 cum across (2↔3) mouth edges",
                  fontweight="bold", loc="left", fontsize=11)
    axC.axhline(0, color="#999", lw=0.4)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle("Direct edge-flux reconstruction vs flux.out col 3 — 14 (2↔3) mouth edges",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
