"""
Mouth-flux cross-check, consistent with the CV volume integral.

Reconstructs the volume flux across the 14 (flag 2↔3) mouth edges from
node velocities (horizontalVelX/Y) and **zCoordinates-derived layer thickness**
— the SAME vertical geometry used by the CV volume/storage integral (NOT
ENV_layer_ht, which run 003 lacks). Compares the cumulative reconstruction
against flux.out col 3 (taken from conservation_check's conservation_VOL.csv),
and substitutes it into the volume budget:

    residual_flux  = ΔV_obs − (src_ramped + ∫ flux.out_col3)
    residual_recon = ΔV_obs − (src_ramped + ∫ Q_recon)

Only the ~15 mouth-edge nodes are loaded, so this is light. Sign convention
matches flux.out col 3: positive = flow 3→2 = bay→estuary = INTO the CV.
"""
from __future__ import annotations
import sys, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "nutrient_budget"))
from core import geometry

RUN = HERE.parents[2] / "runs" / "003_P18_flood_flat"
OUTD = HERE.parent / "_outputs" / "003_P18_flood_flat"
START = np.datetime64("2021-04-01T00:00:00")
M_PER_DEG_LAT = 111_320.0
DRAMP_SS = 10.0


def find_23_edges(hg, flag):
    e2e = defaultdict(list); e2n = {}
    for ei in range(hg["n_elements"]):
        nc = int(hg["elements"][ei][0]); ids = [int(v) for v in hg["elements"][ei][1:1+nc]]
        for k in range(nc):
            a, b = ids[k], ids[(k+1) % nc]; key = (min(a, b), max(a, b))
            e2e[key].append(ei); e2n[key] = (a, b)
    out = []
    for key, el in e2e.items():
        if len(el) != 2: continue
        f0, f1 = flag[el[0]], flag[el[1]]
        if {int(f0), int(f1)} == {2, 3}:
            out.append({"a": e2n[key][0], "b": e2n[key][1],
                        "lo": el[0] if f0 == 2 else el[1],
                        "hi": el[0] if f0 == 3 else el[1]})
    return out


def main(days):
    hg = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    x, y = hg["x"], hg["y"]; cent = hg["centroids"]
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    mlon = M_PER_DEG_LAT * math.cos(math.radians(float(np.mean(y))))
    edges = find_23_edges(hg, flag)
    # geometry: length + normal pointing lo(2)->hi(3)
    geom, nodes = [], set()
    for e in edges:
        a, b = e["a"], e["b"]; nodes.add(a); nodes.add(b)
        dx = (x[b]-x[a])*mlon; dy = (y[b]-y[a])*M_PER_DEG_LAT
        L = math.hypot(dx, dy); nx, ny = (-dy/L, dx/L) if L > 1e-9 else (0., 0.)
        vx = (cent[e["hi"], 0]-cent[e["lo"], 0])*mlon; vy = (cent[e["hi"], 1]-cent[e["lo"], 1])*M_PER_DEG_LAT
        if nx*vx + ny*vy < 0: nx, ny = -nx, -ny          # ensure lo(2)->hi(3)
        geom.append({**e, "L": L, "nx": nx, "ny": ny})
    nodes = np.array(sorted(nodes)); npos = {n: i for i, n in enumerate(nodes.tolist())}
    print(f"{len(edges)} mouth edges, {len(nodes)} unique nodes, total gate length {sum(g['L'] for g in geom):.0f} m")

    n_stacks = int(days) + 1
    def load(var):
        ps = geometry.discover_scribed_stacks(RUN/"outputs", var)[:n_stacks]
        arrs, ts = [], []
        for p in ps:
            ds = xr.open_dataset(p, engine="h5netcdf"); da = ds[var]
            dd = list(da.dims)
            tdim = next(d for d in dd if d.lower() == "time")
            ndim = next(d for d in dd if "node" in d.lower())
            ldim = next(d for d in dd if "vgrid" in d.lower() or "layer" in d.lower())
            sub = da.isel({ndim: nodes.tolist()}).transpose(tdim, ndim, ldim).values
            arrs.append(sub); ts.append(ds["time"].values); ds.close()
        return np.concatenate(arrs, 0), np.concatenate(ts)
    vX, t = load("horizontalVelX")
    vY, _ = load("horizontalVelY")
    Z, _ = load("zCoordinates")          # (T, node, level)
    for A in (vX, vY, Z):
        A[~np.isfinite(A) | (np.abs(A) > 1e6)] = np.nan
    dz = np.diff(Z, axis=2)              # (T, node, layer)
    dz = np.where(np.isfinite(dz), dz, 0.0)
    uL = 0.5*(np.nan_to_num(vX[:, :, 1:]) + np.nan_to_num(vX[:, :, :-1]))   # layer-mean u
    vL = 0.5*(np.nan_to_num(vY[:, :, 1:]) + np.nan_to_num(vY[:, :, :-1]))

    Q = np.zeros(vX.shape[0])            # m^3/s, +=into CV
    for g in geom:
        ia, ib = npos[g["a"]], npos[g["b"]]
        ue = 0.5*(uL[:, ia, :] + uL[:, ib, :]); ve = 0.5*(vL[:, ia, :] + vL[:, ib, :])
        dze = 0.5*(dz[:, ia, :] + dz[:, ib, :])
        # u·n is lo->hi (2->3 = OUT); negate so positive = 3->2 = INTO CV (matches flux.out col3)
        Q += -((ue*g["nx"] + ve*g["ny"]) * g["L"] * dze).sum(axis=1)

    tsec = (t - START)/np.timedelta64(1, "s")
    cum_recon = np.concatenate([[0], np.cumsum(0.5*(Q[1:]+Q[:-1])*np.diff(tsec))])

    # budget terms from conservation_VOL.csv (matched window)
    cdir = OUTD / f"conservation_check_{int(days)}d"
    r = list(csv.DictReader(open(cdir/"conservation_VOL.csv")))
    tc = np.array([np.datetime64(z["time"].replace(" ", "T")) for z in r])
    tcs = (tc-START)/np.timedelta64(1, "s")
    d_obs = np.array([float(z["d_obs"]) for z in r]); cum_flux = np.array([float(z["cum_flux"]) for z in r])
    recon_on_c = np.interp(tcs, tsec, cum_recon)
    # ramped source (src1+2+3)
    v = np.loadtxt(RUN/"vsource.th"); tv = v[:, 0]; Qs = v[:, 1]+v[:, 2]+v[:, 3]
    rr = np.tanh(2*tv/86400/DRAMP_SS); cum = np.concatenate([[0], np.cumsum(0.5*(Qs[1:]*rr[1:]+Qs[:-1]*rr[:-1])*np.diff(tv))])
    src = np.interp(tcs, tv, cum)
    res_flux = d_obs - (src + cum_flux)
    res_recon = d_obs - (src + recon_on_c)
    thru = max(abs(src[-1]), abs(cum_flux[-1]), 1)
    print(f"[{days}d] end cum: flux.out col3 = {cum_flux[-1]:+.3e} | recon(zcoord) = {recon_on_c[-1]:+.3e} m^3 "
          f"(recon/flux = {recon_on_c[-1]/cum_flux[-1]:.2f}x)")
    print(f"      residual with flux.out  = {res_flux[-1]:+.3e} ({100*res_flux[-1]/thru:+.1f}%)")
    print(f"      residual with recon     = {res_recon[-1]:+.3e} ({100*res_recon[-1]/thru:+.1f}%)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    a1.plot(tc, cum_flux/1e6, color="grey", lw=2, label="∫ flux.out col 3 (diagnostic)")
    a1.plot(tc, recon_on_c/1e6, color="tab:blue", lw=2, label="∫ Q_recon (node vel × zCoord dz)")
    a1.set_ylabel("cumulative mouth volume flux (×10⁶ m³)"); a1.set_title(f"Mouth flux: flux.out vs zCoord reconstruction ({days}d)")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2.plot(tc, d_obs/1e6, color="black", lw=2.2, label="observed ΔV")
    a2.plot(tc, (src+cum_flux)/1e6, color="grey", lw=1.6, ls="--", label="predicted (flux.out)")
    a2.plot(tc, (src+recon_on_c)/1e6, color="tab:blue", lw=1.6, ls="--", label="predicted (recon)")
    a2.plot(tc, res_flux/1e6, color="tab:orange", lw=1.4, label="residual (flux.out)")
    a2.plot(tc, res_recon/1e6, color="tab:green", lw=1.4, label="residual (recon)")
    a2.axhline(0, color="k", lw=0.6); a2.set_ylabel("ΔV (×10⁶ m³)"); a2.set_title("Volume closure: flux.out vs reconstruction")
    a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout(); out = OUTD / f"edge_flux_zcoord_{int(days)}d.png"
    fig.savefig(out, dpi=150); print(f"  wrote {out}")


if __name__ == "__main__":
    for d in (14, 53):
        main(d)
