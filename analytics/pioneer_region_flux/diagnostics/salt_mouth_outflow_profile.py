"""
Vertical structure of the mouth exchange over the FLAT-TIDE window (run 003).

The box model showed the 14-day salt budget hinges entirely on the OUTFLOW
end-member salinity: outflow at CV-mean S exports far too much salt; outflow at
a fresh surface-plume S ~closes the budget. This script tests, directly at the
14 flag-2<->3 mouth faces, whether the exchange is two-layered:

    surface layers : outflow (un < 0, leaving CV)  carrying FRESH water
    bottom  layers : inflow  (un > 0, into CV)      carrying SALTY water

(classic estuarine gravitational circulation). If so, the CV salinifies even
with net volume export, because it exports fresh surface water and imports salt
at depth - a transport with little NET volume signature, invisible to flux.out's
column-integrated salt and to a bulk box model.

Method: at each mouth face, each level-pair (layer), each output time in the
window, take the velocity component normal to the face (oriented +into CV),
the layer salinity, and the layer mid-depth normalised 0(bottom)->1(surface)
from zCoordinates. Bin by normalised depth; report the time-mean profile of
normal velocity and salinity, and the dz-weighted inflow/outflow mean salinity.
"""
from __future__ import annotations
import sys, math
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
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
START = np.datetime64("2021-04-01T00:00:00"); M_PER_DEG_LAT = 111_320.0
DAYS = 14
NBIN = 12


def find_23_edges(hg, flag):
    e2e = defaultdict(list); e2n = {}
    for ei in range(hg["n_elements"]):
        nc = int(hg["elements"][ei][0]); ids = [int(v) for v in hg["elements"][ei][1:1+nc]]
        for k in range(nc):
            a, b = ids[k], ids[(k+1) % nc]; key = (min(a, b), max(a, b))
            e2e[key].append(ei); e2n[key] = (a, b)
    out = []
    for key, el in e2e.items():
        if len(el) == 2 and {int(flag[el[0]]), int(flag[el[1]])} == {2, 3}:
            out.append({"a": e2n[key][0], "b": e2n[key][1],
                        "lo": el[0] if flag[el[0]] == 2 else el[1],
                        "hi": el[0] if flag[el[0]] == 3 else el[1]})
    return out


def main():
    hg = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    x, y, cent = hg["x"], hg["y"], hg["centroids"]
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    mlon = M_PER_DEG_LAT * math.cos(math.radians(float(np.mean(y))))
    geom, nodes = [], set()
    for e in find_23_edges(hg, flag):
        a, b = e["a"], e["b"]; nodes.add(a); nodes.add(b)
        dx = (x[b]-x[a])*mlon; dy = (y[b]-y[a])*M_PER_DEG_LAT; L = math.hypot(dx, dy)
        nx, ny = (-dy/L, dx/L) if L > 1e-9 else (0., 0.)
        vx = (cent[e["hi"],0]-cent[e["lo"],0])*mlon; vy = (cent[e["hi"],1]-cent[e["lo"],1])*M_PER_DEG_LAT
        if nx*vx+ny*vy < 0: nx, ny = -nx, -ny       # orient +into CV (toward flag-2/lo side)
        geom.append({**e, "L": L, "nx": nx, "ny": ny})
    nodes = np.array(sorted(nodes)); npos = {n: i for i, n in enumerate(nodes.tolist())}
    print(f"{len(geom)} mouth faces, {len(nodes)} nodes")

    n_stacks = DAYS + 1
    def load(var):
        ps = geometry.discover_scribed_stacks(RUN/"outputs", var)[:n_stacks]; arrs, ts = [], []
        for p in ps:
            ds = xr.open_dataset(p, engine="h5netcdf"); da = ds[var]; dd = list(da.dims)
            td = next(d for d in dd if d.lower() == "time"); nd = next(d for d in dd if "node" in d.lower())
            ld = next(d for d in dd if "vgrid" in d.lower() or "layer" in d.lower())
            arrs.append(da.isel({nd: nodes.tolist()}).transpose(td, nd, ld).values); ts.append(ds["time"].values); ds.close()
        return np.concatenate(arrs, 0), np.concatenate(ts)
    vX, t = load("horizontalVelX"); vY, _ = load("horizontalVelY"); Z, _ = load("zCoordinates"); S, _ = load("salinity")
    keep = (t - START) <= np.timedelta64(DAYS, "D")
    vX, vY, Z, S, t = vX[keep], vY[keep], Z[keep], S[keep], t[keep]
    for A in (vX, vY, Z, S): A[~np.isfinite(A) | (np.abs(A) > 1e6)] = np.nan

    # layer-centred quantities (levels -> layers)
    Zmid = 0.5*(Z[:, :, 1:] + Z[:, :, :-1])                 # (T, node, layer)
    dz = np.abs(np.diff(Z, axis=2)); dz = np.where(np.isfinite(dz), dz, 0.0)
    uL = 0.5*(np.nan_to_num(vX[:, :, 1:]) + np.nan_to_num(vX[:, :, :-1]))
    vL = 0.5*(np.nan_to_num(vY[:, :, 1:]) + np.nan_to_num(vY[:, :, :-1]))
    SL = 0.5*(np.nan_to_num(S[:, :, 1:]) + np.nan_to_num(S[:, :, :-1]))
    # depth BELOW SURFACE (m), per (T,node) layer mid-point (robust to LSC2)
    zs = np.nanmax(Z, axis=2, keepdims=True)                # free surface
    zdepth = zs - Zmid                                      # 0 at surface, + downward
    DMAX = float(np.nanpercentile(zdepth[np.isfinite(zdepth)], 98))
    print(f"  mouth column depth (98th pct) = {DMAX:.2f} m")

    # accumulate per depth-bin: dz-weighted un, S, and signed volume by sign
    edges = np.linspace(0, DMAX, NBIN + 1); ctr = 0.5*(edges[1:] + edges[:-1])
    sum_un = np.zeros(NBIN); sum_S = np.zeros(NBIN); sum_w = np.zeros(NBIN)
    qin_S = qout_S = qin = qout = 0.0
    for g in geom:
        ia, ib = npos[g["a"]], npos[g["b"]]
        um = 0.5*(uL[:, ia, :] + uL[:, ib, :]); vm = 0.5*(vL[:, ia, :] + vL[:, ib, :])
        un = -(um*g["nx"] + vm*g["ny"])                    # +into CV (m/s), per layer
        Se = 0.5*(SL[:, ia, :] + SL[:, ib, :])
        dze = 0.5*(dz[:, ia, :] + dz[:, ib, :])
        zn = 0.5*(zdepth[:, ia, :] + zdepth[:, ib, :])     # depth below surface (m)
        q = un * g["L"] * dze                              # layer volume flux (m³/s), +into CV
        # bin profiles (dz-weighted, time+face mean)
        m = np.isfinite(zn) & (dze > 0)
        bi = np.clip(np.digitize(zn[m], edges) - 1, 0, NBIN - 1)
        w = dze[m]
        np.add.at(sum_un, bi, un[m] * w); np.add.at(sum_S, bi, Se[m] * w); np.add.at(sum_w, bi, w)
        # flux-weighted in/out salinity
        qin += np.where(q > 0, q, 0).sum(); qin_S += np.where(q > 0, q*Se, 0).sum()
        qout += np.where(q < 0, q, 0).sum(); qout_S += np.where(q < 0, q*Se, 0).sum()
    prof_un = sum_un / np.where(sum_w > 0, sum_w, np.nan)
    prof_S = sum_S / np.where(sum_w > 0, sum_w, np.nan)
    S_in = qin_S / qin if qin else float("nan")
    S_out = qout_S / qout if qout else float("nan")

    print(f"\n=== {DAYS}-day flat-tide mouth exchange (flux-weighted) ===")
    print(f"  inflow  : {qin:+.3e} m³/s-sum,  flux-weighted salinity = {S_in:6.2f} psu")
    print(f"  outflow : {qout:+.3e} m³/s-sum,  flux-weighted salinity = {S_out:6.2f} psu")
    print(f"  -> outflow is {'FRESHER' if S_out < S_in else 'saltier'} than inflow by {abs(S_in-S_out):.2f} psu")
    print(f"\n  profile (depth below surface, m):")
    print(f"   depth   un(into CV, m/s)   S(psu)")
    for c, u, s in zip(ctr, prof_un, prof_S):
        if np.isfinite(u):
            print(f"   {c:4.2f}     {u:+8.4f}        {s:6.2f}   {'<- OUT' if u<0 else '-> IN '}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    ax = axes[0]
    ax.plot(prof_un*100, ctr, "o-", color="tab:blue", lw=2)
    ax.axvline(0, color="k", lw=0.8); ax.invert_yaxis()
    ax.set_xlabel("normal velocity into CV  (cm/s)"); ax.set_ylabel("depth below surface (m)")
    ax.set_title("(a) vertical structure of mouth flow\n(+ = inflow, − = outflow)", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.3)
    ax.fill_betweenx(ctr, prof_un*100, 0, where=(prof_un<0), color="tab:blue", alpha=0.12)
    ax.fill_betweenx(ctr, prof_un*100, 0, where=(prof_un>0), color="tab:red", alpha=0.12)
    ax2 = axes[1]
    ax2.plot(prof_S, ctr, "o-", color="tab:red", lw=2); ax2.invert_yaxis()
    ax2.set_xlabel("salinity (psu)"); ax2.set_ylabel("depth below surface (m)")
    ax2.set_title("(b) salinity profile at the mouth", fontweight="bold", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.axvline(S_out, color="tab:blue", ls="--", lw=1.5, label=f"flux-wtd OUTFLOW S = {S_out:.1f}")
    ax2.axvline(S_in, color="tab:green", ls="--", lw=1.5, label=f"flux-wtd INFLOW S = {S_in:.1f}")
    ax2.legend(fontsize=8, loc="best")
    fig.suptitle(f"Pioneer mouth — {DAYS}-day flat-tide vertical exchange structure "
                 f"(time-mean over the 14 flag-2↔3 faces)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTD / "salt_mouth_outflow_profile_003.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
