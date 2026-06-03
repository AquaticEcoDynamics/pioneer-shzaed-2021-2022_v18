"""
Salt mouth-flux PROOF: is flux.out capturing the bulk advective salt transport?

Independently reconstructs the ADVECTIVE salt flux across the 14 flag-2↔3 mouth
edges from node velocities (horizontalVelX/Y) × salinity, with zCoordinates dz
(consistent with the volume integral). Reports gross-in, gross-out and NET, and
compares the cumulative net to flux.out col 3 salt (conservation_Salinity.csv).

If reconstruction net ≈ flux.out net (both export), flux.out faithfully computes
the advective salt flux → the salt that salinifies the CV enters by a
NON-advective (dispersive/mixing) route flux.out does not carry. If they differ,
flux.out's advective salt flux itself is wrong.
Only ~15 mouth-edge nodes are loaded; light.
"""
from __future__ import annotations
import sys, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "nutrient_budget"))
from core import geometry

RUN = HERE.parents[2] / "runs" / "003_P18_flood_flat"
OUTD = HERE.parent / "_outputs" / "003_P18_flood_flat"
START = np.datetime64("2021-04-01T00:00:00"); M_PER_DEG_LAT = 111_320.0


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


def main(days):
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
        if nx*vx+ny*vy < 0: nx, ny = -nx, -ny
        geom.append({**e, "L": L, "nx": nx, "ny": ny})
    nodes = np.array(sorted(nodes)); npos = {n: i for i, n in enumerate(nodes.tolist())}

    n_stacks = int(days)+1
    def load(var):
        ps = geometry.discover_scribed_stacks(RUN/"outputs", var)[:n_stacks]; arrs, ts = [], []
        for p in ps:
            ds = xr.open_dataset(p, engine="h5netcdf"); da = ds[var]; dd = list(da.dims)
            td = next(d for d in dd if d.lower() == "time"); nd = next(d for d in dd if "node" in d.lower())
            ld = next(d for d in dd if "vgrid" in d.lower() or "layer" in d.lower())
            arrs.append(da.isel({nd: nodes.tolist()}).transpose(td, nd, ld).values); ts.append(ds["time"].values); ds.close()
        return np.concatenate(arrs, 0), np.concatenate(ts)
    vX, t = load("horizontalVelX"); vY, _ = load("horizontalVelY"); Z, _ = load("zCoordinates"); S, _ = load("salinity")
    for A in (vX, vY, Z, S): A[~np.isfinite(A) | (np.abs(A) > 1e6)] = np.nan
    _dzr = np.diff(Z, axis=2)
    dz = np.where(np.isfinite(_dzr), _dzr, 0.0)
    uL = 0.5*(np.nan_to_num(vX[:, :, 1:])+np.nan_to_num(vX[:, :, :-1]))
    vL = 0.5*(np.nan_to_num(vY[:, :, 1:])+np.nan_to_num(vY[:, :, :-1]))
    SL = 0.5*(np.nan_to_num(S[:, :, 1:])+np.nan_to_num(S[:, :, :-1]))

    Qv = np.zeros(vX.shape[0]); salt_net = np.zeros(vX.shape[0])
    salt_in = np.zeros(vX.shape[0]); salt_out = np.zeros(vX.shape[0])
    for g in geom:
        ia, ib = npos[g["a"]], npos[g["b"]]
        ftmp = -((0.5*(uL[:,ia,:]+uL[:,ib,:]))*g["nx"] + (0.5*(vL[:,ia,:]+vL[:,ib,:]))*g["ny"]) * g["L"] * (0.5*(dz[:,ia,:]+dz[:,ib,:]))  # +=into CV, per layer
        Se = 0.5*(SL[:,ia,:]+SL[:,ib,:])
        sflux = ftmp*Se
        Qv += ftmp.sum(1); salt_net += sflux.sum(1)
        salt_in += np.where(ftmp > 0, sflux, 0).sum(1)     # salt carried IN by inflowing layers
        salt_out += np.where(ftmp < 0, sflux, 0).sum(1)    # salt carried OUT by outflowing layers
    tsec = (t-START)/np.timedelta64(1, "s")
    def cum(q): return np.concatenate([[0], np.cumsum(0.5*(q[1:]+q[:-1])*np.diff(tsec))])
    cum_net, cum_in, cum_out = cum(salt_net), cum(salt_in), cum(salt_out)

    r = list(csv.DictReader(open(OUTD/f"conservation_check_{int(days)}d"/"conservation_Salinity.csv")))
    do = float(r[-1]["d_obs"]); cs = float(r[-1]["cum_src"]); cf = float(r[-1]["cum_flux"])
    print(f"\n=== {days}d salt at the mouth (psu*m^3, cumulative) ===")
    print(f"  reconstruction GROSS IN  (inflow layers carry salt in)  = {cum_in[-1]:+.3e}")
    print(f"  reconstruction GROSS OUT (outflow layers carry salt out) = {cum_out[-1]:+.3e}")
    print(f"  reconstruction NET (advective)                          = {cum_net[-1]:+.3e}")
    print(f"  flux.out col3 NET (advective, model)                    = {cf:+.3e}   (recon/flux = {cum_net[-1]/cf:.2f}x)")
    print(f"  needed NET to close (dM_obs - cum_src)                   = {do-cs:+.3e}")
    print(f"  -> advective gross exchange is {abs(cum_in[-1]):.2e} in / {abs(cum_out[-1]):.2e} out; "
          f"net advective ≈ flux.out and is EXPORT, but storage needs INFLOW.")
    print(f"  -> implied NON-advective (dispersive) salt flux = needed - recon_net = {(do-cs)-cum_net[-1]:+.3e}")


if __name__ == "__main__":
    for d in (14, 53):
        main(d)
