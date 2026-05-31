#!/usr/bin/env python3
"""
make_flat_tide_elev2D.py  --  build elev2D.th.nc for the 003_P18_flood_flat run.

Reconstructs the open-boundary tidal elevation from the harmonic constants in
002's bctides.in (iettype=3) and writes it as a SCHISM space/time-varying
elevation input (elev2D.th.nc, for iettype=4), BUT with the tide held flat
(0 m) for the first 14 days and ramped on over the following 2 days.

  * days  0 -> 14 : eta = 0           (flat ocean)
  * days 14 -> 16 : eta = ramp(t) * harmonic_tide(t)   (cosine 0 -> 1)
  * days 16+      : eta = harmonic_tide(t)  == identical to 002's iettype=3 tide

SCHISM elevation reconstruction (matches the bktides harmonic boundary):
    eta(node,t) = Sum_k  ff_k * amp_{k,node} * cos( amig_k * t
                                                    - phase_{k,node}
                                                    + (V0+u)_k )
with phase and (V0+u) in radians, t in seconds since run start.

This script lives in analytics/utilities/. By default it reads the harmonic
constants from runs/002_P18_flood/bctides.in (the canonical iettype=3 tide)
and writes runs/003_P18_flood_flat/elev2D.th.nc.

Usage:
    python make_flat_tide_elev2D.py [bctides_in] [out_elev2D_nc]
"""
import os
import sys
import math
import numpy as np
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEG2RAD   = math.pi / 180.0
RNDAY     = 91          # param.nml run length (days); cover a bit beyond
PAD_DAYS  = 1           # extra coverage past rnday
DT_OUT    = 900.0       # output cadence of elev2D.th.nc [s]  (>= model dt=45s)
FLAT_END  = 14.0        # days: tide flat (0 m) until here
RAMP_END  = 16.0        # days: tide fully on from here on
FLAT_VALUE = 0.0        # m, datum of the flat period (harmonic is 0-mean)

# Paths: this script lives in <repo>/analytics/utilities/.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))      # analytics/utilities -> repo root
SRC_DEFAULT = os.path.join(REPO, "runs", "002_P18_flood", "bctides.in")
OUT_DEFAULT = os.path.join(REPO, "runs", "003_P18_flood_flat", "elev2D.th.nc")
SRC    = sys.argv[1] if len(sys.argv) > 1 else SRC_DEFAULT
OUT_NC = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT


def parse_bctides(path):
    """Return (constituents, nnodes, amp, pha) for open boundary 1.

    constituents: list of (name, amig[rad/s], ff, face[deg]) in file order
    amp, pha:     arrays shape (nfreq, nnodes); pha in degrees
    """
    with open(path) as fh:
        raw = fh.readlines()

    # strip trailing comments + blank lines, keep order
    toks = []
    for ln in raw:
        s = ln.split("!", 1)[0].strip()
        if s:
            toks.append(s)

    i = 0
    # line: ntip  cutoff_depth
    ntip = int(toks[i].split()[0]); i += 1
    # skip ntip earth-tidal-potential constituent blocks (name + 'spe amp freq nf face')
    for _ in range(ntip):
        i += 1            # name
        i += 1            # data line
    nbfr = int(toks[i].split()[0]); i += 1
    cons = []
    for _ in range(nbfr):
        name = toks[i]; i += 1
        a, ff, face = (float(x) for x in toks[i].split()[:3]); i += 1
        cons.append((name, a, ff, face))
    nope = int(toks[i].split()[0]); i += 1
    if nope != 1:
        raise SystemExit(f"expected 1 open boundary, got {nope}")
    # boundary flag line: nnodes iettype ifltype itetype isatype iGENtype iAEDtype
    flag = toks[i].split(); i += 1
    nnodes = int(flag[0]); iettype = int(flag[1])
    if iettype != 3:
        raise SystemExit(f"expected iettype=3 in source bctides, got {iettype}")
    amp = np.zeros((nbfr, nnodes))
    pha = np.zeros((nbfr, nnodes))
    for k in range(nbfr):
        cname = toks[i]; i += 1
        if cname != cons[k][0]:
            raise SystemExit(f"constituent order mismatch: {cname} vs {cons[k][0]}")
        for n in range(nnodes):
            a, p = (float(x) for x in toks[i].split()[:2]); i += 1
            amp[k, n] = a
            pha[k, n] = p
    return cons, nnodes, amp, pha


def ramp_tide(t_sec):
    """Cosine 0->1 over [FLAT_END, RAMP_END] days; 0 before, 1 after."""
    td = t_sec / 86400.0
    r = np.zeros_like(td)
    on = td >= RAMP_END
    r[on] = 1.0
    mid = (td > FLAT_END) & (td < RAMP_END)
    r[mid] = 0.5 * (1.0 - np.cos(math.pi * (td[mid] - FLAT_END) / (RAMP_END - FLAT_END)))
    return r


def main():
    src = SRC
    if not os.path.exists(src):
        raise SystemExit(f"bctides source not found: {src}")
    print(f"[read]  harmonic constants from: {src}")
    cons, nnodes, amp, pha = parse_bctides(src)
    nfreq = len(cons)
    print(f"[parse] {nfreq} constituents, {nnodes} boundary nodes")
    for name, a, ff, face in cons:
        print(f"        {name:3s}  amig={a:.6e} rad/s  ff={ff:.5f}  V0+u={face:8.3f} deg")

    amig = np.array([c[1] for c in cons])              # (nfreq,)
    ff   = np.array([c[2] for c in cons])
    face = np.array([c[3] for c in cons]) * DEG2RAD    # rad
    pha_r = pha * DEG2RAD                               # (nfreq, nnodes)

    nt = int(round((RNDAY + PAD_DAYS) * 86400.0 / DT_OUT)) + 1
    t = np.arange(nt) * DT_OUT                          # seconds since start
    print(f"[time]  {nt} steps, dt={DT_OUT:.0f}s, span 0 .. {t[-1]/86400:.2f} days")

    # harmonic tide: eta_h(node,t) = sum_k ff_k amp_kn cos(amig_k t - pha_kn + face_k)
    # build per timestep to keep memory modest: (nt, nnodes)
    eta = np.zeros((nt, nnodes), dtype=np.float64)
    for k in range(nfreq):
        # phase argument (nt, nnodes) = amig_k t (nt,1) - pha (1,nnodes) + face_k
        arg = amig[k] * t[:, None] - pha_r[None, k, :] + face[k]
        eta += ff[k] * amp[None, k, :] * np.cos(arg)

    r = ramp_tide(t)[:, None]                           # (nt,1)
    eta = r * eta + (1.0 - r) * FLAT_VALUE

    # ---- sanity checks ----
    td = t / 86400.0
    i_flat = np.where(td <= FLAT_END)[0]
    i_full = np.where(td >= RAMP_END)[0]
    print(f"[check] flat window  |eta| max = {np.abs(eta[i_flat]).max():.3e} m (expect 0)")
    # reconstruct pure harmonic at a late time to confirm full-on amplitude
    eta_h_full_max = 0.0
    if i_full.size:
        eta_h_full_max = np.abs(eta[i_full]).max()
    print(f"[check] full window  |eta| max = {eta_h_full_max:.3f} m  (per-node tidal range)")
    print(f"[check] node-0 amp sum (approx HAT) = {np.sum(ff*amp[:,0]):.3f} m")

    # ---- write elev2D.th.nc ----
    if os.path.exists(OUT_NC):
        os.remove(OUT_NC)
    ds = Dataset(OUT_NC, "w", format="NETCDF4")
    ds.createDimension("nOpenBndNodes", nnodes)
    ds.createDimension("nLevels", 1)
    ds.createDimension("nComponents", 1)
    ds.createDimension("one", 1)
    ds.createDimension("time", None)            # unlimited

    v_step = ds.createVariable("time_step", "f4", ("one",))
    v_step[:] = np.float32(DT_OUT)
    v_time = ds.createVariable("time", "f8", ("time",))
    v_time[:] = t.astype(np.float64)
    v_ts = ds.createVariable("time_series", "f4",
                             ("time", "nOpenBndNodes", "nLevels", "nComponents"))
    v_ts[:, :, 0, 0] = eta.astype(np.float32)

    ds.title = "Flat-tide elevation BC for 003_P18_flood_flat"
    ds.history = ("harmonic reconstruction of 002 bctides; tide held flat (0 m) "
                  "for first 14 days, cosine ramp to full tide over days 14-16")
    ds.close()
    sz = os.path.getsize(OUT_NC) / 1e6
    print(f"[write] {OUT_NC}  ({sz:.2f} MB)  dims time={nt}, nodes={nnodes}")
    print("[done]  set open-bnd elevation flag to iettype=4 in bctides.in.")


if __name__ == "__main__":
    main()
