"""
Zoomed 3-panel vertical TRC tracer animation for the Pioneer CV.

  - 3 panels stacked TOP-TO-BOTTOM: TRC_tr1, TRC_tr2, TRC_tr3
  - tight zoom on the Pioneer Estuary + mouth
  - colour map has a special GREY band for c ∈ [0, 0.01] so that even
    trace amounts of tr1 (e.g. plume leaving the mouth) are visible
    against the white background of a 0-everywhere panel
  - higher-than-default colour resolution for the c < 0.1 range so the
    structure of the dilute plume is visible
  - (2↔3) mouth transect overlaid as a green line
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import argparse

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from collections import defaultdict

try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nutrient_budget"))      # generic core
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pioneer_region_flux"))  # run_config
from run_config import active_run
from core import geometry


CFG = active_run()
RUN = CFG.run_dir
OUT_MP4 = CFG.out_dir / "TRC_animation_flood_zoomed.mp4"

# ----- ZOOM around the Pioneer CV / mouth -----
XLIM = (149.085, 149.225)
YLIM = (-21.158, -21.110)

# ----- colour scale -----
GREY_THRESHOLD = 0.01    # c ≤ this shown as grey (highlights "trace presence")
CMAP_MAX = 1.0
NDAYS = None             # None = use all available stacks


def build_tracer_cmap():
    """Custom colormap:
        c == 0     : white (background, "absent")
        0 < c ≤ 0.01 : grey  (trace presence — highlights the plume edge)
        0.01 < c ≤ 1: matter-like blue→yellow→red gradient
    Uses BoundaryNorm so the grey band is a fixed pixel-width regardless
    of CMAP_MAX.
    """
    # Base gradient for c > 0.01
    base = LinearSegmentedColormap.from_list(
        "tr_base",
        [
            (0.00, "#1d4ed8"),    # blue
            (0.30, "#7c3aed"),    # purple
            (0.55, "#dc2626"),    # red
            (0.80, "#f97316"),    # orange
            (1.00, "#fde047"),    # yellow
        ],
        N=256,
    )
    # Sample base across [0, 256) — 256 colours for the high-c portion
    base_cols = [base(i / 255) for i in range(256)]

    # Build a ListedColormap with 2 special bins followed by 256 gradient bins:
    #   bin 0   : white  (c < eps  → effectively 0)
    #   bin 1   : grey   (0 < c ≤ 0.01)
    #   bins 2.. : gradient (0.01 < c ≤ CMAP_MAX)
    colors = [(1.0, 1.0, 1.0, 1.0),           # white for c == 0
              (0.65, 0.65, 0.65, 1.0)]         # grey for trace
    colors.extend(base_cols)
    cmap = ListedColormap(colors)

    # Boundaries: tiny eps, then GREY_THRESHOLD, then linear to CMAP_MAX
    eps = 1e-6
    gradient_bounds = np.linspace(GREY_THRESHOLD, CMAP_MAX, 257)
    boundaries = np.concatenate(([0.0, eps], gradient_bounds))
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    return cmap, norm


def find_mouth_edges(hgrid, flag):
    """Return list of (x0,y0)-(x1,y1) for every (2↔3) edge."""
    edges = []
    edge2elem = defaultdict(list)
    edge2nodes = {}
    elements = hgrid["elements"]
    for ei in range(hgrid["n_elements"]):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k+1)%nc]
            key = (min(a, b), max(a, b))
            edge2elem[key].append(ei)
            edge2nodes[key] = (a, b)
    x, y = hgrid["x"], hgrid["y"]
    for key, elems in edge2elem.items():
        if len(elems) != 2: continue
        f0, f1 = flag[elems[0]], flag[elems[1]]
        if {f0, f1} == {2, 3}:
            a, b = edge2nodes[key]
            edges.append([(x[a], y[a]), (x[b], y[b])])
    return edges


def load_tracer_concat(outputs_dir, varname, ndays=None):
    """Load all stacks of a scribed tracer + concatenate along time.
    Returns (arr, times) where arr has shape (T, N_nodes) at the surface layer."""
    paths = geometry.discover_scribed_stacks(outputs_dir, varname)
    if not paths:
        raise FileNotFoundError(f"No {varname}_*.nc in {outputs_dir}")
    arrs, ts = [], []
    for p in paths:
        ds = xr.open_dataset(p, engine="h5netcdf")
        da = ds[varname]
        # Take the surface layer (last in the layer axis for SCHISM convention)
        layer_dim = next((d for d in da.dims
                          if "vgrid" in d.lower() or "layer" in d.lower()), None)
        if layer_dim is not None:
            da = da.isel({layer_dim: -1})
        node_dim = next(d for d in da.dims if "node" in d.lower())
        arr = da.transpose("time", node_dim).values
        arr = np.where(np.isfinite(arr) & (np.abs(arr) <= 1e30), arr, 0.0)
        arrs.append(arr)
        ts.append(ds["time"].values)
        ds.close()
    arr_all = np.concatenate(arrs, axis=0)
    t_all = np.concatenate(ts)
    if ndays is not None:
        t0 = t_all[0]
        keep = (t_all - t0) <= np.timedelta64(int(ndays), "D")
        arr_all = arr_all[keep]
        t_all = t_all[keep]
    return arr_all, t_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=RUN / "outputs")
    ap.add_argument("--output", type=Path, default=OUT_MP4)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--ndays", type=float, default=None,
                    help="Limit to first N days (default: all available)")
    ap.add_argument("--cmap-max", type=float, default=CMAP_MAX,
                    help="Upper colour-bar limit (default 1.0)")
    args = ap.parse_args()

    print("Loading hgrid + fluxflag...")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    triangles = []
    for ei in range(hgrid["n_elements"]):
        nc, *ids = hgrid["elements"][ei]
        if nc == 3:
            triangles.append(ids)
        elif nc == 4:
            triangles.append([ids[0], ids[1], ids[2]])
            triangles.append([ids[0], ids[2], ids[3]])
    triangles = np.asarray(triangles, dtype=np.int32)
    triang = mtri.Triangulation(hgrid["x"], hgrid["y"], triangles)

    print("Building mouth-transect overlay...")
    mouth_edges = find_mouth_edges(hgrid, flag)
    print(f"  found {len(mouth_edges)} (2↔3) edges")

    # Build colormap once
    cmap, norm = build_tracer_cmap()

    # Load all three tracers
    print("\nLoading tracer scribed files...")
    tracers = {}
    times_ref = None
    for v in ("TRC_tr1", "TRC_tr2", "TRC_tr3"):
        print(f"  {v}...")
        arr, t = load_tracer_concat(args.outputs_dir, v, ndays=args.ndays)
        tracers[v] = arr
        if times_ref is None:
            times_ref = t
        print(f"     shape={arr.shape}, time range = {t[0]} .. {t[-1]}")

    T = min(arr.shape[0] for arr in tracers.values())
    times = times_ref[:T]
    for v in tracers:
        tracers[v] = tracers[v][:T]

    # ----- Figure: 3 panels, stacked vertically -----
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True, sharey=True)
    panel_labels = {
        "TRC_tr1": "TRC_tr1 — Pioneer (src #3)",
        "TRC_tr2": "TRC_tr2 — other rivers",
        "TRC_tr3": "TRC_tr3 — ocean",
    }
    tpcs = {}
    for ax, var in zip(axes, ("TRC_tr1", "TRC_tr2", "TRC_tr3")):
        c0 = tracers[var][0]
        tpc = ax.tripcolor(triang, c0, shading="gouraud",
                            cmap=cmap, norm=norm)
        tpcs[var] = tpc
        # mouth transect overlay
        if mouth_edges:
            lc = LineCollection(mouth_edges, colors="#16a34a", linewidths=2.0,
                                 zorder=5)
            ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        ax.set_ylabel("Latitude")
        ax.set_title(panel_labels[var], fontweight="bold", loc="left", fontsize=11)
        ax.grid(True, ls=":", alpha=0.3)
    axes[-1].set_xlabel("Longitude")

    # Shared colourbar (right side)
    cb = fig.colorbar(tpcs["TRC_tr1"], ax=axes, shrink=0.7, pad=0.02, fraction=0.04)
    cb.set_label(
        f"tracer concentration (mmol/m³)\n"
        f"grey band  =  0 < c ≤ {GREY_THRESHOLD}  (trace presence)",
        fontsize=9,
    )

    title = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(frame):
        for var in ("TRC_tr1", "TRC_tr2", "TRC_tr3"):
            tpcs[var].set_array(tracers[var][frame])
        title.set_text(f"t = {np.datetime_as_string(times[frame], unit='m')}")
        return list(tpcs.values()) + [title]

    print(f"\nWriting animation -> {args.output} ({T} frames @ {args.fps} fps)...")
    anim = animation.FuncAnimation(fig, update, frames=T,
                                    interval=1000/args.fps, blit=False)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Done.  {args.output}")


if __name__ == "__main__":
    main()
