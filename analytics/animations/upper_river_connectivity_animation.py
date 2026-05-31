"""
Animated upper-river connectivity check.

For every output timestep, colour each element by:
  - RED   when the cell is dry (water depth ≤ DRY_THRESHOLD m)
  - BLUE  with depth gradient when wet

Highlights how the navigable corridor narrows and widens over the tidal
cycle, especially around the current src #3 location.

The previous run (depress_clutch=.TRUE.) outputs are in outputs_clutch/.
Edit OUTPUTS_DIR if needed.
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from collections import defaultdict

try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nutrient_budget"))  # generic core
from core import geometry

RUN = Path("s:/Matt_Working/schism/P18_flood")
OUTPUTS_DIR = RUN / "outputs_clutch"
OUT_MP4 = Path(__file__).parent / "upper_river_connectivity_animation.mp4"

# ----- TUNABLES -----
XLIM = (149.085, 149.155)
YLIM = (-21.158, -21.135)
DRY_THRESHOLD_M = 0.05         # water depth ≤ this → cell shown red (dry)
SHALLOW_THRESHOLD_M = 0.50     # water depth from DRY_THRESHOLD to this shows as gradient
DEEP_MAX_M = 5.0               # colour cap
FPS = 8
DPI = 160
N_STACKS = None                # None = all available
CURRENT_SRC3_1IDX = 50526
CANDIDATES = [
    ("A", 51619, "#f97316"),
    ("B", 23939, "#06b6d4"),
    ("C", 13008, "#d946ef"),
]


def main():
    print(f"Loading hgrid + fluxflag...")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    elements = hgrid["elements"]
    x, y = hgrid["x"], hgrid["y"]
    cx, cy = hgrid["centroids"][:, 0], hgrid["centroids"][:, 1]
    depth = hgrid["depth"]            # positive = below MSL
    n_elem = hgrid["n_elements"]
    n_nodes = hgrid["n_nodes"]

    # Element bed elevation (positive UP from MSL) + vertex indices for elev avg
    bed_elev = np.zeros(n_elem)
    elem_node_idx = np.zeros((n_elem, 4), dtype=np.int32)
    nc_arr = np.zeros(n_elem, dtype=np.int8)
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        nc_arr[ei] = nc
        bed_elev[ei] = -depth[ids[:nc]].mean()
        for k in range(nc):
            elem_node_idx[ei, k] = ids[k]
    elem_node_w = np.zeros((n_elem, 4), dtype=np.float32)
    for k in range(4):
        elem_node_w[:, k] = np.where(nc_arr > k, 1.0/nc_arr, 0.0)

    # Triangulation
    tris, t2e = [], []
    for ei, e in enumerate(elements):
        nc, *ids = e
        if nc == 3:
            tris.append(ids); t2e.append(ei)
        elif nc == 4:
            tris.append([ids[0], ids[1], ids[2]]); t2e.append(ei)
            tris.append([ids[0], ids[2], ids[3]]); t2e.append(ei)
    tris = np.asarray(tris, dtype=np.int32); t2e = np.asarray(t2e, dtype=np.int32)
    triang = mtri.Triangulation(x, y, tris)
    tri_bed = bed_elev[t2e]

    # (2↔3) mouth edges for overlay
    edge2elem = defaultdict(list); edge2nodes = {}
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k+1)%nc]
            key = (min(a, b), max(a, b))
            edge2elem[key].append(ei); edge2nodes[key] = (a, b)
    mouth_edges = []
    for key, elems in edge2elem.items():
        if len(elems) != 2: continue
        f0, f1 = flag[elems[0]], flag[elems[1]]
        if {f0, f1} == {2, 3}:
            a, b = edge2nodes[key]
            mouth_edges.append([(x[a], y[a]), (x[b], y[b])])

    # Sources
    src_lines = open(RUN / "source_sink.in").read().splitlines()
    n_src = int(src_lines[0].split('!')[0])
    src_elems_1idx = [int(src_lines[1+i].split('!')[0].strip()) for i in range(n_src)]

    # ---- Load elevation across available stacks ----
    print(f"Discovering out2d stacks in {OUTPUTS_DIR}...")
    paths = geometry.discover_scribed_stacks(OUTPUTS_DIR, "out2d")
    if N_STACKS is not None:
        paths = paths[:N_STACKS]
    print(f"  {len(paths)} stacks available")

    elev_list, times_list = [], []
    for p in paths:
        ds = xr.open_dataset(p, engine="h5netcdf")
        elev = ds["elevation"].values        # (T, N)
        elev = np.where(np.isfinite(elev) & (np.abs(elev) <= 1e30), elev, np.nan)
        elev_list.append(elev)
        times_list.append(ds["time"].values)
        ds.close()
    elev_all = np.concatenate(elev_list, axis=0)
    times_all = np.concatenate(times_list)
    print(f"  total timesteps: {elev_all.shape[0]}")

    # Compute per-element elevation (avg of vertex nodes) + water depth = elev - bed
    print(f"Computing element-mean elevations + water depth per timestep...")
    T = elev_all.shape[0]
    h_elem_t = np.zeros((T, n_elem), dtype=np.float32)
    chunk = 32
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        elev_sub = elev_all[t0:t1]            # (Tc, N)
        # Gather node elevations per element
        gathered = elev_sub[:, elem_node_idx]   # (Tc, n_elem, 4)
        elem_elev = (gathered * elem_node_w).sum(axis=-1)  # (Tc, n_elem)
        # Water depth = elev - bed (elev is positive UP, bed is positive UP)
        h_elem_t[t0:t1] = elem_elev - bed_elev[None, :]
    print(f"  water depth range: min={h_elem_t.min():.3f}, max={h_elem_t.max():.3f}")

    # Map to triangulated cells
    h_tri_t = h_elem_t[:, t2e]    # (T, n_tris)

    # ---- Custom colormap: red @ dry, white @ ~shallow, blue gradient @ deep ----
    cmap_wd = LinearSegmentedColormap.from_list(
        "depth_red_blue",
        [
            (0.00, "#dc2626"),     # red: dry
            (0.10, "#fca5a5"),     # light red: barely-dry-side
            (0.20, "#ffffff"),     # white: shallow
            (0.50, "#93c5fd"),     # light blue
            (1.00, "#1e3a8a"),     # dark blue: deep
        ]
    )
    vmin = -DRY_THRESHOLD_M
    vmax = DEEP_MAX_M
    # Map water depth to colour: clip h to [vmin, vmax]

    # ---- Set up animation ----
    fig, ax = plt.subplots(figsize=(14, 7))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    tpc = ax.tripcolor(triang, facecolors=h_tri_t[0], shading="flat",
                        cmap=cmap_wd, norm=norm,
                        edgecolors="#33333333", linewidth=0.10)
    # Mouth transect
    if mouth_edges:
        ax.add_collection(LineCollection(mouth_edges, colors="#16a34a", linewidths=3.0, zorder=5))
    # Current src #3
    ei = CURRENT_SRC3_1IDX - 1
    ax.plot(cx[ei], cy[ei], "o", color="#dc2626", markersize=14,
            markeredgecolor="black", markeredgewidth=1.4, zorder=10)
    ax.annotate("  src #3", xy=(cx[ei], cy[ei]),
                 fontsize=9, color="#7f1d1d", fontweight="bold", zorder=11)
    # Candidates
    for label, e1, color in CANDIDATES:
        ei = e1 - 1
        if XLIM[0] <= cx[ei] <= XLIM[1] and YLIM[0] <= cy[ei] <= YLIM[1]:
            ax.plot(cx[ei], cy[ei], "o", color=color, markersize=12,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=10)
            ax.annotate(f"  {label}", xy=(cx[ei], cy[ei]),
                         fontsize=9, fontweight="bold", color="black", zorder=11)
    # Other sources
    for i, e in enumerate(src_elems_1idx, start=1):
        if i == 3: continue
        ei = e - 1
        if XLIM[0] <= cx[ei] <= XLIM[1] and YLIM[0] <= cy[ei] <= YLIM[1]:
            ax.plot(cx[ei], cy[ei], "x", color="#555555", markersize=8,
                    markeredgewidth=1.4, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, ls=":", alpha=0.3)
    title = ax.set_title("", fontweight="bold", loc="left", fontsize=11)
    cb = fig.colorbar(tpc, ax=ax, orientation="vertical", shrink=0.85, pad=0.02)
    cb.set_label(f"Water depth (m)  —  red ≤ {DRY_THRESHOLD_M} m = dry")
    fig.suptitle("Upper Pioneer River — water-depth connectivity over the tidal cycle",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ---- Animation function ----
    def update(frame):
        tpc.set_array(h_tri_t[frame])
        # Count dry cells in view
        cell_in = ((cx >= XLIM[0]) & (cx <= XLIM[1])
                   & (cy >= YLIM[0]) & (cy <= YLIM[1]))
        dry_in_view = ((h_elem_t[frame, cell_in] <= DRY_THRESHOLD_M)).sum()
        title.set_text(
            f"t = {np.datetime_as_string(times_all[frame], unit='m')}  "
            f"|  dry cells in view: {dry_in_view}/{cell_in.sum()} "
            f"({100*dry_in_view/cell_in.sum():.1f}%)"
        )
        return tpc, title

    print(f"Writing animation to {OUT_MP4} ({T} frames @ {FPS} fps)...")
    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000/FPS, blit=False)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=4000)
    anim.save(OUT_MP4, writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"Done. {OUT_MP4}")


if __name__ == "__main__":
    main()
