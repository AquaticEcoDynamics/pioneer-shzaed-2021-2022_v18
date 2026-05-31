"""
Upper-river bathymetry / water-connectivity check.

Two stacked panels covering the same upper-river area:

  TOP — bed elevation (m, positive UP from MSL), colour scale clipped at the
        chosen minimum tidal level so the deep-channel gradient is visible
        while every shallower cell saturates to the top end of the cmap.

  BOTTOM — binary map: cells with bed elevation HIGHER than the chosen
        minimum tidal level are coloured RED (these cells would dry up at
        low tide and break flow connectivity).

Helps identify "natural dams" upstream of src #3 that may be preventing
freshwater + tracer mass from propagating downstream.

THRESHOLD is settable at the top of the file. Default -1.0 m roughly
matches the local low-tide elevation observed at the upper-river nodes.
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core import geometry

RUN = Path("s:/Matt_Working/schism/P18_flood")
OUT_PNG = Path(__file__).parent / "upper_river_connectivity.png"

# ----- TUNABLE PARAMETERS -----
MIN_TIDAL_HEIGHT_M = -1.0          # threshold (m above MSL); cells with bed
                                   # higher than this go dry at low tide
XLIM = (149.085, 149.155)          # upper-river view window
YLIM = (-21.158, -21.135)
CURRENT_SRC3_1IDX = 50526
# Optional candidate cells to mark on both panels
CANDIDATES = [
    ("A", 51619, "#f97316", "3-4 km E"),
    ("B", 23939, "#06b6d4", "7-8 km E"),
    ("C", 13008, "#d946ef", "11-12 km E"),
]


def main():
    print(f"Threshold = {MIN_TIDAL_HEIGHT_M:.2f} m (cells with bed > this go dry at low tide)")
    print(f"Loading hgrid + fluxflag...")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    elements = hgrid["elements"]
    x, y = hgrid["x"], hgrid["y"]
    cx, cy = hgrid["centroids"][:, 0], hgrid["centroids"][:, 1]
    depth = hgrid["depth"]              # positive = below MSL
    n_elem = hgrid["n_elements"]

    # Bed elevation (positive UP from MSL) per element = mean of vertex bed elevations
    bed_elev = np.zeros(n_elem)
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        bed_elev[ei] = -depth[ids[:nc]].mean()   # NEGATE: depth->bed_elev

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

    # (2<->3) mouth edges for spatial reference (will be far east, off-screen for this zoom)
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

    # ---- Stats for the view window ----
    cell_in = ((cx >= XLIM[0]) & (cx <= XLIM[1])
               & (cy >= YLIM[0]) & (cy <= YLIM[1]))
    n_view = cell_in.sum()
    n_above = (cell_in & (bed_elev > MIN_TIDAL_HEIGHT_M)).sum()
    print(f"In-view cells: {n_view},  above threshold (potentially dry): "
          f"{n_above} ({100*n_above/n_view:.1f}%)")
    print(f"Bed elevation in view: min={bed_elev[cell_in].min():.2f}  "
          f"max={bed_elev[cell_in].max():.2f}  mean={bed_elev[cell_in].mean():.2f}")

    # ---- Plot ----
    fig, (axTop, axBot) = plt.subplots(2, 1, figsize=(14, 11), sharex=True, sharey=True)

    # TOP — bathymetry with colour axis clipped at threshold
    vmin = bed_elev[cell_in].min()
    vmax = MIN_TIDAL_HEIGHT_M
    tpc = axTop.tripcolor(triang, facecolors=tri_bed, shading="flat",
                           cmap="Blues_r", vmin=vmin, vmax=vmax,
                           edgecolors="#33333355", linewidth=0.15)
    if mouth_edges:
        lc = LineCollection(mouth_edges, colors="#16a34a", linewidths=3.0, zorder=5)
        axTop.add_collection(lc)
    # Plot sources and src #3
    for i, e in enumerate(src_elems_1idx, start=1):
        ei = e - 1
        if XLIM[0] <= cx[ei] <= XLIM[1] and YLIM[0] <= cy[ei] <= YLIM[1]:
            if i == 3:
                axTop.plot(cx[ei], cy[ei], "o", color="#dc2626", markersize=14,
                            markeredgecolor="black", markeredgewidth=1.2, zorder=10)
                axTop.annotate("  CURRENT src #3", xy=(cx[ei], cy[ei]),
                                fontsize=9, color="#7f1d1d", fontweight="bold")
            else:
                axTop.plot(cx[ei], cy[ei], "x", color="#555555", markersize=8,
                            markeredgewidth=1.4, zorder=6)
                axTop.annotate(f"  src #{i}", xy=(cx[ei], cy[ei]),
                                fontsize=7, color="#444444")
    # Candidates
    for label, e1, color, descr in CANDIDATES:
        ei = e1 - 1
        if XLIM[0] <= cx[ei] <= XLIM[1] and YLIM[0] <= cy[ei] <= YLIM[1]:
            axTop.plot(cx[ei], cy[ei], "o", color=color, markersize=14,
                        markeredgecolor="black", markeredgewidth=1.3, zorder=10)
            axTop.annotate(f"  {label} ({descr})", xy=(cx[ei], cy[ei]),
                            fontsize=9, fontweight="bold", color="black")

    axTop.set_aspect("equal")
    axTop.set_xlim(XLIM); axTop.set_ylim(YLIM)
    axTop.set_ylabel("Latitude")
    axTop.grid(True, ls=":", alpha=0.3)
    axTop.set_title(f"(a) Bed elevation (m above MSL)  —  colour clipped at "
                    f"min tidal level = {MIN_TIDAL_HEIGHT_M:.2f} m",
                    fontweight="bold", loc="left", fontsize=11)
    cb = fig.colorbar(tpc, ax=axTop, orientation="vertical", shrink=0.85, pad=0.02)
    cb.set_label("Bed elev (m above MSL)")

    # BOTTOM — binary connectivity map
    is_above = (tri_bed > MIN_TIDAL_HEIGHT_M).astype(int)
    cmap_bin = ListedColormap(["#dbeafe", "#dc2626"])   # blue = below threshold (always wet),
                                                        # red  = above threshold (dries at low tide)
    norm_bin = BoundaryNorm([-0.5, 0.5, 1.5], cmap_bin.N)
    axBot.tripcolor(triang, facecolors=is_above, shading="flat",
                     cmap=cmap_bin, norm=norm_bin,
                     edgecolors="#33333355", linewidth=0.15)
    if mouth_edges:
        lc = LineCollection(mouth_edges, colors="#16a34a", linewidths=3.0, zorder=5)
        axBot.add_collection(lc)
    # Sources + candidates again
    for i, e in enumerate(src_elems_1idx, start=1):
        ei = e - 1
        if XLIM[0] <= cx[ei] <= XLIM[1] and YLIM[0] <= cy[ei] <= YLIM[1]:
            if i == 3:
                axBot.plot(cx[ei], cy[ei], "o", color="#dc2626", markersize=14,
                            markeredgecolor="white", markeredgewidth=1.5, zorder=10)
                axBot.annotate("  CURRENT src #3", xy=(cx[ei], cy[ei]),
                                fontsize=9, color="white", fontweight="bold")
            else:
                axBot.plot(cx[ei], cy[ei], "x", color="#555555", markersize=8,
                            markeredgewidth=1.4, zorder=6)
    for label, e1, color, descr in CANDIDATES:
        ei = e1 - 1
        if XLIM[0] <= cx[ei] <= XLIM[1] and YLIM[0] <= cy[ei] <= YLIM[1]:
            axBot.plot(cx[ei], cy[ei], "o", color=color, markersize=14,
                        markeredgecolor="black", markeredgewidth=1.3, zorder=10)
            axBot.annotate(f"  {label} ({descr})", xy=(cx[ei], cy[ei]),
                            fontsize=9, fontweight="bold", color="black")
    axBot.set_aspect("equal")
    axBot.set_xlim(XLIM); axBot.set_ylim(YLIM)
    axBot.set_xlabel("Longitude"); axBot.set_ylabel("Latitude")
    axBot.grid(True, ls=":", alpha=0.3)
    axBot.set_title(f"(b) Connectivity flag — RED cells have bed > {MIN_TIDAL_HEIGHT_M:.2f} m  "
                    f"(dry at low tide).  {n_above}/{n_view} cells red in view ({100*n_above/n_view:.1f}%)",
                    fontweight="bold", loc="left", fontsize=11)

    # Custom legend for the binary panel
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#dbeafe", edgecolor="black", label="bed ≤ threshold (always wet, connected)"),
        Patch(facecolor="#dc2626", edgecolor="black", label=f"bed > threshold (dries at η < {MIN_TIDAL_HEIGHT_M:.2f} m)"),
    ]
    axBot.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.92)

    fig.suptitle("Upper Pioneer River — bathymetry + low-tide connectivity check",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
