"""
Map the candidate cells for relocating src #3 (Pioneer @ Dumbleton).

Shows:
  - the mesh in the estuary area, coloured by element depth
  - current src #3 (BIG RED dot) at element 50526
  - 3 candidate cells (LARGE COLOURED dots) at chosen distances east:
       A) elem 51619    3-4 km east, rat ~2.6   (orange)
       B) elem 23939    7-8 km east, rat ~2.3   (cyan)
       C) elem 13008  11-12 km east, rat ~1.15  (magenta)
  - the (2<->3) Pioneer Mouth transect (green edges)
  - other 13 sources marked with grey 'x'
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
from matplotlib.collections import LineCollection
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core import geometry

RUN = Path("s:/Matt_Working/schism/P18_flood")
OUT_PNG = Path(__file__).parent / "src3_candidates_map.png"

CURRENT_SRC3_1IDX = 50526
CANDIDATES = [
    ("A", 51619, "#f97316", "3-4 km E,  rat~2.6"),
    ("B", 23939, "#06b6d4", "7-8 km E,  rat~2.3"),
    ("C", 13008, "#d946ef", "11-12 km E, rat~1.15"),
]


def main():
    print("Loading hgrid + fluxflag + sources...")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    flag = np.loadtxt(RUN / "fluxflag.prop", dtype=int)[:, 1]
    elements = hgrid["elements"]
    x, y = hgrid["x"], hgrid["y"]
    cx, cy = hgrid["centroids"][:, 0], hgrid["centroids"][:, 1]
    depth = hgrid["depth"]
    n_elem = hgrid["n_elements"]

    # Element-mean depth
    elem_depth = np.zeros(n_elem)
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        elem_depth[ei] = depth[ids[:nc]].mean()

    # Build triangulation
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
    tri_depth = elem_depth[t2e]

    # Find (2<->3) mouth edges for overlay
    edge2elem = defaultdict(list)
    edge2nodes = {}
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

    # Read all 14 sources for context
    src_lines = open(RUN / "source_sink.in").read().splitlines()
    n_src = int(src_lines[0].split('!')[0])
    src_elems_1idx = [int(src_lines[1+i].split('!')[0].strip()) for i in range(n_src)]

    # ---- Plot ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 9))

    # Bathymetry, full view + zoom
    for ax, xlim, ylim, title, edge_lw in [
        (axA, (149.08, 149.225), (-21.16, -21.105),
         "(a) full Pioneer Estuary — depth + candidates", 0.0),
        (axB, (149.085, 149.135), (-21.158, -21.140),
         "(b) zoom on candidate A (3-4 km E) area", 0.20),
    ]:
        # Mask deep/dry: clip depth to (-0.5, 8) for colour scaling
        tri_d = np.clip(tri_depth, -0.5, 8)
        tpc = ax.tripcolor(triang, facecolors=tri_d, shading="flat",
                            cmap="Blues", vmin=0, vmax=6,
                            edgecolors="#333333" if edge_lw > 0 else "none",
                            linewidth=edge_lw)
        # Mouth transect overlay
        if mouth_edges:
            lc = LineCollection(mouth_edges, colors="#16a34a", linewidths=3.0,
                                 zorder=5, label="(2↔3) mouth")
            ax.add_collection(lc)
        # All 14 sources (small grey x)
        for i, e in enumerate(src_elems_1idx, start=1):
            ei = e - 1
            if xlim[0] <= cx[ei] <= xlim[1] and ylim[0] <= cy[ei] <= ylim[1]:
                if i == 3:
                    continue  # we'll plot src #3 separately
                ax.plot(cx[ei], cy[ei], "x", color="#555555", markersize=8,
                        markeredgewidth=1.5, zorder=6)
                ax.annotate(f"  src #{i}", xy=(cx[ei], cy[ei]),
                             fontsize=7, color="#444444", zorder=7)

        # Current src #3 (large red dot)
        ei = CURRENT_SRC3_1IDX - 1
        if xlim[0] <= cx[ei] <= xlim[1] and ylim[0] <= cy[ei] <= ylim[1]:
            ax.plot(cx[ei], cy[ei], "o", color="#dc2626", markersize=16,
                    markeredgecolor="black", markeredgewidth=1.4, zorder=10,
                    label=f"CURRENT src #3 (elem {CURRENT_SRC3_1IDX})")
            ax.annotate(f"  CURRENT\n  (rat=21)",
                         xy=(cx[ei], cy[ei]), fontsize=9, color="#7f1d1d",
                         fontweight="bold", zorder=11)

        # Candidates
        for label, e1, color, descr in CANDIDATES:
            ei = e1 - 1
            if xlim[0] <= cx[ei] <= xlim[1] and ylim[0] <= cy[ei] <= ylim[1]:
                ax.plot(cx[ei], cy[ei], "o", color=color, markersize=14,
                        markeredgecolor="black", markeredgewidth=1.4, zorder=10,
                        label=f"{label}  elem {e1}  {descr}")
                ax.annotate(f"  {label}", xy=(cx[ei], cy[ei]),
                             fontsize=10, fontweight="bold", color="black", zorder=11)

        ax.set_aspect("equal")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # Colourbar for bathymetry on the right
    cbar = fig.colorbar(tpc, ax=[axA, axB], orientation="horizontal",
                         shrink=0.6, pad=0.06, aspect=40)
    cbar.set_label("Element-mean depth (m below MSL)", fontsize=10)

    fig.suptitle("Candidate cells for src #3 relocation — Pioneer Estuary",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
