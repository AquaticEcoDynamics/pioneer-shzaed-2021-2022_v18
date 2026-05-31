"""
Per-edge classification map at the Pioneer Mouth.

For every CV-polygon-perimeter edge in the mouth area, show:
  - which side of the edge is inside the CV
  - the flag of the CV side and the non-CV side
  - whether the edge contributes to flux.out (only if both flags ≥ 0 AND |Δ|=1)

Colour code:
  GREEN, thick   : CV side flag=2, outside flag=3  (TRACKED in col 3)
  RED            : -1 <-> 2  edge (NOT tracked)    — distinguish which side is -1
  ORANGE         : 2 <-> 2   edge (NOT tracked)    — polygon cuts a flag-2 strip
  PURPLE (rare)  : other flag pairs                — NOT tracked

The point is to *see* exactly which edges at the mouth are untracked and
why.
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
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry


def main():
    CFG = active_run()
    run = CFG.run_dir
    cfg = CFG.configs
    out_path = CFG.out_dir / "mouth_edge_classification.png"

    print("Loading geometry...")
    hgrid = geometry.read_hgrid_with_areas(run / "hgrid.gr3")
    flag = np.loadtxt(run / "fluxflag.prop", dtype=int)[:, 1]
    x, y = hgrid["x"], hgrid["y"]
    elements = hgrid["elements"]
    n_elem = hgrid["n_elements"]
    cv_ids = [int(l.split("#", 1)[0].strip())
              for l in open(cfg / "cv_pioneer_estuary_elements.txt")
              if l.strip() and not l.strip().startswith("#")]
    cv_mask = np.zeros(n_elem, dtype=bool); cv_mask[np.array(cv_ids) - 1] = True

    pts = []
    with open(cfg / "cv_pioneer_estuary.csv") as f:
        next(f)
        for line in f:
            lon, lat = map(float, line.strip().split(","))
            pts.append((lon, lat))
    px = [p[0] for p in pts]; py = [p[1] for p in pts]

    # Build edge adjacency
    print("Building edges + classifying...")
    edge2elem = defaultdict(list)
    edge2nodes = {}
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k+1)%nc]
            key = (min(a, b), max(a, b))
            edge2elem[key].append(ei); edge2nodes[key] = (a, b)

    classes = {
        "tracked_23":        {"color": "#16a34a", "lw": 3.2, "label": "TRACKED  CV(2) ↔ outside(3)"},
        "untracked_n1_cv2":  {"color": "#dc2626", "lw": 2.6, "label": "UNTRACKED  CV(2) ↔ outside(-1)"},
        "untracked_n1_cvn1": {"color": "#fb923c", "lw": 2.6, "label": "UNTRACKED  CV(-1) ↔ outside(2)"},
        "untracked_22":      {"color": "#a855f7", "lw": 2.6, "label": "UNTRACKED  CV(2) ↔ outside(2)"},
        "untracked_other":   {"color": "#000000", "lw": 2.0, "label": "UNTRACKED  other"},
    }
    edges_by_class = {k: [] for k in classes}
    for key, elems in edge2elem.items():
        if len(elems) != 2: continue
        e0, e1 = elems
        in0, in1 = cv_mask[e0], cv_mask[e1]
        if in0 == in1: continue
        cv_el = e0 if in0 else e1
        nb_el = e1 if in0 else e0
        f_cv = flag[cv_el]; f_nb = flag[nb_el]
        a, b = edge2nodes[key]
        xy = [(x[a], y[a]), (x[b], y[b])]
        if f_cv == 2 and f_nb == 3:                edges_by_class["tracked_23"].append(xy)
        elif f_cv == 2 and f_nb == -1:             edges_by_class["untracked_n1_cv2"].append(xy)
        elif f_cv == -1 and f_nb == 2:             edges_by_class["untracked_n1_cvn1"].append(xy)
        elif f_cv == 2 and f_nb == 2:              edges_by_class["untracked_22"].append(xy)
        else:                                      edges_by_class["untracked_other"].append(xy)

    print("CV-perimeter edge classification at the mouth area:")
    for k, info in classes.items():
        n = len(edges_by_class[k])
        if n: print(f"  {k:25s}: {n:>3} edges  ({info['label']})")

    # Mesh triangulation
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
    tri_flag = flag[t2e]

    tab10 = cm.get_cmap("tab10", 10)
    palette = [(0.92, 0.92, 0.92, 1.0)]
    max_r = int(max(flag.max(), 1))
    for r in range(1, max_r + 1):
        palette.append(tab10((r - 1) % 10))
    cmap_disc = ListedColormap(palette)
    region_idx = np.where(tri_flag < 0, 0, tri_flag)
    norm = BoundaryNorm(np.arange(-0.5, len(palette) + 0.5, 1.0), cmap_disc.N)

    # Tight zoom on mouth area
    fig, ax = plt.subplots(1, 1, figsize=(11, 10))
    xlim, ylim = (149.205, 149.218), (-21.146, -21.130)

    # Region fill
    ax.tripcolor(triang, facecolors=region_idx, shading="flat",
                 cmap=cmap_disc, norm=norm,
                 edgecolors="#4d4d4d33", linewidth=0.4)

    # CV polygon
    ax.plot(px, py, "-", color="black", linewidth=1.6, alpha=0.6, zorder=4,
            label="CV polygon")

    # CV-interior shading (light blue overlay to show which side is CV)
    cv_tri_mask = cv_mask[t2e]
    ax.tripcolor(triang, facecolors=cv_tri_mask.astype(float),
                 shading="flat", cmap="Blues", alpha=0.18, vmin=0, vmax=2,
                 zorder=2)

    # Draw classified perimeter edges
    for k, info in classes.items():
        lines = edges_by_class[k]
        if not lines: continue
        lc = LineCollection(lines, colors=info["color"], linewidths=info["lw"],
                             capstyle="round", zorder=6,
                             label=f"{info['label']} ({len(lines)})")
        ax.add_collection(lc)

    # Element centroids for context
    cx, cy = hgrid["centroids"][:, 0], hgrid["centroids"][:, 1]
    in_view = (cx >= xlim[0]) & (cx <= xlim[1]) & (cy >= ylim[0]) & (cy <= ylim[1])
    # Mark cells by their flag with small text — but only label flagged cells (not -1)
    for ei in np.where(in_view & (flag > 0))[0]:
        ax.text(cx[ei], cy[ei], str(flag[ei]), fontsize=7,
                ha="center", va="center", color="black", zorder=8,
                weight="bold")

    ax.set_aspect("equal")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(
        "Pioneer Mouth — every CV-perimeter edge classified by flag pair\n"
        "Numbers in cells are their fluxflag values (only positive flags labelled)",
        fontweight="bold", fontsize=11
    )
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        f"flux.out col 3 captures ONLY the {len(edges_by_class['tracked_23'])} GREEN edges; "
        f"all RED/ORANGE/PURPLE edges are at the mouth too but INVISIBLE to flux.out",
        fontsize=11, fontweight="bold", y=0.96
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
