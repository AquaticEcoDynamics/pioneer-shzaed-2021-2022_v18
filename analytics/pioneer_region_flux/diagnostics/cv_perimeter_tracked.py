"""
Diagnostic map: CV polygon perimeter — which segments coincide with a
flux-tracked fluxflag interface vs. which run through unflagged water.

For each edge on the CV polygon boundary (= edges where exactly one of
the two adjacent elements is in the CV), classify by whether it would
contribute to a flux.out row:
    GREEN   :  |Δflag|=1 AND both flags ≥ 0   (recorded)
    RED     :  one side is -1                  (NOT recorded)
    PURPLE  :  flags differ but |Δflag|>1     (NOT recorded)

This is the visualisation that explains why for tr3 the CV ΔM = 19 M mmol
but cumulative flux.out row 3 (mouth) = only 0.87 M mmol.
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
    out_path = CFG.out_dir / "cv_perimeter_tracked.png"

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

    # Polygon for overlay
    pts = []
    with open(cfg / "cv_pioneer_estuary.csv") as f:
        next(f)
        for line in f:
            lon, lat = map(float, line.strip().split(","))
            pts.append((lon, lat))
    px = [p[0] for p in pts]; py = [p[1] for p in pts]

    # Build edges of CV polygon (where exactly one of the two adjacent cells is in CV)
    print("Building edges + classifying CV perimeter...")
    edge2elem = defaultdict(list)
    edge2nodes = {}
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k+1) % nc]
            key = (min(a, b), max(a, b))
            edge2elem[key].append(ei)
            edge2nodes[key] = (a, b)

    perim_green = []   # tracked
    perim_red   = []   # -1 ↔ flagged, untracked
    perim_purple = []  # |Δ|>1, untracked
    perim_orphan = []  # both same flag but across CV boundary
    for key, elems in edge2elem.items():
        if len(elems) != 2:
            continue
        e0, e1 = elems
        in0, in1 = cv_mask[e0], cv_mask[e1]
        if in0 == in1:
            continue   # not a CV-perimeter edge
        f0, f1 = flag[e0], flag[e1]
        a, b = edge2nodes[key]
        xy = [(x[a], y[a]), (x[b], y[b])]
        if f0 >= 0 and f1 >= 0:
            if abs(f0 - f1) == 1:   perim_green.append(xy)
            elif f0 == f1:           perim_orphan.append(xy)
            else:                    perim_purple.append(xy)
        else:
            perim_red.append(xy)

    print(f"  CV perimeter edges classified:")
    print(f"    GREEN  (tracked, |Δflag|=1)    : {len(perim_green)}")
    print(f"    RED    (-1↔flagged, untracked) : {len(perim_red)}")
    print(f"    PURPLE (|Δ|>1, untracked)      : {len(perim_purple)}")
    print(f"    ORPHAN (same flag, untracked)  : {len(perim_orphan)}")

    # Build triangulation for region colouring
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

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 9))

    for ax, xlim, ylim, title, edge_lw in [
        (axA, (149.07, 149.23), (-21.16, -21.11),
         "(a) Full CV view — polygon perimeter classified",
         0.0),
        (axB, (149.195, 149.220), (-21.150, -21.125),
         "(b) Close-up around the (2↔3) mouth transect",
         0.25),
    ]:
        ax.tripcolor(triang, facecolors=region_idx, shading="flat",
                     cmap=cmap_disc, norm=norm,
                     edgecolors="#4d4d4d33" if edge_lw > 0 else "none",
                     linewidth=edge_lw)
        # Plot CV polygon faintly
        ax.plot(px, py, "-", color="black", linewidth=1.0, alpha=0.5,
                zorder=4)
        # Overlay perimeter edges classified
        for lines, color, lw, lab in [
            (perim_green, "#16a34a", 3.0, "TRACKED"),
            (perim_red,   "#dc2626", 2.0, "UNTRACKED (-1)"),
            (perim_purple, "#7c3aed", 2.0, "UNTRACKED (|Δ|>1)"),
            (perim_orphan, "#f59e0b", 2.0, "UNTRACKED (same flag)"),
        ]:
            if not lines:
                continue
            lc = LineCollection(lines, colors=color, linewidths=lw,
                                 capstyle="round", zorder=6)
            ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.3)

    # Legend
    handles = []
    tab10 = cm.get_cmap("tab10", 10)
    unique_regions = sorted(set(int(r) for r in flag if r > 0))
    for r in unique_regions:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=tab10((r - 1) % 10),
                                     label=f"fluxflag region {r}"))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=(0.92, 0.92, 0.92, 1.0),
                                 label="unflagged (-1)"))
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.0,
                              alpha=0.5, label="CV polygon (faint)"))
    handles.append(plt.Line2D([0], [0], color="#16a34a", linewidth=3.0,
                              label=f"TRACKED perimeter ({len(perim_green)} edges)"))
    handles.append(plt.Line2D([0], [0], color="#dc2626", linewidth=2.0,
                              label=f"UNTRACKED (-1) ({len(perim_red)} edges)"))
    if perim_purple:
        handles.append(plt.Line2D([0], [0], color="#7c3aed", linewidth=2.0,
                                  label=f"UNTRACKED (|Δ|>1) ({len(perim_purple)})"))
    if perim_orphan:
        handles.append(plt.Line2D([0], [0], color="#f59e0b", linewidth=2.0,
                                  label=f"UNTRACKED same-flag ({len(perim_orphan)})"))
    fig.legend(handles=handles, loc="lower center", ncols=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95)

    total = len(perim_green) + len(perim_red) + len(perim_purple) + len(perim_orphan)
    tracked_pct = 100 * len(perim_green) / max(total, 1)
    fig.suptitle(
        "CV polygon perimeter — tracked vs untracked by flux.out\n"
        f"perimeter edges: {len(perim_green)} TRACKED / {total} TOTAL "
        f"= {tracked_pct:.1f} % of polygon perimeter is tracked",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
