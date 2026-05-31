"""
Mouth-zoom diagnostic map for fluxflag.prop topology.

Two panels:
    a) Wider view of the Pioneer Mouth area + the eastern half of the CV
    b) Tight close-up on the (region 2 ↔ region 3) transect

Each panel:
    - cells coloured by fluxflag region (or grey if -1)
    - mesh edges drawn lightly
    - CV polygon overlaid
    - ACTIVE flux faces highlighted:
        - GREEN  edges: both adjacent elements flagged AND |Δflag|=1 (these
                        edges contribute to flux.out)
        - RED    edges: one side is flagged, the other is -1 (NOT in flux.out;
                        flow across these is INVISIBLE)
        - PURPLE edges: both sides flagged but |Δflag| > 1 (also not recorded)

For our CV, only the GREEN edges contribute to the row 3 cumulative flux.
The RED edges are the "missing" mouth perimeter.
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
from core import geometry


def build_edges(hgrid):
    """Return:
        edges_xy : (n_edges, 2, 2) array of [(x0,y0), (x1,y1)] per edge
        edge_elems : list of (e0, e1) — element ids adjacent to each edge
                     (e1 = -1 if boundary of the mesh)
    """
    elements = hgrid["elements"]
    x, y = hgrid["x"], hgrid["y"]
    edge2elem = defaultdict(list)
    edge2nodes = {}
    for ei in range(hgrid["n_elements"]):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k + 1) % nc]
            key = (min(a, b), max(a, b))
            edge2elem[key].append(ei)
            edge2nodes[key] = (a, b)
    edges_xy = []
    edge_elems = []
    for key, elems in edge2elem.items():
        a, b = edge2nodes[key]
        edges_xy.append([(x[a], y[a]), (x[b], y[b])])
        edge_elems.append((elems[0], elems[1] if len(elems) > 1 else -1))
    return np.asarray(edges_xy), edge_elems


def classify_edges(edges_xy, edge_elems, flag, cv_mask):
    """Bucket edges by interface type.  Returns dict of LineCollections data."""
    green, red, purple, edge_cv = [], [], [], []
    for (e0, e1), xy in zip(edge_elems, edges_xy):
        if e1 == -1:
            continue   # mesh outer boundary — not in flux.out, skip
        f0, f1 = flag[e0], flag[e1]
        in_cv = (cv_mask[e0] != cv_mask[e1])   # CV boundary edge (used for visual context)
        if f0 >= 0 and f1 >= 0:
            if abs(f0 - f1) == 1:
                green.append(xy)
            elif f0 != f1:
                purple.append(xy)
        elif (f0 == -1) ^ (f1 == -1):   # exactly one is -1
            red.append(xy)
        if in_cv:
            edge_cv.append(xy)
    return {
        "active (|Δflag|=1, both ≥0)": (green, "#16a34a", 1.4),
        "INVISIBLE (-1 ↔ flagged)":    (red,   "#dc2626", 1.4),
        "non-adjacent flags (Δ>1)":    (purple, "#7c3aed", 1.0),
    }, edge_cv


def render_mouth_panel(ax, hgrid, flag, cv_mask, sources, polygon, xlim, ylim,
                       title, edges_xy=None, edge_elems=None, show_active=True,
                       show_source_labels=True, mesh_edge_lw=0.15):
    """One panel."""
    # Triangulation
    triangles, tri_to_elem = [], []
    for ei, e in enumerate(hgrid["elements"]):
        nc, *ids = e
        if nc == 3:
            triangles.append(ids); tri_to_elem.append(ei)
        elif nc == 4:
            triangles.append([ids[0], ids[1], ids[2]]); tri_to_elem.append(ei)
            triangles.append([ids[0], ids[2], ids[3]]); tri_to_elem.append(ei)
    triangles = np.asarray(triangles, dtype=np.int32)
    tri_to_elem = np.asarray(tri_to_elem, dtype=np.int32)
    triang = mtri.Triangulation(hgrid["x"], hgrid["y"], triangles)
    tri_flag = flag[tri_to_elem]

    tab10 = cm.get_cmap("tab10", 10)
    palette = [(0.92, 0.92, 0.92, 1.0)]
    max_r = int(max(flag.max(), 1))
    for r in range(1, max_r + 1):
        palette.append(tab10((r - 1) % 10))
    cmap_disc = ListedColormap(palette)
    region_idx = np.where(tri_flag < 0, 0, tri_flag)
    boundaries = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap_disc.N)
    ax.tripcolor(triang, facecolors=region_idx, shading="flat",
                 cmap=cmap_disc, norm=norm,
                 edgecolors="#4d4d4d66" if mesh_edge_lw > 0 else "none",
                 linewidth=mesh_edge_lw)

    # CV mask overlay — hatch the CV cells
    cv_tri_mask = cv_mask[tri_to_elem]
    # Build CV outline by drawing a thicker boundary of the CV cells using cv_polygon
    if polygon is not None:
        px = [p[0] for p in polygon]
        py = [p[1] for p in polygon]
        ax.plot(px, py, "-", color="black", linewidth=2.0,
                label="CV polygon", zorder=5)

    # Active-flux edges overlay
    if show_active and edges_xy is not None:
        edge_classes, _ = classify_edges(edges_xy, edge_elems, flag, cv_mask)
        for lab, (lines, color, lw) in edge_classes.items():
            if not lines:
                continue
            lc = LineCollection(lines, colors=color, linewidths=lw + 0.4,
                                 capstyle="round", zorder=7)
            ax.add_collection(lc)

    # Sources within view
    for s_idx, (elem_1, lab) in enumerate(sources):
        ei = elem_1 - 1
        cx, cy = hgrid["centroids"][ei]
        if xlim[0] <= cx <= xlim[1] and ylim[0] <= cy <= ylim[1]:
            ax.plot(cx, cy, "o", color="red", markersize=11,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=8)
            if show_source_labels:
                ax.annotate(f"  src #{s_idx+1}", xy=(cx, cy), fontsize=9,
                            color="darkred", fontweight="bold", zorder=9)

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)


def read_sources(source_sink_path):
    sources = []
    with open(source_sink_path) as f:
        n_src = int(f.readline().split("!")[0])
        for _ in range(n_src):
            line = f.readline()
            if "!" in line:
                elem_str, label = line.split("!", 1)
                sources.append((int(elem_str.strip()), label.strip()))
            else:
                sources.append((int(line.strip()), ""))
    return sources


def read_polygon(csv_path):
    pts = []
    with open(csv_path) as f:
        next(f)
        for line in f:
            lon, lat = map(float, line.strip().split(","))
            pts.append((lon, lat))
    return pts


def main():
    run = Path("s:/Matt_Working/schism/P18_flood")
    cfg_dir = Path(__file__).parent / "configs"
    out_path = Path(__file__).parent / "fluxflag_zoom_mouth.png"

    print(f"Loading hgrid + fluxflag + sources from {run} ...")
    hgrid = geometry.read_hgrid_with_areas(run / "hgrid.gr3")
    flag = np.loadtxt(run / "fluxflag.prop", dtype=int)[:, 1]
    sources = read_sources(run / "source_sink.in")
    polygon = read_polygon(cfg_dir / "cv_pioneer_estuary.csv")

    cv_ids = [int(l.split("#", 1)[0].strip())
              for l in open(cfg_dir / "cv_pioneer_estuary_elements.txt")
              if l.strip() and not l.strip().startswith("#")]
    cv_elements = np.array(cv_ids, dtype=int) - 1
    cv_mask = np.zeros(hgrid["n_elements"], dtype=bool)
    cv_mask[cv_elements] = True

    print(f"Building edge enumeration ({hgrid['n_elements']} elements)...")
    edges_xy, edge_elems = build_edges(hgrid)
    print(f"  total edges: {len(edges_xy)}")

    # Edge-class counts for caption
    edge_classes, _ = classify_edges(edges_xy, edge_elems, flag, cv_mask)
    counts = {k: len(v[0]) for k, v in edge_classes.items()}
    print(f"  edge classes (whole mesh):")
    for k, n in counts.items():
        print(f"    {k}: {n}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 8))

    # Panel a — wider view: eastern half of the CV including the mouth
    render_mouth_panel(
        axA, hgrid, flag, cv_mask, sources, polygon,
        xlim=(149.16, 149.225), ylim=(-21.155, -21.110),
        title="(a) Eastern CV + Pioneer Mouth — wide view",
        edges_xy=edges_xy, edge_elems=edge_elems,
        show_active=True, mesh_edge_lw=0.08,
    )

    # Panel b — tight close-up on the (2↔3) interface
    render_mouth_panel(
        axB, hgrid, flag, cv_mask, sources, polygon,
        xlim=(149.196, 149.218), ylim=(-21.146, -21.128),
        title="(b) Close-up of the (region 2 ↔ region 3) transect",
        edges_xy=edges_xy, edge_elems=edge_elems,
        show_active=True, mesh_edge_lw=0.25,
    )

    tab10 = cm.get_cmap("tab10", 10)
    unique_regions = sorted(set(int(r) for r in flag if r > 0))
    handles = [plt.Rectangle((0, 0), 1, 1, color=tab10((r - 1) % 10),
                              label=f"fluxflag region {r}")
               for r in unique_regions]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=(0.92, 0.92, 0.92, 1.0),
                                 label="unflagged (-1)"))
    handles.append(plt.Line2D([0], [0], color="black", linewidth=2.0,
                              label="CV polygon"))
    handles.append(plt.Line2D([0], [0], color="#16a34a", linewidth=2.2,
                              label="ACTIVE flux edge (|Δflag|=1 — in flux.out)"))
    handles.append(plt.Line2D([0], [0], color="#dc2626", linewidth=2.2,
                              label="INVISIBLE edge (-1↔flagged — NOT in flux.out)"))
    handles.append(plt.Line2D([0], [0], color="#7c3aed", linewidth=2.0,
                              label="non-adjacent flags (Δ>1 — NOT in flux.out)"))
    handles.append(plt.Line2D([0], [0], marker="o", color="red", linestyle="",
                              markersize=10, markeredgecolor="black",
                              label="point sources"))
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.95)

    n_active = counts.get("active (|Δflag|=1, both ≥0)", 0)
    n_invis  = counts.get("INVISIBLE (-1 ↔ flagged)", 0)
    n_other  = counts.get("non-adjacent flags (Δ>1)", 0)
    fig.suptitle(
        "Pioneer Mouth — fluxflag.prop topology + active flux faces (P18_flood)\n"
        f"whole-mesh edge counts:  "
        f"ACTIVE (|Δ|=1)={n_active}    "
        f"INVISIBLE (-1↔flagged)={n_invis}    "
        f"Δ>1={n_other}    "
        f"region cells:  r2={int((flag==2).sum())}  r3={int((flag==3).sum())}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
