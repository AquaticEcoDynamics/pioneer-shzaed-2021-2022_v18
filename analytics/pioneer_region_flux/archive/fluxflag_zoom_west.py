"""
Zoom-in maps of the fluxflag regions for the Pioneer Estuary.

Two panels:
    a) full estuary view with all 9 regions + 3 CV sources
    b) close-up around src 3 / region 1 / west tip of region 2
       (this is the area where we've been arguing about how much flux is
       captured at flux.out row 2)

The map colors EACH element by its fluxflag region (so the tiny 11-cell
region 1 is actually visible as a 'patch', not lost in averaging).
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core import geometry


def render_panel(ax, hgrid, flag, sources, polygon, xlim, ylim, title,
                 show_source_labels=True, edge_lw=0.0):
    """Render one map panel."""
    # Build triangulation
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

    # Discrete colormap: -1 -> light grey, regions 1..9 -> tab10
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
                 edgecolors="black" if edge_lw > 0 else "none",
                 linewidth=edge_lw)

    # CV polygon overlay
    if polygon is not None:
        px = [p[0] for p in polygon]
        py = [p[1] for p in polygon]
        ax.plot(px, py, "-", color="black", linewidth=1.6, label="CV polygon")

    # Sources
    for s_idx, (elem_1, lab) in enumerate(sources):
        ei = elem_1 - 1
        cx, cy = hgrid["centroids"][ei]
        if xlim[0] <= cx <= xlim[1] and ylim[0] <= cy <= ylim[1]:
            ax.plot(cx, cy, "o", color="red", markersize=11,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=6)
            if show_source_labels:
                ax.annotate(f"  src #{s_idx+1}\n  {lab[:24]}",
                            xy=(cx, cy), fontsize=8, color="darkred",
                            fontweight="bold", zorder=7)

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)


def read_sources(source_sink_path):
    """Parse source_sink.in -> [(elem_1idx, label), ...] for sources only."""
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
        next(f)  # header
        for line in f:
            lon, lat = map(float, line.strip().split(","))
            pts.append((lon, lat))
    return pts


def main():
    run = Path("s:/Matt_Working/schism/P18")
    cfg_dir = Path(__file__).parent / "configs"
    out_path = Path(__file__).parent / "fluxflag_zoom_west.png"

    hgrid = geometry.read_hgrid_with_areas(run / "hgrid.gr3")
    flag = np.loadtxt(run / "fluxflag.prop", dtype=int)[:, 1]
    sources = read_sources(run / "source_sink.in")
    polygon = read_polygon(cfg_dir / "cv_pioneer_estuary.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 8))

    # Panel a — full estuary view
    render_panel(axA, hgrid, flag, sources, polygon,
                 xlim=(149.07, 149.23), ylim=(-21.16, -21.115),
                 title="(a) Full estuary — all 9 fluxflag regions",
                 edge_lw=0.0)

    # Panel b — close-up on the western end where src 3 + region 1 + west of region 2 live
    render_panel(axB, hgrid, flag, sources, polygon,
                 xlim=(149.085, 149.115), ylim=(-21.152, -21.140),
                 title="(b) Close-up around src 3 + region 1 + west of region 2",
                 edge_lw=0.25)

    # Region legend
    tab10 = cm.get_cmap("tab10", 10)
    unique_regions = sorted(set(int(r) for r in flag if r > 0))
    handles = [plt.Rectangle((0, 0), 1, 1, color=tab10((r - 1) % 10),
                              label=f"fluxflag region {r}")
               for r in unique_regions]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=(0.92, 0.92, 0.92, 1.0),
                                 label="unflagged (-1)"))
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.6,
                              label="CV polygon"))
    handles.append(plt.Line2D([0], [0], marker="o", color="red", linestyle="",
                              markersize=10, markeredgecolor="black",
                              label="point sources"))
    fig.legend(handles=handles, loc="lower center", ncols=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95)

    fig.suptitle(
        "Pioneer Estuary fluxflag.prop topology — diagnostic zoom\n"
        f"region cell counts:  "
        f"r1={int((flag==1).sum())}  r2={int((flag==2).sum())}  "
        f"r3={int((flag==3).sum())}  -1={int((flag==-1).sum())}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
