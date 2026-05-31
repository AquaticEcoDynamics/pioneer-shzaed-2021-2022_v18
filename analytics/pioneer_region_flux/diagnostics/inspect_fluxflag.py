"""
Visualise fluxflag.prop regions and the model's point-source inflow locations
on the SCHISM mesh.

Usage:
    python inspect_fluxflag.py \
        --hgrid       /path/to/hgrid.gr3 \
        --fluxflag    /path/to/fluxflag.prop \
        --source-sink /path/to/source_sink.in \
        --output      fluxflag_map.png \
        [--xlim 149.10 149.40 --ylim -21.40 -21.00]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry


def read_source_sink(path: Path) -> list[tuple[int, str]]:
    """Return list of (element_id_1indexed, label) per source."""
    sources = []
    with open(path) as f:
        n_src = int(f.readline().split("!")[0])
        for _ in range(n_src):
            line = f.readline()
            if "!" in line:
                elem_str, label = line.split("!", 1)
                sources.append((int(elem_str.strip()), label.strip()))
            else:
                sources.append((int(line.strip()), ""))
    return sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hgrid", required=True, type=Path)
    ap.add_argument("--fluxflag", required=True, type=Path)
    ap.add_argument("--source-sink", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    print(f"Reading hgrid: {args.hgrid}")
    hgrid = geometry.read_hgrid_with_areas(args.hgrid)
    print(f"  n_nodes={hgrid['n_nodes']}  n_elements={hgrid['n_elements']}")

    # Build a triangulation. Quads → 2 triangles, track tri_to_elem.
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

    print(f"Reading fluxflag.prop: {args.fluxflag}")
    flag = np.loadtxt(args.fluxflag, dtype=int)[:, 1]   # per-element flag
    if len(flag) != hgrid["n_elements"]:
        print(f"  WARNING: fluxflag has {len(flag)} entries but mesh has "
              f"{hgrid['n_elements']} elements")
    tri_flag = flag[tri_to_elem]
    unique_regions = sorted(set(flag[flag > 0]))
    print(f"  regions present (excl. -1): {unique_regions}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 11))

    # Build a categorical colormap: index 0 = pale grey (for -1), then tab10 for regions 1..10.
    from matplotlib.colors import ListedColormap, BoundaryNorm
    palette = [(0.92, 0.92, 0.92, 1.0)]  # for -1 (default)
    tab10 = cm.get_cmap("tab10", 10)
    for r in range(1, max(unique_regions) + 1 if unique_regions else 1):
        palette.append(tab10((r - 1) % 10))
    cmap_disc = ListedColormap(palette)
    # Remap -1 -> 0 for the colormap index; positive regions -> their value.
    region_idx = np.where(tri_flag < 0, 0, tri_flag)
    boundaries = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap_disc.N)
    pc = ax.tripcolor(triang, facecolors=region_idx, shading="flat",
                      cmap=cmap_disc, norm=norm,
                      edgecolors="none", linewidth=0)

    # Overlay the river source elements as big red dots
    if args.source_sink and args.source_sink.exists():
        sources = read_source_sink(args.source_sink)
        print(f"  found {len(sources)} sources in source_sink.in")
        for elem1, label in sources:
            ei = elem1 - 1
            cx, cy = hgrid["centroids"][ei]
            ax.plot(cx, cy, "o", color="red", markersize=9,
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)
            short_label = label.split("GIS node")[-1].strip().rstrip(",")
            ax.annotate(f"  src #{sources.index((elem1, label))+1}: {short_label[:30]}",
                        xy=(cx, cy), fontsize=7, color="darkred", zorder=6)

    # Legend for fluxflag region colours
    handles = [plt.Rectangle((0, 0), 1, 1, color=tab10((r - 1) % 10), label=f"region {r}")
               for r in unique_regions]
    handles.append(plt.Line2D([0], [0], marker="o", color="red", linestyle="",
                              markersize=8, markeredgecolor="black",
                              label="point sources (source_sink.in)"))
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              title="fluxflag.prop regions", framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"fluxflag.prop regions on mesh — {args.fluxflag.name}")
    if args.xlim: ax.set_xlim(args.xlim)
    if args.ylim: ax.set_ylim(args.ylim)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
