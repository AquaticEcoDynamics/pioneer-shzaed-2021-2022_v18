"""
Build a control-volume polygon + element-ID list from:
  - a few user-supplied corner points (lon, lat)
  - a fluxflag region whose cells form one of the polygon edges

Produces:
  - cv_<name>.csv          polygon vertices (lon, lat) ready to load again later
  - cv_<name>_elements.txt element IDs (1-indexed) inside the polygon
  - cv_<name>_map.png      verification map of the CV overlaid on the mesh

Usage example (Pioneer Estuary, closing via fluxflag region 2):
    python build_cv.py \\
        --hgrid    /path/to/hgrid.gr3 \\
        --fluxflag /path/to/fluxflag.prop \\
        --source-sink /path/to/source_sink.in \\
        --name pioneer_estuary \\
        --close-via-region 2 --close-direction w_to_e \\
        --corner -21.125 149.215 \\
        --corner -21.125 149.08 \\
        --corner -21.15  149.08 \\
        --output-dir .
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.path as mpath
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# --- bootstrap: make the nutrient_budget package importable when run directly ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
# ------------------------------------------------------------------------------
from core import geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hgrid", required=True, type=Path)
    ap.add_argument("--fluxflag", required=True, type=Path)
    ap.add_argument("--source-sink", type=Path, default=None)
    ap.add_argument("--name", required=True,
        help="Short name used in output filenames (e.g. 'pioneer_estuary')")
    ap.add_argument("--corner", action="append", nargs=2, type=float,
        metavar=("LAT", "LON"),
        help="Polygon corner (lat lon). Repeat for each corner. Polygon will be "
             "constructed in the order given; the closing edge follows the "
             "fluxflag region specified by --close-via-region.")
    ap.add_argument("--close-via-region", type=int, default=None,
        help="fluxflag region number whose element centroids close the polygon")
    ap.add_argument("--close-direction", default="w_to_e",
        choices=["w_to_e", "e_to_w", "s_to_n", "n_to_s"],
        help="Order to walk the closing region's centroids")
    ap.add_argument("--close-min-lon", type=float, default=None,
        help="Drop closing-region cells with lon < this value")
    ap.add_argument("--close-max-lon", type=float, default=None,
        help="Drop closing-region cells with lon > this value")
    ap.add_argument("--close-min-lat", type=float, default=None,
        help="Drop closing-region cells with lat < this value")
    ap.add_argument("--close-max-lat", type=float, default=None,
        help="Drop closing-region cells with lat > this value")
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading hgrid: {args.hgrid}")
    hgrid = geometry.read_hgrid_with_areas(args.hgrid)
    print(f"  n_elements={hgrid['n_elements']}")

    print(f"Reading fluxflag.prop: {args.fluxflag}")
    flag = np.loadtxt(args.fluxflag, dtype=int)[:, 1]

    # ---- build polygon ----
    corners_latlon = args.corner or []
    if not corners_latlon:
        raise SystemExit("Pass at least 2 --corner LAT LON pairs.")
    # Convert to (lon, lat) tuples for the polygon
    corner_lonlat = [(lon, lat) for (lat, lon) in corners_latlon]

    closing_pts = []
    if args.close_via_region is not None:
        r_idx = np.where(flag == args.close_via_region)[0]
        if len(r_idx) == 0:
            raise SystemExit(f"No elements with fluxflag={args.close_via_region}")
        cx_r = hgrid["centroids"][r_idx, 0]
        cy_r = hgrid["centroids"][r_idx, 1]
        # Optional bbox filter on the closing trace
        keep = np.ones(len(r_idx), dtype=bool)
        if args.close_min_lon is not None:
            keep &= cx_r >= args.close_min_lon
        if args.close_max_lon is not None:
            keep &= cx_r <= args.close_max_lon
        if args.close_min_lat is not None:
            keep &= cy_r >= args.close_min_lat
        if args.close_max_lat is not None:
            keep &= cy_r <= args.close_max_lat
        n_dropped = (~keep).sum()
        if n_dropped:
            print(f"  closing-region bbox filter dropped {n_dropped} cell(s)")
        r_idx = r_idx[keep]
        cx_r = cx_r[keep]
        cy_r = cy_r[keep]
        if len(r_idx) == 0:
            raise SystemExit("All closing-region cells were filtered out")
        if args.close_direction == "w_to_e":
            order = np.argsort(cx_r)
        elif args.close_direction == "e_to_w":
            order = np.argsort(-cx_r)
        elif args.close_direction == "s_to_n":
            order = np.argsort(cy_r)
        else:
            order = np.argsort(-cy_r)
        closing_pts = [(cx_r[i], cy_r[i]) for i in order]
        print(f"  closing edge: {len(closing_pts)} pts from fluxflag region "
              f"{args.close_via_region} ({args.close_direction})")

    # Full polygon = corners + closing trace + back to first corner
    polygon = list(corner_lonlat) + closing_pts + [corner_lonlat[0]]
    print(f"  polygon has {len(polygon)} vertices")

    # ---- save polygon CSV ----
    poly_path = args.output_dir / f"cv_{args.name}.csv"
    with open(poly_path, "w") as f:
        f.write("lon,lat\n")
        for lon, lat in polygon:
            f.write(f"{lon:.6f},{lat:.6f}\n")
    print(f"  wrote {poly_path}")

    # ---- point-in-polygon: which elements are inside? ----
    cx = hgrid["centroids"][:, 0]
    cy = hgrid["centroids"][:, 1]
    path = mpath.Path([(p[0], p[1]) for p in polygon])
    inside = path.contains_points(np.column_stack([cx, cy]))
    cv_idx = np.where(inside)[0]
    cv_areas_m2 = hgrid["areas"][cv_idx].sum()
    print(f"  {len(cv_idx)} elements inside polygon "
          f"({cv_areas_m2/1e6:.2f} km²)")

    # ---- save element-ID list ----
    elem_path = args.output_dir / f"cv_{args.name}_elements.txt"
    with open(elem_path, "w") as f:
        f.write(f"# Control-volume element IDs (1-indexed) for region '{args.name}'\n")
        f.write(f"# Total elements: {len(cv_idx)}\n")
        f.write(f"# Total surface area: {cv_areas_m2:,.0f} m^2 ({cv_areas_m2/1e6:.2f} km^2)\n")
        for ei in cv_idx:
            f.write(f"{ei + 1}\n")   # 1-indexed
    print(f"  wrote {elem_path}")

    # ---- verification map ----
    map_path = args.output_dir / f"cv_{args.name}_map.png"
    _render_verification_map(hgrid, flag, polygon, cv_idx,
                              args.source_sink, args.xlim, args.ylim,
                              map_path, args.name)
    print(f"  wrote {map_path}")
    print()
    print("To use this CV with budget.py, set in the region YAML:")
    print(f"  control_volume_elements: {elem_path.name}")


def _render_verification_map(hgrid, flag, polygon, cv_idx,
                              source_sink_path, xlim, ylim,
                              out_path, name):
    """Plot mesh + fluxflag regions + polygon + selected CV elements."""
    # Build triangulation with tri_to_elem
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
    cv_mask = np.zeros(hgrid["n_elements"], dtype=bool)
    cv_mask[cv_idx] = True
    tri_in_cv = cv_mask[tri_to_elem]

    # Pseudo-RGBA per triangle:
    # base = pale grey for -1, tab10 colour per fluxflag region, special green tint for CV.
    palette = [(0.92, 0.92, 0.92, 1.0)]
    tab10 = cm.get_cmap("tab10", 10)
    max_r = int(max(flag.max(), 1))
    for r in range(1, max_r + 1):
        palette.append(tab10((r - 1) % 10))
    cmap_disc = ListedColormap(palette)
    region_idx = np.where(tri_flag < 0, 0, tri_flag)
    boundaries = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap_disc.N)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.tripcolor(triang, facecolors=region_idx, shading="flat",
                 cmap=cmap_disc, norm=norm, edgecolors="none", linewidth=0)

    # Overlay CV elements with a translucent green
    cv_facecolor = np.zeros((len(tri_flag), 4))
    cv_facecolor[tri_in_cv] = [0.20, 0.65, 0.30, 0.45]
    # Use a second tripcolor only for CV cells — draw via PolyCollection-style mask
    if tri_in_cv.any():
        # plot CV triangles in green via a separate tripcolor with masked array
        cv_only = np.where(tri_in_cv, 1.0, np.nan)
        from matplotlib.colors import ListedColormap as LC
        ax.tripcolor(triang, facecolors=cv_only, shading="flat",
                     cmap=LC([(0.20, 0.65, 0.30, 0.55)]), vmin=0.5, vmax=1.5)

    # Polygon outline
    px = [p[0] for p in polygon]
    py = [p[1] for p in polygon]
    ax.plot(px, py, "-", color="black", linewidth=1.8, label="CV polygon")
    ax.plot(px, py, "o", color="black", markersize=3)

    # Source dots
    if source_sink_path and source_sink_path.exists():
        with open(source_sink_path) as f:
            n_src = int(f.readline().split("!")[0])
            for s in range(n_src):
                line = f.readline()
                elem_str, *rest = line.split("!", 1)
                ei = int(elem_str.strip()) - 1
                cx_s, cy_s = hgrid["centroids"][ei]
                ax.plot(cx_s, cy_s, "o", color="red", markersize=8,
                        markeredgecolor="black", markeredgewidth=0.7, zorder=5)
                ax.annotate(f"  src#{s+1}", xy=(cx_s, cy_s), fontsize=7,
                            color="darkred", zorder=6)

    # Region legend
    unique_regions = sorted(set(int(r) for r in flag if r > 0))
    handles = [plt.Rectangle((0, 0), 1, 1, color=tab10((r - 1) % 10),
                              label=f"fluxflag region {r}")
               for r in unique_regions]
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.8,
                              label="CV polygon"))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=(0.20, 0.65, 0.30, 0.55),
                                  label="elements in CV"))
    handles.append(plt.Line2D([0], [0], marker="o", color="red", linestyle="",
                              markersize=8, markeredgecolor="black",
                              label="point sources"))
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Control volume: {name} — {len(cv_idx)} elements")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
