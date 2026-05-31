"""
6-panel (2 x 3) animation of nitrogen-related 3D process-rate diagnostics
from aed_data_cmb_*.nc. All are element-centred with a vertical dim; surface
layer is plotted by default (override with --layer bottom or --layer N).

Layout (left-to-right, top-to-bottom):
  Row 1:  NIT_nitrif    NIT_denit      NIT_anammox
  Row 2:  NIT_dnra      OGM_don_min    OGM_pon_hyd

Pairs naturally with animate_Nfluxes.py:
  - Nfluxes  -> 2D sheet (sediment-water + atmospheric exchange) fluxes
  - Nrates   -> 3D water-column biogeochemical process rates

Usage:
  python animate_Nrates.py --n-stacks 1
  python animate_Nrates.py --n-stacks 14 --inset-horizon-days 3
"""

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from animlib import (
    read_hgrid,
    build_animation,
    discover_out2d,
    _open_aed_combined,
    _open_aed_diag_panel,
    _open_out2d_multi,
)


RUN_DIR = Path(
    "/Volumes/Development/schism/Pioneer_17_AED_SCHISM_FY2021_FY2022_2D_dry_year_3h_write_frequency"
)
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_OUT = Path(__file__).parent / "Nrates_animation.mp4"


# 6 panels, row-major order (left-to-right, top-to-bottom).
# All have shape (time, nface, layers) — full 3D process rates.
PANEL_VARS = [
    "NIT_nitrif",   "NIT_denit",    "NIT_anammox",
    "NIT_dnra",     "OGM_don_min",  "OGM_pon_hyd",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--outputs-dir", type=Path, default=RUN_DIR / "outputs",
                   help="Directory containing aed_data_cmb_*.nc and out2d_*.nc.")
    p.add_argument("--aed-cmb", type=Path, default=None,
                   help="Combined AED file; defaults to <outputs-dir>/aed_data_cmb_1.nc.")
    p.add_argument("--out2d", type=Path, default=None,
                   help="Optional out2d_*.nc; auto-discovered if omitted.")
    p.add_argument("--no-dry-mask", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="surface",
                   help="'surface' (default), 'bottom', or integer layer index. "
                        "These are 3D rate fields — vertical-layer selection applies.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", default="berlin",
                   help="Colormap for all six panels (default: berlin via cmcrameri).")
    p.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   default=[149.15, 149.27])
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   default=[-21.325, -21.025])
    p.add_argument("--single-stack", action="store_true")
    p.add_argument("--n-stacks", type=int, default=None,
                   help="Cap the number of auto-discovered stacks (default: all).")
    p.add_argument("--ndays", type=float, default=None)
    p.add_argument("--inset-horizon-days", type=float, default=None,
                   help="Gliding inset x-axis: show only the last N days of history.")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    print(f"Reading hgrid:  {args.hgrid}")
    hgrid = read_hgrid(args.hgrid)
    print(f"  n_nodes={hgrid['n_nodes']}, n_elements={hgrid['n_elements']}")

    multi = not args.single_stack
    aed_path = args.aed_cmb or (args.outputs_dir / "aed_data_cmb_1.nc")
    print(f"Opening AED combined:  {aed_path}")
    ds_cmb_list = _open_aed_combined(aed_path, multi=multi, limit=args.n_stacks)

    panels = []
    for i, vname in enumerate(PANEL_VARS):
        da, td, sd, ld, units = _open_aed_diag_panel(ds_cmb_list, vname, ndays=args.ndays)
        print(f"  variable={vname}  dims={da.dims}  shape={da.shape}  units='{units}'  "
              f"(layer_dim={ld}; 3D pelagic)")
        panels.append({
            "da": da, "time_dim": td, "space_dim": sd, "layer_dim": ld,
            "name": vname, "cmap": args.cmap, "units": units,
            "centering": "elem",
            "show_inset": (i == 0),   # only top-left panel gets the Dry % inset
        })

    # Dry mask
    dry_node = None
    dry_elem = None
    ds_2d_list = []
    if not args.no_dry_mask:
        out2d_path = args.out2d or discover_out2d(aed_path)
        if out2d_path and Path(out2d_path).exists():
            print(f"Opening out2d:  {out2d_path}")
            ds_2d_list, dry_node, dry_elem = _open_out2d_multi(
                out2d_path, multi=multi, limit=args.n_stacks
            )
            if dry_node is not None:
                print(f"  dryFlagNode dims={dry_node.dims} shape={dry_node.shape}")
            if dry_elem is not None:
                print(f"  dryFlagElement dims={dry_elem.dims} shape={dry_elem.shape}")

    fig, anim = build_animation(
        hgrid, panels,
        dry_node=dry_node,
        dry_elem=dry_elem,
        layer_choice=args.layer,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        inset_horizon_days=args.inset_horizon_days,
        nrows=2,
    )

    print(f"Writing animation -> {args.output} (fps={args.fps}, dpi={args.dpi})")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    print("Done.")

    if args.show:
        plt.show()
    plt.close(fig)
    for ds in ds_cmb_list:
        ds.close()
    for ds in ds_2d_list:
        ds.close()


if __name__ == "__main__":
    main()
