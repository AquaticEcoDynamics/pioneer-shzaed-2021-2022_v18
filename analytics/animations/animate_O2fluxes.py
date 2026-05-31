"""
3-panel O2 animation:
  - left:    OXY_oxy        (3D state, node-centred, from scribe file)
  - middle:  OXY_oxy_atm    (2D sheet diagnostic, element-centred, from cmb)
  - right:   OXY_oxy_dsf    (2D sheet diagnostic, element-centred, from cmb)

Same layout / xlim / ylim / colourbar treatment / horizontal cbars as
animate_STO2.py. Only the leftmost panel carries the Dry % inset (with the
gliding x-axis if --inset-horizon-days is set).

Note on 2D sheet diagnostics:
  OXY_oxy_atm and OXY_oxy_dsf are shape (time, nface) — no vertical dimension.
  The script's existing element-centred path handles them naturally:
  layer_dim=None -> _select_layer is a no-op; tripcolor uses shading="flat"
  via the tri_to_elem mapping. No surface/bottom selector applies.

Usage:
  python animate_O2fluxes.py --n-stacks 14 --inset-horizon-days 3
"""

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from animlib import (
    read_hgrid,
    build_animation,
    discover_out2d,
    _open_scribe_panel,
    _open_aed_combined,
    _open_aed_diag_panel,
    _open_out2d_multi,
)


RUN_DIR = Path(
    "/Volumes/Development/schism/Pioneer_17_AED_SCHISM_FY2021_FY2022_2D_dry_year_3h_write_frequency"
)
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_OUT = Path(__file__).parent / "O2fluxes_animation.mp4"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--outputs-dir", type=Path, default=RUN_DIR / "outputs",
                   help="Directory containing OXY_oxy_*.nc, aed_data_cmb_*.nc, out2d_*.nc.")
    p.add_argument("--out2d", type=Path, default=None,
                   help="Optional out2d_*.nc; auto-discovered if omitted.")
    p.add_argument("--no-dry-mask", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="surface",
                   help="'surface' (default), 'bottom', or integer index — applied to "
                        "the 3D OXY_oxy panel; the two flux panels are 2D and ignore this.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", default="berlin",
                   help="Colormap for all three O2 panels (default: berlin via cmcrameri).")
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
    print(f"  n_nodes={hgrid['n_nodes']}, n_elements={hgrid['n_elements']}, "
          f"n_triangles={len(hgrid['triangles'])}")

    multi = not args.single_stack
    panels = []
    ds_list = []

    # ---- LEFT: OXY_oxy from scribe (node-centred, 3D) ----
    oxy_path = args.outputs_dir / "OXY_oxy_1.nc"
    print(f"Opening scribe OXY_oxy:  {oxy_path}")
    ds_o_list, da_o, td_o, sd_o, ld_o, name_o, units_o = _open_scribe_panel(
        oxy_path, ndays=args.ndays, multi=multi, limit=args.n_stacks,
    )
    print(f"  variable={name_o}  dims={da_o.dims}  shape={da_o.shape}")
    panels.append({
        "da": da_o, "time_dim": td_o, "space_dim": sd_o, "layer_dim": ld_o,
        "name": name_o, "cmap": args.cmap, "units": units_o,
        "centering": "node",
        "show_inset": True,   # only the leftmost panel carries the inset
    })
    ds_list.extend(ds_o_list)

    # ---- MIDDLE + RIGHT: 2D sheet diagnostics from aed_data_cmb_*.nc ----
    cmb_path = args.outputs_dir / "aed_data_cmb_1.nc"
    print(f"Opening AED combined:  {cmb_path}")
    ds_cmb_list = _open_aed_combined(cmb_path, multi=multi, limit=args.n_stacks)
    ds_list.extend(ds_cmb_list)

    for vname in ("OXY_oxy_atm", "OXY_oxy_dsf"):
        da, td, sd, ld, units = _open_aed_diag_panel(ds_cmb_list, vname, ndays=args.ndays)
        print(f"  variable={vname}  dims={da.dims}  shape={da.shape}  units='{units}'  "
              f"(layer_dim={ld}; 2D sheet)")
        panels.append({
            "da": da, "time_dim": td, "space_dim": sd, "layer_dim": ld,
            "name": vname, "cmap": args.cmap, "units": units,
            "centering": "elem",
            "show_inset": False,
        })

    # ---- dry mask source ----
    dry_node = None
    dry_elem = None
    ds_2d_list = []
    if not args.no_dry_mask:
        out2d_path = args.out2d or discover_out2d(oxy_path)
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
    )

    print(f"Writing animation -> {args.output} (fps={args.fps}, dpi={args.dpi})")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    print("Done.")

    if args.show:
        plt.show()
    plt.close(fig)
    for ds in ds_list:
        ds.close()
    for ds in ds_2d_list:
        ds.close()


if __name__ == "__main__":
    main()
