"""
3-panel animation: Salinity | Temperature | Oxygen.

Same layout as animate_oxy_diags.py (figsize, aspect, xlim/ylim, anchor,
horizontal colorbars, corner annotations) — just three node-centred SCHISM
scribe-mode variables side-by-side.

Each panel can OPTIONALLY have an inset (Dry % time series). By default only
the leftmost (Salinity) panel shows the inset.

Prerequisites:
  - Salinity is enabled by default (iof_hydro(19) = 1).
  - Temperature requires iof_hydro(18) = 1 in param.nml (currently OFF in this
    run, so temperature_*.nc does not exist yet). The script will warn and
    skip the T panel if the file is missing.
  - OXY_oxy is the AED state variable, scribed when AED is on.

Usage:
  python animate_STO2.py
  python animate_STO2.py --n-stacks 3
  python animate_STO2.py --output OXY_STO2.mp4 --fps 8
"""

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Reuse all the shared infrastructure: hgrid reading, panel setup, dry-mask
# concatenation, layout / aspect / xlim / ylim defaults, friendly labels.
from animlib import (
    read_hgrid,
    build_animation,
    discover_out2d,
    _open_scribe_panel,
    _open_out2d_multi,
)


RUN_DIR = Path(
    "/Volumes/Development/schism/Pioneer_17_AED_SCHISM_FY2021_FY2022_2D_dry_year_3h_write_frequency"
)
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_OUT = Path(__file__).parent / "STO2_animation.mp4"


# Panel spec: (NC filename pattern (stack 1), cmap, show_inset).
# All three are node-centred SCHISM scribe outputs.
PANEL_SPECS = [
    ("salinity_1.nc",    "cmo.haline",  True),    # S — with inset
    ("temperature_1.nc", "cmo.thermal", False),   # T — no inset (requires iof_hydro(18)=1)
    ("OXY_oxy_1.nc",     "berlin",      False),   # O2 — no inset (cmc.berlin via cmcrameri)
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--outputs-dir", type=Path, default=RUN_DIR / "outputs",
                   help="Directory containing the scribed *_1.nc files.")
    p.add_argument("--out2d", type=Path, default=None,
                   help="Optional out2d_*.nc; auto-discovered relative to outputs-dir if omitted.")
    p.add_argument("--no-dry-mask", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="surface",
                   help="'surface' (default), 'bottom', or integer layer index.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   default=[149.15, 149.27])
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   default=[-21.325, -21.025])
    p.add_argument("--single-stack", action="store_true",
                   help="Only animate the named stack files; don't auto-discover siblings.")
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

    for fname, cmap, show_inset in PANEL_SPECS:
        path = args.outputs_dir / fname
        if not path.exists():
            print(f"  WARNING: {path.name} not found in {args.outputs_dir} — "
                  f"SKIPPING this panel. (For temperature, set iof_hydro(18)=1 in param.nml.)")
            continue
        print(f"Opening {path}")
        ds_l, da, td, sd, ld, name, units = _open_scribe_panel(
            path, ndays=args.ndays, multi=multi, limit=args.n_stacks,
        )
        print(f"  variable={name}  dims={da.dims}  shape={da.shape}  units='{units}'")
        panels.append({
            "da": da, "time_dim": td, "space_dim": sd, "layer_dim": ld,
            "name": name, "cmap": cmap, "units": units,
            "centering": "node",
            "show_inset": show_inset,
        })
        ds_list.extend(ds_l)

    if not panels:
        raise SystemExit("No panels available — check outputs-dir.")

    # Dry-mask (out2d): use whichever flag is available — dryFlagNode covers all
    # node-centred panels.
    dry_node = None
    dry_elem = None
    ds_2d_list = []
    if not args.no_dry_mask:
        # Auto-discover relative to the first found panel file
        ref_path = args.outputs_dir / PANEL_SPECS[0][0]
        out2d_path = args.out2d or discover_out2d(ref_path)
        if out2d_path and Path(out2d_path).exists():
            print(f"Opening out2d:  {out2d_path}")
            ds_2d_list, dry_node, dry_elem = _open_out2d_multi(
                out2d_path, multi=multi, limit=args.n_stacks
            )
            if dry_node is not None:
                print(f"  dryFlagNode dims={dry_node.dims} shape={dry_node.shape}")
            if dry_elem is not None:
                print(f"  dryFlagElement dims={dry_elem.dims} shape={dry_elem.shape}")
        else:
            print("  No out2d_*.nc found; continuing without dry mask")

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
