"""
3-panel animation of the conservative tracers TRC_tr1 | TRC_tr2 | TRC_tr3.

Same look & feel as analytics/animate_STO2.py — uses the shared
animate_oxy_diags infrastructure (figsize, aspect, default xlim/ylim,
horizontal colorbars, corner annotations, dry mask). All three panels are
node-centred SCHISM scribe outputs from the P18 run.
"""

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# bundled ffmpeg from imageio-ffmpeg (matches other animate_*.py scripts)
try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

# Reuse shared animation infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))
from animlib import (
    read_hgrid,
    build_animation,
    discover_out2d,
    _open_scribe_panel,
    _open_out2d_multi,
)


RUN_DIR = Path("s:/Matt_Working/schism/P18_flood")
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_OUT = Path(__file__).parent / "TRC_animation_flood.mp4"


# Panel spec: (filename pattern (stack 1), cmap, show_inset).
PANEL_SPECS = [
    ("TRC_tr1_1.nc", "cmo.matter", True),    # src 3 marker
    ("TRC_tr2_1.nc", "cmo.matter", False),   # src 7 marker
    ("TRC_tr3_1.nc", "cmo.matter", False),   # unsourced
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--outputs-dir", type=Path, default=RUN_DIR / "outputs")
    p.add_argument("--out2d", type=Path, default=None)
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
    p.add_argument("--single-stack", action="store_true")
    p.add_argument("--n-stacks", type=int, default=None)
    p.add_argument("--ndays", type=float, default=None)
    p.add_argument("--inset-horizon-days", type=float, default=None)
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
            print(f"  WARNING: {path.name} not found — SKIPPING")
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

    # Dry-mask via out2d
    dry_node = None
    dry_elem = None
    ds_2d_list = []
    if not args.no_dry_mask:
        ref_path = args.outputs_dir / PANEL_SPECS[0][0]
        out2d_path = args.out2d or discover_out2d(ref_path)
        if out2d_path and Path(out2d_path).exists():
            print(f"Opening out2d:  {out2d_path}")
            ds_2d_list, dry_node, dry_elem = _open_out2d_multi(
                out2d_path, multi=multi, limit=args.n_stacks
            )

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
