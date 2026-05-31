"""
Single-panel animation of horizontal velocity magnitude:
    |u| = sqrt(horizontalVelX^2 + horizontalVelY^2)

Reads both SCHISM scribe-mode files (node-centred, 3D) and composes a derived
DataArray on the fly.  Same layout/style as animate_oxy_diags.py (single panel
mode): figsize 3.6 x 8.8 in, zoom 149.15-149.27 x -21.325 to -21.025, horizontal
colourbar, gliding inset on Dry %.

Usage:
  python animate_velocity.py --n-stacks 2
  python animate_velocity.py --n-stacks 14 --inset-horizon-days 3
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from animlib import (
    read_hgrid,
    build_animation,
    discover_out2d,
    identify_dims,
    truncate_to_ndays,
    _expand_stacks,
    _concat_along_time,
    _open_out2d_multi,
)


RUN_DIR = Path(
    "/Volumes/Development/schism/Pioneer_17_AED_SCHISM_FY2021_FY2022_2D_dry_year_3h_write_frequency"
)
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_OUT = Path(__file__).parent / "velocity_animation.mp4"


def _open_velocity_magnitude(outputs_dir, multi=True, limit=None, ndays=None):
    """Open horizontalVelX_*.nc + horizontalVelY_*.nc (auto-discovered stacks),
    concatenate along time, and return a derived |u| DataArray."""
    x_path = outputs_dir / "horizontalVelX_1.nc"
    y_path = outputs_dir / "horizontalVelY_1.nc"
    x_paths = _expand_stacks(x_path, multi=multi, limit=limit)
    y_paths = _expand_stacks(y_path, multi=multi, limit=limit)
    if not x_paths or not y_paths:
        raise FileNotFoundError(
            f"horizontalVelX/Y files not found in {outputs_dir}. "
            "Are iof_hydro(16) / scribed horizontal velocity outputs enabled?"
        )
    print(f"  X stacks: {[p.name for p in x_paths]}")
    print(f"  Y stacks: {[p.name for p in y_paths]}")
    ds_x = [xr.open_dataset(p, engine="h5netcdf", decode_cf=True) for p in x_paths]
    ds_y = [xr.open_dataset(p, engine="h5netcdf", decode_cf=True) for p in y_paths]

    name_x = next(v for v in ds_x[0].data_vars
                  if any(d.lower() == "time" for d in ds_x[0][v].dims))
    name_y = next(v for v in ds_y[0].data_vars
                  if any(d.lower() == "time" for d in ds_y[0][v].dims))
    time_dim_init = next(d for d in ds_x[0][name_x].dims if d.lower() == "time")
    da_x = _concat_along_time([ds[name_x] for ds in ds_x], time_dim_init)
    da_y = _concat_along_time([ds[name_y] for ds in ds_y], time_dim_init)

    # Compose the magnitude as an xarray DataArray (preserves dims & coords).
    da = np.sqrt(da_x ** 2 + da_y ** 2)
    da = da.rename("velocity_magnitude")
    da.attrs["units"] = "m s^-1"
    da.attrs["long_name"] = "Horizontal velocity magnitude"

    if ndays is not None:
        da = truncate_to_ndays(da, time_dim_init, ndays)

    time_dim, node_dim, layer_dim = identify_dims(da, "node")
    return ds_x + ds_y, da, time_dim, node_dim, layer_dim


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--outputs-dir", type=Path, default=RUN_DIR / "outputs",
                   help="Directory containing horizontalVelX_*.nc and horizontalVelY_*.nc.")
    p.add_argument("--out2d", type=Path, default=None)
    p.add_argument("--no-dry-mask", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="surface",
                   help="'surface' (default), 'bottom', or integer layer index.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", default="cmc.batlow",
                   help="Colormap (default: cmc.batlow sequential).")
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
    print(f"  n_nodes={hgrid['n_nodes']}")

    multi = not args.single_stack
    print(f"Opening velocity X/Y in {args.outputs_dir}")
    ds_xy_list, da_v, td, sd, ld = _open_velocity_magnitude(
        args.outputs_dir, multi=multi, limit=args.n_stacks, ndays=args.ndays,
    )
    print(f"  velocity_magnitude  dims={da_v.dims}  shape={da_v.shape}  units='{da_v.attrs.get('units','')}'")

    panels = [{
        "da": da_v, "time_dim": td, "space_dim": sd, "layer_dim": ld,
        "name": "velocity_magnitude",
        "cmap": args.cmap,
        "units": da_v.attrs.get("units", "m s^-1"),
        "centering": "node",
        "show_inset": True,
    }]

    dry_node = None
    dry_elem = None
    ds_2d_list = []
    if not args.no_dry_mask:
        ref = args.outputs_dir / "horizontalVelX_1.nc"
        out2d_path = args.out2d or discover_out2d(ref)
        if out2d_path and Path(out2d_path).exists():
            print(f"Opening out2d:  {out2d_path}")
            ds_2d_list, dry_node, dry_elem = _open_out2d_multi(
                out2d_path, multi=multi, limit=args.n_stacks
            )
            if dry_node is not None:
                print(f"  dryFlagNode dims={dry_node.dims} shape={dry_node.shape}")

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
    for ds in ds_xy_list:
        ds.close()
    for ds in ds_2d_list:
        ds.close()


if __name__ == "__main__":
    main()
