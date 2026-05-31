"""
Animate SCHISM AED dissolved oxygen (OXY_oxy) from the first stack NetCDF file.

Approach mirrors schism_vis_GUI.py::AnimationManager.create_2d_animation:
  - read hgrid.gr3 for node coords + element connectivity
  - build matplotlib.tri.Triangulation
  - use tripcolor with set_array() for fast in-place frame updates
  - render with FuncAnimation, save to MP4

Usage:
  python animate_oxy.py
  python animate_oxy.py --output oxy_animation.mp4 --fps 10
  python animate_oxy.py --hgrid /path/to/hgrid.gr3 --nc /path/to/OXY_oxy_1.nc
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from copy import copy
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

# Register Crameri's scientific colormaps (e.g. 'berlin') under the 'cmc.' prefix.
try:
    import cmcrameri.cm  # noqa: F401  (import registers maps with matplotlib)
except ImportError:
    pass

MUDFLAT_BROWN = "#8B6F47"


def resolve_cmap(name):
    """Get a colormap, accepting both bare ('berlin') and prefixed ('cmc.berlin') names."""
    try:
        return plt.get_cmap(name)
    except (ValueError, KeyError):
        try:
            return plt.get_cmap(f"cmc.{name}")
        except (ValueError, KeyError):
            print(f"  WARNING: colormap '{name}' not found; falling back to viridis")
            return plt.get_cmap("viridis")

# Point matplotlib at the imageio-ffmpeg bundled binary if available, so the
# script works on machines without a system ffmpeg install.
try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


RUN_DIR = Path(
    "/Volumes/Development/schism/Pioneer_17_AED_SCHISM_FY2021_FY2022_2D_dry_year_3h_write_frequency"
)
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_NC = RUN_DIR / "outputs_pawsey" / "OXY_oxy_1.nc"
DEFAULT_OUT = Path(__file__).parent / "OXY_oxy_1_animation.mp4"


def read_hgrid(filename):
    """Read SCHISM hgrid.gr3. Returns dict with x, y, depth, triangles (0-based)."""
    with open(filename, "r") as f:
        f.readline()  # header line
        n_elements, n_nodes = map(int, f.readline().split()[:2])

        x = np.empty(n_nodes)
        y = np.empty(n_nodes)
        depth = np.empty(n_nodes)
        for i in range(n_nodes):
            parts = f.readline().split()
            x[i] = float(parts[1])
            y[i] = float(parts[2])
            depth[i] = float(parts[3])

        triangles = []
        for _ in range(n_elements):
            parts = f.readline().split()
            n_corners = int(parts[1])
            ids = [int(p) - 1 for p in parts[2 : 2 + n_corners]]  # 0-based
            if n_corners == 3:
                triangles.append(ids)
            elif n_corners == 4:
                # split quads into 2 triangles (matches schism_vis_GUI.py)
                triangles.append([ids[0], ids[1], ids[2]])
                triangles.append([ids[0], ids[2], ids[3]])

    return {
        "x": x,
        "y": y,
        "depth": depth,
        "triangles": np.asarray(triangles, dtype=np.int32),
        "n_nodes": n_nodes,
    }


def find_oxy_var(ds):
    """Locate the dissolved oxygen variable. Prefer 'OXY_oxy', fall back to first match."""
    if "OXY_oxy" in ds.data_vars:
        return "OXY_oxy"
    for v in ds.data_vars:
        if "oxy" in v.lower():
            return v
    raise KeyError(f"No OXY_oxy-like variable in dataset. data_vars={list(ds.data_vars)}")


def identify_dims(da):
    """Return (time_dim, node_dim, layer_dim_or_None)."""
    time_dim = next((d for d in da.dims if d.lower() == "time"), da.dims[0])
    node_dim = next(
        (d for d in da.dims if "node" in d.lower() or "nschism" in d.lower()),
        da.dims[1] if len(da.dims) >= 2 else None,
    )
    layer_dim = None
    if da.ndim > 2:
        layer_dim = next(d for d in da.dims if d not in (time_dim, node_dim))
    return time_dim, node_dim, layer_dim


def _select_layer(da, layer_dim, layer_choice):
    """Reduce a DataArray over its vertical dimension to (time, node)."""
    if layer_dim is None:
        return da
    if layer_choice == "surface":
        return da.isel({layer_dim: -1})
    if layer_choice == "bottom":
        return da.isel({layer_dim: 0})
    return da.isel({layer_dim: int(layer_choice)})


def _setup_panel(ax, triang, da2, time_dim, n_times, n_nodes, dry_mask_all,
                 cmap_name, name, units, xlim, ylim):
    """Add tripcolor + colorbar + axes labels to one axis. Returns the pc artist."""
    # Color range from a sample of frames
    sample_idx = np.linspace(0, n_times - 1, num=min(n_times, 20), dtype=int)
    sample = np.asarray(da2.isel({time_dim: sample_idx}).values)
    if dry_mask_all is not None:
        sample = np.where(dry_mask_all[sample_idx], np.nan, sample)
    vmin = float(np.nanmin(sample))
    vmax = float(np.nanmax(sample))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    f0 = np.asarray(da2.isel({time_dim: 0}).values)
    if f0.shape[0] != n_nodes:
        raise ValueError(
            f"[{name}] Frame length {f0.shape[0]} != hgrid n_nodes {n_nodes}. "
            "Check that hgrid.gr3 matches the run that produced this NC file."
        )

    if dry_mask_all is not None:
        # Static brown base shows through wherever the overlay is NaN (dry).
        brown_cmap = ListedColormap([MUDFLAT_BROWN])
        ax.tripcolor(triang, np.ones(n_nodes), cmap=brown_cmap,
                     vmin=0, vmax=1, shading="gouraud")
        f0_masked = np.where(dry_mask_all[0], np.nan, f0)
    else:
        f0_masked = f0

    cmap = copy(resolve_cmap(cmap_name))
    cmap.set_bad(color=(0, 0, 0, 0))  # transparent for dry cells

    pc = ax.tripcolor(triang, f0_masked, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="gouraud")
    cbar = ax.figure.colorbar(pc, ax=ax, shrink=0.85)
    cbar.set_label(f"{name}" + (f" [{units}]" if units else ""))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(name)
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return pc


def build_animation(hgrid, panels, dry_node=None, layer_choice="surface",
                    xlim=None, ylim=None):
    """Build a 1- or N-panel animation. `panels` is a list of dicts:
        {'da', 'time_dim', 'node_dim', 'layer_dim', 'name', 'cmap', 'units'}.
    Returns (fig, anim).
    """
    if not panels:
        raise ValueError("Need at least one panel.")

    triang = mtri.Triangulation(hgrid["x"], hgrid["y"], hgrid["triangles"])
    n_nodes = hgrid["n_nodes"]

    # Reduce each panel's data to (time, node).
    reduced = []
    for p in panels:
        da2 = _select_layer(p["da"], p["layer_dim"], layer_choice)
        reduced.append({**p, "da2": da2})

    # Use the leftmost panel's time axis as the master length; other panels are
    # truncated to match if they're longer (out of caution; same-run files match).
    master_time_dim = reduced[0]["time_dim"]
    n_times = int(reduced[0]["da2"].sizes[master_time_dim])
    for p in reduced[1:]:
        n_p = int(p["da2"].sizes[p["time_dim"]])
        if n_p < n_times:
            n_times = n_p

    # Dry-node mask
    dry_mask_all = None
    dry_pct = None
    if dry_node is not None:
        n_dn = int(dry_node.sizes[dry_node.dims[0]])
        n_use = min(n_times, n_dn)
        if n_use != n_times:
            print(f"  WARNING: dry-node time={n_dn} != panel time={n_times}; using first {n_use}")
            n_times = n_use
        dry_mask_all = (np.asarray(dry_node.values[:n_times]) > 0.5)
        dry_pct = dry_mask_all.sum(axis=1) * 100.0 / n_nodes

    # Master time axis (datetime if available)
    time_values = None
    if master_time_dim in reduced[0]["da2"].coords:
        try:
            time_values = reduced[0]["da2"][master_time_dim].values[:n_times]
        except Exception:
            time_values = None

    # Figure layout: 1 or N axes side by side. Height chosen so width*dpi and
    # height*dpi are both even (H.264 yuv420p requires even pixel dimensions).
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 9))
    if n_panels == 1:
        axes = [axes]

    pcs = []
    for ax, p in zip(axes, reduced):
        pc = _setup_panel(
            ax, triang, p["da2"], p["time_dim"], n_times, n_nodes, dry_mask_all,
            cmap_name=p["cmap"], name=p["name"], units=p["units"], xlim=xlim, ylim=ylim,
        )
        pcs.append(pc)

    # Master time label as a figure-level suptitle
    def fmt_time(frame_idx):
        if time_values is not None:
            return np.datetime_as_string(time_values[frame_idx], unit="h")
        return f"frame {frame_idx}/{n_times - 1}"

    suptitle = fig.suptitle(fmt_time(0), fontsize=12, y=0.98)

    # Dry% inset on the LEFTMOST panel, top-left corner. Reduced x-tick density.
    inset_marker = None
    inset_pct_text = None
    if dry_pct is not None:
        ax_left = axes[0]
        inset = ax_left.inset_axes([0.10, 0.74, 0.30, 0.22])
        x_axis = time_values if time_values is not None else np.arange(n_times)
        inset.plot(x_axis, dry_pct, color="#444444", linewidth=1.0)
        inset.fill_between(x_axis, 0, dry_pct, color=MUDFLAT_BROWN, alpha=0.35)
        (inset_marker,) = inset.plot([x_axis[0]], [dry_pct[0]], "o",
                                     color=MUDFLAT_BROWN, markersize=6,
                                     markeredgecolor="white", markeredgewidth=1.0)
        inset.set_ylim(0, max(1.0, dry_pct.max() * 1.1))
        inset.set_title("Dry %", fontsize=9, pad=2)
        inset.tick_params(labelsize=7)
        # Reduce x-tick density (max 3 ticks)
        inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
        if time_values is not None:
            for label in inset.get_xticklabels():
                label.set_rotation(20)
                label.set_horizontalalignment("right")
        inset.grid(alpha=0.3)
        inset.set_facecolor((1, 1, 1, 0.85))
        inset_pct_text = inset.text(0.98, 0.92, f"{dry_pct[0]:.1f}%",
                                    transform=inset.transAxes, ha="right", va="top",
                                    fontsize=9, fontweight="bold", color=MUDFLAT_BROWN)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def animate(frame):
        for pc, p in zip(pcs, reduced):
            data = np.asarray(p["da2"].isel({p["time_dim"]: frame}).values)
            if dry_mask_all is not None:
                data = np.where(dry_mask_all[frame], np.nan, data)
            pc.set_array(data)
        suptitle.set_text(fmt_time(frame))
        if inset_marker is not None:
            x_val = time_values[frame] if time_values is not None else frame
            inset_marker.set_data([x_val], [dry_pct[frame]])
            inset_pct_text.set_text(f"{dry_pct[frame]:.1f}%")
        return tuple(pcs)

    anim = animation.FuncAnimation(
        fig, animate, frames=n_times, interval=150, blit=False, repeat=True
    )
    return fig, anim


def truncate_to_ndays(da, time_dim, ndays):
    """Truncate a DataArray to the first `ndays` of its time axis (uses the time coord)."""
    if time_dim not in da.coords:
        raise ValueError(f"Cannot truncate by --ndays: no time coord on dim '{time_dim}'")
    times = da.coords[time_dim].values
    # Strictly less-than so a clean N-day window doesn't include an extra endpoint frame.
    cutoff = times[0] + np.timedelta64(int(round(ndays * 24 * 3600)), "s")
    n_keep = int((times < cutoff).sum())
    if n_keep == 0:
        n_keep = 1
    return da.isel({time_dim: slice(0, n_keep)})


def discover_out2d(nc_path):
    """Find an out2d_*.nc file matching the stack number in the OXY filename.

    e.g. OXY_oxy_1.nc -> out2d_1.nc in the same directory.
    """
    nc_path = Path(nc_path)
    stem = nc_path.stem  # e.g. 'OXY_oxy_1'
    # Trailing stack number, e.g. '_1'
    stack = stem.rsplit("_", 1)[-1]
    candidate = nc_path.parent / f"out2d_{stack}.nc"
    if candidate.exists():
        return candidate
    # Fallback: any out2d_*.nc in same dir
    matches = sorted(nc_path.parent.glob("out2d_*.nc"))
    return matches[0] if matches else None


def _open_panel_nc(path, ndays=None):
    """Open a tracer NC file and return (ds, da, time_dim, node_dim, layer_dim, name, units)."""
    ds = xr.open_dataset(path, engine="h5netcdf", decode_cf=True)
    # Pick the first time-varying data_var (skip grid/scalar metadata)
    candidates = [v for v in ds.data_vars
                  if any(d.lower() == "time" for d in ds[v].dims)]
    if not candidates:
        raise KeyError(f"No time-varying data_var in {path}. Found: {list(ds.data_vars)}")
    name = candidates[0]
    da = ds[name]
    time_dim, node_dim, layer_dim = identify_dims(da)
    if ndays is not None:
        da = truncate_to_ndays(da, time_dim, ndays)
    units = da.attrs.get("units", "")
    return ds, da, time_dim, node_dim, layer_dim, name, units


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--nc", type=Path, default=DEFAULT_NC,
                   help="Primary NC (right panel in dual mode).")
    p.add_argument("--nc-left", type=Path, default=None,
                   help="Optional second NC for the LEFT panel (e.g. salinity_1.nc).")
    p.add_argument("--cmap-left", default="cmo.haline",
                   help="Colormap for the left panel (used only with --nc-left).")
    p.add_argument("--out2d", type=Path, default=None,
                   help="Path to out2d_*.nc for dryFlagNode. Auto-discovered if omitted.")
    p.add_argument("--no-dry-mask", action="store_true",
                   help="Disable dry-cell masking and Dry% inset.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="surface",
                   help="'surface' (default), 'bottom', or integer layer index.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--cmap", default="viridis",
                   help="Colormap for the (right) primary panel.")
    p.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"))
    p.add_argument("--ndays", type=float, default=None,
                   help="Truncate animation to the first N days from the start time.")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    print(f"Reading hgrid:  {args.hgrid}")
    hgrid = read_hgrid(args.hgrid)
    print(f"  n_nodes={hgrid['n_nodes']}, n_triangles={len(hgrid['triangles'])}")

    # Right panel (primary)
    print(f"Opening NC (right): {args.nc}")
    ds_r, da_r, td_r, nd_r, ld_r, name_r, units_r = _open_panel_nc(args.nc, ndays=args.ndays)
    units_r = units_r or "mmol/m^3"  # AED default
    print(f"  variable={name_r}  dims={da_r.dims}  shape={da_r.shape}")

    panels = []
    ds_list = [ds_r]

    if args.nc_left is not None:
        print(f"Opening NC (left):  {args.nc_left}")
        ds_l, da_l, td_l, nd_l, ld_l, name_l, units_l = _open_panel_nc(args.nc_left, ndays=args.ndays)
        units_l = units_l or "PSU"  # salinity default if attr missing
        print(f"  variable={name_l}  dims={da_l.dims}  shape={da_l.shape}")
        panels.append({
            "da": da_l, "time_dim": td_l, "node_dim": nd_l, "layer_dim": ld_l,
            "name": name_l, "cmap": args.cmap_left, "units": units_l,
        })
        ds_list.append(ds_l)

    panels.append({
        "da": da_r, "time_dim": td_r, "node_dim": nd_r, "layer_dim": ld_r,
        "name": name_r, "cmap": args.cmap, "units": units_r,
    })

    # Optional out2d file for dry mask (auto-discover relative to --nc)
    ds_2d = None
    dry_node = None
    if not args.no_dry_mask:
        out2d_path = args.out2d or discover_out2d(args.nc)
        if out2d_path and Path(out2d_path).exists():
            print(f"Opening out2d:  {out2d_path}")
            ds_2d = xr.open_dataset(out2d_path, engine="h5netcdf", decode_cf=True)
            if "dryFlagNode" in ds_2d.data_vars:
                dry_node = ds_2d["dryFlagNode"]
                print(f"  dryFlagNode dims={dry_node.dims} shape={dry_node.shape}")
            else:
                print("  WARNING: dryFlagNode not found; continuing without dry mask")
        else:
            print("  No out2d_*.nc found; continuing without dry mask")

    fig, anim = build_animation(
        hgrid, panels,
        dry_node=dry_node,
        layer_choice=args.layer,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
    )

    print(f"Writing animation -> {args.output} (fps={args.fps})")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(args.output, writer=writer, dpi=150)
    print("Done.")

    if args.show:
        plt.show()
    plt.close(fig)
    for ds in ds_list:
        ds.close()
    if ds_2d is not None:
        ds_2d.close()


if __name__ == "__main__":
    main()
