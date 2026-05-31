"""
Animate SCHISM AED dissolved oxygen alongside AED diagnostic variables
(OXY_sat, NIT_nitrif by default) from the combined AED NetCDF file.

Extends animate_oxy.py to also handle element/face-centred AED diagnostic
variables stored in aed_data_cmb_*.nc. Node-centred (scribe-mode) panels and
element-centred (AED-direct) panels can be mixed.

Usage:
  python animlib.py
  python animlib.py --diag-vars OXY_sat NIT_nitrif PHS_frp_swi
  python animlib.py --aed-cmb /path/to/aed_data_cmb_1.nc
  python animlib.py --no-oxy   # only show the diagnostics
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
import matplotlib.dates as mdates

try:
    import cmcrameri.cm  # noqa: F401
except ImportError:
    pass

MUDFLAT_BROWN = "#8B6F47"

# Friendly per-variable labels for the bottom-right annotation. Falls back to
# the raw variable name when not listed.
FRIENDLY_LABEL = {
    "salinity":    "Salinity",
    "temperature": "Temperature",
    "velocity_magnitude": "Velocity",
    "OXY_oxy":     "Oxygen",
    "OXY_oxy_atm": r"O$_2$ atm flux",
    "OXY_oxy_dsf": r"O$_2$ sed flux",
    "OXY_sat":     r"O$_2$ saturation",
    "NIT_nitrif":  "Nitrification",
    "NIT_denit":   "Denitrification",
    "NIT_anammox": "Anammox",
    "NIT_dnra":    "DNRA",
    "NIT_amm":     "Ammonium",
    "NIT_nit":     "Nitrate",
    "NIT_amm_dsf": r"NH$_4$ sed flux",
    "NIT_nit_dsf": r"NO$_3$ sed flux",
    "NIT_din_atm": "DIN atm dep",
    "OGM_don_swi": "DON SWI flux",
    "OGM_pon_swi": "PON SWI flux",
    "OGM_pon_res": "PON resusp.",
    "OGM_don_min": "DON min.",
    "OGM_pon_hyd": "PON hydrolysis",
    "PHS_frp":     "FRP",
    "OGM_doc":     "DOC",
    "OGM_poc":     "POC",
}

# LaTeX-rendered colorbar labels: (chemical/short name, units). Fallback uses
# the raw NetCDF variable name and units attribute. Mathtext uses $...$.
COLORBAR_LABEL = {
    "salinity":    (r"Salinity",             r"PSU"),
    "temperature": (r"Temperature",          r"$^{\circ}C$"),
    "velocity_magnitude": (r"Velocity magnitude", r"$m\ s^{-1}$"),
    "OXY_oxy":     (r"$O_2$",                r"$\mu M$"),
    "OXY_oxy_atm": (r"$O_2$ atm flux",       r"$mmol\ m^{-2}\ d^{-1}$"),
    "OXY_oxy_dsf": (r"$O_2$ sediment flux",  r"$mmol\ m^{-2}\ d^{-1}$"),
    "OXY_sat":     (r"$O_2$ saturation",     r"\%"),
    "NIT_nitrif":  (r"Nitrification",        r"$\mu M\ d^{-1}$"),
    "NIT_denit":   (r"Denitrification",      r"$\mu M\ d^{-1}$"),
    "NIT_anammox": (r"Anammox",              r"$\mu M\ d^{-1}$"),
    "NIT_dnra":    (r"DNRA",                 r"$\mu M\ d^{-1}$"),
    "NIT_amm":     (r"$NH_4^+$",             r"$\mu M$"),
    "NIT_nit":     (r"$NO_3^-$",             r"$\mu M$"),
    "NIT_amm_dsf": (r"$NH_4^+$ sediment flux", r"$mmol\ N\ m^{-2}\ d^{-1}$"),
    "NIT_nit_dsf": (r"$NO_3^-$ sediment flux", r"$mmol\ N\ m^{-2}\ d^{-1}$"),
    "NIT_din_atm": (r"DIN atmospheric deposition", r"$mmol\ N\ m^{-2}\ d^{-1}$"),
    "OGM_don_swi": (r"DON SWI flux",          r"$mmol\ N\ m^{-2}\ d^{-1}$"),
    "OGM_pon_swi": (r"PON SWI flux",          r"$mmol\ N\ m^{-2}\ d^{-1}$"),
    "OGM_pon_res": (r"PON resuspension",      r"$mmol\ N\ m^{-2}\ d^{-1}$"),
    "OGM_don_min": (r"DON mineralisation",    r"$\mu M\ d^{-1}$"),
    "OGM_pon_hyd": (r"PON hydrolysis",        r"$\mu M\ d^{-1}$"),
    "PHS_frp":     (r"$PO_4^{3-}$",          r"$\mu M$"),
    "OGM_doc":     (r"DOC",                  r"$\mu M\ C$"),
    "OGM_poc":     (r"POC",                  r"$\mu M\ C$"),
    "NCS_ss1":     (r"Suspended sediment",   r"$g\ m^{-3}$"),
}

try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


RUN_DIR = Path(
    "/Volumes/Development/schism/Pioneer_17_AED_SCHISM_FY2021_FY2022_2D_dry_year_3h_write_frequency"
)
DEFAULT_HGRID = RUN_DIR / "hgrid.gr3"
DEFAULT_OXY_NC = RUN_DIR / "outputs_pawsey" / "OXY_oxy_1.nc"
DEFAULT_AED_CMB = RUN_DIR / "outputs_pawsey" / "aed_data_cmb_1.nc"
DEFAULT_OUT = Path(__file__).parent / "OXY_diags_animation.mp4"
DEFAULT_DIAG_VARS = ["OXY_sat", "NIT_nitrif"]


def resolve_cmap(name):
    try:
        return plt.get_cmap(name)
    except (ValueError, KeyError):
        try:
            return plt.get_cmap(f"cmc.{name}")
        except (ValueError, KeyError):
            print(f"  WARNING: colormap '{name}' not found; falling back to viridis")
            return plt.get_cmap("viridis")


def read_hgrid(filename):
    """Read SCHISM hgrid.gr3.

    Returns dict with x, y, depth, triangles (0-based), tri_to_elem (each tri's
    source SCHISM element index — quads contribute two triangles to the same
    element), n_nodes, n_elements.
    """
    with open(filename, "r") as f:
        f.readline()
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
        tri_to_elem = []
        for elem_idx in range(n_elements):
            parts = f.readline().split()
            n_corners = int(parts[1])
            ids = [int(p) - 1 for p in parts[2 : 2 + n_corners]]
            if n_corners == 3:
                triangles.append(ids)
                tri_to_elem.append(elem_idx)
            elif n_corners == 4:
                triangles.append([ids[0], ids[1], ids[2]])
                tri_to_elem.append(elem_idx)
                triangles.append([ids[0], ids[2], ids[3]])
                tri_to_elem.append(elem_idx)

    return {
        "x": x,
        "y": y,
        "depth": depth,
        "triangles": np.asarray(triangles, dtype=np.int32),
        "tri_to_elem": np.asarray(tri_to_elem, dtype=np.int32),
        "n_nodes": n_nodes,
        "n_elements": n_elements,
    }


def identify_dims(da, centering):
    """Return (time_dim, space_dim, layer_dim_or_None). `centering` is 'node' or 'elem'."""
    time_dim = next((d for d in da.dims if d.lower() == "time"), da.dims[0])
    if centering == "node":
        space_dim = next(
            (d for d in da.dims if "node" in d.lower() or "nschism" in d.lower()),
            da.dims[1] if len(da.dims) >= 2 else None,
        )
    else:  # elem
        space_dim = next(
            (d for d in da.dims if "face" in d.lower() or "elem" in d.lower()),
            da.dims[1] if len(da.dims) >= 2 else None,
        )
    layer_dim = None
    if da.ndim > 2:
        layer_dim = next(d for d in da.dims if d not in (time_dim, space_dim))
    return time_dim, space_dim, layer_dim


def _select_layer(da, layer_dim, layer_choice):
    if layer_dim is None:
        return da
    if layer_choice == "surface":
        return da.isel({layer_dim: -1})
    if layer_choice == "bottom":
        return da.isel({layer_dim: 0})
    return da.isel({layer_dim: int(layer_choice)})


def _frame_data(panel, frame, hgrid):
    """Get (data array of correct length for tripcolor) for a panel at frame `frame`."""
    raw = np.asarray(panel["da2"].isel({panel["time_dim"]: frame}).values)
    if panel["centering"] == "elem":
        return raw[hgrid["tri_to_elem"]]
    return raw  # node-centred


def _frame_dry_mask(panel, frame, hgrid, dry_state):
    """Return per-render-position dry mask (bool) for this panel at this frame, or None."""
    if panel["centering"] == "node":
        return dry_state.get("node_mask_all", [None])[frame] if dry_state.get("node_mask_all") is not None else None
    else:
        return dry_state.get("elem_mask_tri_all", [None])[frame] if dry_state.get("elem_mask_tri_all") is not None else None


def _setup_panel(ax, triang, panel, hgrid, n_times, dry_state, xlim, ylim):
    """Add tripcolor + colorbar + axes labels to one axis. Returns the pc artist."""
    centering = panel["centering"]
    da2 = panel["da2"]
    time_dim = panel["time_dim"]
    name = panel["name"]
    units = panel["units"]

    sample_idx = np.linspace(0, n_times - 1, num=min(n_times, 20), dtype=int)
    sample = np.asarray(da2.isel({time_dim: sample_idx}).values)
    # Strip out fill values (AED uses ~9.97e36)
    sample = np.where(np.abs(sample) > 1.0e30, np.nan, sample)
    # Apply per-frame dry mask to the sample so colour range ignores dry cells
    if centering == "node" and dry_state.get("node_mask_all") is not None:
        sample = np.where(dry_state["node_mask_all"][sample_idx], np.nan, sample)
    elif centering == "elem" and dry_state.get("elem_mask_all") is not None:
        sample = np.where(dry_state["elem_mask_all"][sample_idx], np.nan, sample)
    vmin = float(np.nanmin(sample)) if np.isfinite(sample).any() else 0.0
    vmax = float(np.nanmax(sample)) if np.isfinite(sample).any() else 1.0
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    n_nodes = hgrid["n_nodes"]
    n_tris = len(hgrid["triangles"])

    # Compute initial frame data + dry mask
    f0 = _frame_data(panel, 0, hgrid)
    f0 = np.where(np.abs(f0) > 1.0e30, np.nan, f0)
    expected = n_nodes if centering == "node" else n_tris
    if f0.shape[0] != expected:
        raise ValueError(
            f"[{name}] Frame length {f0.shape[0]} != expected {expected} for "
            f"centering='{centering}'. Check that hgrid.gr3 matches this run."
        )

    f0_dry = _frame_dry_mask(panel, 0, hgrid, dry_state)
    if f0_dry is not None:
        f0_masked = np.where(f0_dry, np.nan, f0)
    else:
        f0_masked = f0

    cmap = copy(resolve_cmap(panel["cmap"]))
    cmap.set_bad(color=(0, 0, 0, 0))

    shading = "gouraud" if centering == "node" else "flat"

    # Static brown base (a single tripcolor underneath) so dry cells show brown.
    if (centering == "node" and dry_state.get("node_mask_all") is not None) or \
       (centering == "elem" and dry_state.get("elem_mask_all") is not None):
        brown_cmap = ListedColormap([MUDFLAT_BROWN])
        if shading == "gouraud":
            ax.tripcolor(triang, np.ones(n_nodes), cmap=brown_cmap,
                         vmin=0, vmax=1, shading="gouraud")
        else:
            ax.tripcolor(triang, facecolors=np.ones(n_tris), cmap=brown_cmap,
                         vmin=0, vmax=1, shading="flat")

    if shading == "gouraud":
        pc = ax.tripcolor(triang, f0_masked, cmap=cmap,
                          vmin=vmin, vmax=vmax, shading="gouraud")
    else:
        pc = ax.tripcolor(triang, facecolors=f0_masked, cmap=cmap,
                          vmin=vmin, vmax=vmax, shading="flat")

    cbar = ax.figure.colorbar(
        pc, ax=ax,
        orientation="horizontal",
        location="bottom",
        shrink=0.9,
        pad=0.03,
        aspect=30,
    )
    cb_name, cb_units = COLORBAR_LABEL.get(name, (name, units))
    cbar.set_label(f"{cb_name} [{cb_units}]" if cb_units else cb_name)

    ax.tick_params(labelsize=9)
    ax.set_facecolor("#e8e8e8")   # light grey background for non-mesh area
    ax.set_aspect("equal", anchor="C")   # centre data in axes box
    # Half the default x-tick density (avoid crowding under tight xlim).
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Bottom-right corner annotation (friendly variable name).
    label = FRIENDLY_LABEL.get(name, name)
    ax.text(
        0.93, 0.03, label,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, fontweight="bold", color="#222222",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.85, edgecolor="#666666", linewidth=0.5),
    )
    return pc, shading


def build_animation(hgrid, panels, dry_node=None, dry_elem=None,
                    layer_choice="surface", xlim=None, ylim=None,
                    inset_horizon_days=None, nrows=1):
    if not panels:
        raise ValueError("Need at least one panel.")

    triang = mtri.Triangulation(hgrid["x"], hgrid["y"], hgrid["triangles"])
    n_nodes = hgrid["n_nodes"]
    n_elements = hgrid["n_elements"]
    tri_to_elem = hgrid["tri_to_elem"]

    # Reduce each panel's data to (time, space).
    for p in panels:
        p["da2"] = _select_layer(p["da"], p["layer_dim"], layer_choice)

    # Master time length = min over panels
    n_times = min(int(p["da2"].sizes[p["time_dim"]]) for p in panels)

    # Build dry masks (per frame, indexed by render-position)
    dry_state = {"node_mask_all": None, "elem_mask_all": None, "elem_mask_tri_all": None}
    dry_pct = None  # for inset; uses whichever mask is available, preferring node
    if dry_node is not None:
        n_dn = int(dry_node.sizes[dry_node.dims[0]])
        n_use = min(n_times, n_dn)
        if n_use != n_times:
            print(f"  WARNING: dryFlagNode time={n_dn} != panel time={n_times}; using first {n_use}")
            n_times = n_use
        dry_state["node_mask_all"] = (np.asarray(dry_node.values[:n_times]) > 0.5)
        dry_pct = dry_state["node_mask_all"].sum(axis=1) * 100.0 / n_nodes
    if dry_elem is not None:
        n_de = int(dry_elem.sizes[dry_elem.dims[0]])
        n_use = min(n_times, n_de)
        if n_use != n_times:
            print(f"  WARNING: dryFlagElement time={n_de} != panel time={n_times}; using first {n_use}")
            n_times = n_use
        dry_state["elem_mask_all"] = (np.asarray(dry_elem.values[:n_times]) > 0.5)
        # Map element-space mask to triangle-space mask for rendering
        dry_state["elem_mask_tri_all"] = dry_state["elem_mask_all"][:, tri_to_elem]
        if dry_pct is None:
            dry_pct = dry_state["elem_mask_all"].sum(axis=1) * 100.0 / n_elements

    # Master time coord (from first panel)
    master_time_dim = panels[0]["time_dim"]
    time_values = None
    if master_time_dim in panels[0]["da2"].coords:
        try:
            time_values = panels[0]["da2"][master_time_dim].values[:n_times]
        except Exception:
            time_values = None

    n_panels = len(panels)
    ncols = max(1, int(np.ceil(n_panels / max(1, nrows))))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.6 * ncols, 8.8 * nrows),
        constrained_layout=True,
    )
    # h_pad / w_pad add breathing room (in inches) around the outer figure edges
    # so the bottom-mounted colorbars don't sit flush against the figure base.
    fig.set_constrained_layout_pads(h_pad=0.35, w_pad=0.10,
                                    hspace=0.06, wspace=0.04)
    # Flatten axes to 1D list (subplots returns scalar / 1D / 2D depending on layout).
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten().tolist()
    # Hide any unused axes (panels < nrows*ncols).
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    axes = axes[:n_panels]

    pcs, shadings = [], []
    for ax, p in zip(axes, panels):
        pc, shading = _setup_panel(ax, triang, p, hgrid, n_times, dry_state, xlim, ylim)
        pcs.append(pc)
        shadings.append(shading)

    def fmt_time(frame_idx):
        if time_values is not None:
            try:
                return np.datetime_as_string(time_values[frame_idx], unit="h")
            except Exception:
                return f"t={float(time_values[frame_idx]):.0f}s"
        return f"frame {frame_idx}/{n_times - 1}"

    suptitle = fig.suptitle(fmt_time(0), fontsize=12, y=0.98)

    # Per-panel insets — opt-in via p["show_inset"] = True. Currently the only
    # supported inset content is "Dry %" (time series with a running marker).
    # If inset_horizon_days is set, the x-axis glides with the current frame
    # showing only the last horizon-window of history.
    inset_axes_list = [None] * len(panels)
    inset_markers   = [None] * len(panels)
    inset_pct_texts = [None] * len(panels)
    if dry_pct is not None:
        for i, (ax, p) in enumerate(zip(axes, panels)):
            if not p.get("show_inset", False):
                continue
            inset = ax.inset_axes([0.10, 0.70, 0.30, 0.11])
            x_axis = time_values if time_values is not None else np.arange(n_times)
            inset.plot(x_axis, dry_pct, color="#444444", linewidth=1.0)
            inset.fill_between(x_axis, 0, dry_pct, color=MUDFLAT_BROWN, alpha=0.35)
            (m,) = inset.plot([x_axis[0]], [dry_pct[0]], "o",
                              color=MUDFLAT_BROWN, markersize=6,
                              markeredgecolor="white", markeredgewidth=1.0)
            inset.set_ylim(0, max(1.0, dry_pct.max() * 1.1))
            inset.set_title("Dry %", fontsize=9, pad=2)
            inset.tick_params(labelsize=7)
            if (time_values is not None
                    and np.issubdtype(np.asarray(time_values).dtype, np.datetime64)):
                inset.xaxis.set_major_locator(mdates.DayLocator())
                inset.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                for label in inset.get_xticklabels():
                    label.set_rotation(20)
                    label.set_horizontalalignment("right")
            else:
                inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
            inset.grid(True, linestyle="--", alpha=0.45)
            inset.set_facecolor((1, 1, 1, 0.85))
            t = inset.text(0.02, 0.08, f"{dry_pct[0]:.1f}%",
                           transform=inset.transAxes, ha="left", va="bottom",
                           fontsize=9, fontweight="bold", color=MUDFLAT_BROWN)
            inset_axes_list[i] = inset
            inset_markers[i] = m
            inset_pct_texts[i] = t

    # Pre-compute gliding-window helpers if requested.
    glide = inset_horizon_days is not None and inset_horizon_days > 0
    if glide:
        if (time_values is not None
                and np.issubdtype(np.asarray(time_values).dtype, np.datetime64)):
            horizon = np.timedelta64(int(round(inset_horizon_days * 86400)), "s")
            buffer  = np.timedelta64(int(round(inset_horizon_days * 86400 * 0.05)), "s")
        else:
            # Index-based fallback: assume 1 frame per "day" equivalent.
            horizon = inset_horizon_days
            buffer  = max(1, inset_horizon_days * 0.05)

    # constrained_layout=True (set on subplots) handles spacing; no tight_layout needed.

    def animate(frame):
        for pc, p in zip(pcs, panels):
            data = _frame_data(p, frame, hgrid)
            data = np.where(np.abs(data) > 1.0e30, np.nan, data)
            dry = _frame_dry_mask(p, frame, hgrid, dry_state)
            if dry is not None:
                data = np.where(dry, np.nan, data)
            pc.set_array(data)
        suptitle.set_text(fmt_time(frame))
        if any(m is not None for m in inset_markers):
            x_val = time_values[frame] if time_values is not None else frame
            for m, t in zip(inset_markers, inset_pct_texts):
                if m is None:
                    continue
                m.set_data([x_val], [dry_pct[frame]])
                t.set_text(f"{dry_pct[frame]:.1f}%")
            if glide:
                # Gliding window: x-max = now + small buffer; x-min = now - horizon.
                x_max = x_val + buffer
                x_min = x_val - horizon
                for inset in inset_axes_list:
                    if inset is not None:
                        inset.set_xlim(x_min, x_max)
        return tuple(pcs)

    anim = animation.FuncAnimation(
        fig, animate, frames=n_times, interval=150, blit=False, repeat=True
    )
    return fig, anim


def truncate_to_ndays(da, time_dim, ndays):
    if time_dim not in da.coords:
        raise ValueError(f"Cannot truncate by --ndays: no time coord on dim '{time_dim}'")
    times = da.coords[time_dim].values
    cutoff = times[0] + np.timedelta64(int(round(ndays * 24 * 3600)), "s")
    n_keep = int((times < cutoff).sum())
    if n_keep == 0:
        n_keep = 1
    return da.isel({time_dim: slice(0, n_keep)})


def discover_out2d(nc_path):
    nc_path = Path(nc_path)
    stem = nc_path.stem
    stack = stem.rsplit("_", 1)[-1]
    candidate = nc_path.parent / f"out2d_{stack}.nc"
    if candidate.exists():
        return candidate
    matches = sorted(nc_path.parent.glob("out2d_*.nc"))
    return matches[0] if matches else None


def _expand_stacks(path, multi=True, limit=None):
    """Given /dir/foo_N.nc, return sorted sibling stack files /dir/foo_1.nc, foo_2.nc, ...

    If multi is False or `path` lacks a trailing _<int>, returns [path].
    If limit is a positive int, truncates to that many stacks (in numeric order).
    """
    p = Path(path)
    if not multi:
        return [p]
    stem = p.stem
    if "_" not in stem:
        return [p]
    prefix, last = stem.rsplit("_", 1)
    if not last.isdigit():
        return [p]
    matches = [(m, int(m.stem.rsplit("_", 1)[1]))
               for m in p.parent.glob(f"{prefix}_*.nc")
               if "_" in m.stem and m.stem.rsplit("_", 1)[1].isdigit()]
    matches.sort(key=lambda x: x[1])
    out = [m for m, _ in matches] if matches else [p]
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def _concat_along_time(das, time_dim):
    """Concatenate a list of DataArrays along their time dimension."""
    if len(das) == 1:
        return das[0]
    return xr.concat(das, dim=time_dim)


def _open_scribe_panel(path, ndays=None, multi=True, limit=None):
    """Node-centred scribe-mode NC, optionally concatenated across all sibling stacks."""
    paths = _expand_stacks(path, multi=multi, limit=limit)
    print(f"  stacks: {[p.name for p in paths]}")
    ds_list = [xr.open_dataset(p, engine="h5netcdf", decode_cf=True) for p in paths]
    candidates = [v for v in ds_list[0].data_vars
                  if any(d.lower() == "time" for d in ds_list[0][v].dims)]
    if not candidates:
        raise KeyError(f"No time-varying data_var in {paths[0]}. Found: {list(ds_list[0].data_vars)}")
    name = candidates[0]
    time_dim_initial = next(d for d in ds_list[0][name].dims if d.lower() == "time")
    da = _concat_along_time([ds[name] for ds in ds_list], time_dim_initial)
    time_dim, node_dim, layer_dim = identify_dims(da, "node")
    if ndays is not None:
        da = truncate_to_ndays(da, time_dim, ndays)
    units = da.attrs.get("units", "")
    return ds_list, da, time_dim, node_dim, layer_dim, name, units


def _open_aed_combined(path, multi=True, limit=None):
    """Open all aed_data_cmb_*.nc sibling stacks (or just `path` if multi=False)."""
    paths = _expand_stacks(path, multi=multi, limit=limit)
    print(f"  stacks: {[p.name for p in paths]}")
    return [xr.open_dataset(p, engine="h5netcdf", decode_cf=True) for p in paths]


def _open_aed_diag_panel(ds_list, varname, ndays=None):
    """Element-centred AED diagnostic var, concatenated across the given combined-file list."""
    missing = [i for i, ds in enumerate(ds_list) if varname not in ds.data_vars]
    if missing:
        raise KeyError(
            f"'{varname}' missing from {len(missing)} of {len(ds_list)} combined files. "
            f"Available in ds[0]: {list(ds_list[0].data_vars)[:20]} ..."
        )
    time_dim_initial = next(d for d in ds_list[0][varname].dims if d.lower() == "time")
    da = _concat_along_time([ds[varname] for ds in ds_list], time_dim_initial)
    time_dim, elem_dim, layer_dim = identify_dims(da, "elem")
    if ndays is not None:
        da = truncate_to_ndays(da, time_dim, ndays)
    units = da.attrs.get("units", "")
    return da, time_dim, elem_dim, layer_dim, units


def _open_out2d_multi(path, multi=True, limit=None):
    """Open all out2d_*.nc stacks; return (ds_list, dryFlagNode_concat, dryFlagElement_concat)."""
    paths = _expand_stacks(path, multi=multi, limit=limit)
    print(f"  stacks: {[p.name for p in paths]}")
    ds_list = [xr.open_dataset(p, engine="h5netcdf", decode_cf=True) for p in paths]

    def _try_concat(varname):
        if not all(varname in ds.data_vars for ds in ds_list):
            return None
        time_dim_initial = next(d for d in ds_list[0][varname].dims if d.lower() == "time")
        return _concat_along_time([ds[varname] for ds in ds_list], time_dim_initial)

    return ds_list, _try_concat("dryFlagNode"), _try_concat("dryFlagElement")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hgrid", type=Path, default=DEFAULT_HGRID)
    p.add_argument("--oxy-nc", type=Path, default=DEFAULT_OXY_NC,
                   help="Scribe-mode OXY_oxy_*.nc (node-centred). Pass --no-oxy to skip.")
    p.add_argument("--no-oxy", action="store_true",
                   help="Skip the OXY_oxy scribe-mode panel.")
    p.add_argument("--aed-cmb", type=Path, default=DEFAULT_AED_CMB,
                   help="Combined AED file (aed_data_cmb_*.nc) for diagnostic vars.")
    p.add_argument("--diag-vars", nargs="*", default=DEFAULT_DIAG_VARS,
                   help=f"AED diagnostic var names to plot (default: {DEFAULT_DIAG_VARS}).")
    p.add_argument("--out2d", type=Path, default=None,
                   help="Path to out2d_*.nc. Auto-discovered if omitted.")
    p.add_argument("--no-dry-mask", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--layer", default="surface",
                   help="'surface' (default), 'bottom', or integer layer index.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300, help="Output render DPI (default 300).")
    p.add_argument("--inset-horizon-days", type=float, default=None,
                   help="Gliding inset x-axis: show only the last N days of history "
                        "(x-max = current time + 5%% buffer). Omit for full history.")
    p.add_argument("--cmap-oxy", default="viridis", help="Colormap for OXY_oxy panel.")
    p.add_argument("--cmap-diag", default="cmo.matter",
                   help="Colormap for diagnostic panels.")
    p.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   default=[149.15, 149.27])
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   default=[-21.325, -21.025])
    p.add_argument("--single-stack", action="store_true",
                   help="Only animate the named stack files; don't auto-discover siblings.")
    p.add_argument("--n-stacks", type=int, default=None,
                   help="Cap the number of auto-discovered stacks (default: all).")
    p.add_argument("--ndays", type=float, default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    print(f"Reading hgrid:  {args.hgrid}")
    hgrid = read_hgrid(args.hgrid)
    print(f"  n_nodes={hgrid['n_nodes']}, n_elements={hgrid['n_elements']}, "
          f"n_triangles={len(hgrid['triangles'])}")

    multi = not args.single_stack

    panels = []
    ds_list = []

    # Optional OXY_oxy scribe-mode panel (node-centred)
    if not args.no_oxy:
        print(f"Opening scribe OXY:  {args.oxy_nc}")
        ds_o_list, da_o, td_o, sd_o, ld_o, name_o, units_o = _open_scribe_panel(
            args.oxy_nc, ndays=args.ndays, multi=multi, limit=args.n_stacks,
        )
        units_o = units_o or "mmol/m^3"
        print(f"  variable={name_o}  dims={da_o.dims}  shape={da_o.shape}")
        panels.append({
            "da": da_o, "time_dim": td_o, "space_dim": sd_o, "layer_dim": ld_o,
            "name": name_o, "cmap": args.cmap_oxy, "units": units_o,
            "centering": "node",
        })
        ds_list.extend(ds_o_list)

    # Diagnostic panels from combined AED file (element-centred)
    if args.diag_vars:
        print(f"Opening AED combined:  {args.aed_cmb}")
        ds_cmb_list = _open_aed_combined(args.aed_cmb, multi=multi, limit=args.n_stacks)
        ds_list.extend(ds_cmb_list)
        for vname in args.diag_vars:
            da, td, sd, ld, units = _open_aed_diag_panel(ds_cmb_list, vname, ndays=args.ndays)
            print(f"  variable={vname}  dims={da.dims}  shape={da.shape}  units='{units}'")
            panels.append({
                "da": da, "time_dim": td, "space_dim": sd, "layer_dim": ld,
                "name": vname, "cmap": args.cmap_diag, "units": units,
                "centering": "elem",
            })

    if not panels:
        raise SystemExit("No panels to render — pass --no-oxy without --diag-vars leaves nothing.")

    # Preserve existing behaviour: put the Dry % inset on the leftmost panel only.
    panels[0]["show_inset"] = True

    # Dry-mask source (out2d): use both dryFlagNode and dryFlagElement if available.
    dry_node = None
    dry_elem = None
    ds_2d_list = []
    if not args.no_dry_mask:
        ref_for_out2d = args.oxy_nc if not args.no_oxy else args.aed_cmb
        out2d_path = args.out2d or discover_out2d(ref_for_out2d)
        if out2d_path and Path(out2d_path).exists():
            print(f"Opening out2d:  {out2d_path}")
            ds_2d_list, dry_node, dry_elem = _open_out2d_multi(
                out2d_path, multi=multi, limit=args.n_stacks
            )
            if dry_node is not None:
                print(f"  dryFlagNode dims={dry_node.dims} shape={dry_node.shape}")
            if dry_elem is not None:
                print(f"  dryFlagElement dims={dry_elem.dims} shape={dry_elem.shape}")
            if dry_node is None and dry_elem is None:
                print("  WARNING: neither dryFlagNode nor dryFlagElement found; "
                      "continuing without dry mask")
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

    print(f"Writing animation -> {args.output} (fps={args.fps})")
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
