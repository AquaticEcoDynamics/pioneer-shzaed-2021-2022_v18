"""
Mesh geometry helpers for the SCHISM-AED budget package.

Provides:
  - read_hgrid_with_areas(): mesh + element area + element centroid
  - load_layer_thickness(): per-element, per-layer thickness from aed_data_cmb
  - element_volume(): area × layer_ht → (time, elem, layer) volume array
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import math

import numpy as np
import xarray as xr


# Approximate metres per degree at the equator
M_PER_DEG_LAT = 111_320.0


def read_hgrid_with_areas(hgrid_path: Path | str) -> dict:
    """Read SCHISM hgrid.gr3 and compute per-element areas in m².

    Works for both projected (Cartesian metres) and geographic (lon/lat) meshes
    — for lon/lat, uses a flat-Earth approximation centred at the mesh
    centroid (accurate to ~0.5 % over a 0.5-degree domain).

    Returns dict with keys:
        x, y       : node coords (length n_nodes)
        depth      : node depths (length n_nodes)
        elements   : list of [n_corners, node0_idx, node1_idx, ...] per element
        n_nodes, n_elements
        areas      : per-element area in m² (length n_elements)
        centroids  : per-element (x, y) centroid (n_elements x 2), in mesh units
        is_latlon  : True if mesh appears to be in lon/lat
    """
    hgrid_path = Path(hgrid_path)
    with open(hgrid_path) as f:
        f.readline()
        n_elements, n_nodes = map(int, f.readline().split()[:2])
        x = np.empty(n_nodes); y = np.empty(n_nodes); depth = np.empty(n_nodes)
        for i in range(n_nodes):
            parts = f.readline().split()
            x[i] = float(parts[1]); y[i] = float(parts[2]); depth[i] = float(parts[3])
        elements = []
        for _ in range(n_elements):
            parts = f.readline().split()
            n_corners = int(parts[1])
            ids = [int(p) - 1 for p in parts[2:2 + n_corners]]
            elements.append([n_corners] + ids)

    # Detect coordinate system
    is_latlon = abs(x).max() <= 360.0 and abs(y).max() <= 90.0

    # Convert to local metric x/y for area computation
    if is_latlon:
        y_mid = float(np.mean(y))
        m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(y_mid))
        xm = (x - x.mean()) * m_per_deg_lon
        ym = (y - y.mean()) * M_PER_DEG_LAT
    else:
        xm = x.copy(); ym = y.copy()

    # Compute per-element area + centroid
    areas = np.zeros(n_elements)
    centroids = np.zeros((n_elements, 2))
    for ei, e in enumerate(elements):
        nc, *ids = e
        # Triangle area = 0.5 * |x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)|
        xs = xm[ids]; ys = ym[ids]
        if nc == 3:
            a = 0.5 * abs(xs[0]*(ys[1]-ys[2]) + xs[1]*(ys[2]-ys[0]) + xs[2]*(ys[0]-ys[1]))
            cx = (x[ids[0]] + x[ids[1]] + x[ids[2]]) / 3.0
            cy = (y[ids[0]] + y[ids[1]] + y[ids[2]]) / 3.0
        elif nc == 4:
            # Split quad into two triangles (0,1,2) + (0,2,3)
            a1 = 0.5 * abs(xs[0]*(ys[1]-ys[2]) + xs[1]*(ys[2]-ys[0]) + xs[2]*(ys[0]-ys[1]))
            a2 = 0.5 * abs(xs[0]*(ys[2]-ys[3]) + xs[2]*(ys[3]-ys[0]) + xs[3]*(ys[0]-ys[2]))
            a = a1 + a2
            cx = float(np.mean(x[ids]))
            cy = float(np.mean(y[ids]))
        else:
            raise ValueError(f"Unsupported n_corners={nc} for element {ei}")
        areas[ei] = a
        centroids[ei] = (cx, cy)

    return {
        "x": x, "y": y, "depth": depth, "elements": elements,
        "n_nodes": n_nodes, "n_elements": n_elements,
        "areas": areas, "centroids": centroids, "is_latlon": is_latlon,
    }


def discover_cmb_stacks(outputs_dir: Path | str, prefix: str = "aed_data_cmb") -> list[Path]:
    """Return aed_data_cmb_<N>.nc files in numeric stack order."""
    outputs_dir = Path(outputs_dir)
    matches = list(outputs_dir.glob(f"{prefix}_*.nc"))
    # extract trailing integer
    def stack_n(p):
        try:
            return int(p.stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return -1
    matches.sort(key=stack_n)
    return [m for m in matches if stack_n(m) > 0]


def discover_scribed_stacks(outputs_dir: Path | str, varname: str) -> list[Path]:
    """Return <varname>_<N>.nc files in numeric stack order."""
    outputs_dir = Path(outputs_dir)
    matches = list(outputs_dir.glob(f"{varname}_*.nc"))
    def stack_n(p):
        try:
            return int(p.stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return -1
    matches.sort(key=stack_n)
    return [m for m in matches if stack_n(m) > 0]


def open_cmb_concat(outputs_dir: Path | str,
                    n_stacks: Optional[int] = None,
                    prefix: str = "aed_data_cmb") -> xr.Dataset:
    """Open aed_data_cmb_*.nc stacks and concatenate along time, lazily."""
    paths = discover_cmb_stacks(outputs_dir, prefix=prefix)
    if n_stacks is not None:
        paths = paths[:n_stacks]
    if not paths:
        raise FileNotFoundError(f"No {prefix}_*.nc in {outputs_dir}")
    if len(paths) == 1:
        return xr.open_dataset(paths[0], engine="h5netcdf", decode_cf=True)
    # Lazy multi-file open: dask chunks one stack at a time along the time axis
    return xr.open_mfdataset(
        [str(p) for p in paths],
        engine="h5netcdf",
        decode_cf=True,
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={"time": "auto"},
    )


def open_scribed_concat(outputs_dir: Path | str, varname: str,
                        n_stacks: Optional[int] = None) -> xr.DataArray:
    """Open <varname>_<S>.nc stacks and concatenate along time, lazily.

    Returns a single DataArray for `varname`.
    """
    paths = discover_scribed_stacks(outputs_dir, varname)
    if n_stacks is not None:
        paths = paths[:n_stacks]
    if not paths:
        raise FileNotFoundError(f"No {varname}_*.nc in {outputs_dir}")
    if len(paths) == 1:
        return xr.open_dataset(paths[0], engine="h5netcdf", decode_cf=True)[varname]
    ds = xr.open_mfdataset(
        [str(p) for p in paths],
        engine="h5netcdf",
        decode_cf=True,
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={"time": "auto"},
    )
    return ds[varname]


def node_to_element_average(node_data: xr.DataArray, elements: list,
                            elem_dim_name: str = "nSCHISM_hgrid_face") -> xr.DataArray:
    """Convert a node-centred 3D array to element-centred by averaging vertices.

    For a triangle: mean of 3 vertex values. For a quad: mean of 4 vertex
    values. Operates per layer and per time slice.

    Returns DataArray with the node dimension replaced by an element dimension.
    """
    # Find the node dimension
    node_dim = next(d for d in node_data.dims
                    if "node" in d.lower() or "nschism_hgrid_node" in d.lower())
    arr = node_data.values  # numpy
    # axis index of the node dim
    node_axis = list(node_data.dims).index(node_dim)
    n_elem = len(elements)
    # Build new shape with node_dim replaced by n_elem
    new_shape = list(arr.shape)
    new_shape[node_axis] = n_elem
    out = np.empty(new_shape, dtype=np.float32)
    # Slice prep — we need to broadcast over the node axis
    # Simplest robust approach: loop over elements
    for ei, e in enumerate(elements):
        nc, *ids = e
        # Take values at these node ids along node_axis and mean them
        # Build slicing: take node_axis = ids, mean -> single value per other-axis combo
        vals = np.take(arr, ids, axis=node_axis).mean(axis=node_axis)
        # Insert into out at element index ei along node_axis
        idx = [slice(None)] * out.ndim
        idx[node_axis] = ei
        out[tuple(idx)] = vals
    # Build new DataArray
    new_dims = list(node_data.dims)
    new_dims[node_axis] = elem_dim_name
    coords = {d: node_data.coords[d] for d in node_data.coords
              if d in new_dims and d != node_dim}
    return xr.DataArray(out, dims=new_dims, coords=coords, name=node_data.name,
                        attrs=node_data.attrs)
