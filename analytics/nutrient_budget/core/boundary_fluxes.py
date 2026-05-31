"""
Boundary-flux extraction from SCHISM's flux.out file (Term E).

flux.out format with `iflux=2`:
  - one line per (time, tracer/quantity) pair
  - 1 + n_regions columns: [time_days, q_region1, q_region2, ..., q_regionN]
  - tracers/quantities cycle in fixed order each timestep

This module:
  1. Parses flux.out into a (time, region, tracer) array
  2. Identifies which "row indices" (= tracer slots) correspond to which
     named tracers
  3. Aggregates the named-tracer fluxes per region with user-supplied signs
     to produce a single time series of net inflow into the control volume

NOTE: the exact ordering of tracer rows inside one timestep group is build-
specific. SCHISM typically writes: volume, then T, S, then each tracer in
the order their modules were declared. The default ordering is configurable
via boundary_tracer_row_map argument.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
import xarray as xr


# SCHISM iflux=2 writes blocks of (3 + 2 * n_vars) rows per timestep, where
# the first three rows are VOL net / VOL positive (outflow) / VOL negative (inflow),
# then each named variable gets two rows (positive then negative). Net flux per
# variable is computed as data[pos] + data[neg] (the negative row is already
# signed). This list must NOT include VOL — only the 19 mass/heat tracers, in
# the same order they appear in flux.out.
#
# Pioneer P18 build: T + S + 17 AED state variables (NCS + OXY + 2 NIT + 2 PHS
# + 6 OGM + 1 PHY + 4 TRC). For a different build, override via
# parse_flux_out(..., tracer_order=[...]).
DEFAULT_TRACER_ORDER = [
    "Heat",         # 0
    "Salinity",     # 1
    "GEN_1",        # 2  (only present if USE_GEN=ON with ntracer_gen>=1; SCHISM registers GEN after S, before AED)
    "NCS_ss1",      # 3
    "OXY_oxy",      # 4
    "NIT_amm",      # 5
    "NIT_nit",      # 6
    "PHS_frp",      # 7
    "PHS_frp_ads",  # 8
    "OGM_doc",      # 9
    "OGM_poc",      # 10
    "OGM_don",      # 11
    "OGM_pon",      # 12
    "OGM_dop",      # 13
    "OGM_pop",      # 14
    "PHY_mixed",    # 15
    "TRC_tr1",      # 16
    "TRC_tr2",      # 17
    "TRC_tr3",      # 18
    "TRC_age",      # 19
]


def _fix_fortran_number(token: str) -> str:
    """Some flux.out exponentials get written as -0.9756-305 instead of -0.9756E-305.
    Repair by inserting 'E' before the trailing signed exponent if needed."""
    try:
        float(token)
        return token
    except ValueError:
        return re.sub(r'([+-]?\d*\.\d+)([+-]\d+)$', r'\1E\2', token)


def parse_flux_out(path: Path | str,
                   n_regions: int = None,
                   tracer_order: list[str] = None) -> xr.DataArray:
    """Parse SCHISM iflux=2 flux.out into a (time, tracer, region) DataArray.

    Block layout per timestep (verified against schism_step.F90 and the
    schism_run_process_3.ipynb parser):
        row 0           : VOL  net (signed)        m^3/s
        row 1           : VOL  positive (outflow)  m^3/s
        row 2           : VOL  negative (inflow)   m^3/s
        rows 3, 4       : var1 positive, negative  mmol/s
        rows 5, 6       : var2 positive, negative  mmol/s
        ...
        block_len = 3 + 2 * len(tracer_order)

    The net flux per variable is computed as data[pos] + data[neg] (the
    negative row is already signed). Output's 'tracer' coord includes VOL
    plus each named variable.

    Args
    ----
    path          : path to flux.out
    n_regions     : number of fluxflag regions (e.g. 9)
    tracer_order  : names of mass/heat variables in flux.out's block order.
                    Excludes VOL (added automatically). Length must equal
                    (block_len - 3) / 2 where block_len is auto-detected.

    Returns
    -------
    xr.DataArray dims ('time_days', 'tracer', 'region'), values in flux.out's
    native units (m^3/s for VOL, mmol/s for AED tracers).
    """
    path = Path(path)
    if tracer_order is None:
        tracer_order = DEFAULT_TRACER_ORDER

    # Auto-detect n_regions from the first valid numeric line if not given.
    # The new simplified fluxflag.prop layouts produce far fewer columns
    # (e.g. max_flreg=3 gives 4 fields per row: time + 3 region values).
    if n_regions is None:
        with open(path) as f:
            for line in f:
                toks = line.split()
                if len(toks) < 2:
                    continue
                try:
                    float(_fix_fortran_number(toks[0]))
                    n_regions = len(toks) - 1
                    break
                except ValueError:
                    continue
        if n_regions is None:
            raise RuntimeError(f"Could not auto-detect n_regions from {path}")
        print(f"  Auto-detected n_regions = {n_regions} from {path.name}")

    print(f"  Parsing {path} (this may take a moment for large files)...")
    rows = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) < 1 + n_regions:
                continue
            try:
                t = float(_fix_fortran_number(toks[0]))
                vals = [float(_fix_fortran_number(x)) for x in toks[1:1+n_regions]]
                rows.append([t] + vals)
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        raise RuntimeError(f"No numeric data parsed from {path}")
    print(f"  read {arr.shape[0]} rows x {arr.shape[1]} cols from flux.out")

    # Auto-detect block length from the time column
    t_col = arr[:, 0]
    t0 = t_col[0]
    block_len = int(np.searchsorted(t_col, t0, side="right"))
    if block_len < 3:
        raise RuntimeError("Detected block_len < 3 — flux.out structure unrecognised.")
    expected_n_vars = (block_len - 3) // 2
    if 3 + 2 * expected_n_vars != block_len:
        raise RuntimeError(
            f"block_len={block_len} is not of the form 3 + 2*n_vars; "
            f"flux.out structure unrecognised."
        )
    print(f"  detected block_len = 3 + 2*{expected_n_vars} = {block_len} rows/timestep")

    if len(tracer_order) != expected_n_vars:
        raise ValueError(
            f"tracer_order has {len(tracer_order)} names but flux.out implies "
            f"{expected_n_vars} variables (block_len={block_len}). Adjust "
            f"DEFAULT_TRACER_ORDER in boundary_fluxes.py or pass tracer_order=..."
        )

    n_total_rows = arr.shape[0]
    if n_total_rows % block_len != 0:
        n_groups = n_total_rows // block_len
        arr = arr[: n_groups * block_len]
        print(f"  truncated to {n_groups} complete blocks")

    n_t = arr.shape[0] // block_len
    times = arr[::block_len, 0]                    # one time per block

    # Reshape so we can extract sub-rows easily
    block = arr[:, 1:].reshape(n_t, block_len, n_regions)
    vol_net = block[:, 0, :]
    # Per-variable net = positive row + negative row
    nets = np.empty((n_t, len(tracer_order), n_regions), dtype=arr.dtype)
    for j in range(len(tracer_order)):
        ipos = 3 + 2 * j
        ineg = ipos + 1
        nets[:, j, :] = block[:, ipos, :] + block[:, ineg, :]

    # Stack VOL + variables into the tracer axis
    all_tracers = np.concatenate(
        [vol_net[:, None, :], nets], axis=1
    )
    tracer_names = ["VOL"] + list(tracer_order)

    da = xr.DataArray(
        all_tracers,
        dims=("time_days", "tracer", "region"),
        coords={
            "time_days": times,
            "tracer": tracer_names,
            "region": np.arange(1, n_regions + 1),
        },
        attrs={
            "source_file": str(path),
            "block_len": block_len,
            "notes": "Net flux per tracer = pos_row + neg_row (signed). "
                     "VOL in m^3/s, AED tracers in mmol/s.",
        },
    )
    return da


def boundary_flux_term(flux_da: xr.DataArray,
                       boundary_regions: list[dict],
                       tracers_n_per_mol: dict,
                       start_date: str = None) -> xr.DataArray:
    """Compute Term E: net boundary flux of the budget element into the CV.

    For each named tracer with mol-N-per-mol factor:
      summed across all bounding regions (with each region's sign applied)

    Args
    ----
    flux_da            : (time_days, tracer, region) DataArray from parse_flux_out
    boundary_regions   : list of dicts like
                         [{"region_id": 3, "sign": +1, "label": "upstream"}, ...]
    tracers_n_per_mol  : mapping {tracer_name: n_atoms_per_molecule},
                         e.g. {"NIT_amm": 1.0, "PHY_mixed": 16.0/106.0}
    start_date         : ISO date string ("2021-04-01") to convert time_days to datetimes

    Returns
    -------
    xr.DataArray(time) of net N flux INTO the control volume, in mmol N / day.
    flux.out's native units are mmol/s; we multiply by 86400.
    """
    total = None
    for tracer_name, n_per_mol in tracers_n_per_mol.items():
        if tracer_name not in flux_da.coords["tracer"].values:
            print(f"  WARN: boundary tracer '{tracer_name}' not in flux.out tracer order — skipped")
            continue
        tracer_slice = flux_da.sel(tracer=tracer_name)  # (time_days, region)
        # Sum across bounding regions with their signs
        contrib = xr.zeros_like(tracer_slice.isel(region=0))
        for br in boundary_regions:
            rid = br["region_id"]
            sign = br["sign"]
            if rid not in tracer_slice.coords["region"].values:
                print(f"  WARN: region {rid} not in flux.out (max region in file is "
                      f"{tracer_slice.coords['region'].values.max()})")
                continue
            contrib = contrib + sign * tracer_slice.sel(region=rid)
        contrib = contrib * n_per_mol
        total = contrib if total is None else total + contrib

    if total is None:
        raise RuntimeError("No tracers from the budget mapped onto flux.out columns")

    # Convert from mmol/s to mmol/day
    total = total * 86400.0

    # Optionally convert time_days → datetime64
    if start_date is not None:
        t0 = pd.Timestamp(start_date)
        times_dt = t0 + pd.to_timedelta(total["time_days"].values, unit="D")
        total = total.assign_coords(time=("time_days", times_dt)).swap_dims({"time_days": "time"})

    total.attrs["units"] = "mmol N/day"
    total.attrs["long_name"] = "Net boundary N flux into control volume"
    return total
