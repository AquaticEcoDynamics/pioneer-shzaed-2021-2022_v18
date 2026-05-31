"""
Integration functions for the four volume / area-based budget terms.

Term A — Storage: volume-integrate concentration over the control volume.
Term B — Internal rates: volume-integrate process rates.
Term C — Atmospheric flux: area-integrate over the surface of the CV.
Term D — SWI flux: area-integrate over the benthic surface of the CV.

Unit conventions on output:
  Storage:  mmol N (instantaneous, time-resolved)
  Rates/fluxes: mmol N / day (instantaneous, time-resolved)
"""

from __future__ import annotations
from typing import Iterable

import numpy as np
import xarray as xr

from . import geometry


SECONDS_PER_DAY = 86400.0


def _zero_series_like(ds_cmb: xr.Dataset) -> xr.DataArray:
    """A zero (time,) series matching the cmb time axis — used when a budget
    term's diagnostics were not saved to the output (so the term is unavailable
    and treated as zero, with the residual absorbing it)."""
    nt = ds_cmb.sizes.get("time", 1)
    coords = {"time": ds_cmb["time"].values} if "time" in ds_cmb.coords else None
    return xr.DataArray(np.zeros(nt, dtype=float), dims=["time"], coords=coords,
                        name="zero")


def _time_dim_of(da: xr.DataArray) -> str:
    return next(d for d in da.dims if d.lower() == "time")


def _elem_dim_of(da: xr.DataArray) -> str:
    return next(d for d in da.dims
                if "face" in d.lower() or "elem" in d.lower()
                or "nschism_hgrid_face" in d.lower())


def _layer_dim_of(da: xr.DataArray) -> str | None:
    for d in da.dims:
        if "vgrid" in d.lower() or "layer" in d.lower():
            return d
    return None


def _mask_fill(da: xr.DataArray, fill_threshold: float = 1.0e30) -> xr.DataArray:
    """Replace |x| > threshold (NetCDF fill values) with NaN."""
    return da.where(np.abs(da) <= fill_threshold)


# ----------------------------------------------------------------------
# Term A — storage
# ----------------------------------------------------------------------

def storage_term(state_vars: dict,
                 outputs_dir,
                 hgrid: dict,
                 cv_elements: np.ndarray,
                 layer_ht: xr.DataArray,
                 n_stacks: int | None = None) -> dict:
    """Compute Term A: total mass of the budget element in the control volume.

    For each state variable in `state_vars`:
      - Open the scribed NetCDF stacks (node-centred 3D field).
      - Convert to element-centred by averaging vertex values.
      - Volume-integrate over the CV: Σ_elem Σ_layer  c × layer_ht × area.
      - Multiply by `n_per_mol` to convert into the budget element (e.g. mmol N).

    Returns:
        dict with keys:
            'per_var' : {varname: xr.DataArray(time)} in mmol N
            'total'   : xr.DataArray(time) summed across all pools (mmol N)
    """
    elem_areas = hgrid["areas"]                       # (n_elem,)
    elements_list = hgrid["elements"]
    cv_mask = np.zeros(hgrid["n_elements"], dtype=bool)
    cv_mask[cv_elements] = True

    per_var = {}
    total_da = None
    for vname, spec in state_vars.items():
        if spec["source"] == "scribed":
            da = geometry.open_scribed_concat(outputs_dir, vname, n_stacks=n_stacks)
            da_elem = geometry.node_to_element_average(da, elements_list)
        elif spec["source"] == "cmb":
            ds_cmb = geometry.open_cmb_concat(outputs_dir, n_stacks=n_stacks)
            da_elem = ds_cmb[vname]
        else:
            raise ValueError(f"Unknown source {spec['source']} for {vname}")

        da_elem = _mask_fill(da_elem)
        time_dim = _time_dim_of(da_elem)
        elem_dim = _elem_dim_of(da_elem)
        layer_dim = _layer_dim_of(da_elem)

        # Build the cell-volume array on the same grid as da_elem.
        # layer_ht is (time, elem, layer); da_elem is the same.
        # Element area broadcasts as (elem,).
        # Compute mass per cell = c * layer_ht * area, then sum over CV elem + layers.
        # Note: layer_ht's time axis may not match perfectly if scribed stacks
        # and cmb stacks have different output cadence. We align if possible.
        layer_ht_aligned = _align_time(layer_ht, da_elem)

        # Restrict to CV elements
        da_cv = da_elem.isel({elem_dim: cv_mask})
        lh_cv = layer_ht_aligned.isel({_elem_dim_of(layer_ht_aligned): cv_mask})

        area_cv = xr.DataArray(elem_areas[cv_mask], dims=[elem_dim],
                               name="elem_area")
        # volume per cell: lh_cv * area_cv
        vol_cv = lh_cv * area_cv
        mass = (da_cv * vol_cv).sum(dim=[d for d in (elem_dim, layer_dim) if d])
        mass = mass * spec["n_per_mol"]
        mass.attrs["units"] = "mmol N"
        mass.attrs["long_name"] = f"Storage of N in {vname}"
        per_var[vname] = mass

        if total_da is None:
            total_da = mass.copy()
        else:
            total_da = total_da + mass

    total_da.attrs["units"] = "mmol N"
    total_da.attrs["long_name"] = "Total N storage in control volume"
    return {"per_var": per_var, "total": total_da}


def storage_derivative(storage_total: xr.DataArray) -> xr.DataArray:
    """Compute dM/dt for the total-N storage time series, in mmol N/day.

    Uses centred finite differences on the time axis (datetime64 expected).
    """
    time_dim = _time_dim_of(storage_total)
    t = storage_total[time_dim].values
    dt_s = np.diff(t).astype("timedelta64[s]").astype(float)
    # Centred difference: for interior points dM/dt = (M_{i+1} - M_{i-1}) / (dt_{i-1}+dt_i)
    # For ends, use forward / backward diff.
    M = storage_total.values
    dMdt = np.zeros_like(M, dtype=float)
    dMdt[0] = (M[1] - M[0]) / dt_s[0]
    dMdt[-1] = (M[-1] - M[-2]) / dt_s[-1]
    for i in range(1, len(M) - 1):
        dMdt[i] = (M[i+1] - M[i-1]) / (dt_s[i-1] + dt_s[i])
    # Convert from per-second to per-day
    dMdt_per_day = dMdt * SECONDS_PER_DAY
    out = xr.DataArray(dMdt_per_day, dims=[time_dim],
                       coords={time_dim: storage_total[time_dim]},
                       name="dMdt", attrs={"units": "mmol N/day",
                                            "long_name": "Rate of N storage change"})
    return out


# ----------------------------------------------------------------------
# Term B — internal rates (3D, volume-integrated)
# ----------------------------------------------------------------------

def internal_rates_term(rates_spec: dict,
                        ds_cmb: xr.Dataset,
                        hgrid: dict,
                        cv_elements: np.ndarray,
                        layer_ht: xr.DataArray) -> dict:
    """Volume-integrate each 3D rate over the control volume.

    Returns dict with:
        'per_rate' : {rate_name: xr.DataArray(time)} in mmol N/day, with sign
                     applied (sign_in_pool * area * thickness * rate)
        'net_loss' : sum of all rates with sign_in_pool != 0 (the actual budget
                     contribution — internal recyclers cancel out)
    """
    elem_areas = hgrid["areas"]
    cv_mask = np.zeros(hgrid["n_elements"], dtype=bool)
    cv_mask[cv_elements] = True
    elem_dim_layer = _elem_dim_of(layer_ht)
    area_cv = xr.DataArray(elem_areas[cv_mask], dims=[elem_dim_layer])

    per_rate = {}
    net_loss = None
    for rate_name, spec in rates_spec.items():
        if rate_name not in ds_cmb.data_vars:
            print(f"  WARN: rate '{rate_name}' not in cmb dataset — skipping")
            continue
        da = _mask_fill(ds_cmb[rate_name])
        elem_dim = _elem_dim_of(da)
        layer_dim = _layer_dim_of(da)

        # Restrict to CV
        da_cv = da.isel({elem_dim: cv_mask})
        lh_cv = layer_ht.isel({elem_dim_layer: cv_mask})
        vol_cv = lh_cv * area_cv

        # Rate is in mmol/m^3/day; multiply by cell volume to get mmol/day.
        contrib = (da_cv * vol_cv).sum(dim=[d for d in (elem_dim, layer_dim) if d])
        # Apply sign for budget contribution
        signed = contrib * spec["sign_in_pool"]
        signed.attrs["units"] = "mmol N/day"
        signed.attrs["long_name"] = spec.get("label", rate_name)
        signed.attrs["sign_in_pool"] = spec["sign_in_pool"]
        per_rate[rate_name] = signed

        if spec["sign_in_pool"] != 0:
            net_loss = signed if net_loss is None else net_loss + signed

    if net_loss is None:
        # No rates contributed to net loss. If some rates existed but all had
        # sign 0, mirror their time axis; otherwise (no rate diagnostics in the
        # cmb at all) build a zero series over the cmb time axis.
        if per_rate:
            net_loss = xr.zeros_like(next(iter(per_rate.values())))
        else:
            print("  WARN: no rate diagnostics in cmb — Term B set to zero.")
            net_loss = _zero_series_like(ds_cmb)
    net_loss.attrs["units"] = "mmol N/day"
    net_loss.attrs["long_name"] = "Net internal N loss (denitrification + anammox)"
    return {"per_rate": per_rate, "net_loss": net_loss}


# ----------------------------------------------------------------------
# Terms C + D — surface area-integrated fluxes (2D sheet vars)
# ----------------------------------------------------------------------

def surface_flux_term(spec: dict,
                      ds_cmb: xr.Dataset,
                      hgrid: dict,
                      cv_elements: np.ndarray,
                      label: str) -> dict:
    """Area-integrate a set of 2D sheet variables over the CV surface.

    `spec` is a dict like NITROGEN["atm"] or NITROGEN["swi"]: {varname: {sign_into_water}}.

    Returns dict with:
        'per_var' : {varname: xr.DataArray(time)} in mmol N/day (signed)
        'total'   : sum across all vars (xr.DataArray(time))
    """
    elem_areas = hgrid["areas"]
    cv_mask = np.zeros(hgrid["n_elements"], dtype=bool)
    cv_mask[cv_elements] = True
    area_cv_total = elem_areas[cv_mask].sum()  # m^2 — for reporting only

    per_var = {}
    total = None
    for vname, vspec in spec.items():
        if vname not in ds_cmb.data_vars:
            print(f"  WARN: {label} var '{vname}' not in cmb dataset — skipping")
            continue
        da = _mask_fill(ds_cmb[vname])
        elem_dim = _elem_dim_of(da)
        # 2D sheet: shape (time, nface)
        da_cv = da.isel({elem_dim: cv_mask})
        area_cv = xr.DataArray(elem_areas[cv_mask], dims=[elem_dim])
        contrib = (da_cv * area_cv).sum(dim=elem_dim)
        signed = contrib * vspec["sign_into_water"]
        signed.attrs["units"] = "mmol N/day"
        signed.attrs["long_name"] = vspec.get("label", vname)
        per_var[vname] = signed
        total = signed if total is None else total + signed

    if total is None:
        print(f"  WARN: no {label} diagnostics in cmb — surface term set to zero.")
        total = _zero_series_like(ds_cmb)
    total.attrs["units"] = "mmol N/day"
    total.attrs["long_name"] = f"Total {label} flux into CV"
    total.attrs["cv_surface_area_m2"] = float(area_cv_total)
    return {"per_var": per_var, "total": total}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _align_time(donor: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Re-index donor onto target's time axis (nearest-neighbour).

    Used because scribed and cmb files may have different output cadence —
    we use the cmb's layer_ht for volume info but apply it to scribed
    state-variable values at the scribed time points.
    """
    t_donor = _time_dim_of(donor)
    t_target = _time_dim_of(target)
    if t_donor not in donor.coords:
        return donor   # no time coord; assume identity
    target_times = target[t_target].values
    return donor.reindex({t_donor: target_times}, method="nearest")
