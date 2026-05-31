"""
SCHISM-AED nutrient mass-balance budget — CLI orchestrator.

Usage:
    python -m analytics.budget.budget \\
        --region configs/upper_estuary.yaml \\
        --tracer-group nitrogen \\
        --outputs-dir /path/to/run/outputs \\
        --hgrid /path/to/run/hgrid.gr3 \\
        --param-nml /path/to/run/param.nml \\
        --output-dir ./budget_runs/upper_estuary_N

Produces in --output-dir:
    budget_timeseries.csv
    budget_stacked.png
    budget_cumulative.png
    report.html

The region YAML must declare:
    name, description
    control_volume_elements: <file with one element-ID per line>
    boundary_fluxes:  [{region_id, sign, label}, ...]
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re

import numpy as np
import xarray as xr
import yaml

# When running as a script (no package context), make sibling imports work.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- bootstrap: make the nutrient_budget package importable when run directly ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
# ------------------------------------------------------------------------------
from core import geometry, integrators, boundary_fluxes, point_sources, report
from core.tracer_groups import get_group


def _read_param_start_date(param_nml_path: Path) -> str:
    """Pull start_year / start_month / start_day from param.nml.
    Returns 'YYYY-MM-DD' or None on failure."""
    try:
        text = Path(param_nml_path).read_text()
    except (OSError, IOError):
        return None
    def grab(key):
        m = re.search(rf'^\s*{key}\s*=\s*(\d+)', text, re.MULTILINE)
        return int(m.group(1)) if m else None
    y = grab("start_year"); m = grab("start_month"); d = grab("start_day")
    if y and m and d:
        return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def load_region(region_yaml_path: Path) -> tuple[dict, np.ndarray]:
    """Load region config YAML and the control-volume element-ID list.

    Returns (region_config_dict, cv_element_ids_zero_indexed_array).
    """
    cfg = yaml.safe_load(Path(region_yaml_path).read_text())
    cv_file = Path(cfg["control_volume_elements"])
    if not cv_file.is_absolute():
        cv_file = region_yaml_path.parent / cv_file
    # CV file: one element ID per line, # comments ok.
    ids = []
    with open(cv_file) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                ids.append(int(line) - 1)   # 1-indexed → 0-indexed
            except ValueError:
                pass
    return cfg, np.array(ids, dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, type=Path,
                    help="YAML defining the control volume + boundary regions")
    ap.add_argument("--tracer-group", default="nitrogen",
                    help="Tracer group name (currently only 'nitrogen')")
    ap.add_argument("--outputs-dir", required=True, type=Path,
                    help="SCHISM run's outputs/ directory")
    ap.add_argument("--hgrid", required=True, type=Path,
                    help="Path to hgrid.gr3")
    ap.add_argument("--param-nml", type=Path, default=None,
                    help="Optional path to param.nml (for run start date)")
    ap.add_argument("--n-stacks", type=int, default=None,
                    help="Cap the number of NetCDF stacks to read (default: all)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write CSV/PNG/HTML outputs")
    ap.add_argument("--n-regions", type=int, default=9,
                    help="Number of fluxflag regions in flux.out (default 9)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run directory containing source_sink.in, vsource.th, "
                         "msource.th. If omitted, Term F (point sources) is skipped.")
    args = ap.parse_args()

    print(f"=== Loading region config: {args.region} ===")
    region_cfg, cv_elements = load_region(args.region)
    print(f"  region name: {region_cfg['name']}")
    print(f"  CV elements: {len(cv_elements)}")
    print(f"  boundary regions: {[br['region_id'] for br in region_cfg['boundary_fluxes']]}")

    print(f"\n=== Loading hgrid + element areas: {args.hgrid} ===")
    hgrid = geometry.read_hgrid_with_areas(args.hgrid)
    print(f"  n_nodes={hgrid['n_nodes']}  n_elements={hgrid['n_elements']}  "
          f"is_latlon={hgrid['is_latlon']}")
    cv_area_m2 = hgrid["areas"][cv_elements].sum()
    print(f"  CV surface area: {cv_area_m2:,.0f} m² ({cv_area_m2/1e6:.2f} km²)")

    print(f"\n=== Loading combined AED file (cmb) ===")
    ds_cmb = geometry.open_cmb_concat(args.outputs_dir, n_stacks=args.n_stacks)
    print(f"  time points (raw): {ds_cmb.sizes.get('time', '?')}")
    if "time" in ds_cmb.coords:
        t = ds_cmb["time"].values
        _, uniq_idx = np.unique(t, return_index=True)
        if len(uniq_idx) != len(t):
            ds_cmb = ds_cmb.isel(time=np.sort(uniq_idx))
            print(f"  deduplicated overlapping stacks: {len(t)} -> {ds_cmb.sizes['time']}")
    if "ENV_layer_ht" not in ds_cmb.data_vars:
        raise RuntimeError(
            "ENV_layer_ht not in cmb file — required for volume integration. "
            "Add it to d_vars_save in aed.nml and re-run."
        )
    layer_ht = ds_cmb["ENV_layer_ht"]
    print(f"  layer_ht shape={layer_ht.shape}")

    group = get_group(args.tracer_group)
    warnings = []

    print(f"\n=== Term A: volume-integrating storage variables ===")
    A = integrators.storage_term(
        group["state"], args.outputs_dir, hgrid, cv_elements, layer_ht,
        n_stacks=args.n_stacks,
    )
    print(f"  Total N storage time series: shape={A['total'].shape}")

    print(f"\n=== Term A': dM/dt ===")
    dMdt = integrators.storage_derivative(A["total"])

    print(f"\n=== Term B: volume-integrating internal rates ===")
    B = integrators.internal_rates_term(
        group["rates"], ds_cmb, hgrid, cv_elements, layer_ht,
    )
    print(f"  net-loss series: shape={B['net_loss'].shape}")

    print(f"\n=== Term C: atmospheric flux ===")
    C = integrators.surface_flux_term(group["atm"], ds_cmb, hgrid, cv_elements, "atm")
    print(f"  atm-total series: shape={C['total'].shape}")

    print(f"\n=== Term D: sediment-water flux ===")
    D = integrators.surface_flux_term(group["swi"], ds_cmb, hgrid, cv_elements, "swi")
    print(f"  swi-total series: shape={D['total'].shape}")

    print(f"\n=== Term E: boundary fluxes from flux.out ===")
    flux_path = args.outputs_dir / "flux.out"
    if not flux_path.exists():
        raise FileNotFoundError(f"flux.out not found at {flux_path}")
    flux_da = boundary_fluxes.parse_flux_out(flux_path, n_regions=args.n_regions)
    start_date = _read_param_start_date(args.param_nml) if args.param_nml else None
    if start_date is None:
        warnings.append("Could not parse start date from param.nml — flux.out "
                        "times stay in days-since-start; alignment with NC time "
                        "axes may be approximate.")
    E = boundary_fluxes.boundary_flux_term(
        flux_da, region_cfg["boundary_fluxes"],
        group["boundary_tracers_n_per_mol"],
        start_date=start_date,
    )
    print(f"  boundary-net series: shape={E.shape}")

    # Reindex Term E onto the cmb time axis if possible
    if "time" in E.coords:
        t_e = E["time"].values
        _, uniq_e = np.unique(t_e, return_index=True)
        if len(uniq_e) != len(t_e):
            E = E.isel(time=np.sort(uniq_e))
            print(f"  deduplicated flux.out time index: {len(t_e)} -> {E.sizes['time']}")
        E_aligned = E.reindex_like(A["total"], method="nearest")
    else:
        warnings.append("Term E has no datetime axis; assuming alignment by index.")
        E_aligned = E

    print(f"\n=== Term F: point-source (river) input ===")
    F_info = None
    if args.run_dir is None:
        warnings.append("--run-dir not supplied; Term F (point sources) skipped.")
        F_aligned = xr.zeros_like(A["total"])
        F_aligned.attrs["units"] = "mmol N/day"
    else:
        ss_path = args.run_dir / "source_sink.in"
        vs_path = args.run_dir / "vsource.th"
        ms_path = args.run_dir / "msource.th"
        for p in (ss_path, vs_path, ms_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
        F_info = point_sources.point_source_term(
            ss_path, vs_path, ms_path,
            cv_elements=cv_elements,
            tracer_n_per_mol=group["point_source_tracers_n_per_mol"],
            start_date=start_date,
        )
        print(f"  {len(F_info['cv_src_idx'])} source(s) inside CV: "
              f"indices {F_info['cv_src_idx']} (elements {F_info['cv_src_elem']})")
        if F_info["total"].dtype.kind in ("M", "m") or "time" in F_info["total"].coords:
            F_aligned = F_info["total"].reindex_like(A["total"], method="nearest")
        else:
            warnings.append("Term F has no datetime axis; aligning by index.")
            F_aligned = F_info["total"]

    terms = {
        "storage_total": A["total"],
        "storage_per_var": A["per_var"],
        "dMdt": dMdt,
        "internal_per": B["per_rate"],
        "internal_net": B["net_loss"],
        "atm_per":   C["per_var"],
        "atm_total": C["total"],
        "swi_per":   D["per_var"],
        "swi_total": D["total"],
        "boundary":  E_aligned,
        "point_source": F_aligned,
        "point_source_info": F_info,
    }

    print(f"\n=== Writing outputs to {args.output_dir} ===")
    report.write_outputs(terms, args.output_dir, region_cfg,
                         args.tracer_group, warnings=warnings)
    print("\nDone.")


if __name__ == "__main__":
    main()
