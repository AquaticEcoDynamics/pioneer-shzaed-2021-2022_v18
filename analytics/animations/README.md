# Animations

Cross-cutting animation tooling for SCHISM-AED runs — tracer, oxygen, nitrogen,
and physical-state movies. These serve a range of work (tracer transport, oxygen
dynamics, N-budget diagnostics), so they live in one shared place rather than
inside any single analysis package.

## Shared library

**`animlib.py`** (formerly `animate_oxy_diags.py`) is the engine: hgrid reading
with the `tri_to_elem` element→triangle mapping, node- vs element-centred
rendering, dry-cell masking (`dryFlagNode`/`dryFlagElement`), fill-value masking,
multi-stack auto-discovery/concatenation, the gliding time-series inset, and the
FFmpeg writer. It is also runnable standalone as a configurable multi-panel
animator. The thin scripts below import from it.

## Scripts

| Script | Animates | Inputs |
|---|---|---|
| `animlib.py` | configurable multi-panel (scribe vars + AED diagnostics) | scribe `*_*.nc`, `aed_data_cmb_*.nc`, `out2d_*.nc` |
| `animate_oxy.py` | single tracer (1–2 panel); legacy standalone | scribe |
| `animate_O2fluxes.py` | OXY_oxy + O2 atm/dsf fluxes | scribe + cmb |
| `animate_Nfluxes.py` | 6-panel N sediment/atm flux sheets | cmb |
| `animate_Nrates.py` | 6-panel 3D N process rates | cmb |
| `animate_STO2.py` | salinity + temperature + oxygen | scribe |
| `animate_velocity.py` | depth velocity magnitude \|u\| | scribe (velX/velY) |
| `animate_trc.py` | full-domain TRC_tr1/2/3 | scribe |
| `animate_trc_zoomed.py` | Pioneer-mouth TRC plume + transect overlay | scribe |
| `upper_river_connectivity_animation.py` | upper-river wet/dry connectivity | `out2d` |

## Running

```bash
python animate_Nfluxes.py --outputs-dir <run>/outputs --hgrid <run>/hgrid.gr3
python animlib.py --diag-vars OXY_sat NIT_nitrif --aed-cmb <run>/outputs/aed_data_cmb_1.nc
```

Most scripts share the same flags (`--hgrid`, `--outputs-dir`, `--fps`, `--dpi`,
`--xlim/--ylim`, `--ndays`, `--single-stack`, `--n-stacks`, `--layer`, …).

**Cross-package imports:** two scripts depend on the budget packages and
self-bootstrap their paths —
- `animate_trc_zoomed.py` uses `core` (from `../nutrient_budget/`) and
  `run_config` (from `../pioneer_region_flux/`, so it honours `--run 002|003`);
- `upper_river_connectivity_animation.py` uses `core` (from `../nutrient_budget/`).

Rendered `.mp4` files are git-ignored. `animate_trc_zoomed.py` writes to the
per-run `_outputs/<run>/` dir; the others write to their default names — point
them somewhere outside the repo or let git ignore the output.

> Note: the `read_hgrid` in `animlib.py` (with `tri_to_elem`) and
> `core.geometry.read_hgrid_with_areas` are two separate mesh readers. They were
> left as-is in this relocation; reconciling them into one is a future cleanup.
