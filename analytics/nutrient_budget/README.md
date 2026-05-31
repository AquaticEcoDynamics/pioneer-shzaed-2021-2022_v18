# Nutrient budget framework

A generic, region-agnostic mass-balance toolkit for SCHISM-AED runs. Given a
control volume (CV) and a tracer group (currently total nitrogen) it computes:

```
dM/dt = B + C + D + E + F + residual
```

| Term | Meaning | Source |
|---|---|---|
| A | storage `M(t)`, `dM/dt` | scribed state vars + layer height |
| B | internal sinks (denit + anammox) | 3D rate diagnostics |
| C | atmospheric flux | `*_atm` sheet diagnostics |
| D | sediment–water interface flux | `*_swi` / `*_dsf` diagnostics |
| E | net boundary exchange | `flux.out` over `fluxflag.prop` regions |
| F | river point sources | `vsource.th` × `msource.th` |

`residual` is the closure error.

## Layout

```
nutrient_budget/
├── core/                    importable framework package
│   ├── budget.py            CLI entry point — runs the full 6-term budget
│   ├── tracer_groups.py     tracer-pool registry (NITROGEN bundle, signs)
│   ├── geometry.py          hgrid I/O, areas, NetCDF concat, node→element
│   ├── integrators.py       Terms A–D (storage, rates, surface fluxes)
│   ├── boundary_fluxes.py   flux.out parser + Term E aggregator
│   ├── point_sources.py     source_sink/vsource/msource → Term F
│   ├── report.py            CSV / PNG / HTML writers
│   └── build_cv.py          define a CV from a polygon + fluxflag regions
├── configs/                 example_region.yaml (generic template)
└── README.md
```

This package is **transferable** — point it at any SCHISM-AED run and CV. It has
no hard-coded run paths. A region config (YAML) names the CV element list and the
bounding `fluxflag.prop` regions; see `configs/example_region.yaml`.

## Running the budget CLI

```bash
python core/budget.py \
    --region       <region>.yaml \
    --tracer-group nitrogen \
    --outputs-dir  <run>/outputs \
    --hgrid        <run>/hgrid.gr3 \
    --param-nml    <run>/param.nml \
    --n-stacks     14 \
    --output-dir   ./_outputs/<name>
```

Outputs: `budget_timeseries.csv`, `budget_stacked.png`, `budget_cumulative.png`,
`report.html`.

## Defining a control volume

`core/build_cv.py` builds a CV element list from polygon corners plus optional
`fluxflag.prop` region edges, writing `cv_<name>.csv`, `cv_<name>_elements.txt`,
and a verification map.

## Pioneer application

The Pioneer-estuary investigation that drove the development of this framework
(flux-direction / mass-closure debugging across the river mouth) lives in the
sibling package **[`../pioneer_region_flux/`](../pioneer_region_flux/)**, which
imports this `core/` framework and adds run definitions, the Pioneer CV, and the
diagnostic scripts.
