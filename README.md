# Pioneer SCHISM-AED Model

A coupled hydrodynamic–biogeochemical model of the **Pioneer River estuary at
Mackay** (central Queensland, Australia), built with **SCHISM** (unstructured-grid
3D hydrodynamics) and **AED** (Aquatic EcoDynamics biogeochemistry). This
repository holds the model run configurations, the analysis and visualisation
toolkits, and the documentation.

## 📖 Documentation

**The full modelling guide is published at:**
**<https://aquaticecodynamics.github.io/pioneer-shzaed-2021-2022_v18/>**

It walks a new user through the model setup, running it, analysing the outputs,
and the nutrient-budget / conservation-testing work. The guide is a
[Quarto](https://quarto.org) book in [`docs/quarto/`](docs/quarto/); build it
locally with:

```bash
cd docs/quarto
quarto preview     # live dev server
quarto render      # full build → _output/
```

It is rendered and deployed to GitHub Pages automatically on every push that
touches `docs/quarto/**` (see [`.github/workflows/publish.yml`](.github/workflows/publish.yml)).

## Repository layout

```
pioneer-shzaed-2021-2022_v18/
├── runs/                     model run directories + RUN_LOG.md
│   ├── 002_P18_flood/        flood-pulse run
│   └── 003_P18_flood_flat/   flood pulse + 14-day quiescent (flat) tide
├── analytics/                analysis & visualisation code
│   ├── nutrient_budget/      generic, region-agnostic budget framework (core/)
│   ├── pioneer_region_flux/  Pioneer-specific budget investigation + diagnostics
│   ├── animations/           tracer / oxygen / nitrogen animation tooling
│   └── utilities/            run-preparation scripts
└── docs/
    ├── quarto/               the modelling guide (→ GitHub Pages)
    └── quick-start/          annotated file table + file-map figure
```

### `runs/`

Each `runs/<id>/` is a self-contained set of model inputs (`param.nml`,
`aed.nml`, mesh, `*.gr3`, `*.ic`, `*.th`, `sflux/`, sources) plus the
`run_schism_vm.sh` launcher. Generated `outputs/` are **not** committed (only a
`.keep` placeholder is tracked). The run-to-run history and rationale are in
[`runs/RUN_LOG.md`](runs/RUN_LOG.md):

| ID | Run | Purpose |
|----|-----|---------|
| 001 | Pioneer_17 | original baseline (archived, not in this repo) |
| 002 | `002_P18_flood` | Pioneer 18 with an artificial flood pulse + simplified `fluxflag.prop` for budget testing |
| 003 | `003_P18_flood_flat` | copy of 002 with a flat ocean tide for the first 14 days, to isolate flux-direction effects at the mouth |

### `analytics/`

- **`nutrient_budget/`** — a transferable six-term mass-balance framework
  (`core/budget.py` + integrators, geometry, flux/point-source parsers, report
  writer). No hard-coded run paths.
- **`pioneer_region_flux/`** — the Pioneer application: the control-volume
  definition, run selection (`run_config.py`, `--run 002|003`), and the
  conservation-testing diagnostics (`diagnostics/`, with older one-offs in
  `archive/`).
- **`animations/`** — a shared engine (`animlib.py`) plus per-purpose scripts for
  tracer, oxygen, nitrogen and physical-state movies.
- **`utilities/`** — run-preparation scripts (e.g. `make_flat_tide_elev2D.py`
  for the 003 flat-tide boundary).

## Running the model

The model executable `pschism_AED` and the `combine_output11_MPI` post-processor
are built from the [AED_Tools](https://github.com/AquaticEcoDynamics/AED_Tools)
umbrella repo. From a run directory:

```bash
./run_schism_vm.sh                 # runs SCHISM, combines AED outputs, health-checks
NPROC=32 NSCRIBE=8 ./run_schism_vm.sh   # override defaults
```

See the guide's [Running the Model](https://aquaticecodynamics.github.io/pioneer-shzaed-2021-2022_v18/)
section for build, run, combine and health-check details.

## Notes

- Model outputs (per-stack NetCDF, ~tens of GB per run), `.mp4` renders, and
  Quarto build artefacts are git-ignored. Only the small web-encoded movies used
  in the guide (`docs/quarto/assets/videos/`) are committed.
- This is the `v18` (Pioneer 18) iteration of the model.
