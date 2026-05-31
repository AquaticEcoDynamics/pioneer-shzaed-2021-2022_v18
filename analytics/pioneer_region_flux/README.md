# Pioneer region flux & budget investigation

Pioneer-estuary-specific analysis built on the generic budget framework in
[`../nutrient_budget/`](../nutrient_budget/). This is where the **flux-direction
/ mass-closure investigation across the Pioneer river mouth** lives.

```
pioneer_region_flux/
├── run_config.py      run registry (002/003) + Pioneer CV; active_run()
├── configs/           cv_pioneer_estuary_* + pioneer_estuary.yaml
├── diagnostics/       live investigation scripts (14)
├── archive/           superseded / one-off scripts (kept for record)
└── plots/             generated figures
```

Scripts import the generic framework as `from core import …` (resolved from the
sibling `nutrient_budget/`) and run selection as `from run_config import active_run`.

## Running

Target a run **by name** — no path editing:

```bash
python diagnostics/conservation_check.py              # defaults to run 002
python diagnostics/conservation_check.py --run 003 --days 14
python diagnostics/edge_flux_reconstruction.py --run-dir /path/to/any/run
```

`active_run()` resolves via `--run` / `--run-dir` / `SCHISM_RUN` (default 002)
and `--days` / `SCHISM_DAYS` (default 21). Registered runs (`RUNS` in
`run_config.py`): `002` → `s:/Matt_Working/schism/P18_flood`, `003` →
`runs/003_P18_flood_flat`. Each `RunConfig` derives all run-file paths, parses
the start date from `param.nml`, reads `n_regions` from `fluxflag.prop`, and
points at this package's `configs/`. Per-run plots go to `_outputs/<run>/`
(git-ignored).

## Investigation state — why the budget won't close

1. **Transport is conservative** — `TRC_tr1` (AED) ≡ `GEN_1` (SCHISM-native)
   bit-for-bit (`tr1_vs_gen1_compare.py`), so the loss is **not** an AED
   coupling bug.
2. **Source injection loses mass** at high `rat = Q·dt/bigv` in dry/shallow
   cells — `src3_dry_rat_check.py` shows mean `rat≈7` during the pulse
   (~12% retention per step if cell volume doesn't grow).
3. **The mouth flux is undercounted** — `flux.out` only records edges where
   adjacent `fluxflag` values differ by exactly 1 (and neither is −1). Much of
   the CV mouth perimeter is **invisible** to Term E
   (`mouth_edge_classification.py`, `cv_perimeter_tracked.py`); an independent
   velocity-based reconstruction (`edge_flux_reconstruction.py`) cross-checks it.
   Net: cumulative source ≫ cumulative tracked mouth flux (e.g. TRC_tr1 ~3.97e7
   vs ~−2.9e5 mmol).

The `003_P18_flood_flat` run (quiescent tide for 14 days) is designed to
disentangle the source-injection loss from the tidal/mouth flux-direction effect.

### Key diagnostics (in `diagnostics/`)

| Script | Question it answers |
|---|---|
| `tr1_vs_gen1_compare.py` | AED coupling vs SCHISM transport loss? (decisive) |
| `bucket_tr1.py` | CV retention vs a well-mixed bucket bound |
| `global_tr1_check.py` / `global_tr3_check.py` | Lost domain-wide or just outside the CV? |
| `src3_dry_rat_check.py` | The `rat`-driven source-injection loss mechanism |
| `mouth_edge_classification.py` / `cv_perimeter_tracked.py` | Which mouth edges does `flux.out` track? |
| `edge_flux_reconstruction.py` | Independent flux across the mouth vs `flux.out` |
| `conservation_check.py` / `global_balance.py` | Per-variable residual / boundary closure |
| `per_variable_plots.py` | 20-variable diagnostic gallery |
| `check_flux_upwind.py` | POS/NEG/NET breakdown + implied upwind concentration at the gate |
| `inspect_fluxflag.py` | Reusable `fluxflag.prop` map renderer (CLI) |

`archive/` holds earlier one-offs (relocation-candidate maps, transect cross-checks,
single-cell time series).

**Animations** that used to live here (`animate_trc_zoomed.py`, `animate_trc.py`,
`upper_river_connectivity_animation.py`) have moved to
[`../animations/`](../animations/). `animate_trc_zoomed.py` still honours
`--run 002|003` via this package's `run_config.py`.
