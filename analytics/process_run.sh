#!/usr/bin/env bash
#
# process_run.sh — reusable post-processing pipeline for a SCHISM-AED run.
#
# Runs the nutrient budget (two windows), the per-variable diagnostic gallery,
# the conservation/global-balance checks, and the key animations against a
# registered run's combined outputs. Each piece is a *stage* so the stages can
# be driven independently or fanned out in parallel by an orchestrator.
#
# The model outputs must already be combined (scribe NetCDF + aed_data_cmb_*.nc
# + flux.out present in <run>/outputs/). This script does NOT run the model.
#
# Usage:
#   ./process_run.sh <stage> [run_id]
#
#   stage:
#     budget14 budget53        6-term N budget over 14 / 53 stacks (= days)
#     pervar14 pervar53        per_variable_plots gallery, 14 / 53 day window
#     diag14   diag53          conservation_check + global_balance, 14 / 53 day
#     anim_trc anim_trc_zoom   tracer animations (full window)
#     anim_vel anim_sto2       physical-state animations (velocity, S/T/O2)
#     anim_nflux anim_nrates   nitrogen flux / rate animations
#     budgets                  budget14 + budget53
#     diags                    pervar14 pervar53 diag14 diag53
#     anims                    all six animations
#     all                      everything (serial; slow)
#     list                     print the stage names (for orchestration)
#
#   run_id: registered run, default 003.
#
# Environment:
#   SCHISM_PY   path to the python interpreter (default: the Python312 install
#               that carries xarray / scipy / matplotlib / imageio_ffmpeg).
#
# Notes / gotchas baked in here so callers don't have to remember them:
#   * 1 output stack == 1 day  (ihfskip 1920 x dt 45 s = 86400 s).
#   * run 003's fluxflag.prop has 3 regions, so the budget needs --n-regions 3
#     (budget.py defaults to 9 — wrong for this run).
#   * the animation scripts (except animate_trc_zoomed) hard-code run-002 paths,
#     so --hgrid / --outputs-dir / --output are passed explicitly here.
#   * per_variable_plots and conservation_check write to a fixed per-run folder,
#     so the 14d and 53d passes would clobber each other — this script renames
#     each result to a *_<N>d suffix after it runs.
#
set -euo pipefail

# Headless matplotlib: render off-screen, avoid the TkAgg "main thread is not in
# main loop" teardown crash seen when scripts default to an interactive backend.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Memory note: while a run's cmb lacks ENV_layer_ht (e.g. before the aed.nml
# d_vars_save fix is re-run), the budget + diagnostics fall back to loading
# full-domain zCoordinates (~7.5 GB/stack-set), so run the heavy stages
# SERIALLY rather than fanning them out, or they contend for RAM and OOM.

STAGE="${1:-}"
RUN_ID="${2:-003}"

PY="${SCHISM_PY:-C:/Users/aedvmadmin/AppData/Local/Programs/Python/Python312/python.exe}"
ANALYTICS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ANALYTICS/.." && pwd)"

# ---- resolve the run directory (mirrors run_config.RUNS) ----
case "$RUN_ID" in
  002) RUN_DIR="s:/Matt_Working/schism/P18_flood" ;;
  003) RUN_DIR="$REPO/runs/003_P18_flood_flat" ;;
  *)   echo "unknown run '$RUN_ID' (known: 002, 003)"; exit 2 ;;
esac
RUN_NAME="$(basename "$RUN_DIR")"
OUTPUTS="$RUN_DIR/outputs"
HGRID="$RUN_DIR/hgrid.gr3"
PARAM="$RUN_DIR/param.nml"

NB="$ANALYTICS/nutrient_budget"
PRF="$ANALYTICS/pioneer_region_flux"
ANIM="$ANALYTICS/animations"
REGION="$PRF/configs/pioneer_estuary.yaml"
NREG=3
PRF_OUT="$PRF/_outputs/$RUN_NAME"
RENDER_DIR="$PRF_OUT/animations"     # *.mp4 + analytics/**/_outputs/ are git-ignored

# Full-window movies are ~96 frames/day; dpi 300 is overkill for a 50+ day clip.
# Override with ANIM_DPI / ANIM_FPS if you want higher fidelity / different speed.
ADPI="${ANIM_DPI:-150}"
AFPS="${ANIM_FPS:-12}"

banner () { echo ""; echo "=========== $* ==========="; echo ""; }

# rename a fixed-name diagnostic output folder to a window-suffixed one so the
# 14d and 53d passes coexist.   $1 = folder basename, $2 = day suffix
_suffix_outdir () {
  local base="$PRF_OUT/$1" days="$2"
  if [ -d "$base" ]; then
    rm -rf "${base}_${days}d"
    mv "$base" "${base}_${days}d"
    echo "  -> ${base}_${days}d"
  fi
}

# ----------------------------------------------------------------- budget
budget () {  # $1 = window label, $2 = n_stacks
  local win="$1" nst="$2"
  banner "BUDGET  run $RUN_ID  ${win}  (n-stacks=$nst, n-regions=$NREG)"
  "$PY" "$NB/core/budget.py" \
    --region       "$REGION" \
    --tracer-group nitrogen \
    --outputs-dir  "$OUTPUTS" \
    --hgrid        "$HGRID" \
    --param-nml    "$PARAM" \
    --n-stacks     "$nst" \
    --n-regions    "$NREG" \
    --run-dir      "$RUN_DIR" \
    --output-dir   "$NB/_outputs/pioneer_N_${RUN_ID}_${win}"
}

# ----------------------------------------------------------------- diagnostics
pervar () {  # $1 = days
  local days="$1"
  banner "PER-VARIABLE PLOTS  run $RUN_ID  ${days}d"
  "$PY" "$PRF/diagnostics/per_variable_plots.py" --run "$RUN_ID" --days "$days"
  _suffix_outdir "per_variable_plots_simple_fluxflag" "$days"
}

diag () {  # $1 = days
  local days="$1"
  banner "CONSERVATION + GLOBAL BALANCE  run $RUN_ID  ${days}d"
  "$PY" "$PRF/diagnostics/conservation_check.py" --run "$RUN_ID" --days "$days"
  "$PY" "$PRF/diagnostics/global_balance.py"     --run "$RUN_ID" --days "$days"
  _suffix_outdir "conservation_check" "$days"
}

# ----------------------------------------------------------------- animations
# full window: --n-stacks omitted -> the scripts read every available stack.
anim_trc () {
  banner "ANIM trc  run $RUN_ID  (dpi=$ADPI fps=$AFPS)"
  mkdir -p "$RENDER_DIR"
  "$PY" "$ANIM/animate_trc.py" --hgrid "$HGRID" --outputs-dir "$OUTPUTS" \
    --dpi "$ADPI" --fps "$AFPS" --output "$RENDER_DIR/TRC_${RUN_ID}.mp4"
}
anim_trc_zoom () {
  banner "ANIM trc_zoomed  run $RUN_ID  (writes to its own per-run dir)"
  "$PY" "$ANIM/animate_trc_zoomed.py" --run "$RUN_ID" --dpi "$ADPI" --fps "$AFPS"
}
anim_vel () {
  banner "ANIM velocity  run $RUN_ID  (dpi=$ADPI fps=$AFPS)"
  mkdir -p "$RENDER_DIR"
  "$PY" "$ANIM/animate_velocity.py" --hgrid "$HGRID" --outputs-dir "$OUTPUTS" \
    --dpi "$ADPI" --fps "$AFPS" --output "$RENDER_DIR/velocity_${RUN_ID}.mp4"
}
anim_sto2 () {
  banner "ANIM STO2  run $RUN_ID  (dpi=$ADPI fps=$AFPS)"
  mkdir -p "$RENDER_DIR"
  "$PY" "$ANIM/animate_STO2.py" --hgrid "$HGRID" --outputs-dir "$OUTPUTS" \
    --dpi "$ADPI" --fps "$AFPS" --output "$RENDER_DIR/STO2_${RUN_ID}.mp4"
}
anim_nflux () {
  banner "ANIM Nfluxes  run $RUN_ID  (dpi=$ADPI fps=$AFPS)"
  mkdir -p "$RENDER_DIR"
  "$PY" "$ANIM/animate_Nfluxes.py" --hgrid "$HGRID" --outputs-dir "$OUTPUTS" \
    --dpi "$ADPI" --fps "$AFPS" --output "$RENDER_DIR/Nfluxes_${RUN_ID}.mp4"
}
anim_nrates () {
  banner "ANIM Nrates  run $RUN_ID  (dpi=$ADPI fps=$AFPS)"
  mkdir -p "$RENDER_DIR"
  "$PY" "$ANIM/animate_Nrates.py" --hgrid "$HGRID" --outputs-dir "$OUTPUTS" \
    --dpi "$ADPI" --fps "$AFPS" --output "$RENDER_DIR/Nrates_${RUN_ID}.mp4"
}

# ----------------------------------------------------------------- dispatch
case "$STAGE" in
  budget14)      budget 14d 14 ;;
  budget53)      budget 53d 53 ;;
  pervar14)      pervar 14 ;;
  pervar53)      pervar 53 ;;
  diag14)        diag 14 ;;
  diag53)        diag 53 ;;
  anim_trc)      anim_trc ;;
  anim_trc_zoom) anim_trc_zoom ;;
  anim_vel)      anim_vel ;;
  anim_sto2)     anim_sto2 ;;
  anim_nflux)    anim_nflux ;;
  anim_nrates)   anim_nrates ;;
  budgets)       budget 14d 14; budget 53d 53 ;;
  diags)         pervar 14; pervar 53; diag 14; diag 53 ;;
  anims)         anim_trc; anim_trc_zoom; anim_vel; anim_sto2; anim_nflux; anim_nrates ;;
  all)           budget 14d 14; budget 53d 53; pervar 14; pervar 53; diag 14; diag 53; \
                 anim_trc; anim_trc_zoom; anim_vel; anim_sto2; anim_nflux; anim_nrates ;;
  list)          echo "budget14 budget53 pervar14 pervar53 diag14 diag53 anim_trc anim_trc_zoom anim_vel anim_sto2 anim_nflux anim_nrates" ;;
  ""|-h|--help)  sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
  *)             echo "unknown stage '$STAGE' (try: ./process_run.sh list)"; exit 2 ;;
esac
