#!/bin/bash
# -----------------------------------------------------------------------------
# Run SCHISM-AED on a single multi-core VM (no SLURM).
# Mirrors run_schism_sbatch.sh in spirit: timing, captured stderr, exit-code
# check, formatted elapsed time.
#
# Usage:
#   ./run_schism_vm.sh
# Or override at invocation:
#   NPROC=32 NSCRIBE=8 ./run_schism_vm.sh
#
# Combine-only mode (skip SCHISM, just combine whatever stacks exist in
# outputs/ — useful after a prematurely killed run):
#   COMBINE_ONLY=1 ./run_schism_vm.sh
# Optionally narrow the stack range:
#   COMBINE_ONLY=1 COMBINE_BEGIN=1 COMBINE_END=10 ./run_schism_vm.sh
# -----------------------------------------------------------------------------

# --- Configuration ----------------------------------------------------------
# Total MPI ranks (compute + scribe). Set to your VM's core count.
NPROC=${NPROC:-64}

# Scribe ranks reserved for combined NetCDF output. Compute ranks = NPROC - NSCRIBE.
NSCRIBE=${NSCRIBE:-26}  # bumped from 25 — GEN_1 adds 1 scribed variable

# SCHISM-AED executable. Adjust if you've renamed/copied it elsewhere.
BINARY=${BINARY:-/media/fileshare/Matt_Working/schism/AED_Tools/binaries/ubuntu/22.04/pschism_AED_GEN_BLD_STANDALONE_SH_MEM_COMM_TVD-VL}

# Log file (appended to across runs, so previous runs are preserved).
LOG=${LOG:-schism_error.log}

# --- Post-run combine (combine_output11_MPI) --------------------------------
# Set to 0 to skip the combine step entirely.
DO_COMBINE=${DO_COMBINE:-1}

# Combine-only mode. Set COMBINE_ONLY=1 to SKIP the SCHISM run and go straight
# to the combine step over whichever per-rank stacks already exist in outputs/.
# Useful when SCHISM was killed mid-run and you want to salvage the completed
# stacks.
COMBINE_ONLY=${COMBINE_ONLY:-0}

# Path to combine_output11_MPI (built into <schism>/build/bin).
COMBINE_BIN=${COMBINE_BIN:-/media/fileshare/Matt_Working/schism/AED_Tools/schism/build/bin/combine_output11_MPI}

# Number of MPI ranks for the combiner. Default 'auto' = min(NPROC, n_stacks),
# i.e. one rank per stack but never more than the SCHISM rank count.
# Override with an explicit integer if you want a different setting.
COMBINE_NPROC=${COMBINE_NPROC:-auto}

# Stack range to combine (-b begin, -e end) and output prefix (-o).
# COMBINE_END=auto (default) discovers the highest stack number found in
# outputs/aed_data_*_*.nc once the model has finished — so you don't have to
# update this every time the run length changes. Override with an explicit
# integer (e.g. COMBINE_END=14) to combine a specific range.
COMBINE_BEGIN=${COMBINE_BEGIN:-1}
COMBINE_END=${COMBINE_END:-auto}
COMBINE_PREFIX=${COMBINE_PREFIX:-aed_data}

# Separate log for combine step
COMBINE_LOG=${COMBINE_LOG:-combine.log}

# If 1, delete the per-rank aed_data_<rank>_<stack>.nc files once combining
# has SUCCEEDED and all expected cmb files exist. Saves ~tens of GB per run.
# Set to 0 to keep the per-rank files (e.g. if you want to re-run the combine).
DELETE_PER_RANK_AFTER_COMBINE=${DELETE_PER_RANK_AFTER_COMBINE:-1}

# --- Heartbeat (progress indicator while SCHISM is running) -----------------
# Every HEARTBEAT_INTERVAL seconds, prints the most recent timestep line from
# outputs/mirror.out to this script's stdout. Useful when running under tmux
# or `nohup ./run_schism_vm.sh > run.log 2>&1 &` so you can see the model is
# still advancing without tailing mirror.out separately.
HEARTBEAT_ON=${HEARTBEAT_ON:-1}                  # set to 0 to disable
HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-60}      # seconds (default 1 min)

# Parse dt (model timestep, seconds) from param.nml as a fallback for the
# heartbeat's "simulated days" calculation. If TIME= appears in mirror.out
# we prefer that; dt × step_count is the back-up.
DT=$(grep -E '^[[:space:]]*dt[[:space:]]*=' param.nml 2>/dev/null | head -1 \
    | sed -E 's/[^0-9.]*([0-9.]+).*/\1/')
DT=${DT:-1}
# ----------------------------------------------------------------------------

# Heartbeat worker — runs in background. Looks for the latest "TIME STEP="
# line in outputs/mirror.out and prints it; falls back to the file's last
# line if no step pattern is present yet.
_heartbeat_loop() {
    # \r returns cursor to start of line; \033[K clears to end of line so the
    # printed message overwrites whatever was there before.
    while sleep "$HEARTBEAT_INTERVAL"; do
        ts="[$(date +%H:%M:%S)]"
        if [ -f outputs/mirror.out ]; then
            step_line=$(grep -hE 'TIME STEP=' outputs/mirror.out 2>/dev/null | tail -1)
            if [ -n "$step_line" ]; then
                msg=$(echo "$step_line" | tr -s '[:space:]' ' ')
                # Get simulated days: prefer TIME=<sec> from the line, else fall
                # back to step_number × DT.
                time_sec=$(echo "$step_line" | grep -oE 'TIME=[[:space:]]*[0-9.]+' \
                    | head -1 | sed -E 's/TIME=[[:space:]]*//')
                if [ -z "$time_sec" ]; then
                    step_n=$(echo "$step_line" | grep -oE 'TIME STEP=[[:space:]]*[0-9]+' \
                        | head -1 | sed -E 's/TIME STEP=[[:space:]]*//')
                    if [ -n "$step_n" ]; then
                        time_sec=$(awk -v s="$step_n" -v dt="$DT" 'BEGIN { print s*dt }')
                    fi
                fi
                if [ -n "$time_sec" ]; then
                    days=$(awk -v t="$time_sec" 'BEGIN { printf "%.2f", t/86400 }')
                    real_elapsed=$(( $(date +%s) - start_time ))
                    if [ "$real_elapsed" -gt 0 ]; then
                        ratio=$(awk -v sim="$time_sec" -v real="$real_elapsed" \
                                'BEGIN { printf "%.1f", sim/real }')
                        printf '\r⏱️  %s %s  (≈ %s sim-days, 1:%s real:sim)\033[K' \
                            "$ts" "$msg" "$days" "$ratio"
                    else
                        printf '\r⏱️  %s %s  (≈ %s sim-days)\033[K' "$ts" "$msg" "$days"
                    fi
                else
                    printf '\r⏱️  %s %s\033[K' "$ts" "$msg"
                fi
            else
                last=$(tail -1 outputs/mirror.out 2>/dev/null | tr -s '[:space:]' ' ')
                printf '\r⏱️  %s mirror.out: %s\033[K' "$ts" "${last:-(no content yet)}"
            fi
        else
            printf '\r⏱️  %s waiting for outputs/mirror.out...\033[K' "$ts"
        fi
    done
}

# Ensure the heartbeat dies when this script exits (clean OR interrupted).
HEARTBEAT_PID=""
_cleanup_heartbeat() {
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null
        wait "$HEARTBEAT_PID" 2>/dev/null
        HEARTBEAT_PID=""
        # Emit a newline so subsequent output starts on a fresh line rather
        # than continuing the heartbeat's overwrite line.
        printf '\n'
    fi
}
trap _cleanup_heartbeat EXIT INT TERM

# Start time
start_time=$(date +%s)

if [ "$COMBINE_ONLY" = "1" ]; then
    echo ""
    echo "🔁 COMBINE_ONLY mode — skipping SCHISM run, going straight to combine."
    echo "   (Will combine whichever per-rank stacks already exist in outputs/.)"
    exit_code=0   # pretend SCHISM succeeded so the combine block runs
else
    echo ""
    echo "Running SCHISM-AED model..."
    echo "  NPROC   = $NPROC"
    echo "  NSCRIBE = $NSCRIBE  (compute ranks = $((NPROC - NSCRIBE)))"
    echo "  BINARY  = $BINARY"
    echo "  LOG     = $LOG"

    # Sanity check: binary must be on PATH (or supplied as an absolute path)
    if ! command -v "$BINARY" >/dev/null 2>&1 && [ ! -x "$BINARY" ]; then
        echo "❌ Cannot find SCHISM binary '$BINARY'. Put it on PATH or set BINARY=/abs/path/to/pschism_..."
        exit 1
    fi

    # Optional: load modules if the VM uses environment-modules. Comment out or
    # edit if your VM ships netcdf-fortran in the system path (Ubuntu apt does).
    # module use /home/uqmmorav/EasyBuild/modules/all
    # module load netcdf-fortran

    echo -e "\n\n\n NEW RUN\n----- Run started at $(date) -----" >> "$LOG"

    # Start heartbeat (progress prints every $HEARTBEAT_INTERVAL seconds)
    if [ "$HEARTBEAT_ON" = "1" ]; then
        echo "  Heartbeat enabled — progress will print every ${HEARTBEAT_INTERVAL}s."
        echo ""
        _heartbeat_loop &
        HEARTBEAT_PID=$!
    fi

    mpirun -np "$NPROC" "$BINARY" "$NSCRIBE" >> "$LOG" 2>&1
    exit_code=$?

    # Stop heartbeat as soon as mpirun returns
    _cleanup_heartbeat

    if [ $exit_code -ne 0 ]; then
        echo "❌ SCHISM run failed with exit code $exit_code."
        echo "------ ERROR LOG (tail) ------"
        tail -50 "$LOG"
    else
        echo "✅ SCHISM run completed successfully."
    fi

    # End time / formatted elapsed (SCHISM only)
    schism_end_time=$(date +%s)
    schism_elapsed=$(( schism_end_time - start_time ))
    hours=$((schism_elapsed / 3600))
    minutes=$(( (schism_elapsed % 3600) / 60 ))
    seconds=$((schism_elapsed % 60))
    printf "⏱️ Time taken to run SCHISM: %02d:%02d:%02d (hh:mm:ss)\n" \
        $hours $minutes $seconds
fi

# --- Combine per-rank AED files --------------------------------------------
combine_exit=0
if [ "$DO_COMBINE" = "1" ] && [ $exit_code -eq 0 ]; then
    # Auto-detect COMBINE_END if user left the default
    if [ "$COMBINE_END" = "auto" ]; then
        max_stack=$(ls outputs/${COMBINE_PREFIX}_*_*.nc 2>/dev/null \
            | sed -nE 's|.*/'"${COMBINE_PREFIX}"'_[0-9]+_([0-9]+)\.nc$|\1|p' \
            | sort -n | tail -1)
        if [ -n "$max_stack" ]; then
            COMBINE_END=$max_stack
            echo "  (auto-detected COMBINE_END=$COMBINE_END from per-rank files)"
        else
            echo "❌ COMBINE_END=auto but no ${COMBINE_PREFIX}_*_*.nc files in outputs/"
            combine_exit=2
            COMBINE_END=0   # avoid arithmetic errors downstream
        fi
    fi

    # Auto-resolve COMBINE_NPROC = min(NPROC, n_stacks) if left at default
    if [ "$COMBINE_NPROC" = "auto" ]; then
        n_stacks=$(( COMBINE_END - COMBINE_BEGIN + 1 ))
        if [ "$n_stacks" -lt 1 ]; then n_stacks=1; fi
        COMBINE_NPROC=$(( NPROC < n_stacks ? NPROC : n_stacks ))
        echo "  (auto-resolved COMBINE_NPROC=$COMBINE_NPROC = min($NPROC, $n_stacks))"
    fi

    echo ""
    echo "Combining per-rank outputs..."
    echo "  COMBINE_BIN    = $COMBINE_BIN"
    echo "  COMBINE_NPROC  = $COMBINE_NPROC"
    echo "  STACKS         = $COMBINE_BEGIN .. $COMBINE_END"
    echo "  OUTPUT PREFIX  = $COMBINE_PREFIX  (produces ${COMBINE_PREFIX}_cmb_<S>.nc)"

    if [ $combine_exit -ne 0 ]; then
        : # already failed (no per-rank files); skip the actual combine
    elif [ ! -x "$COMBINE_BIN" ]; then
        echo "❌ combine binary not found / not executable: $COMBINE_BIN"
        combine_exit=127
    elif [ ! -d outputs ]; then
        echo "❌ outputs/ directory not found in $(pwd)"
        combine_exit=2
    else
        combine_start=$(date +%s)
        echo -e "\n----- Combine started at $(date) -----" >> "$COMBINE_LOG"
        ( cd outputs && \
          mpirun -np "$COMBINE_NPROC" "$COMBINE_BIN" \
                 -b "$COMBINE_BEGIN" -e "$COMBINE_END" -o "$COMBINE_PREFIX" ) >> "$COMBINE_LOG" 2>&1
        combine_exit=$?
        combine_elapsed=$(( $(date +%s) - combine_start ))
        c_hh=$((combine_elapsed / 3600))
        c_mm=$(( (combine_elapsed % 3600) / 60 ))
        c_ss=$((combine_elapsed % 60))

        if [ $combine_exit -ne 0 ]; then
            echo "❌ Combine failed with exit code $combine_exit."
            echo "------ COMBINE LOG (tail) ------"
            tail -50 "$COMBINE_LOG"
        else
            printf "✅ Combine completed in %02d:%02d:%02d.\n" $c_hh $c_mm $c_ss

            # --- Delete per-rank aed_data files after successful combine ---
            if [ "$DELETE_PER_RANK_AFTER_COMBINE" = "1" ]; then
                # Verify all expected cmb files actually exist before deleting
                cmb_exp=$(( COMBINE_END - COMBINE_BEGIN + 1 ))
                cmb_got=$(ls outputs/${COMBINE_PREFIX}_cmb_*.nc 2>/dev/null | wc -l)
                if [ "$cmb_got" -lt "$cmb_exp" ]; then
                    echo "⚠️  Found only $cmb_got of $cmb_exp expected cmb files — NOT deleting per-rank files."
                else
                    # Per-rank pattern: aed_data_<6-digit-rank>_<stack>.nc
                    # The [0-9]* glob excludes 'cmb' so combined files are safe.
                    pr_count=$(ls outputs/${COMBINE_PREFIX}_[0-9]*_*.nc 2>/dev/null | wc -l)
                    if [ "$pr_count" -gt 0 ]; then
                        pr_size=$(du -ch outputs/${COMBINE_PREFIX}_[0-9]*_*.nc 2>/dev/null \
                            | tail -1 | awk '{print $1}')
                        echo "  Deleting $pr_count per-rank ${COMBINE_PREFIX} files (~$pr_size)..."
                        rm -f outputs/${COMBINE_PREFIX}_[0-9]*_*.nc
                        echo "  ✅ per-rank files deleted; combined cmb files kept."
                    else
                        echo "  (no per-rank ${COMBINE_PREFIX} files to delete)"
                    fi
                fi
            fi
        fi
    fi
elif [ "$DO_COMBINE" = "1" ] && [ $exit_code -ne 0 ]; then
    echo "⚠️  Skipping combine because SCHISM failed (exit $exit_code)."
fi

# --- Health check -----------------------------------------------------------
# Quick sanity checks on the output directory. Doesn't change the exit code
# (SCHISM/combine exit codes are authoritative); just surfaces warnings.
if [ -d outputs ]; then
    echo ""
    echo "--- Run health check ---"
    health_warn=0

    # 1. fatal.error should be empty (0 bytes = no fatal abort)
    if [ ! -e outputs/fatal.error ]; then
        echo "⚠️  fatal.error missing (run may not have started)"
        health_warn=1
    elif [ -s outputs/fatal.error ]; then
        echo "❌ fatal.error is non-empty — SCHISM aborted:"
        sed 's/^/     /' outputs/fatal.error | head -10
        health_warn=1
    else
        echo "✅ fatal.error is empty (no fatal abort)"
    fi

    # 2. mirror.out should exist and end with a finalization message
    if [ ! -f outputs/mirror.out ]; then
        echo "⚠️  mirror.out missing"
        health_warn=1
    else
        echo "✅ mirror.out tail:"
        tail -3 outputs/mirror.out | sed 's/^/     /'
    fi

    # 3. Scan nonfatal_* for any error mentions
    nonfatal_count=$(ls outputs/nonfatal_* 2>/dev/null | wc -l)
    if [ "$nonfatal_count" -gt 0 ]; then
        nonfatal_errs=$(grep -l -i error outputs/nonfatal_* 2>/dev/null | wc -l)
        if [ "$nonfatal_errs" -gt 0 ]; then
            echo "⚠️  $nonfatal_errs of $nonfatal_count nonfatal_<rank> files mention 'error':"
            grep -l -i error outputs/nonfatal_* 2>/dev/null | head -5 | sed 's/^/     /'
            health_warn=1
        else
            echo "✅ no 'error' mentions in $nonfatal_count nonfatal_<rank> files"
        fi
    fi

    # 4. Combined AED files exist (when combine was requested)
    if [ "$DO_COMBINE" = "1" ]; then
        cmb_count=$(ls outputs/${COMBINE_PREFIX}_cmb_*.nc 2>/dev/null | wc -l)
        cmb_expected=$((COMBINE_END - COMBINE_BEGIN + 1))
        if [ "$cmb_count" -eq "$cmb_expected" ]; then
            echo "✅ $cmb_count combined AED file(s) present (${COMBINE_PREFIX}_cmb_${COMBINE_BEGIN}..${COMBINE_END}.nc)"
        elif [ "$cmb_count" -gt 0 ]; then
            echo "⚠️  found $cmb_count combined AED file(s) but expected $cmb_expected"
            health_warn=1
        else
            echo "❌ no combined AED files (${COMBINE_PREFIX}_cmb_*.nc) found"
            health_warn=1
        fi
    fi

    if [ $health_warn -eq 0 ]; then
        echo "✅ All health checks passed — outputs look healthy."
    else
        echo "⚠️  Health checks completed with warnings (see above)."
    fi
else
    echo "⚠️  outputs/ directory not found — skipping health check."
fi
echo ""

# Overall exit: non-zero if either step failed
if [ $exit_code -ne 0 ]; then
    exit $exit_code
fi
exit $combine_exit
