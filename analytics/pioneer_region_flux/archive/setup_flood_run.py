"""
Create a P18_flood/ run directory by copying P18/ inputs (excluding outputs/),
then inject a flow pulse at src #3 (Pioneer @ Dumbleton) and mark it with
TRC_tr2 = 1.0 for tracking.

Pulse parameters:
  - Source: #3 (Pioneer @ Dumbleton, element 50526, vsource col 4 / 0-idx 3)
  - Q during pulse: 100 m^3/s constant
  - Window: day 3 -> day 7 of simulation
      = t in [259200 s, 604800 s]
      = 2021-04-04 00:00 -> 2021-04-08 00:00
  - Tracer: TRC_tr2 raised from 0 to 1.0 at src #3 during pulse only

src #1 and src #2 are left exactly as-is.
"""

from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np


SRC_DIR = Path("s:/Matt_Working/schism/P18")
DST_DIR = Path("s:/Matt_Working/schism/P18_flood")

PULSE_Q = 100.0                # m^3/s
PULSE_T_START = 3 * 86400.0    # 259,200 s
PULSE_T_END   = 7 * 86400.0    # 604,800 s
SRC_IDX_1   = 3                # src #3 (1-indexed)
SRC_IDX_0   = SRC_IDX_1 - 1    # 0-indexed

N_SRC = 14
N_AED = 17
# TRC_tr2 is AED-tracer #15 (1-indexed within AED list). Column in msource =
# 1 (time) + 2*n_src (T,S) + (k-1)*n_src + (src-1)   for AED tracer k.
TRC_TR2_K = 15
TRC_TR2_COL = 1 + 2 * N_SRC + (TRC_TR2_K - 1) * N_SRC + SRC_IDX_0   # 0-indexed
TRC_TR1_COL = 1 + 2 * N_SRC + (14 - 1) * N_SRC + SRC_IDX_0          # for cross-check


def copy_inputs():
    """Copy P18/ -> P18_flood/, skipping outputs/."""
    if DST_DIR.exists():
        raise SystemExit(f"{DST_DIR} already exists — refusing to overwrite. "
                          f"Remove it first or pick a different name.")
    print(f"Copying {SRC_DIR}/ -> {DST_DIR}/  (excluding outputs/)...")
    shutil.copytree(
        SRC_DIR, DST_DIR,
        ignore=shutil.ignore_patterns("outputs", "outputs/*"),
    )
    # Create empty outputs/ for the next run
    (DST_DIR / "outputs").mkdir(exist_ok=True)
    print(f"  done. Confirm: {len(list(DST_DIR.glob('*')))} entries in {DST_DIR}")


def modify_vsource():
    """Set src #3 to PULSE_Q m^3/s for t in [PULSE_T_START, PULSE_T_END]."""
    path = DST_DIR / "vsource.th"
    print(f"\nReading {path}...")
    arr = np.loadtxt(path)
    print(f"  shape: {arr.shape}")
    t = arr[:, 0]
    mask = (t >= PULSE_T_START) & (t <= PULSE_T_END)
    n_pulse = int(mask.sum())
    pre_min  = arr[mask, 1 + SRC_IDX_0].min()
    pre_max  = arr[mask, 1 + SRC_IDX_0].max()
    pre_mean = arr[mask, 1 + SRC_IDX_0].mean()
    print(f"  pulse rows: {n_pulse}  (t = {t[mask].min():.0f} .. {t[mask].max():.0f} s)")
    print(f"  src #3 BEFORE: min={pre_min:.3f}, mean={pre_mean:.3f}, max={pre_max:.3f}")
    arr[mask, 1 + SRC_IDX_0] = PULSE_Q
    print(f"  src #3 AFTER:  set to {PULSE_Q} m^3/s")
    # Sanity: confirm src 1 + 2 are unchanged
    s1 = arr[mask, 1 + 0]
    s2 = arr[mask, 1 + 1]
    print(f"  src #1 in pulse window: min={s1.min():.3f}, mean={s1.mean():.3f}, max={s1.max():.3f}  (untouched)")
    print(f"  src #2 in pulse window: min={s2.min():.3f}, mean={s2.mean():.3f}, max={s2.max():.3f}  (untouched)")
    # Save with same precision as original
    np.savetxt(path, arr, fmt="%.6f", delimiter="\t")
    print(f"  wrote {path}")


def modify_msource():
    """Set src #3 TRC_tr2 to 1.0 for t in [PULSE_T_START, PULSE_T_END]."""
    path = DST_DIR / "msource.th"
    print(f"\nReading {path}...")
    arr = np.loadtxt(path)
    print(f"  shape: {arr.shape}  (expect 1 + 19*14 = 267 cols)")
    t = arr[:, 0]
    mask = (t >= PULSE_T_START) & (t <= PULSE_T_END)
    print(f"  pulse rows in msource: {int(mask.sum())}  "
          f"(t = {t[mask].min():.0f} .. {t[mask].max():.0f} s)")

    pre_tr1 = arr[mask, TRC_TR1_COL]
    pre_tr2 = arr[mask, TRC_TR2_COL]
    print(f"  src #3 TRC_tr1 BEFORE: min={pre_tr1.min()}, mean={pre_tr1.mean()}, max={pre_tr1.max()}  (should be 1.0 always)")
    print(f"  src #3 TRC_tr2 BEFORE: min={pre_tr2.min()}, mean={pre_tr2.mean()}, max={pre_tr2.max()}  (should be 0.0 always)")
    arr[mask, TRC_TR2_COL] = 1.0
    post_tr2 = arr[mask, TRC_TR2_COL]
    print(f"  src #3 TRC_tr2 AFTER:  min={post_tr2.min()}, mean={post_tr2.mean()}, max={post_tr2.max()}  (now 1.0 in pulse)")

    # Confirm TRC_tr2 OUTSIDE the pulse window is still 0
    pre_tr2_outside = arr[~mask, TRC_TR2_COL]
    print(f"  src #3 TRC_tr2 OUTSIDE pulse: min={pre_tr2_outside.min()}, max={pre_tr2_outside.max()}  (should still be 0)")

    np.savetxt(path, arr, fmt="%.6f", delimiter="\t")
    print(f"  wrote {path}")


def main():
    copy_inputs()
    modify_vsource()
    modify_msource()
    print("\n=== Summary ===")
    print(f"Run dir:     {DST_DIR}")
    print(f"Pulse:       src #3 Q = {PULSE_Q} m^3/s for t in [day 3, day 7]")
    print(f"             = [{PULSE_T_START:.0f}, {PULSE_T_END:.0f}] s")
    print(f"             = [2021-04-04 00:00, 2021-04-08 00:00]")
    print(f"Tracer:      TRC_tr2 set to 1.0 at src #3 during pulse (only)")
    print(f"src #1/#2:   untouched")
    print(f"All other:   untouched")
    print()
    print("Next step: run SCHISM from this directory (e.g. via run_schism_vm.sh)")


if __name__ == "__main__":
    main()
