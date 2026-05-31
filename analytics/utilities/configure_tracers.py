"""
Configure the four "tag" tracers GEN_1, TRC_tr1, TRC_tr2, TRC_tr3 for the
002_P18_flood run, per this spec:

  GEN_1    : 1 at src #3 (Pioneer @ Dumbleton) only; 0 at all other sources;
             0 at the ocean BC (bctides.in flag = 0 + IC = 0)
  TRC_tr1  : same as GEN_1 — 1 at src #3 only; 0 elsewhere
  TRC_tr2  : 0 at src #3; 1 at every NON-Pioneer river source; 0 at ocean BC
  TRC_tr3  : 0 at every river source; 1 at ocean BC (already set in AED_16.th)

Idempotent: writes msource.th in place, after backing up to msource.th.preconfig
on the first run.
"""

from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np

RUN = Path(__file__).parent

N_SRC = 14
N_AED = 17
N_GEN = 1
SRC_PIONEER_0IDX = 2   # src #3 in 1-indexed = 0-idx 2 within each tracer's 14-col block

# AED tracer registration order (k=1..17)
AED_TRACER_K = {
    "NCS_ss1": 1, "OXY_oxy": 2, "NIT_amm": 3, "NIT_nit": 4,
    "PHS_frp": 5, "PHS_frp_ads": 6, "OGM_doc": 7, "OGM_poc": 8,
    "OGM_don": 9, "OGM_pon": 10, "OGM_dop": 11, "OGM_pop": 12,
    "PHY_mixed": 13, "TRC_tr1": 14, "TRC_tr2": 15, "TRC_tr3": 16, "TRC_age": 17,
}

# Column index helpers (0-indexed into msource.th row)
def col_GEN(g, s):
    """GEN_1..GEN_n block. g=0 => GEN_1."""
    return 1 + 2*N_SRC + g*N_SRC + s
def col_AED(k, s):
    """AED tracer k (1..17), source s (0..13)."""
    return 1 + 2*N_SRC + N_GEN*N_SRC + (k-1)*N_SRC + s


def main():
    msrc_path = RUN / "msource.th"
    backup    = RUN / "msource.th.preconfig"

    print(f"Reading {msrc_path} ...")
    arr = np.loadtxt(msrc_path)
    print(f"  shape: {arr.shape}")
    expected_cols = 1 + (2 + N_GEN + N_AED) * N_SRC
    assert arr.shape[1] == expected_cols, \
        f"column count mismatch: got {arr.shape[1]}, expected {expected_cols}"

    if not backup.exists():
        shutil.copy(msrc_path, backup)
        print(f"  backed up original -> {backup.name}")
    else:
        print(f"  backup already exists at {backup.name} — preserving it")

    other_srcs = [s for s in range(N_SRC) if s != SRC_PIONEER_0IDX]

    # ---- GEN_1 ----
    print("\nGEN_1: src#3 = 1.0, all other sources = 0.0")
    arr[:, col_GEN(0, SRC_PIONEER_0IDX)] = 1.0
    for s in other_srcs:
        arr[:, col_GEN(0, s)] = 0.0

    # ---- TRC_tr1 (AED tracer k=14) ----
    print("TRC_tr1: src#3 = 1.0, all other sources = 0.0")
    arr[:, col_AED(14, SRC_PIONEER_0IDX)] = 1.0
    for s in other_srcs:
        arr[:, col_AED(14, s)] = 0.0

    # ---- TRC_tr2 (AED tracer k=15) — INVERT: marker for non-Pioneer rivers ----
    print("TRC_tr2: src#3 = 0.0, all OTHER 13 sources = 1.0")
    arr[:, col_AED(15, SRC_PIONEER_0IDX)] = 0.0
    for s in other_srcs:
        arr[:, col_AED(15, s)] = 1.0

    # ---- TRC_tr3 (AED tracer k=16) — 0 at all sources (ocean BC handles it) ----
    print("TRC_tr3: 0.0 at every source (ocean BC = 1.0 from AED_16.th)")
    for s in range(N_SRC):
        arr[:, col_AED(16, s)] = 0.0

    # Sanity print: first-row values for each tracer
    print("\n=== After update (t=0 row) ===")
    def show(label, cols):
        vals = arr[0, cols]
        print(f"  {label:14s}: {' '.join(f'{v:6.1f}' for v in vals)}")
    show("GEN_1",   [col_GEN(0, s) for s in range(N_SRC)])
    show("TRC_tr1", [col_AED(14, s) for s in range(N_SRC)])
    show("TRC_tr2", [col_AED(15, s) for s in range(N_SRC)])
    show("TRC_tr3", [col_AED(16, s) for s in range(N_SRC)])

    # Also confirm time-axis constancy (these should all be flat across time now)
    print()
    for name, k in [("GEN_1", None), ("TRC_tr1", 14), ("TRC_tr2", 15), ("TRC_tr3", 16)]:
        if name == "GEN_1":
            block = arr[:, col_GEN(0, 0):col_GEN(0, 0)+N_SRC]
        else:
            block = arr[:, col_AED(k, 0):col_AED(k, 0)+N_SRC]
        time_std = block.std(axis=0).max()
        print(f"  {name}: max std over time across the 14 src columns = {time_std:.3e} "
              f"(0 means flat in time)")

    print(f"\nWriting {msrc_path} ({msrc_path.stat().st_size/1e6:.1f} MB -> ", end="", flush=True)
    np.savetxt(msrc_path, arr, fmt="%.6f", delimiter="\t")
    print(f"{msrc_path.stat().st_size/1e6:.1f} MB)")

    # --- Verify ocean BCs ---
    print("\n=== Ocean BCs (AED_*.th — should already be correct per spec) ===")
    expectations = {"AED_14.th": (0.0, "TRC_tr1"),
                    "AED_15.th": (0.0, "TRC_tr2"),
                    "AED_16.th": (1.0, "TRC_tr3")}
    for fname, (exp_val, label) in expectations.items():
        f = RUN / fname
        vals = np.loadtxt(f)[:, 1:]
        ok = np.allclose(vals, exp_val)
        status = "✓" if ok else "✗ FIX NEEDED"
        print(f"  {fname} ({label} ocean BC): mean={vals.mean():.3f}, "
              f"expected={exp_val}  {status}")

    print("\nDone.  To revert, copy msource.th.preconfig over msource.th.")


if __name__ == "__main__":
    main()
