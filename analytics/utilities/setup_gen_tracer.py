"""
Set up the GEN tracer inputs for P18_flood:

  1. Create GEN_hvar_1.ic — zeros everywhere, format identical to salt.ic
     except the per-node tracer-value column is 0.0.

  2. Modify msource.th — insert 14 new GEN_1 columns between the T/S
     blocks and the AED block.
       - src #3 (Pioneer Dumbleton, col 4 of the 14-source layout): GEN_1 = 1.0
       - all other 13 sources: GEN_1 = -9999.0 (use ambient = 0)
     Total columns: 267 -> 281.

  Run this after USE_GEN=ON binary is in place but BEFORE launching the
  new SCHISM simulation.
"""

from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np


RUN = Path("s:/Matt_Working/schism/P18_flood")
SALT_IC = RUN / "salt.ic"
GEN_IC  = RUN / "GEN_hvar_1.ic"
MSOURCE = RUN / "msource.th"

N_SRC = 14
SRC3_IDX_1 = 3                # 1-indexed source #
SRC3_IDX_0 = SRC3_IDX_1 - 1   # 0-indexed within an N_SRC block

GEN_VAL_AT_SRC3 = 1.0
GEN_VAL_OTHER   = -9999.0     # "use ambient" sentinel


def make_gen_ic():
    """Build GEN_hvar_1.ic from salt.ic template (same node + element table,
    only the value column zeroed)."""
    print(f"\n=== Creating {GEN_IC.name} from {SALT_IC.name} ===")
    with open(SALT_IC) as fin, open(GEN_IC, "w") as fout:
        header1 = fin.readline()   # original first line, e.g. "salt.ic"
        header2 = fin.readline()   # "n_elements n_nodes"
        fout.write("GEN_hvar_1.ic\n")
        fout.write(header2)
        n_el, n_node = map(int, header2.split())
        print(f"  n_elements={n_el}, n_nodes={n_node}")

        # Node block: id, lon, lat, value -> replace value with 0.0
        n_nonzero_value_lines = 0
        for _ in range(n_node):
            parts = fin.readline().split()
            # Preserve original whitespace style (5-char padding for id, then space-separated floats)
            new_line = f"{int(parts[0]):<6d} {float(parts[1]):14.6f} {float(parts[2]):14.6f}  0.0000000e+00\n"
            fout.write(new_line)

        # Element block: pass through unchanged
        for _ in range(n_el):
            fout.write(fin.readline())

    print(f"  wrote {GEN_IC}")
    # Sanity check
    with open(GEN_IC) as f:
        n_lines = sum(1 for _ in f)
    print(f"  total lines: {n_lines} (expected {2 + n_node + n_el} = 2 + {n_node} + {n_el})")


def modify_msource():
    """Insert 14 GEN_1 columns into msource.th between T/S and AED blocks."""
    print(f"\n=== Modifying {MSOURCE.name} ===")
    arr = np.loadtxt(MSOURCE)
    print(f"  current shape: {arr.shape}")
    n_rows, n_cols = arr.shape
    expected_old = 1 + 2 * N_SRC + 17 * N_SRC
    assert n_cols == expected_old, f"expected {expected_old} cols, got {n_cols}"

    # Backup the original
    backup = MSOURCE.with_suffix(".th.bak")
    if not backup.exists():
        shutil.copy(MSOURCE, backup)
        print(f"  backed up original -> {backup.name}")

    # Layout:
    #   col 0          : time
    #   cols 1..14     : T for src1..src14
    #   cols 15..28    : S for src1..src14
    #   --- insert GEN_1 here (14 cols) ---
    #   cols 29..266   : 17 AED tracers x 14 sources (each contiguous block of 14)
    # New layout:
    #   col 0          : time
    #   cols 1..14     : T
    #   cols 15..28    : S
    #   cols 29..42    : GEN_1 for src1..src14   (NEW)
    #   cols 43..280   : AED (shifted right by N_SRC)
    n_new_cols = N_SRC
    new_arr = np.empty((n_rows, n_cols + n_new_cols), dtype=arr.dtype)
    new_arr[:, 0]      = arr[:, 0]          # time
    new_arr[:, 1:29]   = arr[:, 1:29]       # T + S blocks
    # GEN_1 block — fill with sentinel then overwrite src #3
    new_arr[:, 29:29 + N_SRC] = GEN_VAL_OTHER
    new_arr[:, 29 + SRC3_IDX_0] = GEN_VAL_AT_SRC3
    # AED tracers, shifted
    new_arr[:, 29 + N_SRC:] = arr[:, 29:]

    print(f"  new shape: {new_arr.shape}")
    print(f"  GEN_1 src #3 (col {29 + SRC3_IDX_0}): "
          f"value = {new_arr[0, 29 + SRC3_IDX_0]}")
    other_idxs = [29 + i for i in range(N_SRC) if i != SRC3_IDX_0]
    print(f"  GEN_1 other sources (cols {other_idxs[0]}..{other_idxs[-1]}): "
          f"value = {new_arr[0, other_idxs[0]]} (all 13 of them)")

    # Sanity check unchanged blocks
    assert np.array_equal(new_arr[:, 0:29], arr[:, 0:29]), "T/S block was perturbed!"
    assert np.array_equal(new_arr[:, 29 + N_SRC:], arr[:, 29:]), "AED block was perturbed!"

    # Write back with the same format as the original (np.savetxt uses %.6f tab-separated
    # in the existing setup_flood_run.py — matching that)
    np.savetxt(MSOURCE, new_arr, fmt="%.6f", delimiter="\t")
    print(f"  wrote {MSOURCE} ({MSOURCE.stat().st_size/1e6:.1f} MB)")


def main():
    make_gen_ic()
    modify_msource()
    print("\n=== DONE ===")
    print("Next manual edits required:")
    print("  - param.nml:  gen_wsett = 0.0, lev_tr_source(3) = 0, add iof_gen(1)=1 to &SCHOUT")
    print("  - bctides.in: change '73 3 0 1 2 1' to '73 3 0 1 2 0 1'")
    print("  - aed.nml:    depress_clutch = .TRUE.")


if __name__ == "__main__":
    main()
