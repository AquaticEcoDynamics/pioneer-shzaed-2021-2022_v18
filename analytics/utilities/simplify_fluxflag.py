"""
Simplify fluxflag.prop to a single transect at the Pioneer Mouth.

Backs up the existing file to fluxflag.prop.bak (if no backup exists yet),
then writes a new fluxflag.prop where:
  - cells currently flagged 2 stay flagged 2 (CV-side of mouth)
  - cells currently flagged 3 stay flagged 3 (bay-side of mouth)
  - ALL OTHER cells become -1 (unflagged)

Result: flux.out reports ONLY the (2<->3) Pioneer Mouth interface as
col 3 (14 mesh edges). max_flreg auto-detects to 3 at startup, so the
file shrinks from 9 numerical columns per row down to 3.

Run BEFORE launching the next SCHISM simulation.
"""

from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np


RUN = Path("s:/Matt_Working/schism/P18_flood")
SRC = RUN / "fluxflag.prop"
BACKUP = RUN / "fluxflag.prop.bak"


def main():
    print(f"Reading {SRC}...")
    data = np.loadtxt(SRC, dtype=int)
    elem_id = data[:, 0]
    flag = data[:, 1]
    n_elem = len(elem_id)
    print(f"  {n_elem} elements")

    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
        print(f"  backed up original -> {BACKUP.name}")
    else:
        print(f"  backup already exists at {BACKUP.name}; leaving as is")

    # Cell counts before
    print(f"\nBefore:")
    for r in sorted(np.unique(flag)):
        print(f"  flag={r:>3}: {(flag==r).sum():>7,} cells")

    # New flag layout: keep 2 and 3, everything else -> -1
    new_flag = np.where((flag == 2) | (flag == 3), flag, -1)

    print(f"\nAfter:")
    for r in sorted(np.unique(new_flag)):
        print(f"  flag={r:>3}: {(new_flag==r).sum():>7,} cells")

    # Verify the 2<->3 interface count is preserved (should be 14)
    from collections import defaultdict
    import sys
    sys.path.insert(0, "s:/Matt_Working/schism/analytics")
    from budget import geometry
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    elements = hgrid["elements"]
    edge2elem = defaultdict(list)
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc = int(nc)
        for k in range(nc):
            a, b = ids[k], ids[(k+1)%nc]
            edge2elem[(min(a,b), max(a,b))].append(ei)
    n23 = 0
    for key, elems in edge2elem.items():
        if len(elems) != 2: continue
        f0, f1 = new_flag[elems[0]], new_flag[elems[1]]
        if {f0, f1} == {2, 3}:
            n23 += 1
    print(f"\n(2<->3) edges in new flag layout: {n23}  (expected 14)")
    assert n23 == 14, "Pioneer Mouth interface count changed unexpectedly!"

    # Verify no OTHER tracked interfaces remain
    other = 0
    for key, elems in edge2elem.items():
        if len(elems) != 2: continue
        f0, f1 = new_flag[elems[0]], new_flag[elems[1]]
        if f0 < 0 or f1 < 0 or f0 == f1: continue
        if abs(f0 - f1) == 1 and {f0, f1} != {2, 3}:
            other += 1
    print(f"Other tracked interfaces (|dflag|=1, both>=0, NOT 2-3): {other}  (expected 0)")
    assert other == 0, "Unexpected tracked interface besides (2,3)!"

    # Write new fluxflag.prop in the original format
    print(f"\nWriting new {SRC}...")
    with open(SRC, "w") as f:
        for i in range(n_elem):
            f.write(f"{elem_id[i]}\t{new_flag[i]}\n")
    print(f"  wrote {SRC} ({SRC.stat().st_size/1e6:.2f} MB)")

    print("\nDone. Next SCHISM run will report only the (2<->3) Pioneer Mouth")
    print("interface (col 3) in flux.out. max_flreg=3 auto-detected at startup.")
    print(f"To restore the original: cp {BACKUP.name} {SRC.name}")


if __name__ == "__main__":
    main()
