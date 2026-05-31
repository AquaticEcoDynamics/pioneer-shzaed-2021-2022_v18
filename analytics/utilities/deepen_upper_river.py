"""
Increase bathymetric depth by 2.0 m for every node west of lon = 149.095
in hgrid.gr3 (and hgrid.ll, which is a sister copy in lat/lon).

SCHISM convention: hgrid.gr3 stores depth as POSITIVE-BELOW-MSL on each node
line ("node_id  x  y  depth"). So we add +2.0 to the existing depth values.
A node previously at depth = 1.0 (1 m below MSL) becomes depth = 3.0
(3 m below MSL) — i.e. the bed drops 2 m, the water column is now 2 m deeper.

Affects:
  - the river head where src #3 (Pioneer @ Dumbleton, elem 50526) sits —
    src #3's cell goes from a ~210 m² × 1 m = 210 m³ puddle into a ~210 m² × 3 m
    pool, which drops the per-timestep rat = Q·dt/bigv from ~12 to ~4 for a
    100 m³/s pulse.

Idempotent: writes the .gr3/.ll files in place, with first-run backups to
hgrid.gr3.preconfig / hgrid.ll.preconfig so the change is reversible.
"""

from __future__ import annotations
from pathlib import Path
import shutil

RUN = Path(__file__).parent
LON_CUTOFF = 149.095   # any node with x < this gets deepened
DEPTH_INCREASE = 2.0   # metres (positive = deeper below MSL)
TARGET_FILES = ["hgrid.gr3", "hgrid.ll"]


def deepen(path: Path):
    print(f"\n=== {path.name} ===")
    backup = path.with_suffix(path.suffix + ".preconfig")
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"  backed up original -> {backup.name}")
    else:
        print(f"  backup already exists at {backup.name} — leaving as is")

    # Read the file
    with open(path) as f:
        header = f.readline()                            # "hgrid.gr3" or similar
        counts = f.readline()                            # "n_elements n_nodes"
        n_elem, n_nodes = map(int, counts.split()[:2])
        print(f"  n_nodes = {n_nodes}, n_elements = {n_elem}")
        node_lines = [f.readline() for _ in range(n_nodes)]
        # Element connectivity block — pass through unchanged
        elem_lines = []
        for line in f:
            elem_lines.append(line)
        if len(elem_lines) < n_elem:
            print(f"  ⚠️ only read {len(elem_lines)} element lines, expected {n_elem}")

    # Process the node lines
    modified = 0
    new_node_lines = []
    depth_before_min, depth_before_max = float("inf"), float("-inf")
    depth_after_min,  depth_after_max  = float("inf"), float("-inf")
    for line in node_lines:
        parts = line.split()
        nid   = int(parts[0])
        x     = float(parts[1])
        y     = float(parts[2])
        d     = float(parts[3])
        if x < LON_CUTOFF:
            new_d = d + DEPTH_INCREASE
            modified += 1
            depth_before_min = min(depth_before_min, d)
            depth_before_max = max(depth_before_max, d)
            depth_after_min  = min(depth_after_min,  new_d)
            depth_after_max  = max(depth_after_max,  new_d)
        else:
            new_d = d
        # Preserve original-style formatting:  id  x  y  depth
        # The salt.ic / hgrid.gr3 typically use a node-id width of 5+ chars
        # and floats with ~8 decimals. We'll match the original style with
        # a generous %g for the depth.
        new_node_lines.append(
            f"{nid:<6d} {x:>15.8f} {y:>15.8f} {new_d:>15.8f}\n"
        )

    print(f"  nodes west of lon={LON_CUTOFF}: {modified} (of {n_nodes})")
    if modified > 0:
        print(f"  depth BEFORE: min={depth_before_min:.3f}, max={depth_before_max:.3f}")
        print(f"  depth AFTER : min={depth_after_min:.3f}, max={depth_after_max:.3f}")
    else:
        print(f"  no nodes to modify (nothing west of {LON_CUTOFF}?)")
        return

    # Write back
    with open(path, "w") as f:
        f.write(header)
        f.write(counts)
        f.writelines(new_node_lines)
        f.writelines(elem_lines)
    print(f"  wrote {path.name} ({path.stat().st_size/1e6:.2f} MB)")


def verify_src3():
    """Confirm the change took effect at the src #3 element."""
    print("\n=== Sanity: src #3 element 50526 vertex depths ===")
    h = Path(RUN / "hgrid.gr3")
    nodes_x, nodes_y, nodes_d = [], [], []
    with open(h) as f:
        f.readline()
        n_elem, n_nodes = map(int, f.readline().split()[:2])
        for _ in range(n_nodes):
            parts = f.readline().split()
            nodes_x.append(float(parts[1]))
            nodes_y.append(float(parts[2]))
            nodes_d.append(float(parts[3]))
        # Skip to element 50526
        for ei in range(50525):
            f.readline()
        elem_line = f.readline().split()
        nc = int(elem_line[1])
        vertex_ids = [int(x) - 1 for x in elem_line[2:2+nc]]
    print(f"  element 50526 has {nc} vertices at node ids "
          f"{[v+1 for v in vertex_ids]}")
    for v in vertex_ids:
        print(f"    node {v+1}: lon={nodes_x[v]:.5f}, lat={nodes_y[v]:.5f}, "
              f"depth={nodes_d[v]:.3f} m")


def main():
    print(f"Deepening nodes west of lon={LON_CUTOFF} by +{DEPTH_INCREASE} m")
    for fname in TARGET_FILES:
        p = RUN / fname
        if not p.exists():
            print(f"  ⚠️ {p} not found, skipping")
            continue
        deepen(p)
    verify_src3()
    print("\nDone.  To revert: copy hgrid.gr3.preconfig back over hgrid.gr3 "
          "(and hgrid.ll likewise).")


if __name__ == "__main__":
    main()
