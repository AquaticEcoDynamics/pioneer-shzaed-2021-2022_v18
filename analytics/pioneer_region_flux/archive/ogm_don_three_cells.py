"""
Same as trc1_three_cells.py but for OGM_don.

Plot OGM_don concentration time series at the same three cells:
    cell A — at src 3 (element 50526)
    cell B — a cell tagged fluxflag region 1 (just east of src 3)
    cell C — a cell tagged fluxflag region 3, near lon 149.214 (Pioneer Mouth)
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core import geometry


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    out_path = Path(__file__).parent / "ogm_don_three_cells.png"
    START = np.datetime64("2021-04-01")
    PERIOD_DAYS = 14

    hgrid = geometry.read_hgrid_with_areas(run / "hgrid.gr3")
    flag = np.loadtxt(run / "fluxflag.prop", dtype=int)[:, 1]
    cx, cy = hgrid["centroids"][:, 0], hgrid["centroids"][:, 1]

    elem_A = 50526 - 1
    r1_cells = np.where(flag == 1)[0]
    elem_B = int(r1_cells[len(r1_cells) // 2])
    r3_cells = np.where(flag == 3)[0]
    elem_C = int(r3_cells[np.argmin(np.abs(cx[r3_cells] - 149.214))])

    cells = [
        ("A — at src 3 (elem 50526)", elem_A, "#dc2626"),
        (f"B — region 1 (elem {elem_B+1})", elem_B, "#1d4ed8"),
        (f"C — region 3 near 149.214 (elem {elem_C+1})", elem_C, "#16a34a"),
    ]
    for lab, ei, _ in cells:
        print(f"  {lab}: lon={cx[ei]:.4f}, lat={cy[ei]:.4f}, flag={flag[ei]}")

    elements = hgrid["elements"]
    cell_nodes = {}
    for lab, ei, _ in cells:
        nc, *ids = elements[ei]
        cell_nodes[ei] = [int(n) for n in ids[:int(nc)]]

    print("\nLoading scribed OGM_don across stacks...")
    da = geometry.open_scribed_concat(outputs, "OGM_don", n_stacks=PERIOD_DAYS + 1)
    print(f"  shape: {da.shape}, dims: {da.dims}")
    t_coord = da["time"].values
    _, uniq = np.unique(t_coord, return_index=True)
    if len(uniq) != len(t_coord):
        da = da.isel(time=np.sort(uniq))
    keep_t = (da["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    da = da.isel(time=keep_t)
    print(f"  time points after trim: {da.sizes['time']}")

    node_dim = next(d for d in da.dims if "node" in d.lower())
    layer_dim = next((d for d in da.dims
                      if "vgrid" in d.lower() or "layer" in d.lower()), None)

    series = {}
    for lab, ei, color in cells:
        nodes = cell_nodes[ei]
        sub = da.isel({node_dim: nodes})
        if layer_dim is not None:
            sub_surf = sub.isel({layer_dim: -1})
        else:
            sub_surf = sub
        sub_surf = sub_surf.where(np.abs(sub_surf) <= 1e30)
        ts = sub_surf.mean(dim=node_dim)
        ts_v = ts.compute().values
        series[lab] = (da["time"].values, ts_v, color)
        print(f"  {lab}: mean = {np.nanmean(ts_v):.3f}, "
              f"min = {np.nanmin(ts_v):.3f}, max = {np.nanmax(ts_v):.3f}")

    print(f"\nWriting plot to {out_path}")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for lab, (t, v, c) in series.items():
        ax.plot(t, v, color=c, lw=1.6, label=lab)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_ylabel("OGM_don concentration (surface layer)\nnode-vertex mean  (mmol N / m³)")
    ax.set_xlabel("Date (model time)")
    ax.set_title("OGM_don concentration at three cells\n"
                 "src 3 msource value ≈ 16-22 mmol N / m³",
                 fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
