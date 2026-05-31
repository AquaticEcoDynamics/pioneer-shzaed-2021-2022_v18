"""
Global tr1 mass-conservation check.

Integrates TRC_tr1 × layer_ht × area over the ENTIRE model domain at each
timestep (not just the pioneer_estuary CV) and compares to cumulative
src #3 input. If global ≈ input, mass is conserved (and "missing" CV
mass is just outside the CV polygon). If global ≪ input, SCHISM/AED is
losing mass numerically.
"""

from __future__ import annotations
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry

CFG = active_run()
RUN = CFG.run_dir
OUT_PNG = CFG.out_dir / "global_tr1_check.png"
START = CFG.start
PERIOD_DAYS = CFG.period_days
def main():
    print("=== Loading hgrid ===")
    hgrid = geometry.read_hgrid_with_areas(RUN / "hgrid.gr3")
    elements = hgrid["elements"]
    areas_all = hgrid["areas"].astype(np.float64)
    n_elem = hgrid["n_elements"]
    n_nodes = hgrid["n_nodes"]
    print(f"  n_nodes={n_nodes}, n_elements={n_elem}")
    print(f"  total domain area = {areas_all.sum()/1e6:.2f} km²")

    # Build a vectorised node->element averaging for the FULL mesh.
    # We'll use per-element vertex lookups via padded arrays.
    nc_arr = np.zeros(n_elem, dtype=np.int8)
    elem_node_idx = np.zeros((n_elem, 4), dtype=np.int32)   # node indices, padded
    for ei in range(n_elem):
        nc, *ids = elements[ei]
        nc_arr[ei] = int(nc)
        for k, n in enumerate(ids[:int(nc)]):
            elem_node_idx[ei, k] = int(n)
        # pad: leave as 0 (we'll mask out)
    elem_node_w = np.zeros((n_elem, 4), dtype=np.float32)
    for k in range(4):
        elem_node_w[:, k] = np.where(nc_arr > k, 1.0 / nc_arr, 0.0)

    print("\n=== Loading ENV_layer_ht (full domain) ===")
    ds_cmb = geometry.open_cmb_concat(RUN / "outputs", n_stacks=PERIOD_DAYS + 1)
    t_cmb = ds_cmb["time"].values
    _, uc = np.unique(t_cmb, return_index=True)
    if len(uc) != len(t_cmb):
        ds_cmb = ds_cmb.isel(time=np.sort(uc))
    keep = (ds_cmb["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    ds_cmb = ds_cmb.isel(time=keep)
    lh = ds_cmb["ENV_layer_ht"]
    elem_dim = next(d for d in lh.dims if "face" in d.lower())
    layer_dim = next(d for d in lh.dims if "layer" in d.lower() or "vgrid" in d.lower())
    print(f"  ENV_layer_ht dims = {lh.dims}, shape = {lh.shape}")
    times = ds_cmb["time"].values
    t_sec = (times - times[0]) / np.timedelta64(1, "s")

    # Stream through stacks to limit memory: process each stack of tr1 + lh together
    print("\n=== Streaming tr1 stacks and integrating over whole domain ===")
    M_global = np.zeros(len(t_sec))            # mmol
    V_global = np.zeros(len(t_sec))            # m³ (water volume of full domain)
    n_stacks = PERIOD_DAYS + 1
    paths_tr1 = geometry.discover_scribed_stacks(RUN / "outputs", "TRC_tr1")[:n_stacks]
    paths_cmb = geometry.discover_cmb_stacks(RUN / "outputs")[:n_stacks]
    cursor = 0
    import xarray as xr
    for path_tr, path_cm in zip(paths_tr1, paths_cmb):
        ds_t = xr.open_dataset(path_tr, engine="h5netcdf")
        ds_c = xr.open_dataset(path_cm, engine="h5netcdf")
        da_t = ds_t["TRC_tr1"]
        da_l = ds_c["ENV_layer_ht"]
        # Trim to 14d window
        keep_t = (da_t["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        da_t = da_t.isel(time=keep_t)
        keep_l = (da_l["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        da_l = da_l.isel(time=keep_l)
        T_stack = min(da_t.sizes["time"], da_l.sizes["time"])
        if T_stack == 0:
            ds_t.close(); ds_c.close(); continue

        # tr1 dims: (time, node, layer)  ;  ENV_layer_ht dims: (time, face, layer)
        node_dim = next(d for d in da_t.dims if "node" in d.lower())
        layer_dim_s = next(d for d in da_t.dims
                           if "vgrid" in d.lower() or "layer" in d.lower())

        # Load this stack into memory (manageable per-stack)
        arr_t = da_t.isel(time=slice(0, T_stack)).transpose(
            "time", layer_dim_s, node_dim).values    # (T, L, N)
        arr_l = da_l.isel(time=slice(0, T_stack)).transpose(
            "time", layer_dim, elem_dim).values      # (T, L, E)
        # Defensive masks: tr1 is fine if finite and within fill range;
        # lh must be FINITE AND POSITIVE (negative/NaN means dry or invalid).
        arr_t = np.where(np.isfinite(arr_t) & (np.abs(arr_t) <= 1e30), arr_t, 0.0)
        arr_l = np.where(np.isfinite(arr_l) & (arr_l > 0), arr_l, 0.0)

        # Compute element-mean tr1 via vectorised gather + weighted sum
        # tr1_elem[t, l, e] = sum_k w[e,k] * arr_t[t, l, elem_node_idx[e,k]]
        # Shape: (T, L, E, 4) -- (T*L*E*4) = 96*6*130884*4*4B ~= 1.2 GB peak.
        # To keep memory tame, process in chunks of 16 timesteps.
        T, L, _ = arr_t.shape
        E = n_elem
        chunk = 16
        tr1_dot_lh_sum = np.zeros(T)   # mmol per timestep summed
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            sl = slice(t0, t1)
            # Gather: (T_chunk, L, E, 4)
            gathered = arr_t[sl, :, elem_node_idx]   # (T_chunk, L, E, 4)
            elem_vals = (gathered * elem_node_w).sum(axis=-1)   # (T_chunk, L, E)
            # Multiply by lh × area
            mass_per_cell = elem_vals * arr_l[sl] * areas_all[None, None, :]  # (T_chunk, L, E)
            tr1_dot_lh_sum[sl] = mass_per_cell.sum(axis=(1, 2))
            # Also V_global for this chunk
            vol_per_cell = arr_l[sl] * areas_all[None, None, :]   # (T_chunk, L, E)
            V_global[cursor + t0: cursor + t1] = vol_per_cell.sum(axis=(1, 2))

        M_global[cursor: cursor + T] = tr1_dot_lh_sum
        cursor += T
        print(f"  stack {path_tr.name}: T={T}, "
              f"M peak so far = {M_global[:cursor].max()/1e6:.3f} M mmol, "
              f"V peak so far = {V_global[:cursor].max()/1e6:.3f} M m³")
        ds_t.close(); ds_c.close()

    M_global = M_global[:cursor]
    V_global = V_global[:cursor]
    times = times[:cursor]
    t_sec = t_sec[:cursor]
    C_global = M_global / V_global

    # ---- Cumulative src #3 input ----
    vs = np.loadtxt(RUN / "vsource.th")
    t_vs = vs[:, 0]
    Q3 = vs[:, 3]   # src #3 = numpy col 3
    # tr1 conc at src #3 = 1.0 always; input rate (mmol/s) = Q3 × 1.0
    Q3_on = np.interp(t_sec, t_vs, Q3)
    F_in_t = Q3_on * 1.0    # mmol/s
    cum_in = np.concatenate(
        [[0], np.cumsum(0.5*(F_in_t[1:]+F_in_t[:-1]) * np.diff(t_sec))]
    )

    # ---- Summary ----
    print()
    print("=" * 70)
    print("=== GLOBAL MASS CONSERVATION CHECK FOR TRC_tr1 ===")
    print("=" * 70)
    print(f"Cum src #3 input over 14d        : {cum_in[-1]/1e6:12.3f} M mmol")
    print(f"Peak global tr1 mass in domain   : {M_global.max()/1e6:12.3f} M mmol  "
          f"(at {times[M_global.argmax()]})")
    print(f"Final global tr1 mass (day 14)   : {M_global[-1]/1e6:12.3f} M mmol")
    print(f"Peak global / cum input          : {M_global.max() / cum_in[-1] * 100:12.2f} %")
    print(f"Final global / cum input         : {M_global[-1] / cum_in[-1] * 100:12.2f} %")
    print()
    print(f"Domain total water volume range  : "
          f"[{V_global.min()/1e6:.2f}, {V_global.max()/1e6:.2f}] M m³")
    print(f"Domain total area                : {areas_all.sum()/1e6:.2f} km²")
    print(f"Mean global C_tr1                : {C_global.mean():.6f} mmol/m³")
    print(f"Peak global C_tr1                : {C_global.max():.6f} mmol/m³  "
          f"(at {times[C_global.argmax()]})")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axA, axB, axC = axes

    axA.plot(times, cum_in / 1e6, color="black", lw=2.0,
             label="Cum tr1 INPUT (src #3, ∫ Q × 1.0 dt)")
    axA.plot(times, M_global / 1e6, color="#dc2626", lw=2.0,
             label="GLOBAL tr1 mass (whole domain integral)")
    axA.set_ylabel("M (M mmol)")
    axA.grid(True, ls=":", alpha=0.4)
    axA.legend(loc="upper left", fontsize=10)
    axA.set_title("(a) Global tr1 mass conservation: domain integral vs cumulative input",
                  fontweight="bold", loc="left", fontsize=11)
    axA.axhline(0, color="#999", lw=0.4)

    axB.plot(times, (cum_in - M_global) / 1e6, color="#7c2d12", lw=2.0,
             label="cum INPUT − global mass  (= apparent loss)")
    axB.set_ylabel("Apparent loss (M mmol)")
    axB.grid(True, ls=":", alpha=0.4)
    axB.legend(loc="upper left", fontsize=10)
    axB.set_title("(b) Apparent mass loss: cum input minus global domain mass",
                  fontweight="bold", loc="left", fontsize=11)
    axB.axhline(0, color="#999", lw=0.4)

    axC.plot(times, V_global / 1e6, color="#0f766e", lw=1.6,
             label="Global domain water volume")
    axC.set_ylabel("V_global (M m³)")
    axC.set_xlabel("Date (model time)")
    axC.grid(True, ls=":", alpha=0.4)
    axC.legend(loc="upper left", fontsize=10)
    axC.set_title("(c) Domain-wide water volume (sanity check)",
                  fontweight="bold", loc="left", fontsize=11)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle("TRC_tr1 global mass conservation check (P18_flood)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
