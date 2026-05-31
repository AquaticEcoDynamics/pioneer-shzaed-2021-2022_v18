"""
Per-variable diagnostic plots (20 PNGs, one per variable).

For every model state variable (T, S, + 17 AED, + VOL), produces a 3-panel figure:

Panel a) Cumulative flux time series, 6 lines:
    BLACK   — src 3 incoming mass        (∫ Q_src3 × c_src3 dt)
    GREY    — flux.out row 2 (2 -> 1)    (∫ raw row 2 dt)
    RED     — src 1 incoming mass        (∫ Q_src1 × c_src1 dt)
    ORANGE  — src 2 incoming mass        (∫ Q_src2 × c_src2 dt)
    PINK    — flux.out row 4 (4 -> 3)    (∫ raw row 4 dt)
    BLUE    — flux.out row 3 (3 -> 2)    (∫ raw row 3 dt)  [Pioneer Mouth]

Panel b) Concentration at 3 cells (surface layer):
    A — at src 3              (element 50526)
    B — a region 1 cell       (just east of src 3)
    C — region 3 near 149.214 (Pioneer Mouth)
    (For VOL: panel b is replaced by a "no concentration" placeholder.)

Panel c) Volumetrically-integrated value over the pioneer_estuary control
    volume (Σ_elem Σ_layer  value × layer_ht × area).
    For VOL: total water volume of the CV in m³.
    For other variables: native units × m³ (e.g. mmol N for AED tracers,
    deg C × m³ for Heat, psu × m³ for Salinity).

All raw sign conventions kept (no flipping). 14-day window. Native units.

Output: a subfolder `per_variable_plots_flood/` next to this script with 20 PNGs.
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
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from run_config import active_run
from core import geometry
from core.boundary_fluxes import parse_flux_out
from core.point_sources import (
    parse_source_sink, parse_vsource, parse_msource,
)


# ---- variables to loop over (same order as flux.out) ----
# "VOL" is a synthetic entry handled specially: no scribed file, no
# concentration panel; uses flux.out's VOL tracer and the CV's total volume.
# "GEN_1" was added when the run was reconfigured with USE_GEN=ON and a
# single GEN tracer mirroring TRC_tr1's setup (msource=1.0 at src#3).
VARIABLES = [
    "Heat", "Salinity",
    "NCS_ss1", "OXY_oxy",
    "NIT_amm", "NIT_nit",
    "PHS_frp", "PHS_frp_ads",
    "OGM_doc", "OGM_poc", "OGM_don", "OGM_pon", "OGM_dop", "OGM_pop",
    "PHY_mixed",
    "TRC_tr1", "TRC_tr2", "TRC_tr3", "TRC_age",
    "GEN_1",
    "VOL",
]
GEN_TRACER_ORDER = ["GEN_1"]

# Ocean BC file lookup. AED_<k>.th index k = AED state-var registration order
# (1 = NCS_ss1, 2 = OXY_oxy, 3 = NIT_amm, ..., 17 = TRC_age).
OCEAN_BC_FILE = {
    "Heat":         "TEM_1.th",
    "Salinity":     "bctides.in",      # constant 35.0 (S line under '!S')
    "NCS_ss1":      "AED_1.th",
    "OXY_oxy":      "AED_2.th",
    "NIT_amm":      "AED_3.th",
    "NIT_nit":      "AED_4.th",
    "PHS_frp":      "AED_5.th",
    "PHS_frp_ads":  "AED_6.th",
    "OGM_doc":      "AED_7.th",
    "OGM_poc":      "AED_8.th",
    "OGM_don":      "AED_9.th",
    "OGM_pon":      "AED_10.th",
    "OGM_dop":      "AED_11.th",
    "OGM_pop":      "AED_12.th",
    "PHY_mixed":    "AED_13.th",
    "TRC_tr1":      "AED_14.th",
    "TRC_tr2":      "AED_15.th",
    "TRC_tr3":      "AED_16.th",
    "TRC_age":      "AED_17.th",
    "GEN_1":        "(bctides.in flag=0)",   # ambient (zero) from IC
}
SALINITY_OCEAN_BC_VALUE = 35.0

# scribe filename root for each variable (some differ from the variable name)
SCRIBE_NAME = {
    "Heat":      "temperature",
    "Salinity":  "salinity",
}
# AED names: same as variable name
for v in VARIABLES:
    SCRIBE_NAME.setdefault(v, v)

# Units shown in plot labels (best-effort, native units)
UNITS_PER_VOL = {
    "Heat":     "deg C",
    "Salinity": "psu",
}
# AED tracers — mmol/m^3 in concentration; flux.out cum in m^3 * mmol/m^3 = mmol
DEFAULT_CONC_UNIT = "mmol/m^3"


def cumtrap(rate, t_sec):
    if len(t_sec) < 2:
        return np.zeros_like(rate)
    dt = np.diff(t_sec)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    return np.concatenate([[0.0], np.cumsum(incr)])


def main():
    CFG = active_run()
    run = CFG.run_dir
    outputs = run / "outputs"
    # Output directory — bumped so this re-run doesn't clobber the older one
    out_dir = CFG.out_dir / "per_variable_plots_simple_fluxflag"
    out_dir.mkdir(parents=True, exist_ok=True)
    START = np.datetime64("2021-04-01")
    PERIOD_DAYS = 21

    # ---- mesh + fluxflag ----
    print("Loading hgrid + fluxflag...")
    hgrid = geometry.read_hgrid_with_areas(run / "hgrid.gr3")
    flag = np.loadtxt(run / "fluxflag.prop", dtype=int)[:, 1]
    cx, cy = hgrid["centroids"][:, 0], hgrid["centroids"][:, 1]

    # 3 cells:
    #   A = at src 3 (elem 50526)
    #   B = mid-estuary (closest cell to lon ~149.15, in the wet channel)
    #   C = near the (2↔3) mouth transect (closest flag=3 cell to lon 149.214,
    #       or fallback to nearest flag=2 cell if flag=3 absent)
    elem_A = 50526 - 1
    # B: pick a wet cell at lon ~ 149.15 (channel midway), restricted to
    # cells with positive bathymetric depth so it's actually water.
    depth_node = hgrid["depth"]
    depth_elem = np.array([
        depth_node[hgrid["elements"][ei][1:int(hgrid["elements"][ei][0])+1]].mean()
        for ei in range(hgrid["n_elements"])
    ])
    wet_band = (np.abs(cx - 149.15) < 0.005) & (depth_elem > 0.5) & (np.abs(cy + 21.142) < 0.015)
    if wet_band.any():
        elem_B = int(np.where(wet_band)[0][np.argmin(np.abs(cx[wet_band] - 149.15))])
    else:
        elem_B = elem_A   # fallback
    # C: nearest flag=3 cell (mouth) — else fall back to nearest flag=2
    r3_cells = np.where(flag == 3)[0]
    if len(r3_cells):
        elem_C = int(r3_cells[np.argmin(np.abs(cx[r3_cells] - 149.214))])
        c_label = f"C - flag=3 near 149.214 (elem {elem_C+1})"
    else:
        r2_cells = np.where(flag == 2)[0]
        elem_C = int(r2_cells[np.argmin(np.abs(cx[r2_cells] - 149.214))]) if len(r2_cells) else elem_A
        c_label = f"C - flag=2 near 149.214 (elem {elem_C+1})"
    cells = [
        ("A - at src 3 (elem 50526)",                       elem_A, "black"),
        (f"B - mid-estuary (elem {elem_B+1})",              elem_B, "#6b7280"),
        (c_label,                                            elem_C, "#1d4ed8"),
    ]
    elements = hgrid["elements"]
    cell_nodes = {}
    for _, ei, _ in cells:
        nc, *ids = elements[ei]
        cell_nodes[ei] = [int(n) for n in ids[:int(nc)]]
    # union of all nodes we'll need across all cells
    all_nodes = sorted(set(n for ns in cell_nodes.values() for n in ns))

    # ---- vsource + msource ----
    print("Loading vsource.th + msource.th...")
    src_elems = parse_source_sink(run / "source_sink.in")
    n_src = len(src_elems)
    t_vs, Q = parse_vsource(run / "vsource.th")
    t_ms, conc_aed = parse_msource(run / "msource.th", n_src,
                                   gen_tracer_order=GEN_TRACER_ORDER)
    raw_ms = np.loadtxt(run / "msource.th")
    T_all = raw_ms[:, 1:1 + n_src]
    S_all = raw_ms[:, 1 + n_src:1 + 2 * n_src]
    msource = {"Heat": T_all, "Salinity": S_all, **conc_aed}

    period_s = PERIOD_DAYS * 86400.0
    mvs = t_vs <= period_s
    t_vs = t_vs[mvs]
    Q = Q[mvs]
    src_times = START + (t_vs * 1e9).astype("timedelta64[ns]")

    # ---- flux.out (parse once, reuse) ----
    # n_regions auto-detected from the file (None) — handles both the old
    # 9-region layout and the new simplified mouth-only (max_flreg=3) layout.
    print("Parsing flux.out (slow)...")
    flux_da = parse_flux_out(outputs / "flux.out", n_regions=None)
    t_flux_d = flux_da["time_days"].values
    keep_fx = t_flux_d <= PERIOD_DAYS
    t_flux_d = t_flux_d[keep_fx]
    flux_times = START + (t_flux_d * 86400 * 1e9).astype("timedelta64[ns]")
    t_flux_s = t_flux_d * 86400.0
    available_regions = set(int(r) for r in flux_da["region"].values)
    print(f"  flux.out covers regions: {sorted(available_regions)}")

    # ---- precompute ocean BC mean per variable ----
    print("\nPrecomputing ocean BC mean per variable...")
    ocean_bc_mean = {}
    ocean_bc_label = {}
    for var in VARIABLES:
        if var == "VOL":
            ocean_bc_mean[var] = None
            ocean_bc_label[var] = ""
            continue
        if var == "GEN_1":
            # Ocean BC type = 0 in bctides.in → ambient = 0 from IC
            ocean_bc_mean[var] = 0.0
            ocean_bc_label[var] = OCEAN_BC_FILE[var]
            print(f"  {var:12s}  {OCEAN_BC_FILE[var]}   mean = 0.0 (ambient)")
            continue
        fname = OCEAN_BC_FILE[var]
        ocean_bc_label[var] = fname
        if var == "Salinity":
            ocean_bc_mean[var] = SALINITY_OCEAN_BC_VALUE
            continue
        fpath = run / fname
        if not fpath.exists():
            print(f"  WARN: {fname} not found — skipping ocean BC marker for {var}")
            ocean_bc_mean[var] = None
            continue
        try:
            arr = np.loadtxt(fpath)
            # Col 0 = time (s). Take mean of remaining cols within 14-day window.
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            t_bc = arr[:, 0]
            vals = arr[:, 1:]
            mask = t_bc <= PERIOD_DAYS * 86400.0
            if mask.sum() < 1:
                mask[:] = True
            mean_val = float(vals[mask].mean())
            ocean_bc_mean[var] = mean_val
            print(f"  {var:12s}  {fname:14s}  mean = {mean_val:.4f}")
        except Exception as e:
            print(f"  WARN: failed to read {fname}: {e}")
            ocean_bc_mean[var] = None

    # ---- panel-C precompute: CV element list + node->element averaging op ----
    print("\nLoading CV element list + building averaging operator...")
    cv_path = CFG.cv_elements
    cv_ids_1idx = []
    with open(cv_path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                cv_ids_1idx.append(int(line))
            except ValueError:
                pass
    cv_elements_arr = np.array(cv_ids_1idx, dtype=int) - 1   # 0-indexed
    cv_area = hgrid["areas"][cv_elements_arr].astype(np.float64)
    n_cv = len(cv_elements_arr)

    cv_nodes_set = set()
    cv_elem_node_ids = []
    for ei in cv_elements_arr:
        nc, *ids = elements[ei]
        nodes_e = [int(n) for n in ids[:int(nc)]]
        cv_nodes_set.update(nodes_e)
        cv_elem_node_ids.append((nc, nodes_e))
    cv_nodes_union = np.array(sorted(cv_nodes_set), dtype=int)
    node_pos_cv = {n: i for i, n in enumerate(cv_nodes_union.tolist())}
    n_cv_nodes = len(cv_nodes_union)
    print(f"  CV: {n_cv} elements, {n_cv_nodes} unique vertex nodes, "
          f"area = {cv_area.sum():,.0f} m² ({cv_area.sum()/1e6:.2f} km²)")

    # Sparse averaging matrix A: elem_vals = A @ node_vals  (axis 0 = node)
    rows, cols, data = [], [], []
    for k, (nc, nodes_e) in enumerate(cv_elem_node_ids):
        for n in nodes_e:
            rows.append(k)
            cols.append(node_pos_cv[n])
            data.append(1.0 / nc)
    A_avg = csr_matrix(
        (np.asarray(data, dtype=np.float32),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_cv, n_cv_nodes), dtype=np.float32,
    )
    print(f"  averaging matrix A: shape {A_avg.shape}, nnz {A_avg.nnz}")

    # ---- load layer thickness across stacks, subset to CV elements ----
    # Prefer ENV_layer_ht from aed_data_cmb (face-centred). If it's missing
    # (e.g. when AED diagnostics weren't saved, or depress_clutch was on),
    # fall back to deriving lh from zCoordinates (node-centred SCHISM scribe).
    print("\nLoading layer thickness for CV...")
    ds_cmb = geometry.open_cmb_concat(outputs, n_stacks=PERIOD_DAYS + 1)
    t_cmb = ds_cmb["time"].values
    _, uc = np.unique(t_cmb, return_index=True)
    if len(uc) != len(t_cmb):
        ds_cmb = ds_cmb.isel(time=np.sort(uc))
    keep_c = (ds_cmb["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    ds_cmb = ds_cmb.isel(time=keep_c)

    if "ENV_layer_ht" in ds_cmb.data_vars:
        print("  using ENV_layer_ht from aed_data_cmb (face-centred)")
        lh_full = ds_cmb["ENV_layer_ht"]
        elem_dim_cmb = next(d for d in lh_full.dims if "face" in d.lower())
        layer_dim_cmb = next(d for d in lh_full.dims
                             if "layer" in d.lower() or "vgrid" in d.lower())
        lh_cv_da = lh_full.isel({elem_dim_cmb: cv_elements_arr}).compute()
        lh_cv_arr = lh_cv_da.transpose("time", layer_dim_cmb, elem_dim_cmb).values.astype(np.float64)
        lh_cv_arr = np.where(np.isfinite(lh_cv_arr) & (lh_cv_arr > 0), lh_cv_arr, 0.0)
        t_cmb_aligned = lh_cv_da["time"].values
    else:
        print("  ENV_layer_ht missing — falling back to zCoordinates → "
              "per-element layer thickness via vertex-average")
        # Load zCoordinates (per node, per layer) over the CV's vertex nodes
        da_z = geometry.open_scribed_concat(outputs, "zCoordinates",
                                             n_stacks=PERIOD_DAYS + 1)
        tt = da_z["time"].values
        _, uq = np.unique(tt, return_index=True)
        if len(uq) != len(tt):
            da_z = da_z.isel(time=np.sort(uq))
        keep_z = (da_z["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        da_z = da_z.isel(time=keep_z)
        node_dim = next(d for d in da_z.dims if "node" in d.lower())
        layer_dim_z = next(d for d in da_z.dims
                            if "vgrid" in d.lower() or "layer" in d.lower())
        z_cv_nodes = da_z.isel({node_dim: cv_nodes_union.tolist()}).compute()
        z_arr = z_cv_nodes.transpose("time", layer_dim_z, node_dim).values
        z_arr = np.where(np.isfinite(z_arr) & (np.abs(z_arr) <= 1e30), z_arr, 0.0)
        # Per-node layer thickness = diff over layer axis
        dz_nodes = np.diff(z_arr, axis=1)        # (T, L-1, n_cv_nodes)
        dz_nodes = np.where(dz_nodes > 0, dz_nodes, 0.0)
        T_z, Lm1, N_z = dz_nodes.shape
        # Average node values to elements via the sparse A_avg operator
        dz_flat = dz_nodes.reshape(T_z * Lm1, N_z).astype(np.float32)
        dz_elem_flat = dz_flat @ A_avg.T
        lh_cv_arr = dz_elem_flat.reshape(T_z, Lm1, n_cv).astype(np.float64)
        t_cmb_aligned = z_cv_nodes["time"].values

    print(f"  lh_cv shape (T, L, n_cv) = {lh_cv_arr.shape}")

    # Total CV water volume time series (Σ_elem Σ_layer layer_ht × area)
    vol_total_t = (lh_cv_arr * cv_area[None, None, :]).sum(axis=(1, 2))   # (T,) m³
    print(f"  CV water-volume mean = {vol_total_t.mean():,.0f} m³  "
          f"(min {vol_total_t.min():,.0f}, max {vol_total_t.max():,.0f})")

    # ---- domain-wide precompute (right y-axis of panel C) ----
    # Node-centred dz from zCoordinates over the FULL mesh + node-share areas.
    # Memory: T × (L-1) × n_nodes × 4 bytes. For 21 d × 96 ts × 5 × 67762 ~ 2.6 GB
    # at float32. Streamed per stack and converted to float32 to keep peak low.
    print("\nLoading zCoordinates across stacks for full-domain mass integration...")
    A_node_full = np.zeros(hgrid["n_nodes"], dtype=np.float64)
    for ei in range(hgrid["n_elements"]):
        nc, *ids = elements[ei]
        nc = int(nc)
        share = hgrid["areas"][ei] / nc
        for n in ids[:nc]:
            A_node_full[int(n)] += share
    print(f"  Σ A_node = {A_node_full.sum()/1e6:.2f} km² "
          f"(should equal Σ A_elem = {hgrid['areas'].sum()/1e6:.2f} km²)")

    paths_z = geometry.discover_scribed_stacks(outputs, "zCoordinates")[:PERIOD_DAYS+1]
    dz_full_stacks, t_full_stacks = [], []
    import xarray as xr_local
    for p in paths_z:
        ds_z = xr_local.open_dataset(p, engine="h5netcdf")
        da_z = ds_z["zCoordinates"]
        keep = (da_z["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
        if not keep.any():
            ds_z.close(); continue
        da_z = da_z.isel(time=keep)
        z = da_z.transpose("time", "nSCHISM_vgrid_layers",
                            "nSCHISM_hgrid_node").values
        z = np.where(np.isfinite(z) & (np.abs(z) <= 1e30), z, 0.0)
        dz = np.diff(z, axis=1).astype(np.float32)        # (T, L-1, n_nodes)
        dz = np.where(dz > 0, dz, 0.0)
        dz_full_stacks.append(dz)
        t_full_stacks.append(da_z["time"].values)
        ds_z.close()
    dz_full = np.concatenate(dz_full_stacks, axis=0)       # (T_total, L-1, n_nodes)
    t_domain = np.concatenate(t_full_stacks)
    print(f"  dz_full shape (T, L-1, n_nodes) = {dz_full.shape}, "
          f"~{dz_full.nbytes/1e9:.2f} GB")

    vol_domain_t = (dz_full * A_node_full[None, None, :]).sum(axis=(1, 2))  # (T,) m³
    print(f"  Domain water-volume mean = {vol_domain_t.mean()/1e6:.0f} M m³ "
          f"(min {vol_domain_t.min()/1e6:.0f}, max {vol_domain_t.max()/1e6:.0f})")

    def _panelC_for_scribed(da_s, sc_name, node_dim, layer_dim):
        """Compute volume-integrated time series of a scribed variable.
        Returns (times[T], cv_mass[T], domain_mass[T]).  domain_mass may be
        None if the variable is 2D (no layer dim)."""
        da_cv = da_s.isel({node_dim: cv_nodes_union.tolist()}).compute()
        arr = da_cv.values
        dims = list(da_cv.dims)
        t_ax = dims.index(next(d for d in dims if d.lower() == "time"))
        n_ax = dims.index(node_dim)
        if layer_dim is not None and layer_dim in dims:
            l_ax = dims.index(layer_dim)
            arr_perm = np.transpose(arr, (t_ax, l_ax, n_ax))         # (T, L, N)
            arr_perm = np.where(np.isfinite(arr_perm) & (np.abs(arr_perm) <= 1e30),
                                 arr_perm, 0.0)
            T_s, L_s, N_s = arr_perm.shape
            # Sparse A maps (N -> n_cv); apply per (T*L) slab.
            arr_flat = arr_perm.reshape(T_s * L_s, N_s).astype(np.float32)
            elem_flat = arr_flat @ A_avg.T   # (T*L, n_cv)
            elem_arr = elem_flat.reshape(T_s, L_s, n_cv).astype(np.float64)
            # Truncate to common time length with lh_cv_arr
            T_use = min(T_s, lh_cv_arr.shape[0])

            # Reconcile layer-axis lengths between elem_arr and lh_cv_arr.
            #   - ENV_layer_ht path: lh_cv has L layers, same as the tracer (direct multiply)
            #   - zCoord fallback : lh_cv has L-1 layers (np.diff over the
            #                       L vertical-grid interfaces), so we average
            #                       adjacent tracer values to mid-prism centres
            #                       and integrate trapezoidally.
            if lh_cv_arr.shape[1] == L_s - 1:
                c_mid = 0.5 * (elem_arr[:T_use, :-1, :]
                                + elem_arr[:T_use, 1:, :])             # (T, L-1, n_cv)
                mass_t = (c_mid * lh_cv_arr[:T_use]
                          * cv_area[None, None, :]).sum(axis=(1, 2))
            else:
                mass_t = (elem_arr[:T_use] * lh_cv_arr[:T_use]
                          * cv_area[None, None, :]).sum(axis=(1, 2))

            # ---- Domain-wide mass: node-based integration ----
            # Read the tracer over the FULL mesh (all nodes), transpose to
            # (T, L, n_nodes), and integrate with dz_full + A_node_full.
            da_full = da_s.transpose(
                next(d for d in da_s.dims if d.lower() == "time"),
                next(d for d in da_s.dims
                     if "vgrid" in d.lower() or "layer" in d.lower()),
                next(d for d in da_s.dims if "node" in d.lower()),
            )
            arr_full = da_full.values
            arr_full = np.where(np.isfinite(arr_full) & (np.abs(arr_full) <= 1e30),
                                 arr_full, 0.0).astype(np.float32)
            T_full = min(arr_full.shape[0], dz_full.shape[0])
            c_mid_full = 0.5 * (arr_full[:T_full, :-1, :]
                                 + arr_full[:T_full, 1:, :])             # (T, L-1, n_nodes)
            domain_mass_t = (c_mid_full * dz_full[:T_full]
                              * A_node_full[None, None, :]).sum(axis=(1, 2))
            domain_mass_t = domain_mass_t.astype(np.float64)
            # Free the largest intermediate before returning
            del arr_full, c_mid_full
            # Trim CV mass to the same length so they share a time axis
            T_common = min(T_use, T_full)
            mass_t = mass_t[:T_common]
            domain_mass_t = domain_mass_t[:T_common]
        else:
            arr_perm = np.transpose(arr, (t_ax, n_ax))                # (T, N)
            arr_perm = np.where(np.abs(arr_perm) <= 1e30, arr_perm, 0.0)
            T_s, N_s = arr_perm.shape
            elem_arr = (arr_perm.astype(np.float32) @ A_avg.T).astype(np.float64)
            # 2D var: weight by total column depth × area
            T_use = min(T_s, lh_cv_arr.shape[0])
            lh_col = lh_cv_arr[:T_use].sum(axis=1)                    # (T, n_cv)
            mass_t = (elem_arr[:T_use] * lh_col * cv_area[None, :]).sum(axis=1)
            domain_mass_t = None    # 2D var — domain integral not meaningful here
        return da_cv["time"].values[:mass_t.shape[0]], mass_t, domain_mass_t

    # ---- loop over variables ----
    for var in VARIABLES:
        print(f"\n--- {var} ---")
        is_vol = (var == "VOL")
        sc_name = None if is_vol else SCRIBE_NAME[var]
        var_unit = UNITS_PER_VOL.get(var, DEFAULT_CONC_UNIT) if not is_vol else "m³/s"

        # Will be set below if we successfully load scribed C at src#3 cell.
        # Used to compute "actual injected mass = integral Q_src3 * C_cell dt".
        c_cell_t = None
        c_cell_v = None

        # ---- Panel A inputs ----
        cum_src = {}
        cum_row = {}
        if is_vol:
            # Volume itself: cum src input = ∫ Q dt; cum flux = ∫ VOL(row) dt.
            for si in (0, 1, 2):
                cum_src[si] = cumtrap(Q[:, si], t_vs)
            for row in (2, 3, 4):
                if row in available_regions:
                    raw = flux_da.sel(tracer="VOL", region=row).values[keep_fx]
                    cum_row[row] = cumtrap(raw, t_flux_s)
        else:
            for si in (0, 1, 2):
                c_raw = msource[var][:, si]
                # SCHISM's "use ambient cell value" sentinel in msource.th is
                # -9999.0 (also -9.999E+03). Treat any such value as 0 so it
                # doesn't dominate the cumulative-source integral.
                c_raw = np.where(c_raw <= -1e3, 0.0, c_raw)
                c_i = np.interp(t_vs, t_ms, c_raw)
                rate = Q[:, si] * c_i
                cum_src[si] = cumtrap(rate, t_vs)
            for row in (2, 3, 4):
                if row in available_regions:
                    raw = flux_da.sel(tracer=var, region=row).values[keep_fx]
                    cum_row[row] = cumtrap(raw, t_flux_s)

        # ---- Panels B + C ----
        cell_series = {}
        panelC = None
        panelC_domain = None
        if is_vol:
            panelC = (t_cmb_aligned, vol_total_t,
                      "m³", "Total water volume of CV")
            # Domain water volume on right axis
            panelC_domain = (t_domain, vol_domain_t,
                             "m³", "Domain water volume")
        else:
            try:
                da_s = geometry.open_scribed_concat(outputs, sc_name,
                                                     n_stacks=PERIOD_DAYS + 1)
                tt = da_s["time"].values
                _, uniq = np.unique(tt, return_index=True)
                if len(uniq) != len(tt):
                    da_s = da_s.isel(time=np.sort(uniq))
                keep_s = (da_s["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
                da_s = da_s.isel(time=keep_s)
                node_dim = next(d for d in da_s.dims if "node" in d.lower())
                layer_dim = next((d for d in da_s.dims
                                  if "vgrid" in d.lower() or "layer" in d.lower()), None)
                # Panel B: subset to the 3 cells' vertex nodes (small)
                da_sub = da_s.isel({node_dim: all_nodes}).compute()
                node_pos = {n: i for i, n in enumerate(all_nodes)}
                for lab, ei, color in cells:
                    idxs = [node_pos[n] for n in cell_nodes[ei]]
                    vals = da_sub.isel({node_dim: idxs})
                    if layer_dim is not None:
                        vals = vals.isel({layer_dim: -1})
                    vals = vals.where(np.abs(vals) <= 1e30)
                    ts = vals.mean(dim=node_dim).values
                    cell_series[lab] = (da_sub["time"].values, ts, color)
                    # If this is cell A (= src#3 cell), keep the C_cell series
                    # so we can compute the "actual injected mass" estimate
                    # = ∫ Q_src3 × C_cell_src3 dt  (mass leaving the src cell
                    # per unit time via advection = mass actually entering the
                    # broader domain from this source).
                    if ei == elem_A:
                        c_cell_t = da_sub["time"].values
                        c_cell_v = ts
                # Panel C: volume-integrate over CV AND domain
                try:
                    t_C, v_C, v_dom = _panelC_for_scribed(da_s, sc_name,
                                                          node_dim, layer_dim)
                    panelC = (t_C, v_C, f"{var_unit} × m³",
                              f"Volume-integrated {var} in CV")
                    if v_dom is not None:
                        panelC_domain = (t_C, v_dom, f"{var_unit} × m³",
                                          f"Volume-integrated {var} over DOMAIN")
                    print(f"  CV mass max = {v_C.max():.3e}, "
                          f"domain mass max = "
                          f"{('n/a' if v_dom is None else f'{v_dom.max():.3e}')}")
                except Exception as e:
                    print(f"  WARN: panel C failed for {var}: {e}")
            except Exception as e:
                print(f"  WARN: could not load scribed '{sc_name}': {e}")

        # ---- "Actual injected mass" estimate for src #3 ----
        # SCHISM's source step is (C_old + rat * msource) / (1 + rat) — a
        # relaxation toward msource, NOT a mass injection. The true mass
        # entering the broader domain via this source is the mass advected
        # OUT of the src cell:  integral Q × C_cell dt.
        # (Volumetric flow Q leaves the cell every second carrying C_cell.)
        cum_src_actual = None
        if c_cell_t is not None and not is_vol:
            # interpolate scribed C_cell at src#3 onto vsource time base
            t_c_sec = (c_cell_t - START) / np.timedelta64(1, "s")
            t_c_sec = t_c_sec.astype(float)
            c_at_vs = np.interp(t_vs, t_c_sec, c_cell_v)
            rate_actual = Q[:, 2] * c_at_vs
            cum_src_actual = cumtrap(rate_actual, t_vs)
            print(f"  actual injected (Q_src3 * C_cell) end = "
                  f"{cum_src_actual[-1]:+.3e}  "
                  f"(vs naive Q*msource = {cum_src[2][-1]:+.3e})")

        # ---- plot ----
        fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

        # ---- Panel a ----
        if is_vol:
            src_labels = ["src 3 input (Dumbleton)", "src 1 input (Outlet 21)",
                          "src 2 input (Outlet 59)"]
        else:
            src_labels = ["src 3 input (Dumbleton)", "src 1 input (Outlet 21)",
                          "src 2 input (Outlet 59)"]
        axA.plot(src_times,  cum_src[2],     color="black",   lw=1.8,
                 label="src 3 input  (Dumbleton)")
        if 2 in cum_row:
            axA.plot(flux_times, -cum_row[2],    color="#9ca3af", lw=1.6,
                     label="flux.out row 2 sign-reversed  (1 -> 2, downstream of src 3)")
        axA.plot(src_times,  cum_src[0],     color="#dc2626", lw=1.6,
                 label="src 1 input  (Outlet 21)")
        axA.plot(src_times,  cum_src[1],     color="#f97316", lw=1.6,
                 label="src 2 input  (Outlet 59)")
        if 4 in cum_row:
            axA.plot(flux_times, -cum_row[4],    color="#ec4899", lw=1.6,
                     label="flux.out row 4 sign-reversed  (3 -> 4, downstream of src 1/2)")
        if 3 in cum_row:
            axA.plot(flux_times, cum_row[3],     color="#1d4ed8", lw=1.8,
                     label="flux.out row 3 (3 -> 2)  Pioneer Mouth")
        if cum_src_actual is not None:
            axA.plot(src_times, cum_src_actual, color="#7c3aed", lw=2.0, ls="--",
                      label="TRUE injection est = integral Q_src3 * C_cell dt")
        axA.axhline(0, color="#999", lw=0.4)
        panelA_unit = "m³" if is_vol else f"{var_unit} × m³"
        axA.set_ylabel(f"Cumulative flux\n({panelA_unit})")
        axA.set_title(f"(a) {var} — cumulative flux at 3 sources and 3 transects (14-day, raw signs)",
                      fontweight="bold", loc="left", fontsize=10)
        axA.legend(loc="best", fontsize=8, ncols=2)
        axA.grid(True, ls=":", alpha=0.4)

        # End-of-period numeric annotation, top-left of panel a
        annot_lines = [f"FINAL CUMULATIVE VALUES ({PERIOD_DAYS}d):"]
        annot_lines.append(f"  src 3 (Dumbleton, BLACK) : {cum_src[2][-1]:+.3e}")
        if 2 in cum_row:
            annot_lines.append(f"  flux.out row 2 -reversed (GREY)  : {-cum_row[2][-1]:+.3e}")
        annot_lines.append(f"  src 1 (Outlet 21, RED)   : {cum_src[0][-1]:+.3e}")
        annot_lines.append(f"  src 2 (Outlet 59, ORANGE): {cum_src[1][-1]:+.3e}")
        if 4 in cum_row:
            annot_lines.append(f"  flux.out row 4 -reversed (PINK)  : {-cum_row[4][-1]:+.3e}")
        if 3 in cum_row:
            annot_lines.append(f"  flux.out row 3 raw (BLUE, mouth) : {cum_row[3][-1]:+.3e}")
        if cum_src_actual is not None:
            annot_lines.append(f"  TRUE inj est Q*C_cell (PURPLE)   : {cum_src_actual[-1]:+.3e}")
        axA.text(0.005, 0.995, "\n".join(annot_lines),
                  transform=axA.transAxes, ha="left", va="top",
                  fontsize=8, family="monospace",
                  bbox=dict(facecolor="white", edgecolor="#999",
                            alpha=0.85, pad=4))

        # ---- Panel b ----
        if is_vol:
            axB.text(0.5, 0.5,
                     "No 'concentration' panel for VOL\n(see panel c for CV water volume)",
                     transform=axB.transAxes, ha="center", va="center",
                     fontsize=12, color="grey")
        elif cell_series:
            for lab, (t, v, c) in cell_series.items():
                axB.plot(t, v, color=c, lw=1.6, label=lab)
        else:
            axB.text(0.5, 0.5, f"(no scribed file for '{sc_name}')",
                     transform=axB.transAxes, ha="center", va="center",
                     fontsize=12, color="grey")
        axB.axhline(0, color="#999", lw=0.4)
        axB.set_ylabel(f"Concentration\n({var_unit})")
        axB.set_title(f"(b) {var} — concentration at 3 cells (surface layer)",
                      fontweight="bold", loc="left", fontsize=10)
        if not is_vol and cell_series:
            axB.legend(loc="upper left", fontsize=9)
        axB.grid(True, ls=":", alpha=0.4)

        # ---- right-margin markers: ocean BC star + river circle (panel b only) ----
        if not is_vol:
            trans = axB.get_yaxis_transform()
            ocean_val = ocean_bc_mean.get(var)
            river_val = float(np.interp(t_vs, t_ms, msource[var][:, 2]).mean())
            x_marker = 1.015
            if ocean_val is not None:
                axB.plot(x_marker, ocean_val, marker="*", color="#1d4ed8",
                         markersize=18, markeredgecolor="black", markeredgewidth=0.8,
                         transform=trans, clip_on=False, zorder=20)
                axB.annotate(ocean_bc_label[var],
                              xy=(x_marker, ocean_val), xycoords=trans,
                              xytext=(0, 12), textcoords="offset points",
                              ha="center", fontsize=7, color="#1d4ed8", rotation=90)
            axB.plot(x_marker, river_val, marker="o", color="black",
                     markersize=11, markeredgecolor="black", markeredgewidth=0.8,
                     transform=trans, clip_on=False, zorder=20)
            axB.annotate("msource.th (src 3)",
                          xy=(x_marker, river_val), xycoords=trans,
                          xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=7, color="black", rotation=90)

        # ---- Panel c ----
        # CV-integrated mass on LEFT y-axis (teal)
        # Domain-integrated mass on RIGHT y-axis (red, dashed)
        cv_color  = "#0f766e"
        dom_color = "#dc2626"
        if panelC is not None:
            t_C, v_C, unit_C, label_C = panelC
            axC.plot(t_C, v_C, color=cv_color, lw=1.8, label=label_C)
        else:
            axC.text(0.5, 0.5, "(panel C unavailable)", transform=axC.transAxes,
                     ha="center", va="center", fontsize=12, color="grey")
            unit_C = ""
        axC.axhline(0, color="#999", lw=0.4)
        axC.set_ylabel(f"CV-integrated {var}\n({unit_C})", color=cv_color)
        axC.tick_params(axis="y", labelcolor=cv_color)
        axC.set_xlabel("Date (model time)")
        axC.set_title(f"(c) {var} — volume-integrated over CV (left) and domain (right)",
                      fontweight="bold", loc="left", fontsize=10)
        axC.grid(True, ls=":", alpha=0.4)
        axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        # Right y-axis: domain-wide integral
        if panelC_domain is not None:
            axC2 = axC.twinx()
            t_D, v_D, _, label_D = panelC_domain
            axC2.plot(t_D, v_D, color=dom_color, lw=1.4, ls="--", alpha=0.85,
                      label=label_D)
            axC2.set_ylabel(f"Domain-integrated {var}\n({unit_C})", color=dom_color)
            axC2.tick_params(axis="y", labelcolor=dom_color)
            # Combined legend
            lines_left, labels_left = axC.get_legend_handles_labels()
            lines_right, labels_right = axC2.get_legend_handles_labels()
            axC.legend(lines_left + lines_right, labels_left + labels_right,
                       loc="upper left", fontsize=9)
        elif panelC is not None:
            axC.legend(loc="upper left", fontsize=9)

        fig.suptitle(f"{var}  —  src/transect cumulative fluxes  +  3-cell conc  "
                     f"+  CV volumetric storage",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.93, 0.97])
        out_path = out_dir / f"{var}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"  wrote {out_path.name}")

    print(f"\nDone. {len(VARIABLES)} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
