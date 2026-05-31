"""
Volume-only sanity-check for the Pioneer Estuary N budget.

Assume every cell holds a uniform 1 mg N / L = 1 g N / m^3 of total nitrogen,
and that all three river inputs and the boundary export water are at the same
1 mg N / L. Under this assumption, the N budget reduces to a pure volume
balance, which lets us check that the underlying flows (vsource.th and
flux.out volume) are sensible BEFORE worrying about tracer concentrations
or unit conventions.

Outputs:
    panel a — cumulative inflow N load from each of the 3 CV rivers (positive)
    panel b — cumulative export N load through Pioneer Mouth boundary (negative)
    panel c — total N mass currently in the CV control volume (Σ V × 1 g/m^3)

All in tonnes N over the first 14 days of the P18 run.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nutrient_budget"))
from core import geometry
from core.boundary_fluxes import parse_flux_out
from core.point_sources import parse_source_sink, parse_vsource


def main():
    run = Path("s:/Matt_Working/schism/P18")
    outputs = run / "outputs"
    cv_file = Path(__file__).parent / "configs" / "cv_pioneer_estuary_elements.txt"
    plot_path = Path(__file__).parent / "volume_sanity_check.png"

    START = np.datetime64("2021-04-01")
    PERIOD_DAYS = 14
    G_PER_M3 = 1.0           # 1 mg N / L = 1 g N / m^3
    G_TO_T = 1.0 / 1.0e6     # g -> t

    # ---- 1. CV elements ----
    cv_elements = np.loadtxt(cv_file, dtype=int, comments="#") - 1
    print(f"CV elements: {len(cv_elements)}")

    # ---- 2. Rivers — Q(t) per source from vsource.th ----
    print("\nReading vsource.th + source_sink.in...")
    src_elems_0idx = parse_source_sink(run / "source_sink.in")
    t_vs, Q = parse_vsource(run / "vsource.th")          # Q: (n_t, n_src)
    cv_set = set(int(e) for e in cv_elements)
    cv_src_mask = np.array([int(e) in cv_set for e in src_elems_0idx], dtype=bool)
    cv_src_idx_1 = [i + 1 for i, m in enumerate(cv_src_mask) if m]
    print(f"  CV-interior sources (1-indexed): {cv_src_idx_1}")

    # Trim to first 14 days
    period_s = PERIOD_DAYS * 86400.0
    keep_vs = t_vs <= period_s
    t_vs_d = t_vs[keep_vs] / 86400.0                 # days since start
    Q_cv = Q[keep_vs][:, cv_src_mask]                # (n_t, n_cv_src) m^3/s
    src_times = START + (t_vs_d * 86400 * 1e9).astype("timedelta64[ns]")

    # Cumulative load per source [t N]
    # Load rate = Q × 1 g/m^3 = g N/s; cumulative = trapz(rate, t) in g N
    n_t = len(t_vs_d)
    cum_load_per_src = np.zeros_like(Q_cv)
    if n_t > 1:
        # trapezoidal integration
        dt_s = np.diff(t_vs[keep_vs])  # seconds
        for s in range(Q_cv.shape[1]):
            rate = Q_cv[:, s] * G_PER_M3                                 # g N/s
            incr = 0.5 * (rate[1:] + rate[:-1]) * dt_s                   # g N per step
            cum_load_per_src[1:, s] = np.cumsum(incr)
    cum_load_per_src_t = cum_load_per_src * G_TO_T
    print(f"  Final cumulative river loads (t N):")
    for k, s in enumerate(cv_src_idx_1):
        print(f"    src{s}: {cum_load_per_src_t[-1, k]:.3f}")

    # ---- 3. Pioneer Mouth — VOL(t) from flux.out ----
    # IMPORTANT SCHISM convention: fluxes_tr(itmp1, ...) where itmp1 = MAX of the
    # two adjacent region IDs. So flux through the (2, 3) interface is reported
    # in row 3, NOT row 2. flux.out "region 2" is the (1, 2) interface (i.e.,
    # the tiny "extra" flux-region 1).
    #
    # Sign convention I derived from schism_step.F90:8980-8998:
    #   ftmp > 0  iff  flow goes from HIGH region to LOW region
    # For Pioneer Mouth (2,3), flow 3 -> 2 means INTO our CV (region 2 is inside
    # CV polygon, region 3 is the bay outside). So:
    #   raw +ve at row 3 = into CV
    #   raw -ve at row 3 = out of CV (estuary -> bay)
    print("\nReading flux.out (this may take ~30s)...")
    flux_da = parse_flux_out(outputs / "flux.out", n_regions=9)
    MOUTH_REGION = 3
    MOUTH_SIGN = +1   # +1 because +ve raw at row 3 = INTO CV
    vol_rN = flux_da.sel(tracer="VOL", region=MOUTH_REGION).values        # m^3/s
    t_flux_d = flux_da["time_days"].values
    keep_fx = t_flux_d <= PERIOD_DAYS
    vol_rN = vol_rN[keep_fx]
    t_flux_d = t_flux_d[keep_fx]
    mouth_times = START + (t_flux_d * 86400 * 1e9).astype("timedelta64[ns]")
    print(f"  using flux.out row for region {MOUTH_REGION} (Pioneer Mouth = (2,3) interface)")
    print(f"  flux.out t range kept: {t_flux_d.min():.3f} .. {t_flux_d.max():.3f} d "
          f"({len(t_flux_d)} samples)")
    print(f"  mean raw VOL: {vol_rN.mean():.3f} m^3/s  (cum {vol_rN.mean()*len(t_flux_d)*45/1e6:.2f} Mm^3)")

    # Cumulative N into CV through mouth: rate = MOUTH_SIGN × raw × 1 g/m^3
    if len(t_flux_d) > 1:
        dt_s = np.diff(t_flux_d) * 86400.0
        rate_into_cv = MOUTH_SIGN * vol_rN * G_PER_M3                      # g N/s into CV
        incr = 0.5 * (rate_into_cv[1:] + rate_into_cv[:-1]) * dt_s         # g N/step
        cum_mouth_into_cv = np.concatenate([[0.0], np.cumsum(incr)])       # g N
    else:
        cum_mouth_into_cv = np.zeros_like(vol_rN)
    cum_mouth_t = cum_mouth_into_cv * G_TO_T
    print(f"  Final cumulative mouth load INTO CV (t N): {cum_mouth_t[-1]:+.3f}")

    # ---- 4. CV total volume × 1 g/m^3 from cmb's ENV_layer_ht ----
    print("\nReading hgrid + cmb (lazy) for CV volume...")
    hgrid = geometry.read_hgrid_with_areas(run / "hgrid.gr3")
    n_stacks_needed = PERIOD_DAYS + 1                                   # 1 stack/day approx
    ds_cmb = geometry.open_cmb_concat(outputs, n_stacks=n_stacks_needed)
    # Dedup time
    t_cmb = ds_cmb["time"].values
    _, uniq = np.unique(t_cmb, return_index=True)
    if len(uniq) != len(t_cmb):
        ds_cmb = ds_cmb.isel(time=np.sort(uniq))
    # Subset to <=14 days
    t_keep = (ds_cmb["time"].values - START) <= np.timedelta64(PERIOD_DAYS, "D")
    ds_cmb = ds_cmb.isel(time=t_keep)
    print(f"  cmb time points kept: {ds_cmb.sizes['time']}")
    layer_ht = ds_cmb["ENV_layer_ht"]                                    # (time, face, layer)
    # Restrict to CV elements
    elem_dim = next(d for d in layer_ht.dims if "face" in d.lower())
    cv_mask_full = np.zeros(hgrid["n_elements"], dtype=bool)
    cv_mask_full[cv_elements] = True
    layer_ht_cv = layer_ht.isel({elem_dim: cv_mask_full})
    # Mask fill values
    layer_ht_cv = layer_ht_cv.where(np.abs(layer_ht_cv) <= 1e30)
    # Element area for CV
    area_cv = xr.DataArray(hgrid["areas"][cv_mask_full], dims=[elem_dim])
    # Cell volume: layer_ht × area, sum over layers + elements → total CV volume (m^3)
    layer_dim = next((d for d in layer_ht_cv.dims if "vgrid" in d.lower() or "layer" in d.lower()), None)
    sum_dims = [d for d in (elem_dim, layer_dim) if d]
    vol_cv = (layer_ht_cv * area_cv).sum(dim=sum_dims)
    print("  Computing CV volume time series (forcing dask compute)...")
    vol_cv_arr = vol_cv.compute().values                                   # (time,) m^3
    cv_times = ds_cmb["time"].values
    cv_N_mass_t = vol_cv_arr * G_PER_M3 * G_TO_T                          # t N
    print(f"  CV volume range: {vol_cv_arr.min():.3e} .. {vol_cv_arr.max():.3e} m^3 "
          f"({vol_cv_arr.min()/1e6:.2f} .. {vol_cv_arr.max()/1e6:.2f} Mm^3)")
    print(f"  CV mean depth: {vol_cv_arr.mean() / hgrid['areas'][cv_mask_full].sum():.3f} m")
    print(f"  CV N storage @ 1 g/m^3: {cv_N_mass_t.min():.3f} .. {cv_N_mass_t.max():.3f} t N")

    # ---- 5. 3-panel figure ----
    print(f"\nWriting plot to {plot_path}")
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    src_colors = ["#1d4ed8", "#059669", "#dc2626"]
    src_labels = {1: "src 1 — Outlet Node21 (north)",
                  2: "src 2 — Outlet Node59 (north)",
                  3: "src 3 — Pioneer R. @ Dumbleton (west)"}
    for k, s in enumerate(cv_src_idx_1):
        axA.plot(src_times, cum_load_per_src_t[:, k], color=src_colors[k % 3],
                 lw=1.8, label=src_labels.get(s, f"src {s}"))
    axA.axhline(0, color="#999", lw=0.5)
    axA.set_ylabel("Cumulative input load\n(tonnes N)")
    axA.set_title("(a) Cumulative N input via rivers (1 mg N / L assumed)",
                  fontweight="bold", loc="left")
    axA.legend(loc="best", fontsize=9)
    axA.grid(True, ls=":", alpha=0.4)

    axB.plot(mouth_times, cum_mouth_t, color="#7c3aed", lw=1.8,
             label="Pioneer Mouth (flux.out row 3 = (2,3) interface)")
    axB.axhline(0, color="#999", lw=0.5)
    axB.set_ylabel("Cumulative net load\ninto CV (tonnes N)")
    axB.set_title("(b) Cumulative N transport across Pioneer Mouth (1 mg N / L assumed)\n"
                  "negative = net export from CV to bay",
                  fontweight="bold", loc="left")
    axB.legend(loc="best", fontsize=9)
    axB.grid(True, ls=":", alpha=0.4)

    axC.plot(cv_times, cv_N_mass_t, color="#0f766e", lw=1.8,
             label="CV total N storage")
    axC.set_ylabel("CV inventory (tonnes N)")
    axC.set_title("(c) Total CV N inventory at 1 g N / m³ "
                  "(= 10⁻⁶ × CV water volume in m³)",
                  fontweight="bold", loc="left")
    axC.legend(loc="best", fontsize=9)
    axC.grid(True, ls=":", alpha=0.4)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axC.set_xlabel("Date (model time)")

    fig.suptitle("Pioneer Estuary — volume-only N sanity check (14 days)\n"
                 "Uniform 1 mg N / L everywhere; isolates volumetric flows from tracer dynamics",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Summary
    print()
    print("=== SUMMARY (14-day, 1 mg N/L everywhere) ===")
    sum_rivers = cum_load_per_src_t[-1].sum()
    delta_S = cv_N_mass_t[-1] - cv_N_mass_t[0]
    expected = sum_rivers + cum_mouth_t[-1]
    residual = delta_S - expected
    print(f"  sum river input (panel a, final): {sum_rivers:.3f} t N")
    print(f"  mouth net into CV (panel b, final): {cum_mouth_t[-1]:+.3f} t N")
    print(f"  CV storage range (panel c): "
          f"{cv_N_mass_t.min():.3f} .. {cv_N_mass_t.max():.3f} t N "
          f"(delta_S = {delta_S:+.3f})")
    print(f"  expected delta_S if balanced: {expected:+.3f} t N")
    print(f"  residual (delta_S - input - boundary): {residual:+.3f} t N "
          f"({100*residual/max(abs(delta_S),1e-9):+.1f}% of delta_S)")


if __name__ == "__main__":
    main()
