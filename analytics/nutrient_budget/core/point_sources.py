"""
Term F — point-source (river) tracer input.

SCHISM source/sink injects volume and tracer mass directly into specific mesh
elements via:

    source_sink.in   element IDs of each source (1-indexed)
    vsource.th       time, Q(m^3/s) for each source
    msource.th       time, T(per source), S(per source), then AED tracer(per source)
                     repeated for each AED state variable in the order they are
                     registered by the active &aed_models list.

Units in msource.th match the AED state-variable unit (e.g. mmol N/m^3 for
nitrogen pools, mmol C/m^3 for PHY_mixed). flux.out uses the same units, so
no extra conversion is needed between Term F and Term E.

Term F is the volumetric load of each tracer entering the CV:

    F_t(t) = Σ_{src∈CV}  Q_src(t) × c_{src,t}(t) × n_per_mol[t]   [mmol N / s]
    →       × 86400   [mmol N / day]

where the sum runs only over sources whose target element is inside the
control volume.
"""

from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import xarray as xr


# Default AED state-variable order in msource.th.  This is the registration
# order across the active &aed_models block for the Pioneer P18 run.
# Override via parse_msource(..., aed_tracer_order=[...]) if your build differs.
DEFAULT_AED_TRACER_ORDER = [
    "NCS_ss1",
    "OXY_oxy",
    "NIT_amm",
    "NIT_nit",
    "PHS_frp",
    "PHS_frp_ads",
    "OGM_doc",
    "OGM_poc",
    "OGM_don",
    "OGM_pon",
    "OGM_dop",
    "OGM_pop",
    "PHY_mixed",
    "TRC_tr1",
    "TRC_tr2",
    "TRC_tr3",
    "TRC_age",
]


def parse_source_sink(path: Path) -> np.ndarray:
    """Return 0-indexed element IDs of each source (in declaration order).

    Skips the sinks block at the bottom of source_sink.in.
    """
    elems = []
    with open(path) as f:
        n_src = int(f.readline().split("!")[0])
        for _ in range(n_src):
            line = f.readline()
            elem_str = line.split("!", 1)[0]
            elems.append(int(elem_str.strip()) - 1)   # 1-indexed → 0-indexed
    return np.asarray(elems, dtype=int)


def parse_vsource(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vsource.th -> (time_sec, Q[n_t, n_src])."""
    data = np.loadtxt(path)
    t = data[:, 0]
    Q = data[:, 1:]
    return t, Q


def parse_msource(
    path: Path,
    n_src: int,
    aed_tracer_order: list[str] = None,
    gen_tracer_order: list[str] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read msource.th and return (time_sec, {tracer_name: c[n_t, n_src]}).

    msource.th layout per row (with GEN tracers if any):
        t,  T(1..n_src), S(1..n_src),
            [GEN_1(1..n_src), ..., GEN_n_gen(1..n_src),]
            AED1(1..n_src), AED2(1..n_src), ...

    If gen_tracer_order is provided, those columns are extracted as well
    and inserted into the output dict by name. AED tracers shift right by
    n_gen × n_src in the file layout.
    """
    order = aed_tracer_order or DEFAULT_AED_TRACER_ORDER
    gen_order = gen_tracer_order or []
    n_gen = len(gen_order)
    data = np.loadtxt(path)
    t = data[:, 0]
    expected_cols = 1 + (2 + n_gen + len(order)) * n_src
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"msource.th has {data.shape[1]} columns but expected "
            f"{expected_cols} (= 1 + (T+S+{n_gen} GEN+{len(order)} AED) × {n_src} src). "
            f"Tracer-order assumption probably wrong."
        )
    out = {}
    # GEN block, then AED block
    gen_start = 1 + 2 * n_src
    for i, name in enumerate(gen_order):
        c0 = gen_start + i * n_src
        c1 = c0 + n_src
        out[name] = data[:, c0:c1]
    aed_start = gen_start + n_gen * n_src
    for i, name in enumerate(order):
        c0 = aed_start + i * n_src
        c1 = c0 + n_src
        out[name] = data[:, c0:c1]
    return t, out


def _seconds_to_datetime(t_sec: np.ndarray, start_date: str | None) -> np.ndarray:
    """Convert seconds-since-start to numpy datetime64[ns] given start_date='YYYY-MM-DD'.

    If start_date is None, returns the raw float seconds as a numeric index.
    """
    if start_date is None:
        return t_sec.astype("float64")
    t0 = np.datetime64(start_date)
    return t0 + (t_sec * 1e9).astype("timedelta64[ns]")


def point_source_term(
    source_sink_path: Path,
    vsource_path: Path,
    msource_path: Path,
    cv_elements: np.ndarray,
    tracer_n_per_mol: dict[str, float],
    aed_tracer_order: list[str] = None,
    start_date: str = None,
) -> dict:
    """Compute Term F — point-source nutrient load into the CV.

    Returns:
        {
          "total":    xr.DataArray (time,)  mmol N / day, all CV sources summed
          "per_src":  xr.DataArray (time, src) per-source N load (only CV sources)
          "per_var":  {tracer_name: DataArray(time,) mmol N/day from that pool},
          "cv_src_idx": list of source indices (1-indexed) inside CV,
          "cv_src_elem": list of element IDs (1-indexed) of CV sources,
        }

    `tracer_n_per_mol` lists ONLY the tracers that carry the budget element,
    with their per-mol weight (e.g. PHY_mixed gets 16/106 for N).
    """
    src_elems = parse_source_sink(source_sink_path)
    n_src = len(src_elems)

    cv_set = set(int(e) for e in cv_elements)
    cv_src_mask = np.array([int(e) in cv_set for e in src_elems], dtype=bool)
    cv_src_idx_1 = [int(i) + 1 for i, m in enumerate(cv_src_mask) if m]   # 1-idx for display
    cv_src_elem_1 = [int(src_elems[i]) + 1 for i in range(n_src) if cv_src_mask[i]]

    t_vs, Q = parse_vsource(vsource_path)
    if Q.shape[1] != n_src:
        raise ValueError(
            f"vsource.th has {Q.shape[1]} source columns but source_sink.in "
            f"declares {n_src} sources"
        )
    t_ms, conc = parse_msource(msource_path, n_src, aed_tracer_order=aed_tracer_order)

    # Align msource onto vsource time axis (linear interp) — both typically
    # share the same cadence, but lengths can differ if one was regenerated.
    needs_interp = (len(t_ms) != len(t_vs)) or not np.allclose(t_ms, t_vs)
    if needs_interp:
        for name, c in conc.items():
            c_new = np.empty_like(Q)
            for s in range(n_src):
                c_new[:, s] = np.interp(t_vs, t_ms, c[:, s])
            conc[name] = c_new

    # Mask to CV sources only
    Q_cv = Q[:, cv_src_mask]               # (n_t, n_cv_src)
    src_labels = [f"src{i}_elem{src_elems[i-1]+1}" for i in cv_src_idx_1]

    # Per-tracer N load (mmol N/s) summed over CV sources
    per_var = {}
    total_per_t = np.zeros_like(t_vs)
    for name, npm in tracer_n_per_mol.items():
        if name not in conc:
            raise KeyError(
                f"Tracer '{name}' not in msource columns. Available: "
                f"{list(conc.keys())}"
            )
        c_cv = conc[name][:, cv_src_mask]      # (n_t, n_cv_src), mmol X/m^3
        load_var = (Q_cv * c_cv * npm).sum(axis=1)  # mmol N/s
        per_var[name] = load_var * 86400.0          # mmol N/day
        total_per_t += per_var[name]

    # Per-source total N load (mmol N/day)
    per_src_load = np.zeros_like(Q_cv)
    for name, npm in tracer_n_per_mol.items():
        c_cv = conc[name][:, cv_src_mask]
        per_src_load += Q_cv * c_cv * npm
    per_src_load *= 86400.0

    # Build xarray with datetime axis (if start_date given) or seconds
    time_axis = _seconds_to_datetime(t_vs, start_date)
    time_dim = "time"

    total_da = xr.DataArray(
        total_per_t, dims=(time_dim,), coords={time_dim: time_axis},
        attrs={"units": "mmol N/day",
               "long_name": "Point-source N load into CV (Term F)"},
    )
    per_src_da = xr.DataArray(
        per_src_load,
        dims=(time_dim, "source"),
        coords={time_dim: time_axis, "source": src_labels},
        attrs={"units": "mmol N/day"},
    )
    per_var_da = {
        name: xr.DataArray(arr, dims=(time_dim,), coords={time_dim: time_axis},
                           attrs={"units": "mmol N/day"})
        for name, arr in per_var.items()
    }

    return {
        "total": total_da,
        "per_src": per_src_da,
        "per_var": per_var_da,
        "cv_src_idx": cv_src_idx_1,
        "cv_src_elem": cv_src_elem_1,
    }
