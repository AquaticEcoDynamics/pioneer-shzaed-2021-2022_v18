"""Shared run configuration for the nutrient-budget diagnostics.

Lets every diagnostic target a run *by name* instead of a hard-coded absolute
path, so the same script works on 002 (flood) and 003 (flat-tide), etc.

Select the run via (first match wins):
    --run 003            on the command line
    --run-dir <path>     ad-hoc run directory (overrides the registry)
    SCHISM_RUN=003       environment variable
    (default: 002)

Optional window override:
    --days 14   or   SCHISM_DAYS=14     (default: 21)

Usage in a script:
    from run_config import active_run
    CFG = active_run()
    hgrid   = CFG.hgrid
    outputs = CFG.outputs
    OUT     = CFG.out_dir / "conservation_check"
    START, PERIOD_DAYS, N_REGIONS = CFG.start, CFG.period_days, CFG.n_regions
"""
from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# the generic budget framework lives in the sibling nutrient_budget/ package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nutrient_budget"))
from core.boundary_fluxes import DEFAULT_TRACER_ORDER

PKG = Path(__file__).resolve().parents[0]          # .../analytics/pioneer_region_flux
REPO = Path(__file__).resolve().parents[2]         # repo root
CONFIGS = PKG / "configs"


@dataclass
class RunConfig:
    name: str
    run_dir: Path
    period_days: float = 21.0
    region: int = 3                                 # mouth (2<->3) gate column
    _start: object = None
    _n_regions: object = None
    tracer_order: list = field(default_factory=lambda: list(DEFAULT_TRACER_ORDER))

    # ---- run-file paths (derived from run_dir) ----
    @property
    def outputs(self): return self.run_dir / "outputs"
    @property
    def hgrid(self): return self.run_dir / "hgrid.gr3"
    @property
    def fluxflag(self): return self.run_dir / "fluxflag.prop"
    @property
    def source_sink(self): return self.run_dir / "source_sink.in"
    @property
    def vsource(self): return self.run_dir / "vsource.th"
    @property
    def msource(self): return self.run_dir / "msource.th"
    @property
    def param_nml(self): return self.run_dir / "param.nml"
    @property
    def flux_out(self): return self.outputs / "flux.out"

    # ---- config files (shared, in nutrient_budget/configs) ----
    @property
    def configs(self): return CONFIGS
    @property
    def cv_elements(self): return CONFIGS / "cv_pioneer_estuary_elements.txt"
    @property
    def cv_csv(self): return CONFIGS / "cv_pioneer_estuary.csv"

    # ---- per-run output dir for diagnostics (kept out of git) ----
    @property
    def out_dir(self):
        d = PKG / "_outputs" / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- lazily-derived values ----
    @property
    def start(self):
        if self._start is None:
            self._start = _parse_start(self.param_nml)
        return self._start

    @property
    def n_regions(self):
        if self._n_regions is None:
            self._n_regions = _max_fluxflag(self.fluxflag)
        return self._n_regions


def _parse_start(param_nml):
    vals = {"start_year": 2021, "start_month": 4, "start_day": 1, "start_hour": 0}
    try:
        txt = Path(param_nml).read_text()
        for k in vals:
            m = re.search(rf'^\s*{k}\s*=\s*([0-9.]+)', txt, re.M)
            if m:
                vals[k] = int(float(m.group(1)))
    except OSError:
        pass
    return (np.datetime64(f"{vals['start_year']:04d}-{vals['start_month']:02d}-{vals['start_day']:02d}")
            + np.timedelta64(int(vals["start_hour"]), "h"))


def _max_fluxflag(fluxflag, default=3):
    try:
        flags = np.loadtxt(fluxflag, dtype=int)
        col = flags[:, 1] if flags.ndim == 2 else flags
        return int(col.max())
    except Exception:
        return default


# ---- registry of known runs ----
RUNS = {
    # 002: the flood run that actually holds outputs on this machine
    "002": RunConfig("002_P18_flood", Path("s:/Matt_Working/schism/P18_flood")),
    # 003: flat-tide run (outputs land here once it has been run)
    "003": RunConfig("003_P18_flood_flat", REPO / "runs" / "003_P18_flood_flat"),
}


def _argval(*flags):
    """Return the value following any of `flags` in sys.argv (supports --x val and --x=val)."""
    argv = sys.argv
    for i, a in enumerate(argv):
        for f in flags:
            if a == f and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(f + "="):
                return a.split("=", 1)[1]
    return None


def active_run():
    """Resolve the RunConfig selected by CLI/env, applying optional --days."""
    run_dir = _argval("--run-dir")
    if run_dir:
        cfg = RunConfig(Path(run_dir).name, Path(run_dir))
    else:
        name = _argval("--run") or os.environ.get("SCHISM_RUN", "002")
        if name not in RUNS:
            raise SystemExit(f"unknown run '{name}'; known: {sorted(RUNS)} "
                             f"(or pass --run-dir <path>)")
        cfg = RUNS[name]
    days = _argval("--days") or os.environ.get("SCHISM_DAYS")
    if days:
        cfg.period_days = float(days)
    return cfg
