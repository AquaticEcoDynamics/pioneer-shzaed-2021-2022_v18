"""
CSV / PNG / HTML report writers for the budget pipeline.

Produces three artefacts in the user-supplied output directory:
  - budget_timeseries.csv   : full time series of every term
  - budget_stacked.png      : stacked-area chart of the budget components
  - budget_cumulative.png   : cumulative time-integrals
  - report.html             : narrated HTML walkthrough of all terms,
                              with embedded plots, sanity checks, closure error
"""

from __future__ import annotations
from pathlib import Path
import base64
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


MMOL_TO_TONNES = 14.0067 / 1e9   # mmol N → tonnes N (mass = mmol × g/mol / 1e9 mg/t)


def _as_series(da: xr.DataArray, name: str) -> pd.Series:
    """Convert a 1D time-DataArray to a pandas Series indexed by time."""
    time_dim = next(d for d in da.dims if "time" in d.lower())
    return pd.Series(da.values, index=pd.Index(da[time_dim].values, name="time"),
                     name=name)


def write_outputs(terms: dict, out_dir: Path | str,
                  region_config: dict,
                  group_name: str,
                  warnings: list[str] = None) -> None:
    """Write CSV + PNGs + HTML report.

    `terms` is the dict assembled by budget.py:
        {
          "storage_total":  DataArray,
          "storage_per_var": {var: DataArray, ...},
          "dMdt":            DataArray,
          "internal_per":    {rate: DataArray, ...},
          "internal_net":    DataArray,
          "atm_per":         {var: DataArray, ...},
          "atm_total":       DataArray,
          "swi_per":         {var: DataArray, ...},
          "swi_total":       DataArray,
          "boundary":        DataArray,
        }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = warnings or []

    # ---- 1. CSV time series ------------------------------------------------
    df = pd.DataFrame({
        "storage_mmolN":     _as_series(terms["storage_total"], "storage_mmolN"),
        "dMdt_mmolN_per_d":  _as_series(terms["dMdt"], "dMdt_mmolN_per_d"),
        "B_internal_mmolN_per_d": _as_series(terms["internal_net"], "B_internal"),
        "C_atm_mmolN_per_d":      _as_series(terms["atm_total"], "C_atm"),
        "D_swi_mmolN_per_d":      _as_series(terms["swi_total"], "D_swi"),
        "E_boundary_mmolN_per_d": _as_series(terms["boundary"], "E_boundary"),
        "F_pointsrc_mmolN_per_d": _as_series(terms["point_source"], "F_pointsrc"),
    })
    df["closure_residual_mmolN_per_d"] = (
        df["dMdt_mmolN_per_d"]
        - (df["B_internal_mmolN_per_d"]
           + df["C_atm_mmolN_per_d"]
           + df["D_swi_mmolN_per_d"]
           + df["E_boundary_mmolN_per_d"]
           + df["F_pointsrc_mmolN_per_d"])
    )
    csv_path = out_dir / "budget_timeseries.csv"
    df.to_csv(csv_path, float_format="%.6e")
    print(f"  wrote {csv_path}")

    # ---- 2. Stacked time-series PNG ----------------------------------------
    stacked_path = out_dir / "budget_stacked.png"
    _plot_stacked(df, stacked_path, title=f"{group_name} budget — {region_config['name']}")
    print(f"  wrote {stacked_path}")

    # ---- 3. Cumulative PNG -------------------------------------------------
    cum_path = out_dir / "budget_cumulative.png"
    _plot_cumulative(df, cum_path, title=f"{group_name} budget (cumulative) — {region_config['name']}")
    print(f"  wrote {cum_path}")

    # ---- 4. HTML report ----------------------------------------------------
    html_path = out_dir / "report.html"
    _write_html_report(
        df, terms, region_config, group_name,
        warnings, stacked_path, cum_path, html_path,
    )
    print(f"  wrote {html_path}")


def _plot_stacked(df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(df.index, 0, df["B_internal_mmolN_per_d"],
                    color="#dc2626", alpha=0.5, label="B  internal (denit+anammox, loss)")
    base = df["B_internal_mmolN_per_d"].fillna(0)
    ax.fill_between(df.index, base, base + df["C_atm_mmolN_per_d"].fillna(0),
                    color="#3b82f6", alpha=0.5, label="C  atmospheric input")
    base2 = base + df["C_atm_mmolN_per_d"].fillna(0)
    ax.fill_between(df.index, base2, base2 + df["D_swi_mmolN_per_d"].fillna(0),
                    color="#b45309", alpha=0.5, label="D  sed-water flux")
    base3 = base2 + df["D_swi_mmolN_per_d"].fillna(0)
    ax.fill_between(df.index, base3, base3 + df["E_boundary_mmolN_per_d"].fillna(0),
                    color="#047857", alpha=0.5, label="E  boundary net")
    base4 = base3 + df["E_boundary_mmolN_per_d"].fillna(0)
    ax.fill_between(df.index, base4, base4 + df["F_pointsrc_mmolN_per_d"].fillna(0),
                    color="#7c3aed", alpha=0.5, label="F  point-source (rivers)")
    ax.plot(df.index, df["dMdt_mmolN_per_d"], color="#111827", linewidth=2,
            label="dM/dt  (observed)")
    ax.axhline(0, color="#999", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("mmol N / day")
    ax.legend(loc="upper left", fontsize=9, ncols=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cumulative(df: pd.DataFrame, out_path: Path, title: str):
    cum = df[["B_internal_mmolN_per_d", "C_atm_mmolN_per_d",
              "D_swi_mmolN_per_d", "E_boundary_mmolN_per_d",
              "F_pointsrc_mmolN_per_d"]].copy()
    # Integrate in time: cumtrapz-style. For simplicity, dt in days.
    dt_days = pd.Series(df.index).diff().dt.total_seconds().values / 86400.0
    dt_days[0] = 0.0
    for col in cum.columns:
        cum[col] = (cum[col].fillna(0) * dt_days).cumsum()
    cum["ΔStorage_observed"] = df["storage_mmolN"] - df["storage_mmolN"].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum.index, cum["B_internal_mmolN_per_d"] * MMOL_TO_TONNES,
            label="B internal (denit+anammox)", color="#dc2626")
    ax.plot(cum.index, cum["C_atm_mmolN_per_d"] * MMOL_TO_TONNES,
            label="C atmospheric", color="#3b82f6")
    ax.plot(cum.index, cum["D_swi_mmolN_per_d"] * MMOL_TO_TONNES,
            label="D sed-water", color="#b45309")
    ax.plot(cum.index, cum["E_boundary_mmolN_per_d"] * MMOL_TO_TONNES,
            label="E boundary", color="#047857")
    ax.plot(cum.index, cum["F_pointsrc_mmolN_per_d"] * MMOL_TO_TONNES,
            label="F point-source (rivers)", color="#7c3aed")
    ax.plot(cum.index, cum["ΔStorage_observed"] * MMOL_TO_TONNES,
            label="ΔStorage (observed)", color="#111827", linewidth=2)
    ax.axhline(0, color="#999", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("tonnes N (cumulative)")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _embed_png(png_path: Path) -> str:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%; height:auto;" />'


def _write_html_report(df, terms, region_config, group_name,
                       warnings, stacked_png, cum_png, html_path):
    rep_period = (df.index.min(), df.index.max())
    period_days = (rep_period[1] - rep_period[0]).total_seconds() / 86400.0
    # Integrated totals (cumulative over the period)
    dt_days = pd.Series(df.index).diff().dt.total_seconds().values / 86400.0
    dt_days[0] = 0.0
    totals = {}
    for k in ("B_internal_mmolN_per_d", "C_atm_mmolN_per_d",
              "D_swi_mmolN_per_d", "E_boundary_mmolN_per_d",
              "F_pointsrc_mmolN_per_d"):
        totals[k] = float((df[k].fillna(0) * dt_days).sum())
    delta_storage = float(df["storage_mmolN"].iloc[-1] - df["storage_mmolN"].iloc[0])
    closure = delta_storage - (totals["B_internal_mmolN_per_d"]
                                + totals["C_atm_mmolN_per_d"]
                                + totals["D_swi_mmolN_per_d"]
                                + totals["E_boundary_mmolN_per_d"]
                                + totals["F_pointsrc_mmolN_per_d"])
    initial_storage = float(df["storage_mmolN"].iloc[0])
    closure_pct = abs(closure) / abs(delta_storage) * 100 if delta_storage != 0 else float("nan")

    def t(x):
        return f"{x * MMOL_TO_TONNES:+.2f}"

    boundary_rows = "".join(
        f"<li>region <b>{br['region_id']}</b> (sign {br['sign']:+d}) — {br.get('label', '')}</li>"
        for br in region_config.get("boundary_fluxes", [])
    )

    ps_info = terms.get("point_source_info")
    if ps_info and ps_info.get("cv_src_idx"):
        rows = "".join(
            f"<tr><td>{i}</td><td>{e}</td></tr>"
            for i, e in zip(ps_info["cv_src_idx"], ps_info["cv_src_elem"])
        )
        ps_rows_html = (
            f"<p><b>Sources inside this CV ({len(ps_info['cv_src_idx'])}):</b></p>"
            f"<table><tr><th>src #</th><th>element ID (1-indexed)</th></tr>"
            f"{rows}</table>"
        )
    elif ps_info is None:
        ps_rows_html = "<p class='dim'>Term F skipped (no --run-dir provided).</p>"
    else:
        ps_rows_html = "<p class='dim'>No point sources inside this CV.</p>"

    warn_html = ""
    if warnings:
        warn_html = "<h3>Warnings &amp; sanity notes</h3><ul>" + \
                    "".join(f"<li>{w}</li>" for w in warnings) + "</ul>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{group_name} budget — {region_config['name']}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        max-width: 1100px; margin: 2em auto; color: #1e293b; padding: 0 1em; }}
h1 {{ color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 4px; }}
h2 {{ color: #1e40af; margin-top: 1.6em; border-bottom: 1px solid #cbd5e1; padding-bottom: 2px; }}
h3 {{ color: #3b82f6; }}
table {{ border-collapse: collapse; margin: 0.5em 0; }}
th, td {{ padding: 5px 10px; border: 1px solid #cbd5e1; text-align: left; }}
th {{ background: #f1f5f9; }}
.gain {{ color: #047857; font-weight: bold; }}
.loss {{ color: #dc2626; font-weight: bold; }}
.dim  {{ color: #64748b; }}
code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; }}
.warn {{ background: #fef3c7; border-left: 4px solid #b45309; padding: 6px 10px; }}
</style></head><body>

<h1>{group_name.capitalize()} mass-balance budget</h1>
<p><b>Region:</b> {region_config['name']}<br>
<b>Description:</b> {region_config.get('description', '(none)')}<br>
<b>Period:</b> {rep_period[0]:%Y-%m-%d} to {rep_period[1]:%Y-%m-%d}
({period_days:.1f} days)<br>
<b>Generated:</b> {dt.datetime.now():%Y-%m-%d %H:%M}</p>

<h2>Period totals (tonnes N)</h2>
<table>
<tr><th>Term</th><th>Cumulative (t N)</th><th>Interpretation</th></tr>
<tr><td><b>ΔStorage</b> (observed)</td>
    <td>{t(delta_storage)}</td>
    <td>Change in total-N inventory between start and end of period</td></tr>
<tr><td><b>B</b> — internal (denit + anammox)</td>
    <td class="loss">{t(totals['B_internal_mmolN_per_d'])}</td>
    <td>Sum of net N losses to N<sub>2</sub> gas</td></tr>
<tr><td><b>C</b> — atmospheric DIN deposition</td>
    <td>{t(totals['C_atm_mmolN_per_d'])}</td>
    <td>Net N from atmospheric deposition into water</td></tr>
<tr><td><b>D</b> — sediment-water flux</td>
    <td>{t(totals['D_swi_mmolN_per_d'])}</td>
    <td>Net N from sediment efflux into water column</td></tr>
<tr><td><b>E</b> — boundary inflow</td>
    <td>{t(totals['E_boundary_mmolN_per_d'])}</td>
    <td>Net N transported across the bounding transects (into CV)</td></tr>
<tr><td><b>F</b> — point-source (rivers)</td>
    <td class="gain">{t(totals['F_pointsrc_mmolN_per_d'])}</td>
    <td>Riverine N input via interior source elements (vsource × msource)</td></tr>
<tr><td><b>Closure residual</b><br>(should be ≈ 0)</td>
    <td><b>{t(closure)}</b></td>
    <td>{closure_pct:.1f}% of ΔStorage</td></tr>
</table>

<p class="dim">Initial storage on first timestep: {t(initial_storage)} t N.
Conversion: 1 mmol N × 14.0067 / 1e9 = tonnes N.</p>

<h2>Time series — instantaneous</h2>
<p>Stacked-area: each coloured layer is one budget term contributing to dM/dt.
The black line shows the observed dM/dt computed from finite-differences of
the storage time series. Where the line follows the top of the stacks, the
budget closes.</p>
{_embed_png(stacked_png)}

<h2>Cumulative time-integrals</h2>
<p>Each term integrated from <code>t<sub>0</sub></code> in tonnes N. The
black line is observed ΔStorage. Closure if the coloured lines sum to the
black line.</p>
{_embed_png(cum_png)}

<h2>How each term is computed</h2>

<h3>Term A — Storage (mmol N)</h3>
<p>Volume integral of all N-containing state-variable concentrations over the
control volume's elements and layers:</p>
<p><code>M(t) = Σ<sub>elem∈CV</sub> Σ<sub>layer</sub>
  c(elem, layer, t) × layer_ht × elem_area × n_per_mol</code></p>
<p>State pools summed (all in mmol N/m³ after the per-mol conversion):
NIT_amm, NIT_nit, OGM_don, OGM_pon, PHY_mixed (×16/106 Redfield N:C).</p>

<h3>Term B — Internal rates (mmol N/day)</h3>
<p>Volume integral of process rates that are net losses from the total-N pool:</p>
<ul>
  <li><code>NIT_denit</code> (NO<sub>3</sub> → N<sub>2</sub>, sign −1)</li>
  <li><code>NIT_anammox</code> (NH<sub>4</sub>+NO<sub>2</sub> → N<sub>2</sub>, sign −1)</li>
</ul>
<p>Other rates (nitrification, DNRA, mineralisation, hydrolysis, phyto uptake)
are tracked individually but their <code>sign_in_pool = 0</code> so they
cancel in the total-N closure.</p>

<h3>Term C — Atmospheric flux (mmol N/day)</h3>
<p>Area integral of <code>NIT_din_atm</code> (2D sheet, mmol N/m²/day) over
the surface area of the CV.</p>

<h3>Term D — Sediment-water flux (mmol N/day)</h3>
<p>Sum of 2D sheet diagnostics, area-integrated over the CV:</p>
<ul>
  <li>NIT_amm_dsf, NIT_nit_dsf, OGM_don_swi, OGM_pon_swi, PHY_phy_swi_n</li>
</ul>
<p>All taken as <i>positive into the water column</i>.</p>

<h3>Term E — Boundary inflow (mmol N/day)</h3>
<p>From <code>flux.out</code>: signed sum of N-tracer transports across the
control volume's bounding fluxflag regions:</p>
<ul>{boundary_rows}</ul>
<p>Tracer columns summed (with n-per-mol weighting):
NIT_amm, NIT_nit, OGM_don, OGM_pon, PHY_mixed (×16/106).</p>

<h3>Term F — Point-source (river) input (mmol N/day)</h3>
<p>River nutrient inputs from <code>source_sink.in</code> + <code>vsource.th</code>
+ <code>msource.th</code> for sources whose target element is interior to the
control volume:</p>
<p><code>F(t) = Σ<sub>src∈CV</sub> Σ<sub>t∈N-pools</sub>
  Q<sub>src</sub>(t) × c<sub>src,t</sub>(t) × n_per_mol[t] × 86400</code></p>
<p>msource.th concentrations are in mmol/m³ (same unit as flux.out), so no
extra conversion is needed. Pools summed:
NIT_amm, NIT_nit, OGM_don, OGM_pon, PHY_mixed (×16/106).</p>
{ps_rows_html}

{warn_html}

<h2>Closure-check interpretation</h2>
<p>If closure residual is small relative to ΔStorage (say &lt; 10%), the
budget is closed within numerical accuracy. Common reasons for non-closure:</p>
<ul>
<li><b>Time-cadence mismatch</b> between scribed and cmb outputs (storage is
  read on scribed cadence; rates+fluxes on cmb cadence).</li>
<li><b>flux.out tracer ordering</b> mismatched against
  <code>DEFAULT_TRACER_ORDER</code> in <code>boundary_fluxes.py</code>.</li>
<li><b>fluxflag.prop boundary signs</b> incorrectly specified (positive flux at
  a transect goes IN one direction; the sign in the region config must reflect
  whether that direction is into or out of the CV).</li>
<li><b>msource.th tracer-column ordering</b> — Term F assumes the AED state-var
  registration order in <code>DEFAULT_AED_TRACER_ORDER</code>
  (<code>point_sources.py</code>); override if your build differs.</li>
</ul>

</body></html>"""
    html_path.write_text(html, encoding="utf-8")
