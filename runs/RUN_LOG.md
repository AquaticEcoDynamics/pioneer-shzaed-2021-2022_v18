# Run log

Progress tracker for the simulations under `runs/`. Each run builds on the
previous one. See the repository README (one level up) for model background,
build, and analysis details.

| ID | Name | In repo | Purpose |
|----|------|:-------:|---------|
| 001 | Pioneer_17 | no | Original baseline setup. Archived elsewhere — not committed here. |
| 002 | `002_P18_flood` | yes | Pioneer 18 with an **artificial flood pulse**, plus a simplified `fluxflag.prop` to enable deep testing of the budget / boundary-flux calculations. |
| 003 | `003_P18_flood_flat` | yes | Copy of 002 with the **modified flood pulse** and a **quiescent (flat) ocean tide for the first 14 days** (cosine ramp to full tide over days 14–16). Isolates the previously noted flux-direction inconsistencies across the Pioneer mouth. |

## Notes

- **001 → 002**: introduced the artificial flood and simplified `fluxflag.prop`
  (fewer transect regions) so the boundary-flux budget can be closed and tested
  cleanly. Added the **GEN_1** general tracer. Also enabled
  `iadjust_mass_consv0(3) = 1` (mass-conservation adjustment for the GEN
  module) — **this led to NaNs**.
- **002 → 003**: tide forcing switched from harmonic (`bctides.in` `iettype=3`)
  to space/time-varying (`iettype=4`, `elev2D.th.nc`). Days 0–14 are held flat
  at 0 m; from day 16 the boundary tide is identical to 002. Everything else is
  unchanged. Generator: `analytics/utilities/make_flat_tide_elev2D.py`.
  `iadjust_mass_consv0(3)` is **disabled** (commented out) to avoid the NaNs
  seen in 002.

Run length 91 days (`rnday`), `dt=45 s`, start 2021-04-01 00:00 UTC.
