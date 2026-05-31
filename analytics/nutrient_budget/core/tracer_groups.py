"""
Tracer-group definitions for budget runs.

A tracer group declares everything needed for a closed mass-balance:
- which state-variable NetCDFs hold the pool concentrations (Term A)
- which 3D diagnostic variables are internal rates (Term B), and whether each
  is a net loss from the pool (sign_in_pool != 0) or internal recycling (0)
- which 2D sheet variables are atmospheric fluxes (Term C) and sediment-water
  interface fluxes (Term D)
- which tracer columns in flux.out carry the pool's mass (Term E)

Sign conventions throughout:
    positive  =  into the budget pool / into the water column
    negative  =  out of the budget pool / out of the water column

`n_per_mol`: how many moles of the budget element (N here) each mole of the
storage variable represents. PHY_mixed is mmol C, so its N content is
multiplied by the Redfield N:C ratio (16/106).
"""

REDFIELD_N_OVER_C = 16.0 / 106.0


# -------- Nitrogen budget bundle ---------------------------------------------
# Net N losses = denitrification + anammox (gaseous loss to N2).
# DNRA is NOT a loss — it cycles NO3 -> NH4 within the N pool.
# Nitrification, mineralisation, hydrolysis and uptake all redistribute N
# between pools but are internal to the total-N pool, so sign_in_pool = 0.
NITROGEN = {
    "name": "nitrogen",
    "element_symbol": "N",
    "molar_mass": 14.0067,   # g/mol — for mmol N → tonnes N conversion

    # ------ Term A: state pools (mmol N/m^3) -------------------------------
    # source = 'scribed': read from <name>_<S>.nc node-centred files
    # source = 'cmb'    : read from aed_data_cmb_<S>.nc (element-centred)
    "state": {
        "NIT_amm":   {"source": "scribed", "n_per_mol": 1.0},
        "NIT_nit":   {"source": "scribed", "n_per_mol": 1.0},
        "OGM_don":   {"source": "scribed", "n_per_mol": 1.0},
        "OGM_pon":   {"source": "scribed", "n_per_mol": 1.0},
        "PHY_mixed": {"source": "scribed", "n_per_mol": REDFIELD_N_OVER_C},
    },

    # ------ Term B: 3D internal-process rates (mmol N/m^3/d) ---------------
    # sign_in_pool: +1 = adds N to total pool (atm-like source)
    #               -1 = removes N from total pool (gas loss)
    #                0 = internal cycling between pools (track but cancels in budget)
    "rates": {
        "NIT_nitrif":  {"sign_in_pool":  0, "label": "Nitrification (NH4 -> NO3)"},
        "NIT_denit":   {"sign_in_pool": -1, "label": "Denitrification (NO3 -> N2, loss)"},
        "NIT_anammox": {"sign_in_pool": -1, "label": "Anammox (NH4+NO2 -> N2, loss)"},
        "NIT_dnra":    {"sign_in_pool":  0, "label": "DNRA (NO3 -> NH4, internal)"},
        "OGM_don_min": {"sign_in_pool":  0, "label": "DON mineralisation"},
        "OGM_pon_hyd": {"sign_in_pool":  0, "label": "PON hydrolysis"},
        "PHY_upt_no3": {"sign_in_pool":  0, "label": "Phytoplankton NO3 uptake"},
        "PHY_upt_nh4": {"sign_in_pool":  0, "label": "Phytoplankton NH4 uptake"},
    },

    # ------ Term C: 2D atmospheric flux (mmol N/m^2/d) ---------------------
    "atm": {
        "NIT_din_atm": {"sign_into_water": +1, "label": "DIN atmospheric deposition"},
    },

    # ------ Term D: 2D sediment-water interface flux (mmol N/m^2/d) --------
    # SWI fluxes positive = sediment -> water; negative = water -> sediment.
    "swi": {
        "NIT_amm_dsf":   {"sign_into_water": +1, "label": "NH4 SWI flux"},
        "NIT_nit_dsf":   {"sign_into_water": +1, "label": "NO3 SWI flux"},
        "OGM_don_swi":   {"sign_into_water": +1, "label": "DON SWI flux"},
        "OGM_pon_swi":   {"sign_into_water": +1, "label": "PON SWI flux"},
        "PHY_phy_swi_n": {"sign_into_water": +1, "label": "Phyto N SWI flux"},
    },

    # ------ Term E: flux.out tracer columns that carry N -------------------
    # `tracers_n_per_mol` lists every tracer whose flux.out entries should
    # be aggregated into Term E, with their mol-N-per-mol conversion factor.
    "boundary_tracers_n_per_mol": {
        "NIT_amm":   1.0,
        "NIT_nit":   1.0,
        "OGM_don":   1.0,
        "OGM_pon":   1.0,
        "PHY_mixed": REDFIELD_N_OVER_C,
    },

    # ------ Term F: point-source (river) tracer input ----------------------
    # Names must match entries in DEFAULT_AED_TRACER_ORDER in point_sources.py
    # (i.e. msource.th column order). Units in msource are mmol X/m^3 for AED
    # state variables; same as flux.out, so no extra conversion is needed.
    "point_source_tracers_n_per_mol": {
        "NIT_amm":   1.0,
        "NIT_nit":   1.0,
        "OGM_don":   1.0,
        "OGM_pon":   1.0,
        "PHY_mixed": REDFIELD_N_OVER_C,
    },
}


# Registry of available groups
GROUPS = {
    "nitrogen": NITROGEN,
}


def get_group(name):
    """Return a tracer-group dict by name; raises KeyError if unknown."""
    if name not in GROUPS:
        raise KeyError(
            f"Unknown tracer group '{name}'. Available: {list(GROUPS.keys())}"
        )
    return GROUPS[name]
