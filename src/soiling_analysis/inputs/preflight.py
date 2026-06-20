"""Data-availability preflight checks — run once at load time.

Returns a capability dict keyed C1–C6.  Every schema-dependent section
downstream branches on this dict; nothing hard-fails here.

C1 — per-string orientation present
C2 — per-string commissioning date present
C3 — rear-side irradiance for bifacial modules
C4 — inverter AC capacity + measured inverter AC power rows
C5 — rain available
C6 — economics inputs (tariff + wash cost) present
"""
from __future__ import annotations
import logging

import pandas as pd

log = logging.getLogger(__name__)


def preflight_schema(
    specs: dict,
    raw_df: pd.DataFrame,
    wash_cost_pkr: float | None = None,
) -> dict:
    """Run the six capability checks and return a capability dict.

    Parameters
    ----------
    specs       : output of ``load_specs()``
    raw_df      : the raw measured CSV DataFrame (all levels, before filtering)
    wash_cost_pkr : wash cost per string in PKR; None means not yet supplied

    Returns
    -------
    dict with keys C1–C6 (bool) and ``details`` (human-readable strings).
    """
    details: dict[str, str] = {}

    all_strings = [
        st
        for inv in specs.get("inverters", {}).values()
        for st in inv.get("strings", {}).values()
    ]

    # ------------------------------------------------------------------
    # C1 — per-string orientation
    # ------------------------------------------------------------------
    c1 = bool(
        all_strings
        and all(
            st.get("string_azimuth_deg") is not None
            and st.get("string_tilt_deg") is not None
            for st in all_strings
        )
    )
    details["C1"] = (
        "per-string az/tilt present in Panel sheet"
        if c1
        else "orientation_source=plant_default (Panel sheet missing az/tilt)"
    )

    # ------------------------------------------------------------------
    # C2 — per-string commissioning date
    # ------------------------------------------------------------------
    c2 = bool(
        all_strings
        and all(st.get("string_commissioning_date") is not None for st in all_strings)
    )
    details["C2"] = (
        "per-string commissioning date present"
        if c2
        else "age_source=plant_default (Panel sheet missing commissioning date)"
    )

    # ------------------------------------------------------------------
    # C3 — rear-side irradiance for bifacial
    # ------------------------------------------------------------------
    any_bifacial = any(st.get("bifacial", False) for st in all_strings)
    cols_lower = [str(c).lower() for c in raw_df.columns]
    rear_col_present = any("rear" in cl or "back" in cl for cl in cols_lower)

    if not any_bifacial:
        c3 = False
        details["C3"] = "no bifacial modules — bifacial gain not applicable"
    elif rear_col_present:
        c3 = True
        details["C3"] = "bifacial=Yes and rear irradiance column present — measured gain"
    else:
        c3 = False
        details["C3"] = (
            "bifacial=Yes but no rear irradiance column — "
            "modeled gain (bifacial_gain_source='modeled') or skip"
        )

    # ------------------------------------------------------------------
    # C4 — inverter AC capacity + measured inverter AC power rows
    # ------------------------------------------------------------------
    inv_specs_ok = bool(specs.get("inverters")) and all(
        inv.get("max_ac_power_kw") is not None
        for inv in specs.get("inverters", {}).values()
    )
    inv_rows = (
        raw_df[raw_df["level"] == "inverter"]
        if "level" in raw_df.columns
        else pd.DataFrame()
    )
    c4 = bool(inv_specs_ok and len(inv_rows) > 0)
    details["C4"] = (
        "inverter AC capacity in specs + measured AC power rows present"
        if c4
        else (
            "inverter AC capacity missing from specs" if not inv_specs_ok
            else "no level=='inverter' rows in CSV — AC reconstructed from string DC"
        )
    )

    # ------------------------------------------------------------------
    # C5 — rain available
    # ------------------------------------------------------------------
    c5 = bool(specs.get("plant", {}).get("rain_available", False))
    details["C5"] = (
        "rain data available — recovery anchors and wash-cause attribution active"
        if c5
        else "rain_available=No — wash cause='suspected', recovery anchors rare"
    )

    # ------------------------------------------------------------------
    # C6 — economics inputs (tariff + wash cost)
    # ------------------------------------------------------------------
    # Tariff comes from the workbook plant sheet or falls back to DEFAULT_TARIFF_PKR_PER_KWH.
    # Wash cost must be supplied via CLI (--wash-cost).
    tariff_in_workbook = specs.get("plant", {}).get("tariff_pkr_kwh") is not None
    wash_cost_supplied = wash_cost_pkr is not None
    c6 = wash_cost_supplied  # tariff always has a default; wash cost is the missing piece
    if c6:
        details["C6"] = f"tariff present; wash cost={wash_cost_pkr:.0f} PKR/string supplied"
    elif tariff_in_workbook:
        details["C6"] = "tariff in workbook; wash cost not supplied — supply via --wash-cost"
    else:
        details["C6"] = (
            "tariff defaulted to DEFAULT_TARIFF_PKR_PER_KWH; "
            "wash cost not supplied — economics_inputs='defaulted'"
        )

    result = {
        "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6,
        "details": details,
    }
    log.info(
        "Preflight: C1=%s C2=%s C3=%s C4=%s C5=%s C6=%s",
        c1, c2, c3, c4, c5, c6,
    )
    return result
