"""Normalize ``RequireInputData.xlsx`` into a typed, nested spec structure.

The workbook is the human-facing input (easy to hand-edit). This module is the
single place that knows its column names and quirks; everything downstream
consumes the clean dict / JSON this produces, so a column rename only ever
breaks *here*.

Structure produced by :func:`load_specs`::

    {
      "plant": { site_name, latitude, longitude, altitude_m,
                 size_kwp_dc, size_kw_ac, tilt_deg,
                 ws_azimuth_deg, azimuth_deg,        # raw + N-referenced
                 commissioning_date,
                 irradiance_available, module_temp_available, rain_available },
      "inverters": {
        "<Inverter SN>": {
          <inverter spec fields...>,
          "strings": {
            "MPPT1|PV1": { mppt, pv, <panel spec fields...> },
            ...
          }
        }
      }
    }

Join with the measured CSV: ``inverter_id`` -> Inverter SN, ``mppt_id`` -> MPPT,
``string_id`` -> PV (case-insensitive; the CSV uses ``pv1`` while the sheet uses
``PV1``). Use :func:`string_spec` to resolve a measured row's specs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Default location of the spec workbook inside the repo.
DEFAULT_REQUIRE_INPUT = (
    Path(__file__).resolve().parents[1] / "RequireInputData" / "RequireInputData.xlsx"
)


# ── column matching ─────────────────────────────────────────────────────────
# Robust to degree symbols, unit suffixes and minor header edits: we match on
# lowercased substrings rather than exact header text.

def _col(df: pd.DataFrame, *keywords: str) -> str:
    """Return the column whose lowercased name contains *all* keywords."""
    for c in df.columns:
        cl = str(c).lower()
        if all(k in cl for k in keywords):
            return c
    raise KeyError(f"No column matching {keywords} in {list(df.columns)}")


def _num(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _yesno(value: Any) -> bool:
    return str(value).strip().lower().startswith("y")


def string_key(mppt: str, pv: str) -> str:
    """Canonical key for a string: ``MPPT<n>|PV<n>`` (upper-cased, trimmed)."""
    return f"{str(mppt).strip().upper()}|{str(pv).strip().upper()}"


# ── readers ─────────────────────────────────────────────────────────────────

def _read_plant(path: Path) -> dict:
    df = pd.read_excel(path, sheet_name="Plant")
    if df.empty:
        raise ValueError("Plant sheet is empty")
    r = df.iloc[0]

    ws_azimuth = _num(r[_col(df, "azimuth")])          # south-referenced, east -ve
    # Convert to the analysis convention (0=N, 90=E, 180=S, 270=W).
    azimuth = None if ws_azimuth is None else 180.0 + ws_azimuth

    return {
        "site_name": str(r[_col(df, "site name")]).strip(),
        "latitude": _num(r[_col(df, "latitude")]),
        "longitude": _num(r[_col(df, "longitude")]),
        "altitude_m": _num(r[_col(df, "altitude")]),
        "size_kwp_dc": _num(r[_col(df, "system size", "dc")]),
        "size_kw_ac": _num(r[_col(df, "system size", "ac")]),
        "tilt_deg": _num(r[_col(df, "tilt")]),
        "ws_azimuth_deg": ws_azimuth,
        "azimuth_deg": azimuth,
        "commissioning_date": pd.to_datetime(
            r[_col(df, "commissioning")]
        ).date().isoformat(),
        "irradiance_available": _yesno(r[_col(df, "irradiance")]),
        "module_temp_available": _yesno(r[_col(df, "module temperature")]),
        "rain_available": _yesno(r[_col(df, "rain")]),
    }


def _read_inverters(path: Path) -> dict[str, dict]:
    df = pd.read_excel(path, sheet_name="Inverter")
    sn = _col(df, "inverter sn")
    fields = {
        "manufacturer": (_col(df, "manufacturer"), str),
        "model": (_col(df, "model"), str),
        "capacity_kw_ac": (_col(df, "capacity", "ac"), _num),
        "max_input_voltage_v": (_col(df, "max input voltage"), _num),
        "num_mppts": (_col(df, "number of mppt"), lambda v: int(_num(v))),
        "max_current_per_mppt_a": (_col(df, "max current", "mppt"), _num),
        "max_isc_per_mppt_a": (_col(df, "short circuit", "mppt"), _num),
        "start_voltage_v": (_col(df, "start voltage"), _num),
        "mppt_v_min": (_col(df, "operating voltage", "min"), _num),
        "mppt_v_max": (_col(df, "operating voltage", "max"), _num),
        "nominal_input_voltage_v": (_col(df, "nominal input voltage"), _num),
        "max_efficiency_pct": (_col(df, "efficiency"), _num),
        "nominal_ac_power_kw": (_col(df, "nominal ac active power"), _num),
        "max_ac_power_kw": (_col(df, "max ac active power"), _num),
    }
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        key = str(r[sn]).strip()
        out[key] = {"inverter_sn": key}
        out[key].update({name: cast(r[col]) for name, (col, cast) in fields.items()})
        out[key]["strings"] = {}
    return out


def _read_panels(path: Path) -> list[dict]:
    df = pd.read_excel(path, sheet_name="Panel")
    sn = _col(df, "inverter sn")
    mppt_c = _col(df, "mppt")
    pv_c = _col(df, "pv")
    fields = {
        "string_capacity_w": (_col(df, "string capacity"), _num),
        "num_panels": (_col(df, "number of panels"), lambda v: int(_num(v))),
        "panel_manufacturer": (_col(df, "panel manufacturer"), str),
        "panel_model": (_col(df, "panel model"), str),
        "panel_capacity_wp": (_col(df, "panel capacity"), _num),
        "voc_v": (_col(df, "voc"), _num),
        "isc_a": (_col(df, "isc"), _num),
        "vmp_v": (_col(df, "vmp"), _num),
        "imp_a": (_col(df, "imp"), _num),
        "alpha_isc_pct_per_c": (_col(df, "alpha"), _num),
        "beta_voc_pct_per_c": (_col(df, "beta"), _num),
        "gamma_pmax_pct_per_c": (_col(df, "gamma"), _num),
        "technology": (_col(df, "technology"), str),
        "first_year_degradation_pct": (_col(df, "first year degradation"), _num),
        "annual_degradation_pct": (_col(df, "annual degradation"), _num),
        "bifacial": (_col(df, "bifacial"), _yesno),
        "num_cells": (_col(df, "number of cells"), lambda v: int(_num(v))),
    }
    rows: list[dict] = []
    for _, r in df.iterrows():
        rec = {
            "inverter_sn": str(r[sn]).strip(),
            "mppt": str(r[mppt_c]).strip().upper(),
            "pv": str(r[pv_c]).strip().upper(),
        }
        rec.update({name: cast(r[col]) for name, (col, cast) in fields.items()})
        rows.append(rec)
    return rows


# ── public API ──────────────────────────────────────────────────────────────

def load_specs(path: str | Path = DEFAULT_REQUIRE_INPUT) -> dict:
    """Load the spec workbook into the normalized nested dict (see module doc)."""
    path = Path(path)
    plant = _read_plant(path)
    inverters = _read_inverters(path)

    for panel in _read_panels(path):
        inv_sn = panel["inverter_sn"]
        inv = inverters.get(inv_sn)
        if inv is None:
            # String references an inverter not listed on the Inverter sheet —
            # keep it rather than dropping data silently.
            inv = inverters[inv_sn] = {"inverter_sn": inv_sn, "strings": {}}
        inv["strings"][string_key(panel["mppt"], panel["pv"])] = panel

    return {"plant": plant, "inverters": inverters}


def string_spec(
    specs: dict, inverter_id: str, mppt_id: str, string_id: str
) -> dict | None:
    """Resolve the panel spec for a measured row (case-insensitive on the PV id)."""
    inv = specs["inverters"].get(str(inverter_id).strip())
    if inv is None:
        return None
    return inv["strings"].get(string_key(mppt_id, string_id))


def specs_to_json(specs: dict, indent: int = 2) -> str:
    return json.dumps(specs, indent=indent, ensure_ascii=False, default=str)


# ── CLI: dump the normalized JSON for inspection ──────────────────────────────

if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUIRE_INPUT
    specs = load_specs(src)
    n_inv = len(specs["inverters"])
    n_str = sum(len(i["strings"]) for i in specs["inverters"].values())
    print(specs_to_json(specs))
    print(
        f"\n# loaded: 1 plant, {n_inv} inverters, {n_str} strings",
        file=sys.stderr,
    )
