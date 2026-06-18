"""Load the measured long-format CSV into a tidy per-string time-series frame.

The CSV (exported by ``data_gather_from_mongo_db``) packs four levels into one
file, tagged by a ``level`` column: ``plant`` / ``inverter`` / ``mppt`` /
``string``. Each row only fills the columns belonging to its own level.

Per-string soiling analysis lives at the ``string`` level, but one signal it
needs is not on string rows:

* **irradiance** — present only on ``plant`` rows. We broadcast it onto string
  rows by timestamp (plant rows are unique per timestamp).

``pv_temperature`` *is* already on string rows (enriched upstream from the EMI
sensor), so it is kept as-is.

Unit note: the source column is named ``irradiance_wm2`` but its values peak
around 0.9 at solar noon — i.e. it is actually **kW/m²**, matching the legacy
input format. We surface it as ``irradiance_kw_m2`` so downstream physics does
not silently apply a 1000x error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical join-key columns (also present on string rows).
ID_COLS = ["inverter_id", "mppt_id", "string_id"]

# String-level columns we carry forward (measured signals + ids + metadata).
_STRING_KEEP = [
    "plant_name", "timestamp", "day_hour",
    "inverter_id", "mppt_id", "string_id",
    "string_capacity_kwp", "string_power_kw",
    "string_current_a", "string_voltage_v",
    "string_specific_yield", "string_pr",
    "string_deviation", "string_sy_deviation_vs_inverter",
    "pv_temperature",
]

_IRRADIANCE_SRC = "irradiance_wm2"   # mislabeled in the CSV; values are kW/m²
IRRADIANCE_COL = "irradiance_kw_m2"


def load_measured(csv_path: str | Path) -> pd.DataFrame:
    """Read the long CSV → tidy string-level frame with irradiance broadcast.

    Returns one row per (string, timestamp) carrying the measured signals plus
    an ``irradiance_kw_m2`` column broadcast from the matching ``plant`` row.
    """
    raw = pd.read_csv(csv_path, low_memory=False)
    if "level" not in raw.columns:
        raise ValueError(
            f"{csv_path} has no 'level' column — not the expected long-format CSV"
        )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # Plant-level irradiance keyed by timestamp (unique per timestamp; guard anyway).
    plant = raw[raw["level"] == "plant"]
    irr = (
        plant.dropna(subset=[_IRRADIANCE_SRC])
        .set_index("timestamp")[_IRRADIANCE_SRC]
    )
    irr = irr[~irr.index.duplicated(keep="first")]

    strings = raw[raw["level"] == "string"].copy()
    strings = strings[[c for c in _STRING_KEEP if c in strings.columns]]
    for c in ID_COLS:
        strings[c] = strings[c].astype(str).str.strip()
    strings[IRRADIANCE_COL] = strings["timestamp"].map(irr)

    return strings.reset_index(drop=True)
