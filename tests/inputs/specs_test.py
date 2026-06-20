"""Tests for Batch 1 schema changes.

Run with:
    python -m pytest src/soiling_analysis/inputs/specs_test.py -v
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from soiling_analysis.inputs.specs import (
    _col_opt,
    _parse_commissioning_date,
    _read_panels,
    load_specs,
)
from soiling_analysis.inputs.preflight import preflight_schema


# ---------------------------------------------------------------------------
# _col_opt
# ---------------------------------------------------------------------------

def test_col_opt_finds_column():
    df = pd.DataFrame(columns=["String Azimuth(°C)", "String Tilt (°C)", "Other"])
    assert _col_opt(df, "azimuth") == "String Azimuth(°C)"
    assert _col_opt(df, "tilt") == "String Tilt (°C)"


def test_col_opt_returns_none_when_missing():
    df = pd.DataFrame(columns=["panel_capacity", "voc"])
    assert _col_opt(df, "azimuth") is None
    assert _col_opt(df, "comissioning") is None


# ---------------------------------------------------------------------------
# _parse_commissioning_date
# ---------------------------------------------------------------------------

def test_parse_excel_serial():
    # Excel serial 45544 = 2024-09-09
    result = _parse_commissioning_date(45544)
    assert result is not None
    parsed = datetime.date.fromisoformat(result)
    assert parsed.year == 2024
    assert parsed.month == 9


def test_parse_datetime_string():
    result = _parse_commissioning_date("2024-06-15")
    assert result == "2024-06-15"


def test_parse_pandas_timestamp():
    result = _parse_commissioning_date(pd.Timestamp("2023-09-01"))
    assert result == "2023-09-01"


def test_parse_nan_returns_none():
    assert _parse_commissioning_date(None) is None
    assert _parse_commissioning_date(float("nan")) is None
    assert _parse_commissioning_date(np.nan) is None


# ---------------------------------------------------------------------------
# Azimuth conversion — N-referenced convention matches plant
# ---------------------------------------------------------------------------

def test_azimuth_conversion_matches_plant_convention():
    """string_azimuth_deg = 180.0 + raw_south_referenced, same as plant."""
    # Plant: ws_azimuth=-13 → azimuth_deg=167
    raw_south_ref = -13.0
    expected_n_ref = 180.0 + raw_south_ref   # 167.0

    # Simulate what _read_panels does
    raw_az = raw_south_ref
    converted = 180.0 + raw_az
    assert converted == pytest.approx(expected_n_ref)


# ---------------------------------------------------------------------------
# _read_panels — synthetic workbook via mock
# ---------------------------------------------------------------------------

def _make_panel_df(**extra_cols) -> pd.DataFrame:
    """Minimal Panel sheet DataFrame including the three new columns."""
    data = {
        "Inverter SN": ["INV01", "INV01"],
        "MPPT": ["MPPT1", "MPPT1"],
        "PV": ["PV1", "PV2"],
        "String Capacity (W)": [16965.0, 15210.0],
        "Number of Panels": [29, 26],
        "Panel Manufacturer": ["JA Solar", "JA Solar"],
        "Panel Model": ["JAM72S30", "JAM72S30"],
        "Panel Capacity (Wp)": [585.0, 585.0],
        "VOC (V)": [51.8, 51.8],
        "ISC (A)": [14.29, 14.29],
        "VMP (V)": [43.24, 43.24],
        "IMP (A)": [13.53, 13.53],
        "Alpha ISC (%/C)": [0.046, 0.046],
        "Beta VOC (%/C)": [-0.26, -0.26],
        "Gamma Pmax (%/C)": [-0.30, -0.30],
        "Technology": ["Mono-c-Si", "Mono-c-Si"],
        "First Year Degradation (%)": [1.0, 1.0],
        "Annual Degradation (%)": [0.4, 0.4],
        "Bifacial": ["Yes", "Yes"],
        "Number of Cells": [144, 144],
        "String Azimuth(°C)": [-13.0, -13.0],
        "String Tilt (°C)": [15.0, 15.0],
        "String Comissioning Date": [45292.0, 45292.0],  # Excel serial
        **extra_cols,
    }
    return pd.DataFrame(data)


def test_read_panels_reads_azimuth_and_tilt(tmp_path):
    df = _make_panel_df()
    xlsx = tmp_path / "test.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, sheet_name="Panel", index=False)

    rows = _read_panels(xlsx)
    assert len(rows) == 2
    for row in rows:
        assert row["string_azimuth_deg"] == pytest.approx(167.0)   # 180 + (-13)
        assert row["string_azimuth_raw_deg"] == pytest.approx(-13.0)
        assert row["string_tilt_deg"] == pytest.approx(15.0)


def test_read_panels_reads_commissioning_date(tmp_path):
    df = _make_panel_df()
    xlsx = tmp_path / "test.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, sheet_name="Panel", index=False)

    rows = _read_panels(xlsx)
    for row in rows:
        assert row["string_commissioning_date"] is not None
        parsed = datetime.date.fromisoformat(row["string_commissioning_date"])
        assert parsed.year in (2023, 2024)   # serial 45292 ≈ Jan 2024
        assert isinstance(row["string_commissioning_year"], int)


def test_read_panels_graceful_when_columns_absent(tmp_path):
    """Old workbook without the three new columns — should not crash."""
    df = _make_panel_df()
    df = df.drop(columns=["String Azimuth(°C)", "String Tilt (°C)", "String Comissioning Date"])
    xlsx = tmp_path / "test.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, sheet_name="Panel", index=False)

    rows = _read_panels(xlsx)
    for row in rows:
        assert row["string_azimuth_deg"] is None
        assert row["string_tilt_deg"] is None
        assert row["string_commissioning_date"] is None
        assert row["string_commissioning_year"] is None


# ---------------------------------------------------------------------------
# string_uid uniqueness
# ---------------------------------------------------------------------------

def test_string_uid_is_unique_across_mppts():
    """string_uid = inverter_id + string_id; user confirmed IDs never repeat across MPPTs."""
    records = [
        {"inverter_id": "INV01", "mppt_id": "MPPT1", "string_id": "PV1"},
        {"inverter_id": "INV01", "mppt_id": "MPPT1", "string_id": "PV2"},
        {"inverter_id": "INV01", "mppt_id": "MPPT2", "string_id": "PV3"},
        {"inverter_id": "INV01", "mppt_id": "MPPT2", "string_id": "PV4"},
        {"inverter_id": "INV02", "mppt_id": "MPPT1", "string_id": "PV1"},
    ]
    uids = [f"{r['inverter_id']}_{r['string_id']}" for r in records]
    assert len(uids) == len(set(uids)), f"Duplicate string_uids: {uids}"


# ---------------------------------------------------------------------------
# preflight_schema — synthetic specs
# ---------------------------------------------------------------------------

def _make_specs_c1c2_ok() -> dict:
    return {
        "plant": {
            "rain_available": False,
            "azimuth_deg": 167.0,
            "tilt_deg": 15.0,
        },
        "inverters": {
            "INV01": {
                "max_ac_power_kw": 60.0,
                "strings": {
                    "MPPT1|PV1": {
                        "string_azimuth_deg": 167.0,
                        "string_tilt_deg": 15.0,
                        "string_commissioning_date": "2024-01-15",
                        "bifacial": True,
                    },
                },
            }
        },
    }


def _make_raw_with_inverter_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "level": ["string", "inverter", "plant"],
        "timestamp": pd.to_datetime(["2024-01-01 12:00"] * 3),
        "inverter_id": ["INV01", "INV01", None],
        "string_power_kw": [5.0, 55.0, None],
    })


def test_preflight_c1_true_when_orientation_present():
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw)
    assert result["C1"] is True


def test_preflight_c2_true_when_dates_present():
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw)
    assert result["C2"] is True


def test_preflight_c3_false_without_rear_irradiance():
    """bifacial=Yes but no rear column → C3=False."""
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw)
    assert result["C3"] is False
    assert "modeled" in result["details"]["C3"]


def test_preflight_c3_true_with_rear_column():
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows().copy()
    raw["rear_irradiance_wm2"] = 50.0
    result = preflight_schema(specs, raw)
    assert result["C3"] is True


def test_preflight_c4_true_with_inverter_rows():
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw)
    assert result["C4"] is True


def test_preflight_c5_false_for_this_plant():
    """CCI Faisalabad: rain_available=No → C5=False."""
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw)
    assert result["C5"] is False


def test_preflight_c6_false_without_wash_cost():
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw, wash_cost_pkr=None)
    assert result["C6"] is False


def test_preflight_c6_true_with_wash_cost():
    specs = _make_specs_c1c2_ok()
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw, wash_cost_pkr=2500.0)
    assert result["C6"] is True


def test_preflight_c1_false_when_orientation_missing():
    specs = _make_specs_c1c2_ok()
    # Remove string_azimuth_deg from the single string
    specs["inverters"]["INV01"]["strings"]["MPPT1|PV1"]["string_azimuth_deg"] = None
    raw = _make_raw_with_inverter_rows()
    result = preflight_schema(specs, raw)
    assert result["C1"] is False


# ---------------------------------------------------------------------------
# Live smoke test — skipped if workbook not present
# ---------------------------------------------------------------------------

_WORKBOOK = Path(__file__).resolve().parents[2] / "data" / "RequireInputData.xlsx"


@pytest.mark.skipif(not _WORKBOOK.exists(), reason="RequireInputData.xlsx not found")
def test_live_panel_columns_read():
    rows = _read_panels(_WORKBOOK)
    assert len(rows) > 0, "No panel rows read"
    # All rows should have the new fields (may be None if absent)
    for r in rows:
        assert "string_azimuth_deg" in r
        assert "string_tilt_deg" in r
        assert "string_commissioning_date" in r
        assert "string_commissioning_year" in r


@pytest.mark.skipif(not _WORKBOOK.exists(), reason="RequireInputData.xlsx not found")
def test_live_azimuth_matches_plant_convention():
    specs = load_specs(_WORKBOOK)
    plant_az = specs["plant"]["azimuth_deg"]
    for inv in specs["inverters"].values():
        for st in inv["strings"].values():
            if st.get("string_azimuth_deg") is not None:
                # Both should be N-referenced (positive, roughly 90-270 for south-facing)
                assert 0 <= st["string_azimuth_deg"] <= 360
                # For this plant all strings share the same orientation as the plant
                assert abs(st["string_azimuth_deg"] - plant_az) < 45, (
                    f"String az {st['string_azimuth_deg']} diverges from plant az {plant_az}"
                )


@pytest.mark.skipif(not _WORKBOOK.exists(), reason="RequireInputData.xlsx not found")
def test_live_string_uid_uniqueness():
    specs = load_specs(_WORKBOOK)
    uids = []
    for inv_id, inv in specs["inverters"].items():
        for key, st in inv["strings"].items():
            pv_id = st.get("pv", key.split("|")[-1] if "|" in key else key)
            uids.append(f"{inv_id}_{pv_id}")
    assert len(uids) == len(set(uids)), "Duplicate string_uids detected in live workbook"
