"""Tests for calendar_grid.py (Batch 2).

Covers to_calendar_grid and valid_day_density.
"""
from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import pytest

from soiling_analysis.diagnostics.calendar_grid import to_calendar_grid, valid_day_density


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_daily(dates, nci_values, n_valid=None, rain_values=None) -> pd.DataFrame:
    n = len(dates)
    nci = np.asarray(nci_values, dtype=float)
    n_val = np.ones(n, dtype=int) * 10 if n_valid is None else np.asarray(n_valid, dtype=int)
    rain = np.zeros(n) if rain_values is None else np.asarray(rain_values, dtype=float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "NCI_noon": nci,
        "NCI_corrected_noon": nci,
        "n_valid": n_val,
        "rain_mm": rain,
    })


# ---------------------------------------------------------------------------
# Test 1 — gap rows are inserted for missing calendar days
# ---------------------------------------------------------------------------

def test_gap_rows_inserted():
    """Rows are inserted for the 5 missing days between Jan 1 and Jan 7."""
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 7)]
    df = _make_daily(dates, [0.95, 0.92])
    out = to_calendar_grid(df)
    assert len(out) == 7, f"Expected 7 rows (Jan 1-7), got {len(out)}"


# ---------------------------------------------------------------------------
# Test 2 — is_present flags: True for real data, False for gap rows
# ---------------------------------------------------------------------------

def test_is_present_flags():
    """Original rows have is_present=True; gap rows have is_present=False."""
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)]  # Jan 2 is a gap
    df = _make_daily(dates, [0.95, 0.92])
    out = to_calendar_grid(df)
    assert len(out) == 3
    assert bool(out["is_present"].iloc[0]) is True,  "Jan 1 (present) should be True"
    assert bool(out["is_present"].iloc[1]) is False, "Jan 2 (gap) should be False"
    assert bool(out["is_present"].iloc[2]) is True,  "Jan 3 (present) should be True"


# ---------------------------------------------------------------------------
# Test 3 — NCI metric columns are NOT forward-filled for gap rows
# ---------------------------------------------------------------------------

def test_nci_not_forward_filled():
    """NCI columns remain NaN on gap days; no forward-fill is applied."""
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 5)]  # 3 gap days
    df = _make_daily(dates, [0.95, 0.92])
    out = to_calendar_grid(df)
    gap_rows = out[~out["is_present"]]
    assert len(gap_rows) == 3, "Should have 3 gap rows"
    assert gap_rows["NCI_noon"].isna().all(), "NCI_noon must be NaN for gap rows"
    assert gap_rows["NCI_corrected_noon"].isna().all(), "NCI_corrected_noon must be NaN for gap rows"


# ---------------------------------------------------------------------------
# Test 4 — n_valid=0 and rain_mm=0.0 for gap rows
# ---------------------------------------------------------------------------

def test_counters_zero_for_gaps():
    """n_valid is 0 and rain_mm is 0.0 for inserted gap rows."""
    dates = [datetime.date(2024, 2, 1), datetime.date(2024, 2, 4)]  # 2 gap days
    df = _make_daily(dates, [0.95, 0.92], rain_values=[3.0, 1.0])
    out = to_calendar_grid(df)
    gap_rows = out[~out["is_present"]]
    assert (gap_rows["n_valid"] == 0).all(), "n_valid must be 0 for gap rows"
    assert (gap_rows["rain_mm"] == 0.0).all(), "rain_mm must be 0.0 for gap rows"


# ---------------------------------------------------------------------------
# Test 5 — empty DataFrame is handled without error
# ---------------------------------------------------------------------------

def test_empty_df_returns_empty():
    """to_calendar_grid handles an empty DataFrame without raising."""
    df = pd.DataFrame(columns=["date", "NCI_noon", "n_valid", "rain_mm"])
    out = to_calendar_grid(df)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Test 6 — single-row DataFrame (no gap possible) is returned as-is
# ---------------------------------------------------------------------------

def test_single_row_unchanged():
    """A single-row daily DataFrame has no gap to fill; output has 1 row."""
    df = _make_daily([datetime.date(2024, 6, 1)], [0.93])
    out = to_calendar_grid(df)
    assert len(out) == 1
    assert bool(out["is_present"].iloc[0]) is True


# ---------------------------------------------------------------------------
# Test 7 — valid_day_density: all present
# ---------------------------------------------------------------------------

def test_valid_day_density_all_present():
    is_present = np.ones(5, bool)
    assert valid_day_density(is_present, 0, 4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 8 — valid_day_density: half present
# ---------------------------------------------------------------------------

def test_valid_day_density_half():
    is_present = np.array([True, False, True, False, True, False])
    result = valid_day_density(is_present, 0, 5)
    assert result == pytest.approx(3 / 6)


# ---------------------------------------------------------------------------
# Test 9 — date column contains Python date objects
# ---------------------------------------------------------------------------

def test_date_column_is_python_date():
    """Output date column should contain Python date objects (not Timestamps)."""
    dates = [datetime.date(2024, 3, 1), datetime.date(2024, 3, 5)]
    df = _make_daily(dates, [0.95, 0.92])
    out = to_calendar_grid(df)
    assert isinstance(out["date"].iloc[0], datetime.date), (
        f"Expected datetime.date, got {type(out['date'].iloc[0])}"
    )


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        test_gap_rows_inserted,
        test_is_present_flags,
        test_nci_not_forward_filled,
        test_counters_zero_for_gaps,
        test_empty_df_returns_empty,
        test_single_row_unchanged,
        test_valid_day_density_all_present,
        test_valid_day_density_half,
        test_date_column_is_python_date,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")
