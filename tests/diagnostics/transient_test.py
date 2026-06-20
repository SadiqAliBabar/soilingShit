"""Tests for transient.py (Batch 2 additions).

Key test: gap days (is_present=False) must not be flagged as transient events
even when their NCI value would pass the dip thresholds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from soiling_analysis.diagnostics.config import PipelineConfig
from soiling_analysis.diagnostics.transient import detect_transient_events


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_daily(n: int, nci: np.ndarray, is_present=None) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ip = np.ones(n, bool) if is_present is None else np.asarray(is_present, bool)
    return pd.DataFrame({
        "date": dates,
        "NCI_noon": nci,
        "NCI_corrected_noon": nci,
        "is_present": ip,
        "n_valid": np.where(ip, 10, 0),
        "rain_mm": np.zeros(n),
    })


# ---------------------------------------------------------------------------
# Test 1 — gap day with very low NCI is NOT flagged when is_present=False
# ---------------------------------------------------------------------------

def test_gap_day_not_flagged_as_transient():
    """A day with is_present=False must not appear in the transient output.

    NCI=0.30 on a single gap day surrounded by stable NCI=0.90 would normally
    trigger the severe-transient threshold (0.30 < 0.75 * 0.90 = 0.675 AND
    IQR≈0 so rmed - 2*iqr ≈ 0.90 > 0.30).  The is_present guard must block it.
    """
    n = 50
    nci = np.full(n, 0.90)
    is_present = np.ones(n, bool)
    # Row 25 is a gap day whose NCI would otherwise trigger a severe transient
    nci[25] = 0.30
    is_present[25] = False

    df = _make_daily(n, nci, is_present)
    cfg = PipelineConfig()
    out = detect_transient_events(df, cfg)

    gap_date = pd.Timestamp("2024-01-26").date()  # row 25 = Jan 26
    event_dates = pd.to_datetime(out["date"]).dt.date.tolist()
    assert gap_date not in event_dates, (
        f"Gap day (is_present=False) must not be flagged as transient, "
        f"but {gap_date} found in events: {event_dates}"
    )


# ---------------------------------------------------------------------------
# Test 2 — same scenario WITHOUT is_present guard fires the detection
# ---------------------------------------------------------------------------

def test_without_is_present_dip_is_flagged():
    """Without is_present guard, a row with NCI=0.30 inside a stable series is flagged.

    This is the pre-Batch-2 behaviour and confirms the guard is the differentiator.
    """
    n = 50
    nci = np.full(n, 0.90)
    nci[25] = 0.30  # severe dip, no is_present column

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "NCI_noon": nci,
        "NCI_corrected_noon": nci,
        "n_valid": np.full(n, 10, dtype=int),
        "rain_mm": np.zeros(n),
        # intentionally NO is_present column
    })
    cfg = PipelineConfig()
    out = detect_transient_events(df, cfg)

    gap_date = pd.Timestamp("2024-01-26").date()
    event_dates = pd.to_datetime(out["date"]).dt.date.tolist()
    assert gap_date in event_dates, (
        f"Without is_present guard, NCI=0.30 on row 25 should be flagged, "
        f"but {gap_date} not in events: {event_dates}"
    )


# ---------------------------------------------------------------------------
# Test 3 — real transient is still detected when is_present=True
# ---------------------------------------------------------------------------

def test_real_dip_with_is_present_true_is_flagged():
    """A genuine dip on a day with is_present=True should still be detected."""
    n = 50
    nci = np.full(n, 0.90)
    is_present = np.ones(n, bool)
    nci[25] = 0.30   # severe dip on a REAL day (is_present=True)

    df = _make_daily(n, nci, is_present)
    cfg = PipelineConfig()
    out = detect_transient_events(df, cfg)

    dip_date = pd.Timestamp("2024-01-26").date()
    event_dates = pd.to_datetime(out["date"]).dt.date.tolist()
    assert dip_date in event_dates, (
        f"is_present=True real dip on {dip_date} must still be flagged, "
        f"but found events: {event_dates}"
    )


# ---------------------------------------------------------------------------
# Test 4 — empty DataFrame is handled without error
# ---------------------------------------------------------------------------

def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=["date", "NCI_corrected_noon", "is_present",
                                "n_valid", "rain_mm"])
    cfg = PipelineConfig()
    out = detect_transient_events(df, cfg)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        test_gap_day_not_flagged_as_transient,
        test_without_is_present_dip_is_flagged,
        test_real_dip_with_is_present_true_is_flagged,
        test_empty_df_returns_empty,
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
