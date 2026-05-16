"""Tests for multi-day wash/rain recovery detection (Prompt 6).

Each test builds its own synthetic daily DataFrame with columns:
date, NCI_noon, rain_mm.
"""
from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import pytest

from .config import PipelineConfig
from .wash_detect import detect_wash_events, detect_distributed_recovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_days: int, nci_values, rain_values=None, start="2024-01-01") -> pd.DataFrame:
    """Build a minimal daily DataFrame suitable for detect_wash_events."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    nci = np.asarray(nci_values, dtype=float)
    assert len(nci) == n_days
    rain = np.zeros(n_days) if rain_values is None else np.asarray(rain_values, dtype=float)
    return pd.DataFrame({"date": dates, "NCI_noon": nci, "rain_mm": rain})


# ---------------------------------------------------------------------------
# Test 1 — distributed rain recovery over 3 days is detected
# ---------------------------------------------------------------------------

def test_distributed_rain_recovery_detected():
    """3-day cumulative rise of +3.7pp with prior rain triggers a multi_day event."""
    n = 30
    nci = np.linspace(0.97, 0.88, 20).tolist()   # days 0-19: declining

    # days 20-22: step up +0.01, +0.015, +0.012 cumulatively from day-19 value
    base = nci[-1]
    nci += [base + 0.010, base + 0.025, base + 0.037]  # incremental rises

    # days 23-29: stable at ~0.925
    nci += [0.925] * 7

    assert len(nci) == n

    rain = np.zeros(n)
    rain[19] = 8.0   # heavy rain on day 19 (day before window starts at idx 20)

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    result = detect_wash_events(df, cfg)

    events = result["events_df"]
    # Should have exactly one event
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"

    evt = events.iloc[0]
    # Event date should be day index 22 (last day of the 3-day window)
    expected_date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=22)).date()
    assert evt["event_date"] == expected_date, \
        f"event_date={evt['event_date']}, expected={expected_date}"

    assert "Rain" in evt["cause"], f"cause={evt['cause']}"
    assert evt["detection_method"] == "multi_day", \
        f"detection_method={evt['detection_method']}"

    # current_segment_df should start from the event date
    cur = result["current_segment_df"]
    assert len(cur) > 0
    assert cur["date"].min().date() == expected_date


# ---------------------------------------------------------------------------
# Test 2 — single-day event is detected and not duplicated
# ---------------------------------------------------------------------------

def test_single_day_event_not_duplicated():
    """A 4pp single-day jump is caught by the existing detector; no duplicate."""
    n = 30
    nci = np.linspace(0.95, 0.90, 15).tolist()
    # Day 15: big jump (+0.04)
    nci.append(nci[-1] + 0.04)
    nci += [nci[-1]] * (n - 16)
    assert len(nci) == n

    rain = np.zeros(n)
    rain[15] = 12.0

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    result = detect_wash_events(df, cfg)

    events = result["events_df"]
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"

    evt = events.iloc[0]
    expected_date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=15)).date()
    assert evt["event_date"] == expected_date
    assert evt["detection_method"] == "single_day"

    assert result["n_single_day_events"] == 1
    assert result["n_multi_day_events"] == 0


# ---------------------------------------------------------------------------
# Test 3 — non-monotone window is rejected
# ---------------------------------------------------------------------------

def test_non_monotone_window_rejected():
    """A window with an intra-day dip below tolerance is not accepted."""
    n = 30
    nci = np.linspace(0.97, 0.88, 20).tolist()
    base = nci[-1]
    # +0.02, -0.01, +0.025 — day 21 dips (non-monotone)
    nci += [base + 0.020, base + 0.010, base + 0.035]
    nci += [nci[-1]] * 7
    assert len(nci) == n

    rain = np.zeros(n)
    rain[19] = 8.0

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    # tolerance is -0.005; the -0.01 dip is below it
    result = detect_wash_events(df, cfg)

    multi_events = result["events_df"][
        result["events_df"]["detection_method"] == "multi_day"
    ]
    assert len(multi_events) == 0, \
        f"Expected no multi_day event for non-monotone window, got {len(multi_events)}"


# ---------------------------------------------------------------------------
# Test 4 — window without prior declining trend or rain is rejected
# ---------------------------------------------------------------------------

def test_no_trend_no_rain_rejected():
    """Flat NCI with no rain and no negative slope should not trigger detection."""
    n = 30
    nci = [0.95] * 20
    # Small rises over 3 days (+0.01 each) — cumulative = 0.03, just at threshold
    nci += [0.96, 0.97, 0.98]
    nci += [0.98] * 7
    assert len(nci) == n

    rain = np.zeros(n)  # no rain

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    result = detect_wash_events(df, cfg)

    assert len(result["events_df"]) == 0, \
        f"Expected no events for flat-NCI no-rain scenario, got {len(result['events_df'])}"


# ---------------------------------------------------------------------------
# Test 5 — post-rain drying delay cause correction
# ---------------------------------------------------------------------------

def test_drying_delay_cause_correction():
    """Single-day detector on day 15 (rain_mm=0) relabels to Rain via lookback."""
    n = 30
    nci = np.linspace(0.95, 0.90, 14).tolist()
    # Day 14: NCI unchanged (panels still muddy after rain on day 13)
    nci.append(nci[-1])
    # Day 15: NCI jumps +0.04
    nci.append(nci[-1] + 0.04)
    nci += [nci[-1]] * (n - 16)
    assert len(nci) == n

    rain = np.zeros(n)
    rain[13] = 15.0   # heavy rain on day 13; day 15 has rain_mm=0

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    result = detect_wash_events(df, cfg)

    events = result["events_df"]
    assert len(events) >= 1

    # The event on day 15 should have cause "Rain" (not "Manual wash (suspected)")
    expected_date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=15)).date()
    evt = events[events["event_date"] == expected_date]
    assert len(evt) == 1, f"Event on day 15 not found; all events: {events}"
    assert "Rain" in evt.iloc[0]["cause"], \
        f"Expected Rain cause after drying-delay correction, got: {evt.iloc[0]['cause']}"


# ---------------------------------------------------------------------------
# Test 6 — overlapping windows do not double-fire
# ---------------------------------------------------------------------------

def test_overlapping_windows_no_double_fire():
    """Days 20-22 and 21-23 both qualify; only the first window fires."""
    n = 30
    nci = np.linspace(0.97, 0.88, 20).tolist()
    base = nci[-1]
    # Each day rises: +0.012, +0.013, +0.012, +0.013 (overlapping valid windows)
    nci += [base + 0.012, base + 0.025, base + 0.037, base + 0.050]
    nci += [nci[-1]] * 6
    assert len(nci) == n

    rain = np.zeros(n)
    rain[19] = 8.0

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    result = detect_wash_events(df, cfg)

    multi_events = result["events_df"][
        result["events_df"]["detection_method"] == "multi_day"
    ]
    assert len(multi_events) == 1, \
        f"Expected exactly 1 multi_day event (no double-fire), got {len(multi_events)}"


# ---------------------------------------------------------------------------
# Test 7 — n_multi_day_events in return dict
# ---------------------------------------------------------------------------

def test_n_events_counts_in_return_dict():
    """One single-day event + one distributed event → correct counters."""
    n = 60

    # First segment: declining, then a big single-day jump at day 20
    nci = np.linspace(0.97, 0.90, 20).tolist()
    nci.append(nci[-1] + 0.04)   # single-day jump, day 20
    nci += [nci[-1]] * 9          # stable plateau days 21-29

    # Second segment: declining, then 3-day distributed recovery at days 47-49
    nci += np.linspace(0.935, 0.900, 17).tolist()  # days 30-46
    base2 = nci[-1]
    nci += [base2 + 0.012, base2 + 0.025, base2 + 0.038]  # days 47-49
    nci += [nci[-1]] * 10  # days 50-59

    assert len(nci) == n

    rain = np.zeros(n)
    rain[20] = 12.0  # rain on single-day event day
    rain[46] = 8.0   # rain day before distributed window

    df = _make_df(n, nci, rain)
    cfg = PipelineConfig()
    result = detect_wash_events(df, cfg)

    assert result["n_single_day_events"] == 1, \
        f"n_single_day_events={result['n_single_day_events']}"
    assert result["n_multi_day_events"] == 1, \
        f"n_multi_day_events={result['n_multi_day_events']}"
    assert len(result["events_df"]) == 2, \
        f"Total events={len(result['events_df'])}"


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_distributed_rain_recovery_detected()
    print("Test 1 passed")
    test_single_day_event_not_duplicated()
    print("Test 2 passed")
    test_non_monotone_window_rejected()
    print("Test 3 passed")
    test_no_trend_no_rain_rejected()
    print("Test 4 passed")
    test_drying_delay_cause_correction()
    print("Test 5 passed")
    test_overlapping_windows_no_double_fire()
    print("Test 6 passed")
    test_n_events_counts_in_return_dict()
    print("Test 7 passed")
    print("All tests passed.")
