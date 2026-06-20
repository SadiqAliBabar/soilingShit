"""Tests for multi-day wash/rain recovery detection (Prompt 6).

Each test builds its own synthetic daily DataFrame with columns:
date, NCI_noon, rain_mm.
"""
from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import pytest

from soiling_analysis.diagnostics.config import PipelineConfig
from soiling_analysis.diagnostics.wash_detect import detect_wash_events, detect_distributed_recovery


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
# Test 8 — 5-day gap suppresses phantom single-day step (Batch 2)
# ---------------------------------------------------------------------------

def _make_df_grid(dates, nci_values, is_present_mask, rain_values=None) -> pd.DataFrame:
    """Build a calendar-gridded daily DataFrame with an is_present column."""
    nci = np.asarray(nci_values, dtype=float)
    is_pres = np.asarray(is_present_mask, dtype=bool)
    rain = np.zeros(len(dates)) if rain_values is None else np.asarray(rain_values, dtype=float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "NCI_noon": nci,
        "rain_mm": rain,
        "is_present": is_pres,
        "n_valid": np.where(is_pres, 10, 0),
    })


def test_8_gap_suppresses_phantom_step():
    """5-day calendar gap prevents a phantom single-day step from being detected.

    OLD path (no is_present): 20 consecutive rows; the 4pp NCI jump at row 10
    looks like a 1-day wash event — the step detector fires.

    NEW path (is_present column, 5-day gap): gap_days=6 > max_step_gap_days+1=3,
    so the pair (last-present-before-gap, first-present-after-gap) is skipped.
    No event is detected.
    """
    cfg = PipelineConfig()

    # --- OLD: consecutive rows, no is_present, phantom step at row 10 ---
    nci_old = np.concatenate([
        np.linspace(0.95, 0.88, 10),  # declining trend (rows 0-9)
        np.full(10, 0.92),             # jump to 0.92 then stable (rows 10-19)
    ])
    df_old = _make_df(20, nci_old)
    result_old = detect_wash_events(df_old, cfg)
    assert result_old["n_single_day_events"] >= 1, (
        f"Old gappy code should detect phantom step at row 10, "
        f"got n_single_day_events={result_old['n_single_day_events']}"
    )

    # --- NEW: 25 rows with a 5-day gap, is_present masks the gap ---
    n_new = 25
    nci_new = np.concatenate([
        np.linspace(0.95, 0.88, 10),  # days 1-10: declining, present
        np.full(5, np.nan),            # days 11-15: gap, absent
        np.full(10, 0.92),             # days 16-25: stable, present
    ])
    is_pres = np.concatenate([np.ones(10, bool), np.zeros(5, bool), np.ones(10, bool)])
    dates_new = pd.date_range("2024-01-01", periods=n_new, freq="D")
    df_new = _make_df_grid(dates_new, nci_new, is_pres)

    result_new = detect_wash_events(df_new, cfg)
    assert result_new["n_single_day_events"] == 0, (
        f"Grid-aware code must suppress phantom step across 5-day gap, "
        f"got n_single_day_events={result_new['n_single_day_events']}"
    )
    assert result_new["n_multi_day_events"] == 0, (
        f"No multi-day event either; got {result_new['n_multi_day_events']}"
    )


# ---------------------------------------------------------------------------
# Test 9 — 14-calendar-day density gate sets baseline_clean to NaN (Batch 2)
# ---------------------------------------------------------------------------

def test_9_density_gate_sets_baseline_nan():
    """Sparse 14-calendar-day look-back → baseline_clean=NaN; without grid it is finite.

    When has_grid=True and only 2 present days fall in the 14-calendar-day
    window before the event, density = 2/14 ≈ 0.14 < min_valid_day_density=0.4,
    so baseline_clean is left as NaN.

    Without a grid the old row-based path always sets density_ok=True, so
    baseline_clean is computed from whatever rows are available.

    Scenario (with grid, 29 rows):
      - Rows 0-25 (days 1-26): absent (large gap)
      - Row 26 (day 27): NCI=0.95 (present, starts declining trend)
      - Row 27 (day 28): NCI=0.88 (present, declining)
      - Row 28 (day 29): NCI=0.92 (present, event; +4pp from row 27)

    14 calendar days back from day 29 = day 15. Only rows 26-27 are present in
    [day 15, day 29) → n_present_lb=2, density=2/14≈0.14 < 0.4 → NaN.

    The declining row26→row27 is enough to make pre_slope negative so the event
    is not blocked by the pre-slope gate.
    """
    cfg = PipelineConfig()

    # --- WITH grid ---
    n = 29
    nci_g = np.full(n, np.nan)
    is_pres = np.zeros(n, bool)
    nci_g[26] = 0.95   # present, high (pre-event)
    nci_g[27] = 0.88   # present, declining
    nci_g[28] = 0.92   # present, event (+0.04 from row 27)
    is_pres[26] = is_pres[27] = is_pres[28] = True

    dates_g = pd.date_range("2024-01-01", periods=n, freq="D")
    df_grid = _make_df_grid(dates_g, nci_g, is_pres)

    result_grid = detect_wash_events(df_grid, cfg)
    # pre_slope at row 28 is driven by the row26→row27 decline and is negative
    # so the pre-slope gate passes; event is detected
    assert result_grid["n_single_day_events"] >= 1, (
        f"Event should be detected (negative pre_slope from row26→row27 decline), "
        f"got n_events={result_grid['n_events']}: {result_grid['explainability']}"
    )
    events = result_grid["events_df"]
    single = events[events["detection_method"] == "single_day"]
    assert len(single) >= 1, "At least one single-day event expected"
    baseline = single.iloc[-1]["baseline_clean"]
    assert not np.isfinite(baseline), (
        f"Sparse look-back (2 days / 14 = 14% < 40%) must yield NaN baseline_clean, "
        f"got {baseline}"
    )

    # --- WITHOUT grid: 5 consecutive rows, density_ok always True ---
    # [0.97, 0.95, 0.92, 0.88, 0.92] — declining then +0.04 at row 4
    df_no_grid = _make_df(5, [0.97, 0.95, 0.92, 0.88, 0.92])
    result_no_grid = detect_wash_events(df_no_grid, cfg)
    if result_no_grid["n_single_day_events"] >= 1:
        no_grid_events = result_no_grid["events_df"]
        no_grid_single = no_grid_events[no_grid_events["detection_method"] == "single_day"]
        baseline_ng = no_grid_single.iloc[-1]["baseline_clean"]
        assert np.isfinite(baseline_ng), (
            f"Without grid, density_ok=True, baseline_clean must be finite, "
            f"got {baseline_ng}"
        )


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
    test_8_gap_suppresses_phantom_step()
    print("Test 8 passed")
    test_9_density_gate_sets_baseline_nan()
    print("Test 9 passed")
    print("All tests passed.")
