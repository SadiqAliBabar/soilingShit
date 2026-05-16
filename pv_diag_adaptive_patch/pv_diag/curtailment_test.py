"""Standalone tests for the voltage-rise curtailment detector (Prompt 5).

Each test builds its own synthetic DataFrame, runs the relevant function,
and asserts the expected outcome.  No external fixtures required.

Run with:
    python -m pytest pv_diag/curtailment_test.py -v
or directly:
    python pv_diag/curtailment_test.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pv_diag.config import PipelineConfig
from pv_diag.constants import QUALITY_FLAGS
from pv_diag.curtailment import (
    detect_voltage_rise_curtailment,
    curtailment_summary,
    quantify_curtailment_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> PipelineConfig:
    """Return a PipelineConfig with small window so tests are fast.

    n_modules=12 gives voc_str_stc = 51.8 * 12 = 621.6 V, so synthetic
    V values of 590-640 satisfy condition 5 (V >= 0.75 * 621.6 = 466.2 V).
    """
    cfg = PipelineConfig()
    cfg.module.n_modules               = 12   # voc_str_stc ~621 V, matches test data
    cfg.curt_vr_min_poa                = 200.0
    cfg.curt_vr_vdc_rise_rate          = 0.5
    cfg.curt_vr_pdc_flat_threshold     = 5.0
    cfg.curt_vr_poa_falling_threshold  = -2.0
    cfg.curt_vr_vdc_min_fraction       = 0.75
    cfg.curt_vr_window_min             = 10.0   # 2 rows at 5-min resolution
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _base_df(n: int = 60) -> pd.DataFrame:
    """60 rows of 5-min data, all clean."""
    ts = pd.date_range("2024-06-01 07:00", periods=n, freq="5min")
    return pd.DataFrame({
        "ts":    ts,
        "V":     np.full(n, 590.0),
        "P":     np.full(n, 7800.0),
        "POA":   np.full(n, 820.0),
        "qflag": np.zeros(n, dtype=np.int64),
    })


VR_FLAG = QUALITY_FLAGS["CURT_VOLTAGE_RISE"]


# ---------------------------------------------------------------------------
# Test 1 — canonical voltage-rise event IS detected
# ---------------------------------------------------------------------------

def test_canonical_vr_event_detected():
    """Rows 24-35 with rising V, falling P, stable POA must get the VR flag."""
    df = _base_df(60)
    event_rows = list(range(24, 36))  # 12 rows

    # Rising V: 600 -> 640 over 12 rows (~0.67 V/min at 5-min data)
    df.loc[event_rows, "V"] = np.linspace(600, 640, len(event_rows))
    # Falling P: start at baseline (7800) and fall to 7000 — no upward jump
    df.loc[event_rows, "P"] = np.linspace(7800, 7000, len(event_rows))
    # POA stable — not falling
    df.loc[event_rows, "POA"] = 850.0

    cfg = _make_cfg()
    out = detect_voltage_rise_curtailment(df, cfg, freq_min=5.0)

    flagged = ((out["qflag"].values & VR_FLAG) > 0)
    # At least the interior event rows (after rolling window warms up) must be flagged
    interior = list(range(26, 36))
    assert flagged[interior].all(), \
        f"Interior event rows should be flagged, got: {flagged[event_rows]}"
    # Rows well outside the event window should not be flagged
    outer = list(range(0, 20)) + list(range(40, 60))
    assert not flagged[outer].any(), "Non-event rows should not be flagged"


# ---------------------------------------------------------------------------
# Test 2 — cloud transient is NOT flagged
# ---------------------------------------------------------------------------

def test_cloud_transient_not_flagged():
    """Fast-falling POA during the event excludes the rows via condition 4."""
    df = _base_df(60)
    event_rows = list(range(24, 36))

    # POA drops sharply and deeply (fast cloud shadow) — dG/dt very negative
    df.loc[event_rows, "POA"] = np.linspace(820, 100, len(event_rows))
    # V rises (natural MPP shift as G drops) and P drops; P starts at baseline
    df.loc[event_rows, "V"]   = np.linspace(600, 640, len(event_rows))
    df.loc[event_rows, "P"]   = np.linspace(7800, 1000, len(event_rows))

    cfg = _make_cfg()
    out = detect_voltage_rise_curtailment(df, cfg, freq_min=5.0)

    flagged = ((out["qflag"].values & VR_FLAG) > 0)
    # Skip row 24 (boundary): its rolling dG/dt window spans the pre-event
    # baseline (POA=820) and the first drop row, giving dG/dt≈0 — ambiguous.
    # All subsequent interior cloud rows must NOT be flagged.
    interior_cloud = list(range(25, 36))
    assert not flagged[interior_cloud].any(), \
        f"Cloud-transient rows must NOT be flagged (POA falling fast), got: {flagged[event_rows]}"


# ---------------------------------------------------------------------------
# Test 3 — low irradiance rows are NOT flagged
# ---------------------------------------------------------------------------

def test_low_irradiance_not_flagged():
    """POA below curt_vr_min_poa must prevent flagging regardless of V/P."""
    df = _base_df(60)
    df["POA"] = 150.0   # below default min of 200 W/m²
    # V rising, P flat — would normally trigger
    df["V"] = np.linspace(580, 640, 60)
    df["P"] = 7800.0

    cfg = _make_cfg()
    out = detect_voltage_rise_curtailment(df, cfg, freq_min=5.0)

    flagged = ((out["qflag"].values & VR_FLAG) > 0)
    assert not flagged.any(), "Low-irradiance rows must NOT be flagged"


# ---------------------------------------------------------------------------
# Test 4 — already-flagged CURT_STATE rows are not double-flagged
# ---------------------------------------------------------------------------

def test_already_curt_state_not_double_flagged():
    """Rows with CURT_STATE set must not additionally get CURT_VOLTAGE_RISE."""
    df = _base_df(60)
    event_rows = list(range(24, 36))

    # Conditions that would trigger VR detector
    df.loc[event_rows, "V"]   = np.linspace(600, 640, len(event_rows))
    df.loc[event_rows, "P"]   = np.linspace(8000, 7200, len(event_rows))
    df.loc[event_rows, "POA"] = 850.0
    # But these rows already carry CURT_STATE
    df.loc[event_rows, "qflag"] = QUALITY_FLAGS["CURT_STATE"]

    cfg = _make_cfg()
    out = detect_voltage_rise_curtailment(df, cfg, freq_min=5.0)

    qf = out["qflag"].values
    # CURT_STATE preserved
    assert ((qf[event_rows] & QUALITY_FLAGS["CURT_STATE"]) > 0).all(), \
        "CURT_STATE flag must be preserved"
    # CURT_VOLTAGE_RISE NOT added
    assert not ((qf[event_rows] & VR_FLAG) > 0).any(), \
        "CURT_VOLTAGE_RISE must NOT be added on top of CURT_STATE"


# ---------------------------------------------------------------------------
# Test 5 — curtailment_summary reflects voltage-rise rows
# ---------------------------------------------------------------------------

def test_curtailment_summary_vr_counts():
    """curtailment_summary must count VR-flagged rows correctly."""
    df = _base_df(60)
    event_rows = list(range(24, 36))

    df.loc[event_rows, "V"]   = np.linspace(600, 640, len(event_rows))
    df.loc[event_rows, "P"]   = np.linspace(7800, 7000, len(event_rows))  # starts at baseline
    df.loc[event_rows, "POA"] = 850.0
    # Add Pmp_exp so energy estimate works
    df["Pmp_exp"] = 8200.0

    cfg = _make_cfg()
    out = detect_voltage_rise_curtailment(df, cfg, freq_min=5.0)
    summary = curtailment_summary(out, freq_min=5.0)

    n_vr = summary["n_curt_voltage_rise"]
    assert n_vr > 0, f"Expected >0 VR rows, got {n_vr}"
    assert summary["curt_voltage_rise_pct"] > 0, \
        "curt_voltage_rise_pct should be positive"

    # Also verify quantify_curtailment_loss consistency
    loss = quantify_curtailment_loss(out, cfg, freq_min=5.0)
    assert loss["curtailment_loss_total_kwh"] >= loss["curtailment_loss_voltage_rise_kwh"], \
        "Total curtailment kWh must be >= VR component"
    # Legacy key present
    assert "total_curt_kwh" in loss, "Legacy key total_curt_kwh must be present"


# ---------------------------------------------------------------------------
# Test 6 — no false positives on clean midday data
# ---------------------------------------------------------------------------

def test_no_false_positives_clean_midday():
    """Normal clear-day I-V tracking (V and P both rising then falling) must not trigger."""
    n = 60
    ts = pd.date_range("2024-06-01 07:00", periods=n, freq="5min")
    # POA rises to noon then falls — smooth bell curve
    t = np.linspace(0, np.pi, n)
    poa = 900.0 * np.sin(t) + 50.0
    # V and P track POA normally: both rise in morning, fall in afternoon
    V = 560.0 + 40.0 * np.sin(t)    # slight rise with irradiance (temp effect)
    P = poa * 8.0                    # proportional to irradiance
    df = pd.DataFrame({
        "ts":    ts,
        "V":     V,
        "P":     P,
        "POA":   poa,
        "qflag": np.zeros(n, dtype=np.int64),
    })

    cfg = _make_cfg()
    out = detect_voltage_rise_curtailment(df, cfg, freq_min=5.0)

    flagged = ((out["qflag"].values & VR_FLAG) > 0)
    assert not flagged.any(), \
        f"Clean midday data must produce zero VR flags, got {flagged.sum()}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_canonical_vr_event_detected,
        test_cloud_transient_not_flagged,
        test_low_irradiance_not_flagged,
        test_already_curt_state_not_double_flagged,
        test_curtailment_summary_vr_counts,
        test_no_false_positives_clean_midday,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
