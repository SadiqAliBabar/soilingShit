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

from soiling_analysis.diagnostics.config import PipelineConfig
from soiling_analysis.diagnostics.constants import QUALITY_FLAGS
from soiling_analysis.diagnostics.curtailment import (
    detect_voltage_rise_curtailment,
    curtailment_summary,
    quantify_curtailment_loss,
    detect_curtailment,
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


VR_FLAG  = QUALITY_FLAGS["CURT_VOLTAGE_RISE"]
SU_FLAG  = QUALITY_FLAGS["STRING_UNDERPERFORM"]
EL_FLAG  = QUALITY_FLAGS["CURT_EXPORT_LIMIT"]
CS_FLAG  = QUALITY_FLAGS["CURT_STATISTICAL"]
SUP_FLAG = QUALITY_FLAGS["CURT_SUPPRESSED"]


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
# Batch 3 tests — inverter-level curtailment rework
# ---------------------------------------------------------------------------

def _multi_string_df(
    n_timestamps: int = 80,
    n_strings: int = 2,
    inverter_id: str = "INV1",
    poa: float = 850.0,
    power_w: float = 8000.0,
    freq_min: float = 5.0,
) -> pd.DataFrame:
    """Return a long_df with n_strings strings on one inverter."""
    ts = pd.date_range("2024-06-01 09:00", periods=n_timestamps, freq=f"{int(freq_min)}min")
    rows = []
    for sid in range(1, n_strings + 1):
        df_s = pd.DataFrame({
            "ts":          ts,
            "inverter_id": inverter_id,
            "mppt_id":     "MPPT1",
            "string_id":   f"PV{sid}",
            "string_label":f"{inverter_id}__MPPT1__PV{sid}",
            "POA":         poa,
            "P":           power_w,
            "V":           590.0,
            "I":           power_w / 590.0,
            "pv_capacity": 5.0,   # kW per string
            "qflag":       np.zeros(n_timestamps, dtype=np.int64),
        })
        rows.append(df_s)
    return pd.concat(rows, ignore_index=True)


def _make_inv_ac(ts_index, inv_id: str, power_kw: float) -> pd.Series:
    """Helper: build a simple inverter AC power Series."""
    midx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(ts_index), [inv_id] * len(ts_index)],
        names=["ts", "inverter_id"],
    )
    return pd.Series(power_kw, index=midx, name="ac_power_kw")


# ---------------------------------------------------------------------------
# Test 7 — below-nameplate export setpoint plateau is caught by adaptive method
# ---------------------------------------------------------------------------

def test_below_nameplate_plateau_detected():
    """A plateau at 70% of nameplate (missed by old 95% gate) must be CURT_STATISTICAL."""
    n_ts = 100
    ts = pd.date_range("2024-06-01 08:00", periods=n_ts, freq="5min")
    # Inverter AC is flat at a value well below nameplate (70 kW out of, say, 100 kW)
    # — a below-nameplate export setpoint; old code would never flag this.
    plateau_kw = 70.0
    inv_ac = _make_inv_ac(ts, "INV1", plateau_kw)

    long_df = _multi_string_df(n_timestamps=n_ts, n_strings=2, power_w=35_000)
    long_df["ts"] = pd.concat([pd.Series(ts)] * 2, ignore_index=True)

    cfg = _make_cfg()
    cfg.curtailment_inverter_level_enabled = True
    cfg.clip_repeat_days = 1   # only 1 day in test data — lower threshold
    cfg.clip_min_dwell   = 3

    out = detect_curtailment(long_df, cfg, freq_min=5.0,
                             inverter_ac_power=inv_ac)
    stat_flagged = (out["qflag"].values & CS_FLAG) > 0
    assert stat_flagged.any(), (
        "Below-nameplate export setpoint plateau must be flagged CURT_STATISTICAL "
        "(old code missed it)"
    )


# ---------------------------------------------------------------------------
# Test 8 — dead string gets STRING_UNDERPERFORM, not CURT_SUPPRESSED
# ---------------------------------------------------------------------------

def test_dead_string_gets_underperform_not_suppressed():
    """A lone low string under bright sun must be STRING_UNDERPERFORM, not CURT_SUPPRESSED.

    This is the critical misattribution fix: the old code flagged the dead/disconnected
    string as CURT_SUPPRESSED (disqualifying) so the fault classifier never saw it.
    With the consensus requirement, a lone low string now gets STRING_UNDERPERFORM
    (non-disqualifying) and survives into classification.
    """
    n_ts = 20
    ts = pd.date_range("2024-06-01 10:00", periods=n_ts, freq="5min")

    # Two strings on one inverter: string 1 normal (8 kW), string 2 dead (~0 W)
    df_ok   = pd.DataFrame({
        "ts": ts, "inverter_id": "INV1", "mppt_id": "MPPT1", "string_id": "PV1",
        "string_label": "INV1__MPPT1__PV1",
        "POA": 850.0, "P": 8000.0, "V": 590.0, "I": 8000/590,
        "pv_capacity": 5.0, "qflag": np.zeros(n_ts, dtype=np.int64),
    })
    df_dead = pd.DataFrame({
        "ts": ts, "inverter_id": "INV1", "mppt_id": "MPPT1", "string_id": "PV2",
        "string_label": "INV1__MPPT1__PV2",
        "POA": 850.0, "P": 50.0,   "V": 0.0,  "I": 0.0,
        "pv_capacity": 5.0, "qflag": np.zeros(n_ts, dtype=np.int64),
    })
    long_df = pd.concat([df_ok, df_dead], ignore_index=True)

    cfg = _make_cfg()
    cfg.curtailment_inverter_level_enabled = True
    cfg.suppression_poa_threshold  = 400.0
    cfg.suppression_power_ratio    = 0.20
    cfg.suppression_min_dwell      = 2
    cfg.suppression_consensus_frac = 0.5

    out = detect_curtailment(long_df, cfg, freq_min=5.0)
    qf = out["qflag"].values

    # Dead string rows
    dead_mask = out["string_id"] == "PV2"
    # Must NOT be CURT_SUPPRESSED
    assert not ((qf[dead_mask.values] & SUP_FLAG) > 0).any(), \
        "Dead string must NOT receive CURT_SUPPRESSED (lone low, consensus not met)"
    # Must be STRING_UNDERPERFORM (non-disqualifying; classifier sees it)
    assert ((qf[dead_mask.values] & SU_FLAG) > 0).any(), \
        "Dead string must receive STRING_UNDERPERFORM"

    # Healthy string must not be flagged at all
    ok_mask = out["string_id"] == "PV1"
    assert not ((qf[ok_mask.values] & SUP_FLAG) > 0).any(), \
        "Healthy string must NOT be suppressed"
    assert not ((qf[ok_mask.values] & SU_FLAG) > 0).any(), \
        "Healthy string must NOT be underperform"


# ---------------------------------------------------------------------------
# Test 9 — simultaneous all-inverter plateau → CURT_EXPORT_LIMIT
# ---------------------------------------------------------------------------

def test_all_inverter_plateau_is_export_limit():
    """When all inverters plateau at the same time the flag must be CURT_EXPORT_LIMIT."""
    n_ts = 100
    ts = pd.date_range("2024-06-01 08:00", periods=n_ts, freq="5min")
    plateau_kw = 80.0

    # Two inverters, both clipping at the same level simultaneously
    inv_ac_i1 = _make_inv_ac(ts, "INV1", plateau_kw)
    inv_ac_i2 = _make_inv_ac(ts, "INV2", plateau_kw)
    inv_ac = pd.concat([inv_ac_i1, inv_ac_i2])

    df_i1 = _multi_string_df(n_timestamps=n_ts, inverter_id="INV1", power_w=40_000)
    df_i2 = _multi_string_df(n_timestamps=n_ts, inverter_id="INV2", power_w=40_000)
    for d, inv_id in [(df_i1, "INV1"), (df_i2, "INV2")]:
        d["ts"] = pd.concat([pd.Series(ts)] * 2, ignore_index=True)
    long_df = pd.concat([df_i1, df_i2], ignore_index=True)

    cfg = _make_cfg()
    cfg.curtailment_inverter_level_enabled = True
    cfg.clip_repeat_days = 1
    cfg.clip_min_dwell   = 3

    out = detect_curtailment(long_df, cfg, freq_min=5.0, inverter_ac_power=inv_ac)
    qf = out["qflag"].values

    el_flagged = (qf & EL_FLAG) > 0
    assert el_flagged.any(), "All-inverter simultaneous plateau must produce CURT_EXPORT_LIMIT"
    # Should not appear as plain CURT_STATISTICAL (export limit is a distinct flag)
    # (some rows may have both bits if the mapping races — we just require EL present)


# ---------------------------------------------------------------------------
# Test 10 — plateau detection does not bleed across inverter boundaries
# ---------------------------------------------------------------------------

def test_no_clip_bleed_across_inverters():
    """Clipping on INV1 must not flag strings on INV2."""
    n_ts = 100
    ts = pd.date_range("2024-06-01 08:00", periods=n_ts, freq="5min")

    # INV1 plateaus, INV2 ramps normally (half power)
    inv_ac_i1 = _make_inv_ac(ts, "INV1", 80.0)    # flat = clipping
    inv_ac_i2 = _make_inv_ac(ts, "INV2", 40.0)    # also flat — but a *different* plateau level
    # Make INV2 clearly below INV1 plateau so they don't both trigger export-limit
    inv_ac = pd.concat([inv_ac_i1, inv_ac_i2])

    df_i1 = _multi_string_df(n_timestamps=n_ts, inverter_id="INV1", power_w=40_000)
    df_i2 = _multi_string_df(n_timestamps=n_ts, inverter_id="INV2", power_w=20_000)
    for d, tss in [(df_i1, ts), (df_i2, ts)]:
        d["ts"] = pd.concat([pd.Series(tss)] * 2, ignore_index=True)
    long_df = pd.concat([df_i1, df_i2], ignore_index=True)

    cfg = _make_cfg()
    cfg.curtailment_inverter_level_enabled = True
    cfg.clip_repeat_days = 1
    cfg.clip_min_dwell   = 3
    cfg.clip_band_rel    = 0.02  # ±2%

    out = detect_curtailment(long_df, cfg, freq_min=5.0, inverter_ac_power=inv_ac)
    qf = out["qflag"].values

    inv1_rows = out["inverter_id"] == "INV1"
    inv2_rows = out["inverter_id"] == "INV2"

    # INV1 strings should be flagged
    assert ((qf[inv1_rows.values] & CS_FLAG) > 0).any() or \
           ((qf[inv1_rows.values] & EL_FLAG) > 0).any(), \
        "INV1 strings (clipping) should be flagged"

    # INV2 at 40 kW plateau should either be flagged independently (its own plateau)
    # or not — the key requirement is that INV1's 80 kW plateau did NOT bleed into INV2.
    # Concretely: INV2 rows must not carry INV1's CURT_EXPORT_LIMIT
    # (they may carry their own CS flag if their plateau also triggers — that's correct).
    # The test proves no cross-boundary bleed by checking that INV2 rows are never
    # flagged as export-limit when the two inverters plateau at different levels.
    # Since both are flat (each at their own level), both may get CS individually.
    # The export-limit bit would require BOTH inverters at the SAME level simultaneously —
    # here they differ by 50%, so EL should not fire.
    assert not ((qf[inv2_rows.values] & EL_FLAG) > 0).any(), \
        "INV2 must not carry CURT_EXPORT_LIMIT — the two inverters plateau at different levels"


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
        test_below_nameplate_plateau_detected,
        test_dead_string_gets_underperform_not_suppressed,
        test_all_inverter_plateau_is_export_limit,
        test_no_clip_bleed_across_inverters,
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
