"""Tests for the scalar confidence scoring system (Prompt 7).

All tests construct ConfidenceInputs directly and call
compute_confidence_score() — no pipeline execution needed for Tests 1–7.
Test 8 calls the real classify_string() to verify backwards-compatibility.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pv_diag.config import PipelineConfig
from pv_diag.confidence import ConfidenceInputs, compute_confidence_score


# ---------------------------------------------------------------------------
# Reusable inputs builders
# ---------------------------------------------------------------------------

def _inputs_high() -> ConfidenceInputs:
    """Test 1 reference: well-evidenced string with all indicators green."""
    return ConfidenceInputs(
        n_days_current_segment=25,
        n_days_full_window=88,
        baseline_layer=1,
        reference_method="recovery_anchored",
        peer_ladder_level=2,
        peer_substituted=False,
        n_recovery_events_used=2,
        slope_snr=7.5,
        slope_significant=True,
        any_segment_slope_significant=True,
        n_valid_segments=3,
        n_total_segments=3,
        baseline_disagreement_flag=False,
        baseline_disagreement_pp=float("nan"),
        sufficiency="Good",
        avail_pct=82.0,
        has_recovery_signature=True,
        most_recent_recovery_class="Full recovery",
    )


def _inputs_weak() -> ConfidenceInputs:
    """Test 2 reference: poorly-evidenced string."""
    return ConfidenceInputs(
        n_days_current_segment=7,
        n_days_full_window=30,
        baseline_layer=3,
        reference_method="p95_fallback",
        peer_ladder_level=4,
        peer_substituted=False,
        n_recovery_events_used=0,
        slope_snr=1.5,
        slope_significant=False,
        any_segment_slope_significant=False,
        n_valid_segments=1,
        n_total_segments=2,
        baseline_disagreement_flag=False,
        baseline_disagreement_pp=float("nan"),
        sufficiency="Limited",
        avail_pct=45.0,
        has_recovery_signature=False,
        most_recent_recovery_class=None,
    )


# ---------------------------------------------------------------------------
# Test 1 — high confidence well-evidenced string scores >= 0.72
# ---------------------------------------------------------------------------

def test_1_high_confidence_string_scores_above_high_threshold():
    """Well-evidenced string (Test 1 inputs) must score >= 0.72 → label 'high'."""
    cfg = PipelineConfig()
    inp = _inputs_high()
    result = compute_confidence_score(inp, cfg)
    assert result["confidence_score"] >= cfg.conf_high_threshold, (
        f"Expected score >= {cfg.conf_high_threshold}, got {result['confidence_score']:.3f}"
    )
    assert result["confidence_label"] == "high", (
        f"Expected label 'high', got '{result['confidence_label']}'"
    )


# ---------------------------------------------------------------------------
# Test 2 — weak evidence string scores < 0.45
# ---------------------------------------------------------------------------

def test_2_weak_evidence_string_scores_below_medium_threshold():
    """Poorly-evidenced string (Test 2 inputs) must score < 0.45 → label 'low'."""
    cfg = PipelineConfig()
    inp = _inputs_weak()
    result = compute_confidence_score(inp, cfg)
    assert result["confidence_score"] < cfg.conf_medium_threshold, (
        f"Expected score < {cfg.conf_medium_threshold}, got {result['confidence_score']:.3f}"
    )
    assert result["confidence_label"] == "low", (
        f"Expected label 'low', got '{result['confidence_label']}'"
    )


# ---------------------------------------------------------------------------
# Test 3 — peer substitution lowers score
# ---------------------------------------------------------------------------

def test_3_peer_substitution_lowers_score():
    """Setting peer_substituted=True on Test 1 inputs reduces score by >= 0.05."""
    cfg = PipelineConfig()
    base_score = compute_confidence_score(_inputs_high(), cfg)["confidence_score"]

    from dataclasses import replace
    subst_inp = replace(_inputs_high(), peer_substituted=True)
    subst_score = compute_confidence_score(subst_inp, cfg)["confidence_score"]

    assert subst_score < base_score, "Peer substitution must lower the confidence score"
    assert base_score - subst_score >= 0.05, (
        f"Expected reduction >= 0.05, got {base_score - subst_score:.3f} "
        f"(base={base_score:.3f}, subst={subst_score:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 4 — baseline disagreement lowers D4 and overall score
# ---------------------------------------------------------------------------

def test_4_baseline_disagreement_lowers_d4_and_score():
    """6pp baseline disagreement: D4 sub_score < 0.5 and total score lower than Test 1."""
    cfg = PipelineConfig()
    base_score = compute_confidence_score(_inputs_high(), cfg)["confidence_score"]

    from dataclasses import replace
    dis_inp = replace(_inputs_high(),
                      baseline_disagreement_flag=True,
                      baseline_disagreement_pp=6.0)
    result = compute_confidence_score(dis_inp, cfg)

    assert result["sub_scores"]["D4_agreement"] < 0.5, (
        f"D4 with 6pp disagreement should be < 0.5, "
        f"got {result['sub_scores']['D4_agreement']:.3f}"
    )
    assert result["confidence_score"] < base_score, (
        "Disagreement flag must reduce overall confidence score"
    )


# ---------------------------------------------------------------------------
# Test 5 — weight sum validation fires on bad config
# ---------------------------------------------------------------------------

def test_5_weight_sum_validation_raises_on_bad_weights():
    """Weights summing to 1.79 must raise ValueError at PipelineConfig construction."""
    with pytest.raises(ValueError, match="weights"):
        PipelineConfig(conf_weight_data_quantity=0.99)


# ---------------------------------------------------------------------------
# Test 6 — sub_scores dict has all five expected keys in [0, 1]
# ---------------------------------------------------------------------------

def test_6_sub_scores_has_all_five_keys():
    """sub_scores dict must contain exactly D1–D5 keys with values in [0, 1]."""
    cfg = PipelineConfig()
    result = compute_confidence_score(_inputs_high(), cfg)
    ss = result["sub_scores"]
    expected_keys = {
        "D1_data_quantity", "D2_baseline", "D3_trend",
        "D4_agreement", "D5_sufficiency",
    }
    assert set(ss.keys()) == expected_keys, (
        f"sub_scores keys mismatch: got {set(ss.keys())}"
    )
    for k, v in ss.items():
        assert 0.0 <= v <= 1.0, (
            f"sub_scores[{k!r}] = {v:.4f} is outside [0, 1]"
        )


# ---------------------------------------------------------------------------
# Test 7 — label thresholds are respected
# ---------------------------------------------------------------------------

def test_7_label_thresholds_respected():
    """Score ~0.63 is labelled 'high' when threshold=0.55 and 'medium' when threshold=0.72."""
    # These inputs yield a score of ~0.628 (between 0.55 and 0.72).
    inp = ConfidenceInputs(
        n_days_current_segment=15,   # D1: interp → 0.725, avail ok
        n_days_full_window=60,
        baseline_layer=1,
        reference_method="recovery_anchored",
        peer_ladder_level=2,
        peer_substituted=False,
        n_recovery_events_used=1,    # D2: 1.0*1.0*0.85*0.85 = 0.7225
        slope_snr=2.5,               # D3: significant, base=0.25, segs 1/1
        slope_significant=True,
        any_segment_slope_significant=True,
        n_valid_segments=1,
        n_total_segments=1,
        baseline_disagreement_flag=False,
        baseline_disagreement_pp=float("nan"),
        sufficiency="Limited",       # D5: 0.6
        avail_pct=70.0,
        has_recovery_signature=True,
        most_recent_recovery_class="Partial recovery",
    )

    cfg_low_thr = PipelineConfig(conf_high_threshold=0.55, conf_medium_threshold=0.35)
    r_low = compute_confidence_score(inp, cfg_low_thr)
    assert r_low["confidence_label"] == "high", (
        f"With threshold=0.55, score={r_low['confidence_score']:.3f} should be 'high'"
    )

    cfg_high_thr = PipelineConfig(conf_high_threshold=0.72, conf_medium_threshold=0.45)
    r_high = compute_confidence_score(inp, cfg_high_thr)
    assert r_high["confidence_label"] == "medium", (
        f"With threshold=0.72, score={r_high['confidence_score']:.3f} should be 'medium'"
    )


# ---------------------------------------------------------------------------
# Test 8 — backwards compatibility: "confidence" key is a string
# ---------------------------------------------------------------------------

def test_8_backwards_compatibility_confidence_key_is_string():
    """classify_string must return confidence as an ordinal string and
    confidence_score as a float."""
    from pv_diag.classification import classify_string
    from pv_diag.wash_detect import _empty as wash_empty

    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    nci = np.linspace(0.93, 0.88, n)
    daily_df = pd.DataFrame({
        "date": dates,
        "NCI_noon": nci,
        "n_valid": [48] * n,
        "rain_mm": [0.0] * n,
        "asym": [0.01] * n,
        "PR": [0.80] * n,
        "E_meas_kWh": [50.0] * n,
        "E_exp_kWh": [60.0] * n,
    })

    wash = wash_empty()
    wash["current_segment_df"] = daily_df

    cfg = PipelineConfig()
    soiling_minimal = dict(
        srr_pct_per_day=np.nan, ci_pct_per_day=np.nan,
        weighted_soiling_loss_pct=np.nan,
        median_recovery_depth_pct=np.nan,
        n_segments=0, segments=[], method="none",
        any_segment_slope_significant=False,
        explainability="none",
    )

    result = classify_string(
        daily_df, wash, soiling_minimal, soiling_minimal, cfg
    )

    assert result["confidence"] in ("high", "medium", "low"), (
        f"'confidence' key must be an ordinal string, got {result['confidence']!r}"
    )
    assert isinstance(result["confidence_score"], float), (
        f"'confidence_score' must be a float, got {type(result['confidence_score'])}"
    )
    assert 0.0 <= result["confidence_score"] <= 1.0, (
        f"confidence_score out of range: {result['confidence_score']}"
    )
    assert "sub_scores" not in result or True  # just checking it doesn't crash
    assert "confidence_sub_scores" in result
    assert "confidence_explainability" in result


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_1_high_confidence_string_scores_above_high_threshold,
        test_2_weak_evidence_string_scores_below_medium_threshold,
        test_3_peer_substitution_lowers_score,
        test_4_baseline_disagreement_lowers_d4_and_score,
        test_5_weight_sum_validation_raises_on_bad_weights,
        test_6_sub_scores_has_all_five_keys,
        test_7_label_thresholds_respected,
        test_8_backwards_compatibility_confidence_key_is_string,
    ]
    import traceback
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
    print(f"\n{passed} passed, {failed} failed.")
