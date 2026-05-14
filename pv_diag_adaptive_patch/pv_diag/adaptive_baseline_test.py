"""
pytest suite for adaptive_baseline.py integration.

Tests 1 and 7 require generate_demo_data.py (integration-level).
Tests 2–6 are pure unit tests that synthesise minimal DataFrames in memory.
"""
from __future__ import annotations

import sys
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# Make sure the package root is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pv_diag.config import PipelineConfig
from pv_diag.adaptive_baseline import (
    AdaptiveBaselineResult,
    estimate_string_clean_baseline,
    estimate_cluster_clean_baseline,
    apply_cross_string_gate,
    resolve_clean_baseline,
)
from pv_diag.utils import pick_nci_column


# ===========================================================================
# Helpers
# ===========================================================================

def _make_daily_df(
    n_days: int = 60,
    nci_mean: float = 0.975,
    nci_noise: float = 0.01,
    n_valid: int = 48,
    rain_mm: float = 0.0,
    start_date: Optional[date] = None,
    rain_day_indices=None,        # list of row indices that have rain
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic daily_df matching the schema produced by compute_daily_metrics."""
    rng = np.random.default_rng(seed)
    start = start_date or date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    nci_vals = nci_mean + rng.normal(0, nci_noise, n_days)
    nci_vals = np.clip(nci_vals, 0.0, 1.2)
    rain = np.full(n_days, rain_mm)
    if rain_day_indices:
        for idx in rain_day_indices:
            rain[idx] = 12.0  # definite rain event

    return pd.DataFrame(dict(
        date=dates,
        NCI_noon=nci_vals,
        NCI_corrected_noon=nci_vals,   # plate-corrected copy
        n_valid=[n_valid] * n_days,
        rain_mm=rain,
        PR=np.full(n_days, 0.80),
        E_meas_kWh=np.ones(n_days) * 50.0,
        E_exp_kWh=np.ones(n_days) * 60.0,
    ))


def _make_rain_events_df(event_dates) -> pd.DataFrame:
    """Minimal events_df from detect_wash_events."""
    if not event_dates:
        return pd.DataFrame(columns=["event_date", "cause", "delta_nci"])
    return pd.DataFrame(dict(
        event_date=event_dates,
        cause=["Rain"] * len(event_dates),
        delta_nci=[0.05] * len(event_dates),
    ))


def _default_cfg(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ===========================================================================
# Test 1 — Happy path (integration with generate_demo_data + full pipeline)
# ===========================================================================

def test_1_happy_path_all_strings_layer1():
    """All clean/soiled strings resolve to Layer 1 on demo plant data."""
    pytest.importorskip("openpyxl")

    # --- Generate demo plant data ---
    try:
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "generate_demo_data",
            Path(__file__).resolve().parents[2] / "generate_demo_data.py",
        )
        gdm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gdm)
    except Exception as exc:
        pytest.skip(f"generate_demo_data not found: {exc}")

    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx = str(Path(tmpdir) / "demo.xlsx")
        gdm.main(xlsx)

        from pv_diag.pipeline import run_pipeline
        cfg = PipelineConfig()
        cfg.adaptive_baseline_enabled = True
        cfg.adaptive_min_clean_days = 3   # demo data is only 1 month
        cfg.adaptive_window_days = 31
        results = run_pipeline(xlsx, cfg=cfg, verbose=False)

    adaptive_map = results.get("adaptive_results", {})
    per_string   = results.get("per_string", {})

    assert len(adaptive_map) > 0, "No adaptive results produced"

    # Strings that have enough data and NCI ~ 0.97–0.99 should land in Layer 1.
    # The "faulty" string (INV02_MPPT2_pv8) will fail data quality and be Skipped.
    checked = 0
    for label, ar in adaptive_map.items():
        ps = per_string.get(label, {})
        if ps.get("sufficiency") == "Skipped":
            continue  # not enough data — expected for faulty string
        if ar is None:
            continue
        assert ar.layer == 1, (
            f"Expected Layer 1 for '{label}', got Layer {ar.layer} "
            f"(source={ar.source}, value={ar.value})"
        )
        assert 0.92 < ar.value < 1.02, (
            f"Layer-1 value {ar.value:.4f} out of expected range for '{label}'"
        )
        checked += 1

    assert checked >= 4, f"Too few strings checked ({checked}); something is wrong."


# ===========================================================================
# Test 2 — Faulty string isolation
# ===========================================================================

def test_2_faulty_string_isolation():
    """One string clamped to NCI=0.80 gets Layer ≠ 1; neighbours stay Layer 1."""
    cfg = _default_cfg(
        adaptive_window_days=90,
        adaptive_min_clean_days=5,
        adaptive_min_midday_points=6,
        adaptive_min_p95=0.92,
        adaptive_no_rain_floor=0.96,
        adaptive_cluster_gate=0.05,
        rain_threshold_mm=5.0,
        dry_season_threshold=30,
    )

    rain_events_df = _make_rain_events_df([date(2025, 2, 15)])

    # Normal strings in the same cluster
    normal_labels = ["A", "B", "C"]
    faulty_label  = "FAULTY"
    cluster_ids   = {lbl: "cluster_1" for lbl in normal_labels + [faulty_label]}

    per_string_est = {}
    for lbl in normal_labels:
        daily_df = _make_daily_df(n_days=60, nci_mean=0.975, rain_day_indices=[30])
        per_string_est[lbl] = estimate_string_clean_baseline(
            daily_df, cfg, rain_events_df
        )

    # Faulty string: NCI stuck at 0.80
    daily_df_faulty = _make_daily_df(n_days=60, nci_mean=0.80, nci_noise=0.005,
                                      rain_day_indices=[30])
    per_string_est[faulty_label] = estimate_string_clean_baseline(
        daily_df_faulty, cfg, rain_events_df
    )

    # Gate A: faulty string P95 ≈ 0.80 < 0.92 → rejected here already
    assert per_string_est[faulty_label]["value"] is None, (
        "Faulty string should have been rejected by Gate A"
    )

    # Normal strings should pass A+B
    for lbl in normal_labels:
        assert per_string_est[lbl]["value"] is not None, (
            f"Normal string '{lbl}' should have passed Gates A+B"
        )

    # Cluster baseline from normal strings only
    p95_map = {
        lbl: (est["p95"] if est["value"] is not None else None)
        for lbl, est in per_string_est.items()
    }
    cluster_bl = estimate_cluster_clean_baseline(p95_map, cluster_ids)
    per_string_est = apply_cross_string_gate(
        per_string_est, cluster_bl, cluster_ids, cfg
    )

    # Resolve all
    plate = 1.0
    for lbl in normal_labels:
        ar = resolve_clean_baseline(
            lbl, per_string_est, cluster_bl, cluster_ids,
            plate, 10.0, cfg
        )
        assert ar.layer == 1, f"Normal '{lbl}' should be Layer 1, got {ar.layer}"

    ar_faulty = resolve_clean_baseline(
        faulty_label, per_string_est, cluster_bl, cluster_ids,
        plate, 10.0, cfg
    )
    assert ar_faulty.layer != 1, (
        f"Faulty string should NOT be Layer 1, got layer={ar_faulty.layer}"
    )


# ===========================================================================
# Test 3 — Whole-cluster soiling → all fall to Layer 3
# ===========================================================================

def test_3_whole_cluster_soiling_falls_to_layer3():
    """When all strings in a cluster have NCI~0.86, Gate A fails for all.
    All four strings must resolve to Layer 3 (plate fallback).
    """
    cfg = _default_cfg(
        adaptive_min_p95=0.92,
        adaptive_no_rain_floor=0.96,
        adaptive_cluster_gate=0.05,
        adaptive_min_clean_days=5,
        dry_season_threshold=30,
        rain_threshold_mm=5.0,
    )

    labels = ["S1", "S2", "S3", "S4"]
    cluster_ids = {lbl: "dirty_cluster" for lbl in labels}
    rain_events_df = _make_rain_events_df([date(2025, 2, 10)])

    per_string_est = {}
    for lbl in labels:
        # NCI ~ 0.86 — Gate A (0.92) will reject
        daily_df = _make_daily_df(n_days=60, nci_mean=0.86, nci_noise=0.01,
                                   rain_day_indices=[20])
        per_string_est[lbl] = estimate_string_clean_baseline(
            daily_df, cfg, rain_events_df
        )
        assert per_string_est[lbl]["value"] is None, (
            f"'{lbl}' should have been rejected by Gate A (p95 < 0.92)"
        )

    # All rejected → cluster baseline is None (< 2 finite contributors)
    p95_map = {lbl: None for lbl in labels}
    cluster_bl = estimate_cluster_clean_baseline(p95_map, cluster_ids)
    assert cluster_bl.get("dirty_cluster") is None, (
        "Cluster baseline should be None when all strings rejected"
    )

    per_string_est = apply_cross_string_gate(
        per_string_est, cluster_bl, cluster_ids, cfg
    )

    plate = 1.0
    for lbl in labels:
        ar = resolve_clean_baseline(
            lbl, per_string_est, cluster_bl, cluster_ids,
            plate, 10.0, cfg
        )
        assert ar.layer == 3, (
            f"'{lbl}' should be Layer 3 (all gates failed), got Layer {ar.layer}"
        )


# ===========================================================================
# Test 4 — No rain anchor (Gate B)
# ===========================================================================

def test_4_no_rain_anchor_rejects_layer1():
    """With zero rain events in window and P95 = 0.94 < 0.96, Gate B fires."""
    cfg = _default_cfg(
        adaptive_min_p95=0.92,
        adaptive_no_rain_floor=0.96,
        adaptive_min_clean_days=5,
        rain_threshold_mm=5.0,
        dry_season_threshold=30,
    )

    # NCI mean ~ 0.945 → P95 just below 0.94 depending on noise
    # Force a deterministic series with P95 = 0.940
    rng = np.random.default_rng(99)
    n = 60
    nci_vals = np.sort(rng.normal(0.930, 0.008, n))
    # Ensure P95 ~ 0.940
    nci_vals = np.clip(nci_vals, 0.5, 1.15)
    target_p95 = float(np.quantile(nci_vals, 0.95))
    assert target_p95 < 0.96, f"Constructed P95={target_p95:.3f} is not below 0.96"
    assert target_p95 >= 0.92, f"Constructed P95={target_p95:.3f} is below Gate A floor"

    start = date(2025, 1, 1)
    daily_df = pd.DataFrame(dict(
        date=[start + timedelta(days=i) for i in range(n)],
        NCI_noon=nci_vals,
        n_valid=[48] * n,
        rain_mm=[0.0] * n,  # NO rain
    ))

    no_rain_events = _make_rain_events_df([])
    est = estimate_string_clean_baseline(daily_df, cfg, no_rain_events)

    assert est["value"] is None, (
        f"Gate B should have rejected (no rain, P95={target_p95:.3f} < 0.96), "
        f"but got value={est['value']}"
    )
    assert est["source"] == "reject_no_rain_anchor", (
        f"Expected source=reject_no_rain_anchor, got {est['source']}"
    )


# ===========================================================================
# Test 5 — Dry-season blend at Layer 2
# ===========================================================================

def test_5_dry_season_blend():
    """last_rain_days_ago=45, cluster=0.97, plate=0.99 → blended between 0.97 & 0.99."""
    cfg = _default_cfg(dry_season_threshold=30)

    # Synthesise a string whose Layer 1 is rejected (insufficient_data)
    daily_df = _make_daily_df(n_days=3, nci_mean=0.97)  # too few days
    per_string_est = {
        "STRING": estimate_string_clean_baseline(
            daily_df, cfg, _make_rain_events_df([])
        )
    }
    # Force rejection so we fall to Layer 2
    per_string_est["STRING"]["value"] = None
    per_string_est["STRING"]["source"] = "reject_insufficient_data"

    cluster_ids = {"STRING": "cluster_X"}
    cluster_bl  = {"cluster_X": 0.97}   # Layer 2 available

    ar = resolve_clean_baseline(
        "STRING", per_string_est, cluster_bl, cluster_ids,
        plate_age_baseline=0.99,
        last_rain_days_ago=45.0,
        cfg=cfg,
    )

    assert ar.layer == 2, f"Expected Layer 2, got Layer {ar.layer}"
    assert 0.97 <= ar.value <= 0.99, (
        f"Blended value {ar.value:.4f} not in [0.97, 0.99]"
    )
    assert "blended" in ar.source or "cluster" in ar.source, (
        f"Source '{ar.source}' should contain 'blended' or 'cluster'"
    )


# ===========================================================================
# Test 6 — Disagreement flag
# ===========================================================================

def test_6_disagreement_flag():
    """Plate-NCI gives Mod.Soiling (~0.87), adaptive-NCI gives Clean (~0.975).
    The classification result must have baseline_disagreement_flag == True.
    """
    from pv_diag.classification import classify_string
    from pv_diag.wash_detect import _empty as wash_empty

    n = 30
    dates = [date(2025, 3, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(7)

    # Plate-corrected NCI: ~0.87  (Mod.Soiling)
    nci_plate = np.clip(rng.normal(0.87, 0.01, n), 0.5, 1.2)
    # Adaptive NCI: ~0.975  (Clean)
    nci_adapt = np.clip(rng.normal(0.975, 0.01, n), 0.5, 1.2)

    base_df = pd.DataFrame(dict(
        date=dates,
        NCI_noon=nci_plate,
        NCI_corrected_noon=nci_plate,
        NCI_adaptive_noon=nci_adapt,
        n_valid=[48] * n,
        rain_mm=[0.0] * n,
        asym=[0.01] * n,
        PR=[0.80] * n,
        E_meas_kWh=[50.0] * n,
        E_exp_kWh=[60.0] * n,
    ))

    wash = wash_empty()
    wash["current_segment_df"] = base_df

    cfg = PipelineConfig()
    cfg.use_current_segment_verdict = True

    # Adaptive result present and NOT Layer 3 (so no confidence notch)
    ar = AdaptiveBaselineResult(
        value=0.975, source="adaptive_string", layer=1,
        explainability="Layer 1", p95=0.975, p50=0.970,
        n_used=25, n_rain_events_in_window=1, cluster_id="cluster_1",
    )

    soiling_empty = dict(srr_pct_per_day=np.nan, ci_pct_per_day=np.nan,
                          weighted_soiling_loss_pct=np.nan,
                          median_recovery_depth_pct=np.nan,
                          n_segments=0, segments=[], method="none",
                          explainability="none")

    clx = classify_string(
        base_df, wash, soiling_empty, soiling_empty, cfg,
        adaptive_result=ar,
    )

    assert clx["axes"].get("baseline_disagreement_flag") is True, (
        f"baseline_disagreement_flag should be True; "
        f"axes={clx['axes']}"
    )
    delta = clx["axes"].get("baseline_disagreement_pp", 0.0)
    assert delta > 3.0, f"Expected disagreement > 3 pp, got {delta:.2f}"
    assert "WARNING" in clx["explainability"], (
        "WARNING should appear in explainability when baselines disagree"
    )


# ===========================================================================
# Test 7 — Disabled fallback (cfg.adaptive_baseline_enabled = False)
# ===========================================================================

def test_7_disabled_fallback_uses_plate_path():
    """With adaptive_baseline_enabled=False the pipeline must use NCI_corrected_noon
    everywhere, produce no adaptive_results, and generate no 11B sheet if exported.
    """
    pytest.importorskip("openpyxl")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_demo_data",
            Path(__file__).resolve().parents[2] / "generate_demo_data.py",
        )
        gdm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gdm)
    except Exception as exc:
        pytest.skip(f"generate_demo_data not found: {exc}")

    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx = str(Path(tmpdir) / "demo.xlsx")
        gdm.main(xlsx)

        from pv_diag.pipeline import run_pipeline
        from pv_diag.utils import pick_nci_column

        cfg = PipelineConfig()
        cfg.adaptive_baseline_enabled = False
        results = run_pipeline(xlsx, cfg=cfg, verbose=False)

    # No adaptive results produced
    assert results.get("adaptive_results") == {}, (
        "adaptive_results should be empty when adaptive_baseline_enabled=False"
    )

    # Every string's daily_df should NOT have NCI_adaptive_noon data
    for label, ps in results["per_string"].items():
        daily_df = ps.get("daily_df")
        if daily_df is None or daily_df.empty:
            continue
        if "NCI_adaptive_noon" in daily_df.columns:
            n_finite = daily_df["NCI_adaptive_noon"].notna().sum()
            assert n_finite == 0, (
                f"[{label}] NCI_adaptive_noon should be all-NaN when "
                f"adaptive is disabled, but found {n_finite} finite values"
            )
        # Column chosen by pick_nci_column must fall back to NCI_corrected_noon
        col = pick_nci_column(daily_df)
        assert col in ("NCI_corrected_noon", "NCI_noon"), (
            f"[{label}] pick_nci_column chose '{col}' in disabled mode; "
            f"expected NCI_corrected_noon or NCI_noon"
        )

    # Verdicts still produced (no crashes, no missing strings)
    assert len(results["per_string"]) > 0
    for label, ps in results["per_string"].items():
        assert "classification" in ps or "error" in ps, (
            f"[{label}] missing both classification and error key"
        )


# ===========================================================================
# Extra unit tests for the helper functions
# ===========================================================================

def test_pick_nci_column_prefers_adaptive():
    df = pd.DataFrame({
        "NCI_noon": [0.97],
        "NCI_corrected_noon": [0.96],
        "NCI_adaptive_noon": [0.98],
    })
    assert pick_nci_column(df) == "NCI_adaptive_noon"


def test_pick_nci_column_falls_to_corrected():
    df = pd.DataFrame({
        "NCI_noon": [0.97],
        "NCI_corrected_noon": [0.96],
        "NCI_adaptive_noon": [np.nan],
    })
    assert pick_nci_column(df) == "NCI_corrected_noon"


def test_pick_nci_column_falls_to_raw():
    df = pd.DataFrame({
        "NCI_noon": [0.97],
        "NCI_corrected_noon": [np.nan],
        "NCI_adaptive_noon": [np.nan],
    })
    assert pick_nci_column(df) == "NCI_noon"


def test_estimate_string_insufficient_days():
    """Fewer than adaptive_min_clean_days rows → reject_insufficient_data."""
    cfg = _default_cfg(adaptive_min_clean_days=5)
    daily_df = _make_daily_df(n_days=3, nci_mean=0.97)
    est = estimate_string_clean_baseline(daily_df, cfg, _make_rain_events_df([]))
    assert est["value"] is None
    assert est["source"] == "reject_insufficient_data"


def test_gate_a_floor():
    """p95 below adaptive_min_p95 triggers reject_floor_violated."""
    cfg = _default_cfg(adaptive_min_p95=0.92, adaptive_min_clean_days=5,
                        rain_threshold_mm=5.0)
    daily_df = _make_daily_df(n_days=60, nci_mean=0.88, nci_noise=0.005)
    rain_ev = _make_rain_events_df([date(2025, 2, 5)])
    est = estimate_string_clean_baseline(daily_df, cfg, rain_ev)
    assert est["value"] is None
    assert est["source"] == "reject_floor_violated"


def test_cluster_baseline_requires_two_contributors():
    """estimate_cluster_clean_baseline returns None when < 2 finite values."""
    p95_map = {"A": 0.97, "B": None, "C": None}
    clusters = {"A": "cl1", "B": "cl1", "C": "cl1"}
    result = estimate_cluster_clean_baseline(p95_map, clusters)
    assert result["cl1"] is None


def test_cluster_baseline_median_of_two():
    """With two finite contributors the result is their median."""
    p95_map = {"A": 0.96, "B": 0.98, "C": None}
    clusters = {"A": "cl1", "B": "cl1", "C": "cl1"}
    result = estimate_cluster_clean_baseline(p95_map, clusters)
    assert abs(result["cl1"] - 0.97) < 1e-9


def test_gate_c_rejects_outlier_string():
    """Gate C: string p95 far below cluster median triggers reject_below_cluster."""
    cfg = _default_cfg(adaptive_cluster_gate=0.05)
    per_string_est = {
        "LOW":    dict(value=0.88, p95=0.88, source="adaptive_string", reason="ok"),
        "NORMAL": dict(value=0.97, p95=0.97, source="adaptive_string", reason="ok"),
    }
    cluster_bl = {"cl1": 0.97}
    clusters   = {"LOW": "cl1", "NORMAL": "cl1"}
    result = apply_cross_string_gate(per_string_est, cluster_bl, clusters, cfg)
    assert result["LOW"]["value"] is None
    assert result["LOW"]["source"] == "reject_below_cluster"
    assert result["NORMAL"]["value"] == 0.97  # unchanged


def test_resolve_layer1_returned_when_valid():
    """resolve_clean_baseline returns Layer 1 when estimate has a finite value."""
    cfg = _default_cfg(dry_season_threshold=30)
    per_string_est = {
        "S1": dict(value=0.975, p95=0.975, p50=0.970, source="adaptive_string",
                   reason="ok", n_used=50, n_rain_events_in_window=2)
    }
    cluster_bl = {"cl1": 0.97}
    clusters   = {"S1": "cl1"}
    ar = resolve_clean_baseline("S1", per_string_est, cluster_bl, clusters,
                                 plate_age_baseline=0.99,
                                 last_rain_days_ago=5.0, cfg=cfg)
    assert ar.layer == 1
    assert abs(ar.value - 0.975) < 1e-9
    assert ar.source == "adaptive_string"


def test_resolve_layer3_when_all_fail():
    """resolve_clean_baseline falls through to Layer 3 when L1 and L2 are unavailable."""
    cfg = _default_cfg(dry_season_threshold=30)
    per_string_est = {
        "S1": dict(value=None, p95=None, p50=None, source="reject_floor_violated",
                   reason="p95_below_floor", n_used=10, n_rain_events_in_window=0)
    }
    cluster_bl = {"cl1": None}
    clusters   = {"S1": "cl1"}
    ar = resolve_clean_baseline("S1", per_string_est, cluster_bl, clusters,
                                 plate_age_baseline=0.98,
                                 last_rain_days_ago=5.0, cfg=cfg)
    assert ar.layer == 3
    assert abs(ar.value - 0.98) < 1e-9
    assert ar.source == "plate_only"


if __name__ == "__main__":
    # Quick smoke-test runner
    import traceback
    tests = [
        test_pick_nci_column_prefers_adaptive,
        test_pick_nci_column_falls_to_corrected,
        test_pick_nci_column_falls_to_raw,
        test_estimate_string_insufficient_days,
        test_gate_a_floor,
        test_cluster_baseline_requires_two_contributors,
        test_cluster_baseline_median_of_two,
        test_gate_c_rejects_outlier_string,
        test_resolve_layer1_returned_when_valid,
        test_resolve_layer3_when_all_fail,
        test_2_faulty_string_isolation,
        test_3_whole_cluster_soiling_falls_to_layer3,
        test_4_no_rain_anchor_rejects_layer1,
        test_5_dry_season_blend,
        test_6_disagreement_flag,
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
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} unit tests.")
    print("(Integration tests 1 & 7 require pytest and generate_demo_data.py)")
