"""
pytest suite for Prompt 3: AOI correction and trend-based loss.

Test 1 — IAM correction reduces within-day NCI scatter.
Test 2 — Uncapped slope is stored; capped slope used only for loss.
Test 3 — mean_nci_based_loss_pct survives as a secondary field.
"""
from __future__ import annotations

import sys
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pv_diag.config import ModuleConfig, PipelineConfig
from pv_diag.daily import compute_daily_metrics, compute_iam
from pv_diag.orientation import _solar_position
from pv_diag.soiling import extract_soiling_trend


# ---------------------------------------------------------------------------
# Test 1 — IAM correction reduces within-day NCI scatter
# ---------------------------------------------------------------------------

def test_1_iam_correction_reduces_within_day_scatter():
    """NCI_relative must be flatter than NCI across the midday window.

    We build a synthetic clear day where I = Imp_exp * IAM (physically correct
    clean string). Without IAM correction, NCI = IAM and varies with sun angle
    even within the 11-13 h window. With correction, NCI_relative = 1.0 exactly.
    """
    lat, lon = 30.0, 70.0
    azimuth_surf, tilt_surf = 180.0, 25.0
    b0 = 0.05
    day_str = "2025-06-15"

    # Naive timestamps representing local solar time (lon=70 → UTC+4.67 h).
    # The function treats naive timestamps as local, so hr_loc maps directly
    # to these hours and the midday window 11–13 h aligns with solar noon.
    times_naive = pd.date_range(f"{day_str} 05:00", f"{day_str} 19:00",
                                freq="5min")
    ts_idx = pd.DatetimeIndex(times_naive)

    # Solar position (function interprets naive timestamps via lon offset)
    sp = _solar_position(ts_idx, lat, lon)
    zen_r = np.radians(sp["zenith"].values)
    az_sun_r = np.radians(sp["azimuth"].values)
    tilt_r = math.radians(tilt_surf)
    az_surf_r = math.radians(azimuth_surf)

    # AOI formula from the prompt
    cos_aoi = (np.cos(zen_r) * math.cos(tilt_r) +
               np.sin(zen_r) * math.sin(tilt_r) * np.cos(az_sun_r - az_surf_r))
    cos_aoi = np.clip(cos_aoi, 0.0, 1.0)
    aoi_deg = np.degrees(np.arccos(cos_aoi))
    iam = compute_iam(aoi_deg, b0)

    # Clear-sky POA proportional to cos(zenith) — simple but adequate for geometry test
    poa = np.maximum(1000.0 * np.cos(zen_r), 0.0)
    poa = np.where(sp["zenith"].values >= 90.0, 0.0, poa)

    # Cell temperature via same NOCT formula that celltemp.py will use internally
    Tc = 25.0 + (poa / 800.0) * 20.0

    plate = ModuleConfig()
    Gn = poa / 1000.0
    dT = Tc - 25.0
    Imp_exp = plate.imp_stc * Gn * (1.0 + plate.alpha_isc * dT)

    # I = Imp_exp * IAM : a perfectly clean string with proper optical losses.
    # This makes NCI = IAM (varies with angle) and NCI_relative = 1.0 (flat).
    I = Imp_exp * iam

    df = pd.DataFrame({
        "ts":    times_naive,
        "I":     I,
        "V":     np.full(len(times_naive), plate.vmp_str_stc),
        "P":     I * plate.vmp_str_stc,
        "POA":   poa,
        "qflag": np.zeros(len(times_naive), dtype=np.int64),
    })

    cfg = PipelineConfig()
    cfg.site.lat = lat
    cfg.site.lon = lon

    compute_daily_metrics(df, plate, cfg=cfg, baseline=1.0,
                          azimuth=azimuth_surf, tilt=tilt_surf)

    assert "NCI_relative" in df.columns, "NCI_relative column not written to df"

    # Filter to quality-OK midday rows with sensible irradiance
    hr = pd.to_datetime(df["ts"]).dt.hour + pd.to_datetime(df["ts"]).dt.minute / 60.0
    midday_mask = (df["POA"].values > 100) & (hr >= 11.0) & (hr <= 13.0)
    nci_mid = df.loc[midday_mask, "NCI"].dropna().values
    nci_rel_mid = df.loc[midday_mask, "NCI_relative"].dropna().values

    assert len(nci_mid) >= 5, f"Too few midday rows ({len(nci_mid)}) to compare scatter"
    assert len(nci_rel_mid) >= 5, "Too few valid NCI_relative midday rows"

    std_nci = float(np.std(nci_mid))
    std_nci_rel = float(np.std(nci_rel_mid))

    assert std_nci_rel < std_nci, (
        f"NCI_relative (std={std_nci_rel:.6f}) should be flatter than "
        f"NCI (std={std_nci:.6f}) inside the midday window. "
        f"The IAM correction should remove the within-day optical-loss shape."
    )


# ---------------------------------------------------------------------------
# Test 2 — Uncapped slope is stored; capped slope drives loss
# ---------------------------------------------------------------------------

def test_2_uncapped_slope_is_stored():
    """srr_pct_per_day must be uncapped; srr_capped_pct_per_day applies the cap.

    We synthesise a perfectly linear NCI decline at -0.05/day (well beyond
    the old -0.03 cap). The trimmed LR will recover slope=-0.05 exactly on
    perfect data. The top-level srr must equal -5.0 %/day (uncapped) while
    the per-segment srr_capped must equal -3.0 %/day, and the headline
    weighted_soiling_loss_pct must reflect the capped slope, not the raw one.
    """
    n = 10
    start = date(2025, 3, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    # Perfect linear decline: slope = -0.05 NCI/day
    nci = np.array([1.0 - 0.05 * i for i in range(n)])

    daily_df = pd.DataFrame(dict(
        date=dates,
        NCI_noon=nci,
        NCI_corrected_noon=nci,
    ))
    wash_result = {"events_df": pd.DataFrame()}
    cfg = PipelineConfig()
    cfg.min_days_for_trend = 7
    cfg.soiling_loss_cap = 0.50  # 50 % maximum

    result = extract_soiling_trend(daily_df, wash_result, cfg)

    # Top-level SRR must be uncapped
    assert result["srr_pct_per_day"] == pytest.approx(-5.0, abs=0.1), (
        f"srr_pct_per_day should be -5.0 (uncapped), got {result['srr_pct_per_day']:.3f}"
    )

    # Segment must expose the capped value separately
    assert len(result["segments"]) >= 1, "Expected at least one segment"
    seg = result["segments"][0]
    assert "srr_capped_pct_per_day" in seg, "srr_capped_pct_per_day missing from segment"
    assert seg["srr_capped_pct_per_day"] == pytest.approx(-3.0, abs=0.1), (
        f"srr_capped_pct_per_day should be -3.0, got {seg['srr_capped_pct_per_day']:.3f}"
    )

    # Headline weighted loss must use capped slope (30 %) not raw (50 %)
    # capped loss = clip(0.03 * 100 * 10 days, 0, 50) = 30.0 %
    # raw   loss  = clip(0.05 * 100 * 10 days, 0, 50) = 50.0 %
    wt_loss = result["weighted_soiling_loss_pct"]
    assert wt_loss == pytest.approx(30.0, abs=1.0), (
        f"weighted_soiling_loss_pct should be ~30 %% (capped slope-based), "
        f"got {wt_loss:.2f} %%"
    )


# ---------------------------------------------------------------------------
# Test 3 — mean_nci_based_loss_pct survives as a secondary field
# ---------------------------------------------------------------------------

def test_3_mean_nci_based_loss_still_present():
    """mean_nci_based_loss_pct must appear in each segment dict.

    Its value must match the old clip(1 - mean_nci, 0, cap) * 100 formula
    so that existing Excel diagnostics that read the field keep working.
    """
    n = 20
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    # Stable soiled NCI — zero slope, mean = 0.85
    nci = np.full(n, 0.85)

    daily_df = pd.DataFrame(dict(
        date=dates,
        NCI_noon=nci,
        NCI_corrected_noon=nci,
    ))
    wash_result = {"events_df": pd.DataFrame()}
    cfg = PipelineConfig()
    cfg.min_days_for_trend = 7
    cfg.soiling_loss_cap = 0.50

    result = extract_soiling_trend(daily_df, wash_result, cfg)

    assert len(result["segments"]) >= 1, "Expected at least one segment"
    for seg in result["segments"]:
        if not np.isfinite(seg.get("slope_per_day", float("nan"))):
            continue  # skip insufficient-data segments
        assert "mean_nci_based_loss_pct" in seg, (
            "mean_nci_based_loss_pct missing from valid segment dict"
        )
        expected = float(np.clip(1.0 - 0.85, 0.0, 0.50) * 100.0)  # = 15.0
        assert abs(seg["mean_nci_based_loss_pct"] - expected) < 0.5, (
            f"mean_nci_based_loss_pct should be ≈{expected:.1f} %%, "
            f"got {seg['mean_nci_based_loss_pct']:.2f} %%"
        )


# ---------------------------------------------------------------------------
# Smoke runner (python daily_test.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        test_1_iam_correction_reduces_within_day_scatter,
        test_2_uncapped_slope_is_stored,
        test_3_mean_nci_based_loss_still_present,
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
