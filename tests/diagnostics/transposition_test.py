"""Tests for transposition.py (Batch 5).

Three assertions from the playbook:
  T1. A synthetic east-facing and west-facing string share the WS GHI but
      get distinct transposed POA with the correct AM/PM tilt pattern.
  T2. Co-oriented case returns the measured POA unchanged (early-exit path).
  T3. compute_daily_metrics produces different NCI_relative_noon for two
      orientations given identical raw current — proving the IAM wiring works.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from soiling_analysis.diagnostics.transposition import transpose_poa, _erbs, _haydavies, _extraterrestrial


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LAT, LON = 31.45, 73.13      # Faisalabad

def _make_ts(date: str, hours: list[float]) -> pd.DatetimeIndex:
    base = pd.Timestamp(date)
    return pd.DatetimeIndex([base + pd.Timedelta(hours=h) for h in hours])


def _south_poa(hours: np.ndarray) -> np.ndarray:
    """Synthetic south-facing POA: sine curve peaking at noon."""
    return np.clip(800.0 * np.sin(np.pi * (hours - 6) / 12.0), 0.0, None)


# ---------------------------------------------------------------------------
# T1: east vs west get AM-heavy vs PM-heavy transposed POA
# ---------------------------------------------------------------------------

def test_east_west_am_pm_asymmetry():
    """East-facing strings should have higher transposed POA in AM; west in PM."""
    summer_date = "2024-06-15"
    # Use AM and PM hours (solar time approx = local time for Pakistan ≈ UTC+5)
    am_hours = [8.0, 9.0, 10.0]    # morning
    pm_hours = [14.0, 15.0, 16.0]  # afternoon

    ts_am = _make_ts(summer_date, am_hours)
    ts_pm = _make_ts(summer_date, pm_hours)

    # WS is south-facing at 15°
    ws_tilt, ws_az = 15.0, 180.0
    poa_am = np.full(len(ts_am), 500.0)
    poa_pm = np.full(len(ts_pm), 500.0)

    # East-facing (az=90) vs west-facing (az=270)
    east_am = transpose_poa(ts_am, poa_am, ws_tilt, ws_az,
                            target_tilt=15.0, target_azimuth=90.0,
                            lat=LAT, lon=LON, model="haydavies")
    west_am = transpose_poa(ts_am, poa_am, ws_tilt, ws_az,
                            target_tilt=15.0, target_azimuth=270.0,
                            lat=LAT, lon=LON, model="haydavies")

    east_pm = transpose_poa(ts_pm, poa_pm, ws_tilt, ws_az,
                            target_tilt=15.0, target_azimuth=90.0,
                            lat=LAT, lon=LON, model="haydavies")
    west_pm = transpose_poa(ts_pm, poa_pm, ws_tilt, ws_az,
                            target_tilt=15.0, target_azimuth=270.0,
                            lat=LAT, lon=LON, model="haydavies")

    # East faces the sun in the morning → east_am > west_am
    assert east_am.mean() > west_am.mean(), (
        f"East AM mean {east_am.mean():.1f} should exceed west AM {west_am.mean():.1f}")

    # West faces the sun in the afternoon → west_pm > east_pm
    assert west_pm.mean() > east_pm.mean(), (
        f"West PM mean {west_pm.mean():.1f} should exceed east PM {east_pm.mean():.1f}")

    # All values non-negative
    assert (east_am >= 0).all() and (west_pm >= 0).all()


# ---------------------------------------------------------------------------
# T2: co-oriented surface returns measured POA unchanged (early exit)
# ---------------------------------------------------------------------------

def test_cooriented_identity():
    """Same orientation as WS → transposed POA equals input (early-exit path)."""
    ts = _make_ts("2024-06-15", [8.0, 10.0, 12.0, 14.0, 16.0])
    poa_in = np.array([200.0, 600.0, 900.0, 650.0, 250.0])

    poa_out = transpose_poa(ts, poa_in,
                            ws_tilt=15.0, ws_azimuth=167.0,
                            target_tilt=15.0, target_azimuth=167.0,
                            lat=LAT, lon=LON)
    np.testing.assert_array_equal(poa_out, poa_in)


# ---------------------------------------------------------------------------
# T3: compute_daily_metrics produces different NCI_relative_noon for two
#     orientations when IAM is wired correctly
# ---------------------------------------------------------------------------

def test_iam_orientation_wiring():
    """NCI_relative_noon differs between south-facing and east-facing strings
    given identical raw current — the IAM removes incidence-angle variation."""
    from soiling_analysis.diagnostics.daily import compute_daily_metrics
    from soiling_analysis.diagnostics.config import ModuleConfig, PipelineConfig

    plate = ModuleConfig(
        voc_stc=45.0, vmp_stc=36.0, isc_stc=10.0, imp_stc=9.0,
        alpha_isc=0.0004, beta_voc=-0.003, gamma_pmp=-0.004,
        n_modules=20, technology="mono-c-Si", cells_in_series=72,
    )
    cfg = PipelineConfig()
    cfg.site.lat = LAT
    cfg.site.lon = LON
    cfg.plant.default_azimuth = 180.0
    cfg.plant.default_tilt = 15.0
    cfg.daily_grid_enabled = False        # off so we get a single row for comparison
    cfg.adaptive_min_midday_points = 4   # lower threshold so 8 pts pass easily

    # Create a synthetic one-day dataset: midday window at 15-minute intervals
    base = pd.Timestamp("2024-06-15")
    ts_list = [base + pd.Timedelta(hours=h)
               for h in [11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0]]
    n = len(ts_list)

    # Both strings see the same raw current and WS POA
    df_base = pd.DataFrame({
        "ts":          ts_list,
        "I":           [9.0] * n,
        "V":           [36.0] * n,
        "P":           [9.0 * 36.0] * n,
        "POA":         [900.0] * n,
        "T_module":    [45.0] * n,
        "qflag":       [0] * n,
        "inverter_state": [0] * n,
        "plant":       ["test"] * n,
        "inverter_id": ["INV1"] * n,
        "mppt_id":     ["M1"] * n,
        "string_id":   ["S1"] * n,
        "string_label": ["test__INV1__M1__S1"] * n,
        "pv_capacity": [plate.vmp_stc * plate.imp_stc * plate.n_modules / 1000.0] * n,
    })

    # South-facing string
    df_south = df_base.copy()
    daily_south = compute_daily_metrics(
        df_south, plate, None, cfg, 1.0, 30.0,
        azimuth=180.0, tilt=15.0,
    )

    # East-facing string (identical data, different orientation for IAM)
    df_east = df_base.copy()
    daily_east = compute_daily_metrics(
        df_east, plate, None, cfg, 1.0, 30.0,
        azimuth=90.0, tilt=15.0,
    )

    south_nci = daily_south["NCI_relative_noon"].dropna()
    east_nci  = daily_east["NCI_relative_noon"].dropna()

    assert len(south_nci) > 0, "South string produced no valid NCI_relative_noon"
    assert len(east_nci)  > 0, "East string produced no valid NCI_relative_noon"

    # At solar noon an east-facing surface has a larger AOI than south-facing
    # → lower IAM → dividing by smaller IAM raises NCI_relative more.
    # The two values must differ (orientation wiring is active).
    diff = abs(float(south_nci.mean()) - float(east_nci.mean()))
    assert diff > 1e-4, (
        f"NCI_relative_noon should differ between south ({south_nci.mean():.4f}) "
        f"and east ({east_nci.mean():.4f}) orientations; diff={diff:.6f}"
    )


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

def test_erbs_preserves_energy():
    """DHI + DNI*cos_zen ≈ GHI (Erbs energy balance)."""
    ghi = np.array([0.0, 200.0, 500.0, 800.0, 1000.0])
    cos_zen = np.array([0.0, 0.4, 0.7, 0.85, 1.0])
    gon = np.full_like(ghi, 1367.0)
    dhi, dni = _erbs(ghi, cos_zen, gon)
    ghi_reconstructed = dhi + dni * cos_zen
    # Where cos_zen > 0 the reconstruction should agree within a few W/m²
    ok = cos_zen > 0.05
    np.testing.assert_allclose(
        ghi_reconstructed[ok], ghi[ok], atol=5.0,
        err_msg="Erbs decomposition violates GHI = DHI + DNI*cos_zen"
    )


def test_haydavies_nonnegative():
    """Hay-Davies output is always non-negative."""
    n = 50
    rng = np.random.default_rng(0)
    ghi = rng.uniform(0, 1000, n)
    cos_zen = rng.uniform(0.1, 1.0, n)
    gon = np.full(n, 1367.0)
    dhi, dni = _erbs(ghi, cos_zen, gon)
    cos_aoi = rng.uniform(0.0, 1.0, n)
    poa = _haydavies(dni, dhi, ghi, cos_aoi, cos_zen,
                     tilt=15.0, albedo=0.20, gon=gon)
    assert (poa >= 0).all(), "Hay-Davies produced negative POA"


def test_extraterrestrial_range():
    """Extraterrestrial irradiance should stay in plausible range (~1320-1415 W/m²)."""
    doy = np.arange(1, 366, dtype=float)
    gon = _extraterrestrial(doy)
    assert gon.min() > 1300.0 and gon.max() < 1450.0
