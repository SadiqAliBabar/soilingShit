"""Clear-sky quality service (Batch 6).

Computes the clear-sky index Kc = measured_POA / clearsky_POA and a
rolling-coefficient-of-variation stability flag.  Used at exactly two
estimation sites:
  1. _select_quality_days() in pipeline.py  — day-level filter for SDM fitting
  2. The NCI_noon interval mask in compute_daily_metrics (daily.py)

Design note — where Kc is NOT used:
  Physical validity gates (absolute W/m²) are NOT replaced by Kc.  Those
  live in quality.py (G_LOW), losses.py, curtailment.py brightness floors,
  and the existing >100 W/m² mask in daily.py.
  Kc gates *estimation quality*; absolute irradiance gates *physical validity*
  and *energy accounting*.  Keep them separate.

Empty-set policy (Batch 7):
  When day_kc_quality_dates() returns an empty set (e.g. a monsoon or smog
  period where every midday has Kc << 1), the caller MUST NOT fail or fit the
  SDM on the remaining garbage data.  Instead it should trigger the five-tier
  cascade in adaptive_baseline.resolve_clean_baseline (widen window → peer →
  hold-last-good → dry-blend → plate) and suppress the SDM refit.
  Use is_kc_empty_set() to detect this condition at the call site.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .orientation import compute_clearsky_poa


# ---------------------------------------------------------------------------
# Kc computation
# ---------------------------------------------------------------------------

def compute_kc(
    ts_idx: pd.DatetimeIndex,
    poa_measured: np.ndarray,
    lat: float,
    lon: float,
    azimuth: float = 180.0,
    tilt: float = 25.0,
    altitude: float = 217.0,
) -> pd.Series:
    """Return Kc = measured_POA / clearsky_POA per timestamp.

    Entries where clearsky_POA < 50 W/m² (sun below/near horizon) are set to
    NaN — those intervals are not estimation sites regardless of Kc.
    """
    if len(ts_idx) == 0:
        return pd.Series([], dtype=float, name="Kc")
    cs = compute_clearsky_poa(ts_idx, lat, lon, azimuth, tilt, altitude)
    cs_arr = np.maximum(cs.values.astype(float), 0.0)
    poa_arr = np.asarray(poa_measured, dtype=float)
    kc = np.where(cs_arr >= 50.0, poa_arr / np.where(cs_arr > 0, cs_arr, 1.0), np.nan)
    return pd.Series(kc, index=ts_idx, name="Kc")


# ---------------------------------------------------------------------------
# Per-interval quality mask
# ---------------------------------------------------------------------------

def clearsky_quality_mask(
    ts_idx: pd.DatetimeIndex,
    poa_measured: np.ndarray,
    cfg: PipelineConfig,
    lat: float,
    lon: float,
    azimuth: float = 180.0,
    tilt: float = 25.0,
    altitude: float = 217.0,
) -> np.ndarray:
    """Boolean mask: True = passes Kc clearness + rolling-CV stability gate.

    A sample is clear-sky quality when:
      |Kc − 1| ≤ cfg.kc_tolerance          (clearness: close to the clear-sky model)
      rolling-CV of Kc ≤ cfg.kc_stability_cv_max  (stability: sky not rapidly varying)

    Rows where Kc is NaN (clearsky_POA < 50 W/m²) are False.
    """
    n = len(ts_idx)
    if n == 0:
        return np.zeros(0, dtype=bool)

    kc = compute_kc(ts_idx, poa_measured, lat, lon, azimuth, tilt, altitude)
    kc_vals = kc.values.astype(float)

    # Clearness gate
    clearness = np.abs(kc_vals - 1.0) <= cfg.kc_tolerance

    # Stability gate — rolling CV of Kc
    if n > 1:
        diffs = pd.Series(ts_idx).diff().dt.total_seconds().dropna()
        freq_sec = float(diffs.median()) if len(diffs) > 0 else 300.0
        freq_min = max(1.0, min(freq_sec / 60.0, 60.0))
    else:
        freq_min = 5.0

    n_win = max(2, int(round(cfg.kc_window_min / freq_min)))
    kc_ser = pd.Series(kc_vals, dtype=float)
    roll_mean = kc_ser.rolling(n_win, min_periods=max(1, n_win // 2)).mean()
    roll_std  = kc_ser.rolling(n_win, min_periods=max(1, n_win // 2)).std(ddof=0)
    safe_mean = roll_mean.values.copy()
    safe_mean[safe_mean <= 0.05] = np.nan
    cv_arr = np.where(
        np.isfinite(safe_mean),
        roll_std.values / safe_mean,
        np.inf,
    )
    stable = cv_arr <= cfg.kc_stability_cv_max

    mask = clearness & stable & np.isfinite(kc_vals)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Day-level quality gate (used by _select_quality_days in pipeline.py)
# ---------------------------------------------------------------------------

def day_kc_quality_dates(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    lat: float,
    lon: float,
    azimuth: float = 180.0,
    tilt: float = 25.0,
    altitude: float = 217.0,
    midday_window: tuple = (11.0, 13.0),
    min_kc_frac: float = 0.30,
) -> set:
    """Return set of dates where midday has ≥ min_kc_frac Kc-quality intervals.

    This replaces the fixed 600 W/m² peak-POA floor in _select_quality_days.
    A clear winter day with POA peak of 450 W/m² but Kc ≈ 1.0 passes; a
    summer cloudy day with POA peak 700 W/m² but Kc = 0.6 does not.
    """
    if len(df) == 0 or "ts" not in df.columns:
        return set()

    ts = pd.to_datetime(df["ts"])
    ts_idx = pd.DatetimeIndex(ts)
    if ts_idx.tz is not None:
        ts_local = ts.dt.tz_convert(None)
    else:
        ts_local = ts

    hr = (ts_local.dt.hour + ts_local.dt.minute / 60.0).values
    poa_arr = pd.to_numeric(df["POA"], errors="coerce").fillna(0.0).values

    try:
        kc_mask = clearsky_quality_mask(
            ts_idx, poa_arr, cfg, lat, lon, azimuth, tilt, altitude
        )
    except Exception as exc:
        warnings.warn(f"[B6] clearsky_quality_mask failed ({exc}); no Kc quality days")
        return set()

    dates_arr = ts_local.dt.date.values
    good_dates: set = set()

    for d in np.unique(dates_arr):
        day_mask    = dates_arr == d
        midday_mask = (hr >= midday_window[0]) & (hr <= midday_window[1]) & day_mask
        n_midday    = int(midday_mask.sum())
        if n_midday == 0:
            continue
        n_kc_ok = int((kc_mask & midday_mask).sum())
        if n_kc_ok / n_midday >= min_kc_frac:
            good_dates.add(d)

    return good_dates


# ---------------------------------------------------------------------------
# Batch 7 — empty-set policy helpers
# ---------------------------------------------------------------------------

def is_kc_empty_set(good_dates: set, min_days: int = 1) -> bool:
    """Return True when the Kc quality date set is empty or below *min_days*.

    Call this after day_kc_quality_dates() to detect a monsoon/smog window
    before trying to fit the SDM.  When True, the caller must NOT fit on the
    remaining data — instead, trigger the five-tier cascade in
    adaptive_baseline.resolve_clean_baseline.

    Parameters
    ----------
    good_dates : set
        Output of day_kc_quality_dates().
    min_days : int
        Minimum number of Kc-quality days required.  Default 1.
    """
    return len(good_dates) < min_days
