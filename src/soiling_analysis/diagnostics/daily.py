"""Daily metrics + per-row write-back of Imp_exp/Pmp_exp/NCI/NCI_corrected.

Key design choice on the baseline:
    Imp_exp  is the *nameplate* expected current (no age correction).
    Pmp_exp  IS age-corrected (used for soiling loss accounting; we don't
             want to pay the customer for age-related output loss).
    NCI               = I / Imp_exp_nameplate          → "vs new module"
    NCI_corrected     = NCI / age_baseline             → "vs same-age clean"
    NCI_adaptive      = NCI / adaptive_clean_ref       → "vs adaptive clean ref"
                        (only written when adaptive_clean_ref is not None)

    NCI_corrected_noon  = midday median of NCI_corrected (legacy verdict input)
    NCI_adaptive_noon   = midday median of NCI_adaptive  (adaptive verdict input)

The existing NCI / NCI_corrected columns are NEVER removed or renamed.
Downstream modules pick the best available column via utils.pick_nci_column().
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from .config import ModuleConfig, PipelineConfig
from .celltemp import estimate_cell_temp
from .orientation import _solar_position
from .utils import _is_ok


def compute_iam(aoi_deg: np.ndarray, b0: float) -> np.ndarray:
    """ASHRAE incidence angle modifier: IAM = 1 - b0 * (1/cos(aoi) - 1).

    b0=0 means no reflection losses. At normal incidence (AOI=0), IAM=1.
    At large AOI the optical transmission decreases. Clipped to [0, 1] so the
    linear model's divergence near grazing incidence does not produce negatives.
    """
    aoi_r = np.radians(np.asarray(aoi_deg, dtype=float))
    iam = 1.0 - b0 * (1.0 / np.clip(np.cos(aoi_r), 1e-6, None) - 1.0)
    return np.clip(iam, 0.0, 1.0)


def compute_daily_metrics(
    df: pd.DataFrame,
    plate: ModuleConfig,
    sdm_params: Optional[dict] = None,
    cfg: Optional[PipelineConfig] = None,
    baseline: float = 1.0,
    freq_min: float = 5.0,
    adaptive_clean_ref: Optional[float] = None,
    azimuth: Optional[float] = None,
    tilt: Optional[float] = None,
) -> pd.DataFrame:
    """Modify *df* in place to add per-row diagnostic columns; return daily agg.

    Parameters
    ----------
    df : DataFrame
        Raw string timeseries (columns: ts, I, V, P, POA, qflag, …).
    plate : ModuleConfig
        Nameplate parameters.
    sdm_params : dict or None
        Single-diode model fit result (currently unused in column math but
        passed through for future use).
    cfg : PipelineConfig or None
    baseline : float
        Degradation age-baseline (1.0 = no correction).
    freq_min : float
        Timestep in minutes (used to convert power to energy).
    adaptive_clean_ref : float or None
        When provided, additionally computes per-row NCI_adaptive =
        NCI / adaptive_clean_ref and the daily NCI_adaptive_noon median.
        The existing NCI and NCI_corrected columns are left unchanged.
    azimuth : float or None
        Surface azimuth in degrees (south=180). Defaults to cfg.plant.default_azimuth.
    tilt : float or None
        Surface tilt in degrees. Defaults to cfg.plant.default_tilt.

    Returns
    -------
    DataFrame
        One row per calendar date with columns including NCI_noon,
        NCI_corrected_noon, NCI_relative_noon (IAM-corrected), and (when
        adaptive_clean_ref is given) NCI_adaptive_noon.
    """
    ts = pd.to_datetime(df["ts"])
    if getattr(ts.dt, "tz", None) is not None and cfg is not None:
        try:
            ts_local = ts.dt.tz_convert(cfg.site.tz)
        except Exception:
            ts_local = ts
    else:
        ts_local = ts
    df["date"]   = ts_local.dt.date
    df["hr_loc"] = ts_local.dt.hour + ts_local.dt.minute / 60

    Tc, _ = estimate_cell_temp(df, plate, cfg)
    df["__Tc"] = Tc.values

    # SCHEMA-DEP C1 (Batch 5): use transposed POA for the expected-current
    # calculation when available and transposition is enabled.  The measured
    # POA stays in df["POA"] for quality masks and energy accounting so
    # consumers can compare both columns.
    if (cfg is not None
            and getattr(cfg, "poa_transposition_enabled", True)
            and "POA_transposed" in df.columns
            and pd.to_numeric(df["POA_transposed"], errors="coerce").notna().any()):
        Gn = pd.to_numeric(df["POA_transposed"], errors="coerce").fillna(0.0).values / 1000.0
    else:
        Gn = df["POA"].fillna(0).values / 1000.0

    # SCHEMA-DEP C3 (Batch 5): bifacial gain scales expected irradiance.
    # Applied only when the string is bifacial (from string_specs) and the
    # feature is enabled.  Kc gates estimation; this scales physical input.
    _bifacial_flag = (cfg is not None
                      and getattr(cfg, "bifacial_gain_enabled", False)
                      and "bifacial" in df.columns
                      and len(df) > 0
                      and bool(df["bifacial"].iloc[0]))
    if _bifacial_flag:
        if ("POA_rear" in df.columns
                and pd.to_numeric(df["POA_rear"], errors="coerce").notna().any()):
            rear_Gn = pd.to_numeric(df["POA_rear"], errors="coerce").fillna(0.0).values / 1000.0
            Gn = Gn + getattr(cfg, "bifaciality", 0.70) * rear_Gn
        else:
            Gn = Gn * getattr(cfg, "bifacial_gain_default", 1.05)

    dT = df["__Tc"].fillna(25).values - 25

    # NAMEPLATE expected current (NO baseline) — for NCI
    Imp_exp_nameplate = plate.imp_stc * Gn * (1 + plate.alpha_isc * dT)
    df["Imp_exp"] = Imp_exp_nameplate
    df["NCI"] = df["I"] / np.where(Imp_exp_nameplate > 0.05,
                                    Imp_exp_nameplate, np.nan)

    # AGE-CORRECTED expected power — for soiling loss accounting
    Pmp_exp_w = (plate.pmp_str_stc * Gn * (1 + plate.gamma_pmp * dT)
                 * float(baseline))
    df["Pmp_exp"] = Pmp_exp_w
    df["NCI_baseline"]  = float(baseline)
    df["NCI_corrected"] = df["NCI"] / max(float(baseline), 0.5)

    # IAM correction — vectorised AOI across all rows, then NCI_relative = NCI / IAM.
    # Using ASHRAE model with per-surface azimuth/tilt so that within-day IAM
    # variation is removed from the NCI signal before taking the midday median.
    _surf_az = (azimuth if azimuth is not None
                else (cfg.plant.default_azimuth if cfg is not None else 180.0))
    _surf_tilt = (tilt if tilt is not None
                  else (cfg.plant.default_tilt if cfg is not None else 25.0))
    _lat = cfg.site.lat if cfg is not None else 31.4504
    _lon = cfg.site.lon if cfg is not None else 73.1350
    _b0 = cfg.iam_b0 if cfg is not None else 0.05

    _ts_idx = pd.DatetimeIndex(ts)
    _sp = _solar_position(_ts_idx, _lat, _lon)
    _zen_r = np.radians(_sp["zenith"].values)
    _az_sun_r = np.radians(_sp["azimuth"].values)
    _tilt_r = np.radians(_surf_tilt)
    _az_surf_r = np.radians(_surf_az)
    _cos_aoi = (np.cos(_zen_r) * np.cos(_tilt_r) +
                np.sin(_zen_r) * np.sin(_tilt_r) * np.cos(_az_sun_r - _az_surf_r))
    _cos_aoi = np.clip(_cos_aoi, 0.0, 1.0)
    _aoi_deg = np.degrees(np.arccos(_cos_aoi))
    df["__IAM"] = compute_iam(_aoi_deg, _b0)
    # Mask rows at extreme incidence angles where the ASHRAE model loses validity.
    _iam_safe = np.where(df["__IAM"].values > 0.05, df["__IAM"].values, 1.0)
    df["NCI_relative"] = np.where(df["__IAM"].values > 0.05,
                                   df["NCI"].values / _iam_safe,
                                   np.nan)

    # ADAPTIVE per-row column (added only when a reference is supplied)
    _has_adaptive = adaptive_clean_ref is not None and float(adaptive_clean_ref) > 0.1
    if _has_adaptive:
        safe_ref = max(float(adaptive_clean_ref), 0.1)
        df["NCI_adaptive"] = df["NCI"] / safe_ref
    elif "NCI_adaptive" in df.columns:
        # Keep column but re-fill with NaN to avoid stale values from a
        # previous call if df is reused.
        df["NCI_adaptive"] = np.nan

    mask_ok = _is_ok(df["qflag"].values) & (df["POA"].values > 100)

    # Batch 6: Kc+stability mask for NCI_noon estimation.
    # Applied ONLY to the estimation columns (NCI_noon, NCI_*_noon).
    # The absolute >100 W/m² mask above stays for physical validity and
    # energy accounting — Kc gates estimation; W/m² gates physical validity.
    _kc_noon_mask: Optional[np.ndarray] = None
    if cfg is not None and getattr(cfg, "clearsky_quality_enabled", True):
        try:
            from .clearsky_quality import clearsky_quality_mask as _ckq_mask
            _poa_for_kc = pd.to_numeric(df["POA"], errors="coerce").fillna(0.0).values
            _kc_noon_mask = _ckq_mask(
                pd.DatetimeIndex(ts), _poa_for_kc, cfg,
                lat=_lat, lon=_lon,
                azimuth=_surf_az, tilt=_surf_tilt,
                altitude=float(getattr(cfg.site, "altitude", 217.0)),
            )
        except Exception as _kc_err:
            import warnings as _w
            _w.warn(f"[B6] Kc interval mask failed ({_kc_err}); "
                    f"falling back to POA-only mask for NCI_noon")
            _kc_noon_mask = None

    # Minimum number of valid midday points required to compute a reliable
    # NCI_noon median. Days with fewer surviving rows are set to NaN to
    # avoid misleading near-zero drops in the soiling dashboard.
    min_pts = cfg.adaptive_min_midday_points if cfg is not None else 6

    dt_h = freq_min / 60.0
    rows = []
    for date, sub in df.groupby("date"):
        idxs = df.index.get_indexer(sub.index)
        s_ok = sub[mask_ok[idxs]]

        # Kc-quality subset for NCI_noon estimation (Batch 6).
        # Energy accounting (E_meas, E_exp, n_valid) and AM/PM keep s_ok.
        if _kc_noon_mask is not None:
            s_ok_est = sub[mask_ok[idxs] & _kc_noon_mask[idxs]]
        else:
            s_ok_est = s_ok

        midday_est = (s_ok_est["hr_loc"] >= 11) & (s_ok_est["hr_loc"] <= 13)
        am_w       = (s_ok    ["hr_loc"] >= 7.5)  & (s_ok    ["hr_loc"] <= 9.5)
        pm_w       = (s_ok    ["hr_loc"] >= 14.5) & (s_ok    ["hr_loc"] <= 16.5)

        E_meas = float((sub["P"].clip(lower=0).fillna(0) * dt_h).sum() / 1000.0)
        E_exp  = float((sub["Pmp_exp"].clip(lower=0).fillna(0) * dt_h).sum() / 1000.0)
        PR     = E_meas / E_exp if E_exp > 0 else np.nan

        row = dict(
            date=date, PR=PR,
            NCI_noon=(s_ok_est.loc[midday_est, "NCI"].median()
                      if midday_est.sum() >= min_pts else np.nan),
            NCI_am  =(s_ok.loc[am_w,   "NCI"].median()
                      if am_w.sum() >= min_pts else np.nan),
            NCI_pm  =(s_ok.loc[pm_w,   "NCI"].median()
                      if pm_w.sum() >= min_pts else np.nan),
            NCI_corrected_noon=(s_ok_est.loc[midday_est, "NCI_corrected"].median()
                                if midday_est.sum() >= min_pts else np.nan),
            NCI_relative_noon=(s_ok_est.loc[midday_est, "NCI_relative"].median()
                               if "NCI_relative" in s_ok_est.columns
                               and midday_est.sum() >= min_pts else np.nan),
            NCI_baseline=float(baseline),
            E_meas_kWh=E_meas, E_exp_kWh=E_exp,
            n_valid=len(s_ok),
            rain_mm=(sub["rainfall"].sum() * dt_h
                     if "rainfall" in sub.columns else 0.0),
        )

        # Adaptive noon median — only when the per-row column was computed
        if _has_adaptive and "NCI_adaptive" in s_ok_est.columns:
            row["NCI_adaptive_noon"] = (
                s_ok_est.loc[midday_est, "NCI_adaptive"].median()
                if midday_est.sum() >= min_pts else np.nan
            )
        else:
            row["NCI_adaptive_noon"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    # Prefer IAM-corrected denominator for asymmetry; it removes the within-day
    # optical-loss shape so that AM/PM imbalance reflects true orientation mismatch
    # rather than incidence-angle artefact. Fall back to NCI_noon per row when
    # NCI_relative_noon is NaN (e.g. low-irradiance days or first-pass mode).
    if "NCI_relative_noon" in out.columns:
        _asym_denom = (out["NCI_relative_noon"]
                       .where(out["NCI_relative_noon"].notna(), out["NCI_noon"])
                       .replace(0, np.nan))
    else:
        _asym_denom = out["NCI_noon"].replace(0, np.nan)
    out["asym"] = (out["NCI_pm"] - out["NCI_am"]).abs() / _asym_denom

    # Apply the continuous calendar grid so downstream consumers (wash_detect,
    # transient, soiling) never mistake a multi-day gap for a one-day step.
    if cfg is not None and cfg.daily_grid_enabled and len(out) > 0:
        from .calendar_grid import to_calendar_grid
        out = to_calendar_grid(out)

    return out