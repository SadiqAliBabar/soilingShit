"""Curtailment detection (state + statistical + voltage-rise) + loss quantification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .constants import QUALITY_FLAGS, STATE_NAME

# Bitmask for already-detected curtailment types (used to avoid double-flagging)
_ALREADY_CURT = QUALITY_FLAGS["CURT_STATE"] | QUALITY_FLAGS["CURT_STATISTICAL"]


def detect_voltage_rise_curtailment(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    freq_min: float = 5.0,
) -> pd.DataFrame:
    """Detect soft curtailment caused by grid voltage rise.

    When the inverter throttles output due to high grid voltage, it walks the
    DC operating point up the I-V curve: Vdc rises while Pdc is flat or
    falling, even though irradiance is stable or increasing.  This is invisible
    to state-flag and clipping detectors but leaves a clear signature in the
    V/P/G relationship.

    All five conditions must be simultaneously true on the same row:
      1. POA >= cfg.curt_vr_min_poa         — meaningful generation period
      2. dV/dt >= cfg.curt_vr_vdc_rise_rate  — Vdc is rising
      3. dP/dt <= cfg.curt_vr_pdc_flat_threshold — Pdc flat or falling
      4. dG/dt >= cfg.curt_vr_poa_falling_threshold — POA not falling fast
      5. V >= cfg.curt_vr_vdc_min_fraction * Voc_str_stc — not in startup

    Does NOT flag rows that already carry CURT_STATE or CURT_STATISTICAL.
    Returns a copy; does not modify df in place.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    qf = df["qflag"].values.astype(np.int64).copy() if "qflag" in df.columns \
        else np.zeros(n, dtype=np.int64)

    poa = pd.to_numeric(df.get("POA", pd.Series([np.nan] * n, index=df.index)),
                        errors="coerce").fillna(0.0)
    V   = pd.to_numeric(df.get("V",   pd.Series([np.nan] * n, index=df.index)),
                        errors="coerce").fillna(0.0)
    P   = pd.to_numeric(df.get("P",   pd.Series([np.nan] * n, index=df.index)),
                        errors="coerce").fillna(0.0)

    # Rolling window size in rows
    win = max(2, int(round(cfg.curt_vr_window_min / freq_min)))

    # Rates of change (per minute): diff gives per-interval, divide by freq_min
    dV_dt = V.diff().rolling(win, min_periods=2).mean() / freq_min
    dP_dt = P.diff().rolling(win, min_periods=2).mean() / freq_min
    dG_dt = poa.diff().rolling(win, min_periods=2).mean() / freq_min

    voc_str = cfg.module.voc_str_stc
    vdc_min = cfg.curt_vr_vdc_min_fraction * voc_str

    # Five boolean conditions (all vectorised)
    c1_poa_ok   = poa.values >= cfg.curt_vr_min_poa
    c2_v_rising = dV_dt.values >= cfg.curt_vr_vdc_rise_rate
    c3_p_flat   = dP_dt.values <= cfg.curt_vr_pdc_flat_threshold
    c4_poa_stable = dG_dt.values >= cfg.curt_vr_poa_falling_threshold
    c5_vdc_ok   = V.values >= vdc_min

    # Not already flagged for curtailment
    not_already_curt = (qf & _ALREADY_CURT) == 0

    vr_flag = c1_poa_ok & c2_v_rising & c3_p_flat & c4_poa_stable & c5_vdc_ok & not_already_curt
    qf[vr_flag] |= QUALITY_FLAGS["CURT_VOLTAGE_RISE"]

    df["qflag"] = qf
    return df


def detect_curtailment(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    freq_min: float = 5.0,
) -> pd.DataFrame:
    """Add CURT_STATISTICAL, CURT_SUPPRESSED, and CURT_VOLTAGE_RISE quality flags.

    Runs three detectors in order:
      1. Statistical plateau (power clipping near AC limit).
      2. Suppression (bright sun, but power far below expected — grid-commanded).
      3. Voltage-rise throttling (Vdc rising while Pdc flat, irradiance stable).

    The voltage-rise detector must run last because it skips rows already
    carrying CURT_STATE or CURT_STATISTICAL to avoid double-flagging.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df
    q = df["qflag"].values.astype(np.int64).copy() if "qflag" in df.columns \
        else np.zeros(n, dtype=np.int64)

    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    P   = pd.to_numeric(df["P"],   errors="coerce").fillna(0).values

    # ---- 1. Statistical plateau: P clamped near inverter envelope on bright days ----
    p_cap = float(cfg.site.p_ac_max_kw) * 1000.0 / max(cfg.site.n_strings_per_inv, 1)
    high_poa  = poa > 800
    near_cap  = P > (1 - cfg.clip_band_pct) * p_cap
    statistical = high_poa & near_cap

    if statistical.any():
        diffs  = np.diff(statistical.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= cfg.clip_min_dwell:
                q[s:e] |= QUALITY_FLAGS["CURT_STATISTICAL"]

    # ---- 2. Suppression: bright sun but power very low ----
    # Catches grid-commanded suppression where the inverter still reports a
    # normal running state code (so CURT_STATE never fires).
    if "Pmp_exp" in df.columns:
        Pmp_exp = pd.to_numeric(df["Pmp_exp"], errors="coerce").fillna(0).values
    else:
        p_str_stc = float(cfg.site.p_ac_max_kw) * 1000.0 / max(cfg.site.n_strings_per_inv, 1)
        Pmp_exp = p_str_stc * (poa / 1000.0)

    bright_sun  = poa > cfg.suppression_poa_threshold
    power_ratio = np.where(Pmp_exp > 10, P / Pmp_exp, 1.0)
    suppressed  = bright_sun & (power_ratio < cfg.suppression_power_ratio)

    if suppressed.any():
        diffs  = np.diff(suppressed.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= cfg.suppression_min_dwell:
                q[s:e] |= QUALITY_FLAGS["CURT_SUPPRESSED"]

    df["qflag"] = q

    # ---- 3. Voltage-rise soft curtailment ----
    # Runs on the updated df so it can skip rows that already carry
    # CURT_STATE (set by quality.py) or CURT_STATISTICAL (set above).
    df = detect_voltage_rise_curtailment(df, cfg, freq_min=freq_min)
    return df


def curtailment_summary(df: pd.DataFrame, freq_min: float = 5.0) -> dict:
    """Summarise curtailment counts and estimated energy loss by type.

    Returns counts/fractions for all three curtailment types:
    CURT_STATE, CURT_STATISTICAL, and CURT_VOLTAGE_RISE.
    """
    if len(df) == 0 or "qflag" not in df.columns:
        return dict(
            n_curt_state=0, n_curt_stat=0, n_curt_voltage_rise=0,
            n_curt_total=0, curt_pct=0.0,
            curt_hours_state=0.0, curt_hours_stat=0.0,
            curt_voltage_rise_pct=0.0, curt_voltage_rise_kwh=0.0,
            top_state_codes="",
        )

    qf  = df["qflag"].values.astype(np.int64)
    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    day = poa > 50
    day_n = max(int(day.sum()), 1)

    cs = int(((qf & QUALITY_FLAGS["CURT_STATE"])       > 0).sum())
    ck = int(((qf & QUALITY_FLAGS["CURT_STATISTICAL"]) > 0).sum())
    vr = int(((qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0).sum())

    _all_curt = (QUALITY_FLAGS["CURT_STATE"]
                 | QUALITY_FLAGS["CURT_STATISTICAL"]
                 | QUALITY_FLAGS["CURT_VOLTAGE_RISE"])
    tot = int((((qf & _all_curt) > 0) & day).sum())

    # State-code histogram for curtailed rows
    state_codes = []
    if "inverter_state" in df.columns:
        curt_mask = ((qf & QUALITY_FLAGS["CURT_STATE"]) > 0)
        if curt_mask.any():
            sc = pd.Series(df.loc[curt_mask, "inverter_state"]).value_counts().head(3)
            state_codes = [f"{int(k)}:{STATE_NAME.get(int(k),'?')}({v})"
                           for k, v in sc.items()]

    h = freq_min / 60.0

    # Estimated energy lost to voltage-rise curtailment
    vr_kwh = 0.0
    vr_mask = (qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0
    if vr_mask.any() and "P" in df.columns:
        P_obs = pd.to_numeric(df["P"], errors="coerce").fillna(0).values
        if "Pmp_exp" in df.columns:
            P_exp = pd.to_numeric(df["Pmp_exp"], errors="coerce").fillna(0).values
        else:
            P_exp = P_obs  # no expected — delta is zero; avoids fabricating numbers
        delta_kw = np.maximum(P_exp - P_obs, 0.0) / 1000.0
        vr_kwh = float((delta_kw[vr_mask] * h).sum())

    return dict(
        n_curt_state=cs, n_curt_stat=ck, n_curt_voltage_rise=vr,
        n_curt_total=tot,
        curt_pct=100.0 * tot / day_n,
        curt_hours_state=cs * h, curt_hours_stat=ck * h,
        curt_voltage_rise_pct=100.0 * vr / day_n,
        curt_voltage_rise_kwh=vr_kwh,
        top_state_codes=", ".join(state_codes),
    )


def quantify_curtailment_loss(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    freq_min: float = 5.0,
) -> dict:
    """Compute lost kWh and PKR due to curtailment, split by type.

    Splits the total curtailment loss into three components:
      - curtailment_loss_state_kwh       (CURT_STATE rows)
      - curtailment_loss_statistical_kwh (CURT_STATISTICAL rows)
      - curtailment_loss_voltage_rise_kwh (CURT_VOLTAGE_RISE rows)
      - curtailment_loss_total_kwh        (sum of all three)

    The legacy key ``total_curt_kwh`` is kept as an alias for
    curtailment_loss_total_kwh so downstream code does not break.
    """
    n = len(df)
    if n == 0:
        return _empty_curt_loss(cfg)

    qf  = df["qflag"].values.astype(np.int64)
    poa = df["POA"].fillna(0).values

    # Expected power (W)
    if "Pmp_exp" not in df.columns:
        plate = cfg.module
        Gn    = poa / 1000.0
        Tc    = df.get("T_module", pd.Series(25.0, index=df.index)).fillna(25).values
        Pmp_exp_w = plate.pmp_str_stc * Gn * (1 + plate.gamma_pmp * (Tc - 25))
    else:
        Pmp_exp_w = df["Pmp_exp"].fillna(0).values

    P_obs_w = df["P"].fillna(0).values
    dt_h    = freq_min / 60.0

    def _loss_kwh(mask: np.ndarray) -> tuple[pd.Series, float]:
        """Return (per-row dP series in kW, total kWh) for a curtailment mask."""
        daylight = poa > 100
        active   = mask & daylight
        dP_w  = np.where(active, np.maximum(Pmp_exp_w - P_obs_w, 0.0), 0.0)
        dP_kw = pd.Series(dP_w / 1000.0, index=df.index)
        return dP_kw, float(dP_kw.sum() * dt_h)

    mask_state = (qf & QUALITY_FLAGS["CURT_STATE"])       > 0
    mask_stat  = (qf & QUALITY_FLAGS["CURT_STATISTICAL"]) > 0
    mask_vr    = (qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0
    mask_all   = mask_state | mask_stat | mask_vr

    dP_state, kwh_state = _loss_kwh(mask_state)
    _,         kwh_stat  = _loss_kwh(mask_stat)
    _,         kwh_vr    = _loss_kwh(mask_vr)
    dP_all,    kwh_total = _loss_kwh(mask_all)

    revenue_pkr = kwh_total * cfg.site.tariff

    ts       = pd.to_datetime(df["ts"])
    ts_naive = ts.dt.tz_convert(None) if getattr(ts.dt, "tz", None) else ts
    dates    = ts_naive.dt.date
    daily_kwh = (dP_all * dt_h).groupby(dates).sum()

    period_days    = max((ts_naive.max() - ts_naive.min()).days, 1)
    annualised_kwh = kwh_total / period_days * 365.0
    annualised_pkr = revenue_pkr / period_days * 365.0

    n_curt = int(mask_all.sum())
    expl = (f"{n_curt} curtailed daylight samples -> "
            f"{kwh_total:.2f} kWh deficit in {period_days} days -> "
            f"{cfg.site.currency} {revenue_pkr:,.0f} "
            f"[state={kwh_state:.2f} stat={kwh_stat:.2f} vr={kwh_vr:.2f}]")

    return dict(
        per_row_dP_kw=dP_all,
        # Split components
        curtailment_loss_state_kwh=float(kwh_state),
        curtailment_loss_statistical_kwh=float(kwh_stat),
        curtailment_loss_voltage_rise_kwh=float(kwh_vr),
        curtailment_loss_total_kwh=float(kwh_total),
        # Legacy alias — keeps downstream code intact
        total_curt_kwh=float(kwh_total),
        total_curt_pkr=float(revenue_pkr),
        n_curt_intervals=n_curt,
        daily_curt_kwh=daily_kwh,
        period_days=int(period_days),
        annualised_kwh=float(annualised_kwh),
        annualised_pkr=float(annualised_pkr),
        method="plate_pmp_minus_observed",
        explainability=expl,
    )


def _empty_curt_loss(cfg):
    return dict(
        per_row_dP_kw=pd.Series(dtype=float),
        curtailment_loss_state_kwh=0.0,
        curtailment_loss_statistical_kwh=0.0,
        curtailment_loss_voltage_rise_kwh=0.0,
        curtailment_loss_total_kwh=0.0,
        total_curt_kwh=0.0,
        total_curt_pkr=0.0,
        n_curt_intervals=0,
        daily_curt_kwh=pd.Series(dtype=float),
        period_days=0,
        annualised_kwh=0.0,
        annualised_pkr=0.0,
        method="plate_pmp_minus_observed",
        explainability="no data",
    )
