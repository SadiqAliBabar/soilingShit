"""Curtailment detection (state + statistical) + loss quantification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .constants import QUALITY_FLAGS, STATE_NAME


def detect_curtailment(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Add CURT_STATISTICAL bit if power plateaus at high POA."""
    df = df.copy()
    n = len(df)
    if n == 0: return df
    q = df["qflag"].values.astype(np.int64).copy() if "qflag" in df.columns else np.zeros(n, dtype=np.int64)

    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    P   = pd.to_numeric(df["P"],   errors="coerce").fillna(0).values

    # Statistical plateau: P clamped near inverter envelope on bright days
    p_cap = float(cfg.site.p_ac_max_kw) * 1000.0 / max(cfg.site.n_strings_per_inv, 1)
    high_poa = poa > 800
    near_cap = P > (1 - cfg.clip_band_pct) * p_cap
    statistical = high_poa & near_cap

    # Need at least clip_min_dwell consecutive samples
    if statistical.any():
        # run-length encoding
        diffs = np.diff(statistical.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= cfg.clip_min_dwell:
                q[s:e] |= QUALITY_FLAGS["CURT_STATISTICAL"]

    df["qflag"] = q
    return df


def curtailment_summary(df: pd.DataFrame) -> dict:
    if len(df) == 0 or "qflag" not in df.columns:
        return dict(n_curt_state=0, n_curt_stat=0, n_curt_total=0,
                    curt_pct=0.0, curt_hours_state=0.0, curt_hours_stat=0.0,
                    top_state_codes="")
    qf = df["qflag"].values.astype(np.int64)
    cs = int(((qf & QUALITY_FLAGS["CURT_STATE"])        > 0).sum())
    ck = int(((qf & QUALITY_FLAGS["CURT_STATISTICAL"])  > 0).sum())
    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    day = poa > 50
    day_n = max(int(day.sum()), 1)
    tot = int((((qf & (QUALITY_FLAGS["CURT_STATE"]|QUALITY_FLAGS["CURT_STATISTICAL"])) > 0) & day).sum())

    # State-code histogram for curtailed rows
    state_codes = []
    if "inverter_state" in df.columns:
        curt_mask = ((qf & QUALITY_FLAGS["CURT_STATE"]) > 0)
        if curt_mask.any():
            sc = pd.Series(df.loc[curt_mask, "inverter_state"]).value_counts().head(3)
            state_codes = [f"{int(k)}:{STATE_NAME.get(int(k),'?')}({v})"
                           for k, v in sc.items()]
    # Assume 5 min interval if not provided
    h = 5 / 60.0
    return dict(n_curt_state=cs, n_curt_stat=ck, n_curt_total=tot,
                curt_pct=100.0 * tot / day_n,
                curt_hours_state=cs * h, curt_hours_stat=ck * h,
                top_state_codes=", ".join(state_codes))


def quantify_curtailment_loss(df: pd.DataFrame, cfg: PipelineConfig,
                              freq_min: float = 5.0) -> dict:
    """Compute lost kWh and PKR due to curtailment."""
    n = len(df)
    if n == 0: return _empty_curt_loss(cfg)
    qf = df["qflag"].values.astype(np.int64)
    is_curt = (((qf & (QUALITY_FLAGS["CURT_STATE"] |
                       QUALITY_FLAGS["CURT_STATISTICAL"])) > 0) &
               (df["POA"].fillna(0).values > 100))

    if "Pmp_exp" not in df.columns:
        plate = cfg.module
        Gn  = df["POA"].fillna(0).values / 1000.0
        Tc  = df.get("T_module", pd.Series(25.0, index=df.index)).fillna(25).values
        Pmp_exp_w = plate.pmp_str_stc * Gn * (1 + plate.gamma_pmp * (Tc - 25))
    else:
        Pmp_exp_w = df["Pmp_exp"].fillna(0).values

    P_obs_w = df["P"].fillna(0).values
    dP_w    = np.where(is_curt, np.maximum(Pmp_exp_w - P_obs_w, 0.0), 0.0)
    dP_kw   = pd.Series(dP_w / 1000.0, index=df.index, name="delta_P_curt_kw")

    dt_h        = freq_min / 60.0
    energy_kwh  = float(dP_kw.sum() * dt_h)
    revenue_pkr = energy_kwh * cfg.site.tariff

    ts = pd.to_datetime(df["ts"])
    ts_naive = ts.dt.tz_convert(None) if getattr(ts.dt, "tz", None) else ts
    dates = ts_naive.dt.date
    daily_kwh = (dP_kw * dt_h).groupby(dates).sum()

    period_days = max((ts_naive.max() - ts_naive.min()).days, 1)
    annualised_kwh = energy_kwh / period_days * 365.0
    annualised_pkr = revenue_pkr / period_days * 365.0

    expl = (f"{int(is_curt.sum())} curtailed daylight samples -> "
            f"{energy_kwh:.2f} kWh deficit in {period_days} days -> "
            f"{cfg.site.currency} {revenue_pkr:,.0f}")
    return dict(per_row_dP_kw=dP_kw, total_curt_kwh=float(energy_kwh),
                total_curt_pkr=float(revenue_pkr),
                n_curt_intervals=int(is_curt.sum()),
                daily_curt_kwh=daily_kwh, period_days=int(period_days),
                annualised_kwh=float(annualised_kwh),
                annualised_pkr=float(annualised_pkr),
                method="plate_pmp_minus_observed", explainability=expl)


def _empty_curt_loss(cfg):
    return dict(per_row_dP_kw=pd.Series(dtype=float),
        total_curt_kwh=0.0, total_curt_pkr=0.0, n_curt_intervals=0,
        daily_curt_kwh=pd.Series(dtype=float),
        period_days=0, annualised_kwh=0.0, annualised_pkr=0.0,
        method="plate_pmp_minus_observed", explainability="no data")
