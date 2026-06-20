"""Combine soiling + curtailment losses, and cleaning-economics recommendation."""
from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
from .config import PipelineConfig


def quantify_string_losses(df, daily_df, curt_loss, cfg: PipelineConfig,
                           freq_min: float = 5.0,
                           classification_verdict: str = "") -> dict:
    if df is None or len(df) == 0:
        return _empty(cfg)
    dt_h = freq_min / 60.0

    is_fault_verdict = (classification_verdict == "Fault / degradation — investigate")

    if set(["Pmp_exp","P","NCI_corrected"]).issubset(df.columns):
        soil_mask = (df["POA"].fillna(0) > 100) & (df["NCI_corrected"].fillna(1) < 0.99)
        dP = (df["Pmp_exp"].fillna(0) - df["P"].fillna(0)).clip(lower=0)
        soil_w = float((dP[soil_mask].sum()) * dt_h)
        raw_soiling_kwh = soil_w / 1000.0
    else:
        raw_soiling_kwh = 0.0

    # When the verdict is a non-soiling defect, the energy gap exists but washing
    # will not recover it — report it separately rather than as soiling loss.
    if is_fault_verdict:
        soiling_kwh = 0.0
        unattributed_loss_kwh = raw_soiling_kwh
    else:
        soiling_kwh = raw_soiling_kwh
        # Compute unattributed from daily_df if available for completeness.
        if daily_df is not None and not daily_df.empty and \
                {"E_exp_kWh", "E_meas_kWh"}.issubset(daily_df.columns):
            gap = float(
                (daily_df["E_exp_kWh"].fillna(0) - daily_df["E_meas_kWh"].fillna(0))
                .clip(lower=0).sum()
            )
            unattributed_loss_kwh = max(gap - soiling_kwh, 0.0)
        else:
            unattributed_loss_kwh = 0.0

    soiling_pkr = soiling_kwh * cfg.site.tariff
    curt_kwh = float(curt_loss.get("total_curt_kwh", 0.0))
    curt_pkr = float(curt_loss.get("total_curt_pkr", 0.0))
    total_kwh = soiling_kwh + curt_kwh
    total_pkr = soiling_pkr + curt_pkr

    period_days = int(curt_loss.get("period_days", 0))
    if period_days <= 0:
        ts = pd.to_datetime(df["ts"])
        period_days = max((ts.max() - ts.min()).days, 1)
    annualised_kwh = total_kwh / period_days * 365.0
    annualised_pkr = total_pkr / period_days * 365.0

    expl = (f"soiling={soiling_kwh:.1f} kWh, curt={curt_kwh:.1f} kWh, "
            f"period={period_days} d")
    if is_fault_verdict:
        expl += (f"; fault verdict — soiling_loss zeroed, "
                 f"unattributed={unattributed_loss_kwh:.1f} kWh")

    return dict(
        soiling_kwh=float(soiling_kwh), soiling_pkr=float(soiling_pkr),
        curtailment_kwh=float(curt_kwh), curtailment_pkr=float(curt_pkr),
        total_avoidable_kwh=float(total_kwh),
        total_avoidable_pkr=float(total_pkr),
        annualised_kwh=float(annualised_kwh),
        annualised_pkr=float(annualised_pkr),
        period_days=int(period_days),
        unattributed_loss_kwh=float(unattributed_loss_kwh),
        explainability=expl)


def aggregate_plant_losses(per_string, cfg: PipelineConfig) -> dict:
    keys = ("soiling_kwh","soiling_pkr","curtailment_kwh","curtailment_pkr",
            "total_avoidable_kwh","total_avoidable_pkr",
            "annualised_kwh","annualised_pkr")
    tot = {k: 0.0 for k in keys}
    period = 0
    for d in per_string.values():
        if not d: continue
        for k in keys:
            tot[k] += float(d.get(k, 0.0) or 0.0)
        period = max(period, int(d.get("period_days", 0) or 0))
    tot["period_days"] = period
    tot["n_strings"] = len(per_string)
    tot["currency"] = cfg.site.currency
    tot["tariff_per_kwh"] = cfg.site.tariff
    return tot


def _empty(cfg):
    return dict(soiling_kwh=0.0, soiling_pkr=0.0,
                curtailment_kwh=0.0, curtailment_pkr=0.0,
                total_avoidable_kwh=0.0, total_avoidable_pkr=0.0,
                annualised_kwh=0.0, annualised_pkr=0.0,
                period_days=0, unattributed_loss_kwh=0.0,
                explainability="no data")


# ---------------------------------------------------------------------------
# Batch 9: Cleaning-economics recommendation
# ---------------------------------------------------------------------------

def cleaning_economics(
    string_label: str,
    soiling_result: dict,
    daily_df,
    cfg: PipelineConfig,
    pv_capacity_kw: float = 0.0,
    last_wash_date=None,
    ref_date=None,
) -> dict:
    """Cleaning-economics recommendation for one string.

    Formula (per playbook §9):
      daily_loss_pkr  = |srr_frac/day| × expected_daily_energy_kWh × tariff
      payback_days    = wash_cost_pkr / daily_loss_pkr
      recommend clean when cumulative_loss_since_last_clean ≥ wash_cost_pkr

    Returns fields: daily_loss_pkr, payback_days, recommended_cleaning_date,
    recoverable_kwh, recoverable_pkr, recommendation, economics_inputs.
    """
    _no_rec = dict(
        string_label=string_label,
        daily_loss_pkr=0.0,
        payback_days=np.nan,
        recommended_cleaning_date=None,
        recoverable_kwh=np.nan,
        recoverable_pkr=np.nan,
        recommendation="No cleaning recommended",
        economics_inputs="n/a",
    )

    srr_pct = soiling_result.get("srr_pct_per_day") if soiling_result else None
    if srr_pct is None or not np.isfinite(srr_pct) or srr_pct >= 0.0:
        return _no_rec

    srr_frac_per_day = abs(float(srr_pct)) / 100.0

    # Resolve wash cost
    wash_cost_pkr: float | None = None
    economics_inputs = "defaulted"
    if cfg.wash_cost_per_string_pkr is not None:
        wash_cost_pkr = float(cfg.wash_cost_per_string_pkr)
        economics_inputs = "wash_cost_per_string_pkr"
    elif cfg.wash_cost_per_kw_pkr is not None and pv_capacity_kw > 0:
        wash_cost_pkr = float(cfg.wash_cost_per_kw_pkr) * pv_capacity_kw
        economics_inputs = "wash_cost_per_kw_pkr"

    # Expected daily energy: prefer measured from daily_df, fall back to capacity estimate
    expected_daily_kwh: float = np.nan
    if daily_df is not None and not (
            hasattr(daily_df, "empty") and daily_df.empty):
        if "E_exp_kWh" in daily_df.columns:
            v = pd.to_numeric(daily_df["E_exp_kWh"], errors="coerce")
            if v.notna().sum() > 0:
                expected_daily_kwh = float(v.mean())
    if not np.isfinite(expected_daily_kwh) and pv_capacity_kw > 0:
        # Conservative fallback: 4.5 peak-sun-hours for central Pakistan
        expected_daily_kwh = pv_capacity_kw * 4.5

    if not np.isfinite(expected_daily_kwh) or expected_daily_kwh <= 0:
        return dict(**_no_rec, economics_inputs=economics_inputs,
                    recommendation="Insufficient energy data for economics")

    daily_loss_pkr = srr_frac_per_day * expected_daily_kwh * cfg.site.tariff

    if daily_loss_pkr <= 0:
        return dict(**_no_rec, economics_inputs=economics_inputs)

    today = ref_date or datetime.date.today()
    days_since_clean: int | None = None
    if last_wash_date is not None:
        try:
            lwd = (last_wash_date if isinstance(last_wash_date, datetime.date)
                   else pd.to_datetime(last_wash_date).date())
            days_since_clean = max((today - lwd).days, 0)
        except Exception:
            pass

    if wash_cost_pkr is not None and wash_cost_pkr > 0:
        payback_days = wash_cost_pkr / daily_loss_pkr

        if days_since_clean is not None and days_since_clean * daily_loss_pkr >= wash_cost_pkr:
            recommendation = "Clean now — cumulative loss exceeds wash cost"
            recoverable_pkr = days_since_clean * daily_loss_pkr
            recoverable_kwh = recoverable_pkr / cfg.site.tariff
            recommended_date = str(today)
        else:
            recommendation = f"Clean in ~{payback_days:.0f} days from last clean"
            recoverable_pkr = wash_cost_pkr  # recoverable at breakeven
            recoverable_kwh = recoverable_pkr / cfg.site.tariff
            if last_wash_date is not None and days_since_clean is not None:
                days_remaining = max(payback_days - days_since_clean, 0)
                rec_dt = today + datetime.timedelta(days=int(days_remaining))
                recommended_date = str(rec_dt)
            else:
                recommended_date = None
    else:
        payback_days = np.nan
        recoverable_pkr = np.nan
        recoverable_kwh = np.nan
        recommended_date = None
        recommendation = ("Soiling active — wash cost not configured; "
                          f"daily loss ~{daily_loss_pkr:.0f} {cfg.site.currency}")
        economics_inputs = "defaulted"

    return dict(
        string_label=string_label,
        daily_loss_pkr=float(daily_loss_pkr),
        payback_days=(float(payback_days) if np.isfinite(payback_days) else np.nan),
        recommended_cleaning_date=recommended_date,
        recoverable_kwh=(float(recoverable_kwh) if np.isfinite(recoverable_kwh) else np.nan),
        recoverable_pkr=(float(recoverable_pkr) if np.isfinite(recoverable_pkr) else np.nan),
        recommendation=recommendation,
        economics_inputs=economics_inputs,
    )


def aggregate_plant_economics(per_string_econ: dict) -> dict:
    """Sum recoverable losses across strings; collect all recommendations."""
    total_pkr = 0.0
    total_kwh = 0.0
    n_clean_now = 0
    for d in per_string_econ.values():
        if not d:
            continue
        pkr = d.get("recoverable_pkr") or 0.0
        kwh = d.get("recoverable_kwh") or 0.0
        if np.isfinite(pkr):
            total_pkr += pkr
        if np.isfinite(kwh):
            total_kwh += kwh
        if (d.get("recommendation") or "").startswith("Clean now"):
            n_clean_now += 1
    return dict(
        total_recoverable_pkr=total_pkr,
        total_recoverable_kwh=total_kwh,
        n_strings_clean_now=n_clean_now,
        n_strings=len(per_string_econ),
    )
