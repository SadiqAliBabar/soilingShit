"""Combine soiling + curtailment losses."""
from __future__ import annotations
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
