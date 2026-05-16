"""Data sufficiency decision per string."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .constants import QUALITY_FLAGS
from .utils import _is_ok


def compute_data_availability(df: pd.DataFrame, cfg: PipelineConfig,
                              freq_min: float = 5.0) -> dict:
    if len(df) == 0:
        return dict(n_total=0, n_daylight=0, n_ok=0, avail_pct=0.0,
                    curt_pct=0.0, fault_pct=0.0, standby_pct=0.0,
                    max_gap_days=999, n_days=0,
                    n_curt_voltage_rise=0, curt_voltage_rise_pct=0.0)
    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    daylight = poa > 50
    n_dl = int(daylight.sum())
    qf = df["qflag"].values.astype(np.int64)

    # Total curtailment bitmask — all three types combined
    _all_curt = (QUALITY_FLAGS["CURT_STATE"]
                 | QUALITY_FLAGS["CURT_STATISTICAL"]
                 | QUALITY_FLAGS["CURT_VOLTAGE_RISE"])
    n_ok   = int((_is_ok(qf) & daylight).sum())
    n_curt = int(((qf & _all_curt) > 0).sum())
    n_curt_vr = int(((qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0).sum())
    n_fault = int(((qf & QUALITY_FLAGS["INVERTER_FAULT"]) > 0).sum())
    n_stby  = int(((qf & QUALITY_FLAGS["STANDBY"]) > 0).sum())

    ts = pd.to_datetime(df["ts"])
    if getattr(ts.dt, "tz", None): ts = ts.dt.tz_convert(None)
    dates = ts.dt.date
    valid_dates = set(d for d, _ok in zip(dates, _is_ok(qf) & daylight) if _ok)
    if valid_dates:
        ds = sorted(valid_dates)
        gaps = [(ds[i+1] - ds[i]).days - 1 for i in range(len(ds)-1)]
        max_gap = max(gaps) if gaps else 0
    else:
        max_gap = 999
    n_days = len(valid_dates)

    return dict(
        n_total=len(df), n_daylight=n_dl, n_ok=n_ok,
        avail_pct=100.0 * n_ok / max(n_dl, 1),
        curt_pct=100.0 * n_curt / max(n_dl, 1),
        fault_pct=100.0 * n_fault / max(len(df), 1),
        standby_pct=100.0 * n_stby / max(len(df), 1),
        max_gap_days=int(max_gap),
        n_days=int(n_days),
        n_curt_voltage_rise=n_curt_vr,
        curt_voltage_rise_pct=100.0 * n_curt_vr / max(n_dl, 1),
    )


def decide_sufficiency(dq: dict, cfg: PipelineConfig):
    """Return (verdict, reason)."""
    if dq["fault_pct"] >= cfg.suff_fault_pct_skip:
        return "Skipped", f"fault rate {dq['fault_pct']:.0f}% >= skip threshold"
    if dq["avail_pct"] <= cfg.suff_avail_pct_skip:
        return "Skipped", f"availability {dq['avail_pct']:.0f}% <= skip threshold"
    if (dq["avail_pct"] >= cfg.suff_good_avail_pct
            and dq["curt_pct"] <= cfg.suff_good_curt_pct
            and dq["max_gap_days"] <= cfg.suff_max_gap_days):
        return "Good", f"availability {dq['avail_pct']:.0f}%, curt {dq['curt_pct']:.0f}%"
    if (dq["avail_pct"] >= cfg.suff_limited_avail_pct
            and dq["curt_pct"] <= cfg.suff_limited_curt_pct):
        return "Limited", f"availability {dq['avail_pct']:.0f}%, curt {dq['curt_pct']:.0f}%"
    return "Poor", f"availability {dq['avail_pct']:.0f}%, curt {dq['curt_pct']:.0f}%, gap {dq['max_gap_days']} d"
