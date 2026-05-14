"""Wash / rain-recovery detection.

Detects upward step in daily NCI after a downward trend.  Uses RAW delta for
step detection (median smoothing would attenuate single-day jumps) but uses
a small leading mean for the pre-event "trend" check.

Column selection: uses pick_nci_column() to prefer NCI_adaptive_noon >
NCI_corrected_noon > NCI_noon.  No other algorithmic change.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .utils import pick_nci_column


def detect_wash_events(daily_df: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Detect wash / rain-recovery step-ups in daily NCI.

    Parameters
    ----------
    daily_df : DataFrame
        Output of compute_daily_metrics.  Must contain at least one of
        NCI_adaptive_noon, NCI_corrected_noon, NCI_noon.
    cfg : PipelineConfig

    Returns
    -------
    dict with keys:
        events_df, current_segment_df, most_recent_event, n_events,
        explainability
    """
    if daily_df is None or len(daily_df) == 0:
        return _empty()

    df = daily_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    col = pick_nci_column(df)

    # Require at least some finite data
    if df[col].notna().sum() < 2:
        return _empty()

    raw    = df[col].values.astype(float)
    smooth = pd.Series(raw).rolling(3, min_periods=1, center=True).median().values
    rain   = df["rain_mm"].values if "rain_mm" in df.columns else np.zeros(len(df))

    # Raw delta (preserves single-day jumps)
    d_raw = np.diff(raw, prepend=raw[0])
    # Smoothed delta (for trend context)
    pre_slope = (pd.Series(smooth).diff().rolling(3, min_periods=2).mean()
                  .shift(1).fillna(0).values)

    events = []
    skip_until = -1
    for i in range(1, len(df)):
        if i < skip_until:
            continue
        if not np.isfinite(d_raw[i]) or not np.isfinite(raw[i]):
            continue
        if d_raw[i] < cfg.wash_step_thr:
            continue
        # Require a downward trend before (or flat with negative slope)
        if pre_slope[i] >= -0.001:
            continue

        rain_today = float(rain[i]) if np.isfinite(rain[i]) else 0.0
        rain_near  = float(np.nansum(rain[max(0, i-1):min(len(df), i+2)]))
        if rain_today >= cfg.rain_threshold_mm:
            cause = "Rain"
        elif rain_near > 1.0:
            cause = "Rain (light)"
        else:
            cause = "Manual wash (suspected)"

        pre_low_window = raw[max(0, i-3):i]
        pre_low    = float(np.nanmin(pre_low_window)) if len(pre_low_window) else np.nan
        post_window = raw[i:min(len(df), i+4)]
        post_high  = float(np.nanmedian(post_window)) if len(post_window) else np.nan
        look_back  = raw[max(0, i-14):i]
        baseline_clean = float(np.nanmax(look_back)) if len(look_back) else np.nan

        denom = baseline_clean - pre_low
        completeness = (1.0 if not np.isfinite(denom) or denom <= 1e-6
                        else float(np.clip((post_high - pre_low) / denom, 0, 1.2)))

        if completeness >= cfg.wash_full_recovery_pct:
            cls = "Full recovery"
        elif completeness >= cfg.wash_partial_recovery_pct:
            cls = "Partial recovery"
        else:
            cls = "Minimal recovery"

        events.append(dict(
            event_date=df["date"].iloc[i].date(),
            cause=cause, delta_nci=float(d_raw[i]),
            pre_event_low=pre_low, post_event_high=post_high,
            baseline_clean=baseline_clean,
            completeness=float(completeness),
            recovery_class=cls, rain_mm_today=rain_today,
        ))
        skip_until = i + int(cfg.wash_window_days) + 1

    events_df = (pd.DataFrame(events) if events else pd.DataFrame(
        columns=["event_date", "cause", "delta_nci", "pre_event_low",
                 "post_event_high", "baseline_clean", "completeness",
                 "recovery_class", "rain_mm_today"]))

    if events:
        last_evt = events[-1]
        cur_mask = df["date"].dt.date >= last_evt["event_date"]
        cur_df   = df.loc[cur_mask].reset_index(drop=True)
        most_recent = last_evt
    else:
        cur_df = df.copy()
        most_recent = None

    expl_lines = []
    if not events:
        expl_lines.append("no wash/rain recovery events detected")
    else:
        for e in events:
            expl_lines.append(f"{e['event_date']}: {e['cause']} "
                              f"(ΔNCI=+{e['delta_nci']*100:.1f}pp, "
                              f"{e['recovery_class']} {e['completeness']*100:.0f}%)")

    return dict(events_df=events_df, current_segment_df=cur_df,
                most_recent_event=most_recent, n_events=len(events),
                nci_col_used=col,
                explainability="; ".join(expl_lines))


def _empty():
    return dict(
        events_df=pd.DataFrame(columns=["event_date", "cause", "delta_nci",
            "pre_event_low", "post_event_high", "baseline_clean",
            "completeness", "recovery_class", "rain_mm_today"]),
        current_segment_df=pd.DataFrame(),
        most_recent_event=None, n_events=0,
        nci_col_used="none",
        explainability="no data")
