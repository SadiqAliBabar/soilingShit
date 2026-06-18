"""Wash / rain-recovery detection.

Detects upward step in daily NCI after a downward trend.  Uses RAW delta for
step detection (median smoothing would attenuate single-day jumps) but uses
a small leading mean for the pre-event "trend" check.

Column selection: uses pick_nci_column() to prefer NCI_adaptive_noon >
NCI_corrected_noon > NCI_noon.  No other algorithmic change.

Prompt 6 additions:
  - _adjust_event_date_for_drying_delay: cause-label correction for post-rain drying
  - detect_distributed_recovery: multi-day cumulative recovery detector
  - detect_wash_events updated to merge both detector outputs and expose
    n_single_day_events / n_multi_day_events in the return dict.
"""
from __future__ import annotations
from datetime import date as _date
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .utils import pick_nci_column


# ---------------------------------------------------------------------------
# Part E — drying-delay cause-label correction
# ---------------------------------------------------------------------------

def _adjust_event_date_for_drying_delay(
    event_date: _date,
    rain_series: pd.Series,
    nci_series: pd.Series,
    cfg: PipelineConfig,
) -> str:
    """Correct cause label when NCI recovery lags the rain spike by 1-2 days.

    When heavy rain falls on day D but panels are still muddy, the NCI
    recovery only appears on D+1 or D+2 after drying.  The single-day
    detector fires on the recovery day (D+1/D+2) when rain_mm is already
    zero, so the default cause is "Manual wash (suspected)".  This helper
    looks back cfg.wash_rain_lookback_days from event_date; if rain exceeding
    cfg.rain_threshold_mm is found, it returns "Rain" instead.

    Returns the corrected cause string, or None if no correction is needed.
    Does NOT move event_date — date is already correct, only label needed fixing.
    """
    # rain_series must be indexed by date or integer-positional; we accept either.
    # Caller passes a slice aligned to the lookback window.
    if rain_series is None or len(rain_series) == 0:
        return None
    window_rain = pd.to_numeric(rain_series, errors="coerce").fillna(0.0)
    if window_rain.max() >= cfg.rain_threshold_mm:
        return "Rain"
    return None


# ---------------------------------------------------------------------------
# Part A — multi-day distributed recovery detector
# ---------------------------------------------------------------------------

def detect_distributed_recovery(
    df: pd.DataFrame,
    existing_events: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Detect wash/rain recovery events that unfold over multiple days.

    The single-day detector (detect_wash_events) requires a 3pp NCI jump in a
    single day.  This function catches recoveries where the same total NCI rise
    is distributed across 2-N consecutive days — a signature of light rain,
    partial washing, or post-rain drying delay.

    Algorithm:
      For each candidate window of width 2 to cfg.wash_multiday_max_days:
        - Compute cumulative NCI rise over the window.
        - Accept if: cumulative >= threshold, no single day in window
          individually triggered, window does not overlap existing events,
          each day in window is non-declining (within tolerance), and
          either rain is present in the lookback window OR pre-slope is
          negative.
      Event date is the LAST day of the window.
    """
    _SCHEMA = ["event_date", "cause", "delta_nci", "pre_event_low",
               "post_event_high", "baseline_clean", "completeness",
               "recovery_class", "rain_mm_today", "detection_method"]

    empty = pd.DataFrame(columns=_SCHEMA)

    if df is None or len(df) < 3:
        return empty

    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    col = pick_nci_column(df)
    raw = df[col].values.astype(float)
    rain = df["rain_mm"].values.astype(float) if "rain_mm" in df.columns else np.zeros(len(df))

    # Pre-slope: same computation as single-day detector
    smooth = pd.Series(raw).rolling(3, min_periods=1, center=True).median().values
    pre_slope = (pd.Series(smooth).diff().rolling(3, min_periods=2).mean()
                 .shift(1).fillna(0).values)

    # Build set of dates already covered by existing events (lockout)
    locked_dates: set = set()
    if existing_events is not None and len(existing_events) > 0:
        for ed in existing_events["event_date"]:
            locked_dates.add(pd.to_datetime(ed).date())

    n = len(df)
    new_events = []
    # Track indices consumed by already-found distributed events (lockout within this run)
    consumed_indices: set = set()

    for start in range(1, n):
        for width in range(2, cfg.wash_multiday_max_days + 1):
            end = start + width - 1  # inclusive end index
            if end >= n:
                break

            # Skip if any index in window is already consumed
            window_indices = set(range(start, end + 1))
            if window_indices & consumed_indices:
                break  # longer windows from same start also blocked

            # Condition 2: no single day in the window individually triggered
            # (we check this by verifying delta for each day < wash_step_thr)
            d_raw_window = np.diff(raw[start - 1:end + 1])  # length = width
            if np.any(d_raw_window >= cfg.wash_step_thr):
                continue

            # Condition 4: each day non-declining within tolerance
            if np.any(d_raw_window < cfg.wash_multiday_monotone_tolerance):
                continue

            # Condition 1: cumulative rise meets threshold
            if not np.isfinite(raw[end]) or not np.isfinite(raw[start - 1]):
                continue
            delta_cumulative = raw[end] - raw[start - 1]
            if delta_cumulative < cfg.wash_multiday_step_thr:
                continue

            # Condition 3: no overlap with existing events
            window_dates = {df["date"].iloc[k].date() for k in range(start, end + 1)}
            if window_dates & locked_dates:
                continue

            # Condition 5a: rain in window + lookback
            rain_lookback_start = max(0, start - cfg.wash_multiday_rain_lookback_days)
            rain_window = rain[rain_lookback_start:end + 1]
            rain_present = bool(np.nansum(rain_window) >= cfg.rain_threshold_mm)

            # Condition 5b: negative pre-slope before window
            slope_before = float(pre_slope[start])
            slope_negative = slope_before < -0.001

            if not rain_present and not slope_negative:
                continue

            # Cause label
            if rain_present:
                cause = "Rain (distributed)"
            else:
                cause = "Manual wash (distributed, suspected)"

            # Completeness — same as single-day detector
            pre_low_window = raw[max(0, start - 3):start]
            pre_low = float(np.nanmin(pre_low_window)) if len(pre_low_window) else np.nan
            post_window = raw[end:min(n, end + 4)]
            post_high = float(np.nanmedian(post_window)) if len(post_window) else np.nan
            look_back = raw[max(0, start - 14):start]
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

            rain_today = float(rain[end]) if np.isfinite(rain[end]) else 0.0
            event_date = df["date"].iloc[end].date()

            new_events.append(dict(
                event_date=event_date,
                cause=cause,
                delta_nci=float(delta_cumulative),
                pre_event_low=pre_low,
                post_event_high=post_high,
                baseline_clean=baseline_clean,
                completeness=float(completeness),
                recovery_class=cls,
                rain_mm_today=rain_today,
                detection_method="multi_day",
            ))

            # Lock out this window so overlapping windows don't double-fire
            consumed_indices |= window_indices
            locked_dates |= window_dates
            break  # move to next start position once a window is accepted

    if not new_events:
        return empty
    return pd.DataFrame(new_events, columns=_SCHEMA)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_wash_events(daily_df: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Detect wash / rain-recovery step-ups in daily NCI.

    Runs the original single-day step detector then calls
    detect_distributed_recovery() to catch multi-day cumulative events missed
    by the single-day threshold.  Combined events drive segment splitting so
    soiling regression sees clean segments.

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
        explainability, n_single_day_events, n_multi_day_events
    """
    if daily_df is None or len(daily_df) == 0:
        return _empty()

    df = daily_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    col = pick_nci_column(df)

    if df[col].notna().sum() < 2:
        return _empty()

    raw    = df[col].values.astype(float)
    smooth = pd.Series(raw).rolling(3, min_periods=1, center=True).median().values
    rain   = df["rain_mm"].values if "rain_mm" in df.columns else np.zeros(len(df))

    d_raw = np.diff(raw, prepend=raw[0])
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
        if pre_slope[i] >= -0.001:
            continue

        rain_today = float(rain[i]) if np.isfinite(rain[i]) else 0.0
        rain_near  = float(np.nansum(rain[max(0, i-1):min(len(df), i+2)]))
        if rain_today >= cfg.rain_threshold_mm:
            cause = "Rain"
        elif rain_near > 1.0:
            cause = "Rain (light)"
        else:
            # Part E: check lookback window for drying-delay correction
            lookback_start = max(0, i - cfg.wash_rain_lookback_days)
            rain_lookback_slice = pd.Series(rain[lookback_start:i])
            corrected = _adjust_event_date_for_drying_delay(
                df["date"].iloc[i].date(), rain_lookback_slice, None, cfg
            )
            cause = corrected if corrected is not None else "Manual wash (suspected)"

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
            detection_method="single_day",
        ))
        skip_until = i + int(cfg.wash_window_days) + 1

    _COLS = ["event_date", "cause", "delta_nci", "pre_event_low",
             "post_event_high", "baseline_clean", "completeness",
             "recovery_class", "rain_mm_today", "detection_method"]

    events_df = (pd.DataFrame(events) if events
                 else pd.DataFrame(columns=_COLS))

    # Part B — run multi-day detector and merge
    additional_events = detect_distributed_recovery(df, events_df, cfg)

    combined_events = pd.concat([events_df, additional_events], ignore_index=True)
    combined_events = combined_events.sort_values("event_date").reset_index(drop=True)

    n_single = int((combined_events["detection_method"] == "single_day").sum())
    n_multi  = int((combined_events["detection_method"] == "multi_day").sum())

    if len(combined_events) > 0:
        last_evt_row = combined_events.iloc[-1]
        last_evt = last_evt_row.to_dict()
        cur_mask = df["date"].dt.date >= last_evt["event_date"]
        cur_df   = df.loc[cur_mask].reset_index(drop=True)
        most_recent = last_evt
    else:
        cur_df = df.copy()
        most_recent = None

    # Part C — explainability with detection_method and summary header
    expl_lines = []
    total = n_single + n_multi
    expl_lines.append(
        f"n_single_day={n_single}, n_multi_day={n_multi}, total={total} events"
    )

    if len(combined_events) == 0:
        expl_lines.append("no wash/rain recovery events detected")
    else:
        for _, e in combined_events.iterrows():
            method = e.get("detection_method", "single_day")
            if method == "multi_day":
                day_label = "cumulative over multiple days"
                expl_lines.append(
                    f"{e['event_date']}: {e['cause']} "
                    f"(ΔNCI=+{e['delta_nci']*100:.1f}pp {day_label}, "
                    f"{e['recovery_class']} {e['completeness']*100:.0f}%, "
                    f"method={method})"
                )
            else:
                expl_lines.append(
                    f"{e['event_date']}: {e['cause']} "
                    f"(ΔNCI=+{e['delta_nci']*100:.1f}pp, "
                    f"{e['recovery_class']} {e['completeness']*100:.0f}%, "
                    f"method={method})"
                )

    return dict(
        events_df=combined_events,
        current_segment_df=cur_df,
        most_recent_event=most_recent,
        n_events=total,
        nci_col_used=col,
        explainability="; ".join(expl_lines),
        n_single_day_events=n_single,
        n_multi_day_events=n_multi,
    )


def _empty():
    _COLS = ["event_date", "cause", "delta_nci", "pre_event_low",
             "post_event_high", "baseline_clean", "completeness",
             "recovery_class", "rain_mm_today", "detection_method"]
    return dict(
        events_df=pd.DataFrame(columns=_COLS),
        current_segment_df=pd.DataFrame(),
        most_recent_event=None, n_events=0,
        nci_col_used="none",
        explainability="no data",
        n_single_day_events=0,
        n_multi_day_events=0,
    )
