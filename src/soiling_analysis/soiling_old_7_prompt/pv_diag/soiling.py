"""Segment-aware soiling-trend extraction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .utils import pick_nci_column


def is_slope_significant(slope: float, se: float, cfg: PipelineConfig) -> bool:
    """Return True only when the slope is both large enough and distinguishable from noise.

    Two independent gates prevent noisy-but-tiny or large-but-uncertain slopes from
    triggering a soiling verdict:
      1. |slope| > cfg.soiling_slope_significance  — operationally meaningful magnitude.
      2. |slope| / (se + 1e-9) > cfg.soiling_slope_snr — trend not buried in residual noise.
    """
    if not np.isfinite(slope) or not np.isfinite(se):
        return False
    abs_slope = abs(slope)
    snr = abs_slope / (se + 1e-9)
    return (abs_slope > cfg.soiling_slope_significance) and (snr > cfg.soiling_slope_snr)


def has_recovery_signature(wash_result: dict) -> bool:
    """Return True if any wash event shows Full or Partial recovery.

    Checks every event in events_df (not just most_recent_event) so that a string
    that was washed several months ago but has since re-soiled still carries the
    recovery signal.  Re-detection is not performed here — this reads the already
    classified recovery_class field produced by wash_detect.
    """
    events_df = wash_result.get("events_df", None)
    if events_df is not None and not events_df.empty and "recovery_class" in events_df.columns:
        recovery_classes = events_df["recovery_class"].dropna()
        return bool((recovery_classes == "Full recovery").any() or
                    (recovery_classes == "Partial recovery").any())
    # Also check most_recent_event as a fallback for callers that only populate that field.
    me = wash_result.get("most_recent_event")
    if me:
        return me.get("recovery_class") in ("Full recovery", "Partial recovery")
    return False


def _trimmed_lr(x, y, trim_pct=0.10):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 4:
        return dict(slope=np.nan, intercept=np.nan, n=n, se=np.nan, r2=np.nan, kept=0)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    n_trim = int(np.floor(trim_pct * n))
    if n_trim > 0:
        order = np.argsort(np.abs(resid))
        keep = order[: max(n - n_trim, 4)]
        x = x[keep]; y = y[keep]
    if len(x) < 4:
        return dict(slope=a, intercept=b, n=n, se=np.nan, r2=np.nan, kept=len(x))
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se_slope = (np.sqrt(ss_res / max(len(x) - 2, 1))
                / (np.sqrt(np.sum((x - x.mean()) ** 2)) + 1e-9))
    return dict(slope=float(a), intercept=float(b), n=int(n), kept=int(len(x)),
                se=float(se_slope), r2=float(r2))


def extract_soiling_trend(daily_df: pd.DataFrame, wash_result: dict,
                          cfg: PipelineConfig) -> dict:
    """Full-window soiling trend using the best available NCI column.

    Uses pick_nci_column() to prefer NCI_adaptive_noon > NCI_relative_noon
    > NCI_corrected_noon > NCI_noon.

    Key result fields:
      srr_pct_per_day           — uncapped weighted-average slope (%/day)
      weighted_soiling_loss_pct — trend-based: clip(|capped_slope|*100*n_days, 0, cap*100)
                                  averaged across segments (headline metric)
      segments[*].srr_capped_pct_per_day   — per-segment capped slope (loss calc only)
      segments[*].mean_nci_based_loss_pct  — old level-based loss (diagnostic only)
    """
    if daily_df is None or len(daily_df) == 0:
        return _empty_soiling()
    df = daily_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    col = pick_nci_column(df)

    if df[col].notna().sum() < cfg.min_days_for_trend:
        return _empty_soiling(reason=f"only {df[col].notna().sum()} valid days")

    events_df = wash_result.get("events_df", pd.DataFrame())
    event_dates = []
    if events_df is not None and not events_df.empty:
        event_dates = sorted(pd.to_datetime(events_df["event_date"]).dt.date.tolist())

    seg_bounds = []
    seg_start = df["date"].iloc[0].date()
    for ed in event_dates:
        seg_bounds.append((seg_start, ed))
        seg_start = ed
    seg_bounds.append((seg_start, df["date"].iloc[-1].date()))

    segments_out = []
    for (s0, s1) in seg_bounds:
        sub = df[(df["date"].dt.date >= s0) & (df["date"].dt.date <= s1)]
        if sub[col].notna().sum() < 4:
            segments_out.append(dict(start=s0, end=s1, n_days=len(sub),
                slope_per_day=np.nan, slope_pct_per_day=np.nan,
                srr_capped_pct_per_day=np.nan, se=np.nan,
                r2=np.nan, soiling_loss_pct=np.nan,
                mean_nci_based_loss_pct=np.nan,
                mean_nci=float(sub[col].mean()) if sub[col].notna().any() else np.nan))
            continue
        x = (sub["date"] - sub["date"].min()).dt.days.values.astype(float)
        y = sub[col].values.astype(float)
        fit = _trimmed_lr(x, y)
        # Raw slope is reported as-is; the cap is applied only for the loss calculation
        # so that absurdly steep noisy segments don't produce absurd loss estimates.
        slope_raw = float(fit["slope"])
        slope_capped = float(np.clip(slope_raw, -0.03, 0.01))
        mean_nci = float(np.nanmean(y))
        # Trend-based loss: how much NCI accumulates over the segment at the capped rate.
        accumulated_loss_pct = float(np.clip(
            abs(slope_capped) * 100.0 * len(sub), 0.0, cfg.soiling_loss_cap * 100.0
        ))
        # Level-based loss kept as secondary diagnostic (old headline formula).
        mean_nci_based_loss_pct = float(
            np.clip(1.0 - mean_nci, 0.0, cfg.soiling_loss_cap) * 100.0
        )
        seg_se = fit["se"]
        slope_snr = float(abs(slope_raw) / (seg_se + 1e-9)) if np.isfinite(seg_se) else 0.0
        segments_out.append(dict(start=s0, end=s1, n_days=len(sub),
            slope_per_day=slope_raw, slope_pct_per_day=slope_raw * 100,
            srr_capped_pct_per_day=slope_capped * 100,
            se=seg_se, r2=fit["r2"],
            soiling_loss_pct=accumulated_loss_pct,
            mean_nci_based_loss_pct=mean_nci_based_loss_pct,
            mean_nci=mean_nci,
            slope_significant=is_slope_significant(slope_raw, seg_se, cfg),
            slope_snr=slope_snr))

    valid = [s for s in segments_out if np.isfinite(s["slope_per_day"])]
    if not valid:
        return _empty_soiling(reason="no valid segments")
    w = np.array([s["n_days"] for s in valid], dtype=float)
    sl = np.array([s["slope_per_day"] for s in valid])
    se = np.array([s["se"] if np.isfinite(s["se"]) else 0.0 for s in valid])
    wt_slope = float(np.average(sl, weights=w))
    wt_se = float(np.sqrt(np.average(se**2, weights=w)))
    ci = cfg.confidence_z * wt_se
    losses = np.array([s["soiling_loss_pct"] for s in valid])
    wt_loss = float(np.average(losses, weights=w))

    rd = []
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            d = (float(ev["baseline_clean"]) - float(ev["pre_event_low"])) * 100.0
            if np.isfinite(d) and d > 0:
                rd.append(d)
    median_recovery = float(np.median(rd)) if rd else np.nan

    expl = [f"{len(valid)}/{len(segments_out)} valid segments; "
            f"weighted SRR={wt_slope*100:.3f} %/day (±{ci*100:.3f} pp); "
            f"nci_col={col}"]
    for s in segments_out:
        if np.isfinite(s["slope_per_day"]):
            expl.append(f"  {s['start']}->{s['end']}: "
                        f"slope={s['slope_pct_per_day']:.3f}%/day, "
                        f"loss={s['soiling_loss_pct']:.1f}%, n={s['n_days']}")

    any_sig = any(s.get("slope_significant", False) for s in valid)
    return dict(srr_pct_per_day=wt_slope * 100.0, ci_pct_per_day=ci * 100.0,
                weighted_soiling_loss_pct=wt_loss,
                median_recovery_depth_pct=median_recovery,
                n_segments=len(segments_out), segments=segments_out,
                method="segment_weighted_trimmed_lr",
                nci_col_used=col,
                any_segment_slope_significant=any_sig,
                explainability="\n".join(expl))


def extract_soiling_current_segment(daily_df, wash_result, cfg):
    cur = wash_result.get("current_segment_df", pd.DataFrame())
    if cur is None or cur.empty:
        return _empty_soiling("no current segment")
    return extract_soiling_trend(cur, dict(events_df=pd.DataFrame()), cfg)


def _empty_soiling(reason="no data"):
    return dict(srr_pct_per_day=np.nan, ci_pct_per_day=np.nan,
                weighted_soiling_loss_pct=np.nan,
                median_recovery_depth_pct=np.nan, n_segments=0, segments=[],
                method="none", nci_col_used="none",
                any_segment_slope_significant=False,
                explainability=reason)
