"""Segment-aware soiling-trend extraction."""
from __future__ import annotations
import warnings
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


def _robust_lr(x, y, method: str = "theilsen") -> dict:
    """Robust slope estimator (Theil-Sen default, Huber alternative).

    Theil-Sen is the IEC/NREL standard for soiling-rate regression (RdTools
    uses it for SRR).  Its breakdown point is ~29% — far better than trimmed
    OLS, which is ~10%.  Huber is an alternative with similar robustness.
    Both are graceful: a single high-leverage curtailed or transient sample
    that slipped the pre-filter will not tilt the fit.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 4:
        return dict(slope=np.nan, intercept=np.nan, n=n, se=np.nan, r2=np.nan, kept=0)

    if method == "theilsen":
        try:
            from scipy.stats import theilslopes
            res = theilslopes(y, x, 0.95)
            slope = float(res.slope)
            intercept = float(res.intercept)
            # Theil-Sen 95% CI → approximate SE for is_slope_significant
            se = float((res.high_slope - res.low_slope) / (2 * 1.96))
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return dict(slope=slope, intercept=intercept, n=int(n), kept=int(n),
                        se=max(se, 1e-12), r2=float(r2))
        except Exception as e:
            warnings.warn(f"[soiling] Theil-Sen failed ({e}); falling back to trimmed OLS")

    if method == "huber":
        try:
            from sklearn.linear_model import HuberRegressor
            X_2d = x.reshape(-1, 1)
            hreg = HuberRegressor(epsilon=1.35, max_iter=300)
            hreg.fit(X_2d, y)
            slope = float(hreg.coef_[0])
            intercept = float(hreg.intercept_)
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            se = (np.sqrt(ss_res / max(n - 2, 1)) /
                  (np.sqrt(np.sum((x - x.mean()) ** 2)) + 1e-9))
            return dict(slope=slope, intercept=intercept, n=int(n), kept=int(n),
                        se=float(se), r2=float(r2))
        except Exception as e:
            warnings.warn(f"[soiling] Huber failed ({e}); falling back to trimmed OLS")

    # Fallback / "trimmed_ols" explicit path
    return _trimmed_lr(x, y)


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
                          cfg: PipelineConfig,
                          transient_dates: set | None = None) -> dict:
    """Full-window soiling trend using the best available NCI column.

    Uses pick_nci_column() to prefer NCI_adaptive_noon > NCI_relative_noon
    > NCI_corrected_noon > NCI_noon.

    Batch 8 additions:
      - Robust regression (Theil-Sen/Huber) via cfg.robust_soiling_regression.
      - transient_dates: set of date objects to exclude from fitting AND from
        segment-boundary detection (transients ≠ wash events).

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

    _transient_set = set(transient_dates) if transient_dates else set()
    _prefilter_on  = getattr(cfg, "transient_prefilter_enabled", True) and bool(_transient_set)

    events_df = wash_result.get("events_df", pd.DataFrame())
    event_dates = []
    if events_df is not None and not events_df.empty:
        raw_dates = sorted(pd.to_datetime(events_df["event_date"]).dt.date.tolist())
        # Batch 8: exclude transient dates from segment boundaries —
        # a transient dip+rebound is not a segment-splitting wash event.
        event_dates = [ed for ed in raw_dates
                       if not (_prefilter_on and ed in _transient_set)]

    seg_bounds = []
    seg_start = df["date"].iloc[0].date()
    for ed in event_dates:
        seg_bounds.append((seg_start, ed))
        seg_start = ed
    seg_bounds.append((seg_start, df["date"].iloc[-1].date()))

    _regressor    = getattr(cfg, "robust_soiling_regression", "theilsen")
    segments_out = []
    _min_density = cfg.min_valid_day_density if cfg is not None else 0.4
    _grid_on     = getattr(cfg, "daily_grid_enabled", False) if cfg is not None else False

    for (s0, s1) in seg_bounds:
        sub = df[(df["date"].dt.date >= s0) & (df["date"].dt.date <= s1)]
        n_valid = sub[col].notna().sum()
        # Minimum absolute count gate (unchanged)
        if n_valid < 4:
            segments_out.append(dict(start=s0, end=s1, n_days=len(sub),
                slope_per_day=np.nan, slope_pct_per_day=np.nan,
                srr_capped_pct_per_day=np.nan, se=np.nan,
                r2=np.nan, soiling_loss_pct=np.nan,
                mean_nci_based_loss_pct=np.nan,
                mean_nci=float(sub[col].mean()) if sub[col].notna().any() else np.nan))
            continue
        # Density gate: only apply when the grid is on (calendar segment length is meaningful)
        if _grid_on:
            window_days = max((pd.Timestamp(s1) - pd.Timestamp(s0)).days + 1, 1)
            if n_valid / window_days < _min_density:
                segments_out.append(dict(start=s0, end=s1, n_days=window_days,
                    slope_per_day=np.nan, slope_pct_per_day=np.nan,
                    srr_capped_pct_per_day=np.nan, se=np.nan,
                    r2=np.nan, soiling_loss_pct=np.nan,
                    mean_nci_based_loss_pct=np.nan,
                    mean_nci=float(sub[col].mean()) if sub[col].notna().any() else np.nan))
                continue

        # Batch 8: exclude transient days from the fitting arrays only;
        # the segment time-window (n_days) is unchanged for weighting purposes.
        if _prefilter_on:
            fit_sub = sub[~sub["date"].dt.date.isin(_transient_set)]
        else:
            fit_sub = sub

        x = (fit_sub["date"] - fit_sub["date"].min()).dt.days.values.astype(float)
        y = fit_sub[col].values.astype(float)

        if _regressor == "trimmed_ols":
            fit = _trimmed_lr(x, y)
        else:
            fit = _robust_lr(x, y, method=_regressor)

        # Raw slope is reported as-is; the cap is applied only for the loss calculation
        # so that absurdly steep noisy segments don't produce absurd loss estimates.
        slope_raw = float(fit["slope"])
        slope_capped = float(np.clip(slope_raw, -0.03, 0.01))
        mean_nci = float(np.nanmean(sub[col].values.astype(float)))
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
    _method_used = (f"segment_weighted_{_regressor}"
                    if "_regressor" in dir() else "segment_weighted_trimmed_lr")
    return dict(srr_pct_per_day=wt_slope * 100.0, ci_pct_per_day=ci * 100.0,
                weighted_soiling_loss_pct=wt_loss,
                median_recovery_depth_pct=median_recovery,
                n_segments=len(segments_out), segments=segments_out,
                method=_method_used,
                nci_col_used=col,
                any_segment_slope_significant=any_sig,
                transient_days_excluded=len(_transient_set),
                explainability="\n".join(expl))


def extract_soiling_current_segment(daily_df, wash_result, cfg,
                                    transient_dates: set | None = None):
    cur = wash_result.get("current_segment_df", pd.DataFrame())
    if cur is None or cur.empty:
        return _empty_soiling("no current segment")
    return extract_soiling_trend(cur, dict(events_df=pd.DataFrame()), cfg,
                                 transient_dates=transient_dates)


def _empty_soiling(reason="no data"):
    return dict(srr_pct_per_day=np.nan, ci_pct_per_day=np.nan,
                weighted_soiling_loss_pct=np.nan,
                median_recovery_depth_pct=np.nan, n_segments=0, segments=[],
                method="none", nci_col_used="none",
                any_segment_slope_significant=False,
                explainability=reason)
