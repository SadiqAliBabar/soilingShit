"""Adaptive per-string clean NCI baseline estimation.

Architecture mirrors NREL RdTools soiling_srr:
  Layer 1 — per-string P95 of recent high-quality NCI days (Gates A, B, C).
  Layer 2 — cluster-median fallback (with optional dry-season plate blend).
  Layer 3 — plate-age baseline (existing path, always succeeds).

All thresholds live in PipelineConfig; no magic numbers here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveBaselineResult:
    """Full provenance record for one string's resolved clean reference."""
    value: float
    source: str
    layer: int
    explainability: str
    p95: Optional[float]
    p50: Optional[float]
    n_used: int
    n_rain_events_in_window: int
    cluster_id: str


# ---------------------------------------------------------------------------
# Layer 1 — per-string estimate
# ---------------------------------------------------------------------------

def estimate_string_clean_baseline(
    daily_df: pd.DataFrame,
    cfg: PipelineConfig,
    rain_events: Any,  # events_df DataFrame or list of dicts from detect_wash_events
) -> dict:
    """Compute the per-string clean-NCI estimate from recent high-quality days.

    Parameters
    ----------
    daily_df : DataFrame
        Output of compute_daily_metrics for one string.  Must contain columns
        ``NCI_noon``, ``n_valid``, ``rain_mm``.
    cfg : PipelineConfig
        Threshold configuration.
    rain_events : DataFrame or list of dicts
        Wash/rain events from detect_wash_events (``events_df`` key).

    Returns
    -------
    dict with keys:
        value        – float P95 (or None if any gate fails)
        source       – "adaptive_string" | "reject_*"
        reason       – human-readable reason string
        n_used       – int, surviving rows count
        p50, p95, p99 – float quantiles (None when insufficient data)
        n_rain_events_in_window – int
    """
    if daily_df is None or len(daily_df) == 0:
        return dict(value=None, reason="no_data", n_used=0,
                    source="reject_no_data", p50=None, p95=None, p99=None,
                    n_rain_events_in_window=0)

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    window_start = max_date - pd.Timedelta(days=cfg.adaptive_window_days)

    # Restrict to adaptive window
    df = df[df["date"] >= window_start].copy()

    # ---- Build row-filter mask ----
    nci_col = pd.to_numeric(df["NCI_noon"], errors="coerce")

    if "n_valid" in df.columns:
        n_valid_col = pd.to_numeric(df["n_valid"], errors="coerce").fillna(0)
    else:
        # Fallback: assume all days have enough points
        n_valid_col = pd.Series(
            float(cfg.adaptive_min_midday_points), index=df.index
        )

    if "rain_mm" in df.columns:
        rain_col = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0.0)
    else:
        rain_col = pd.Series(0.0, index=df.index)

    mask = (
        nci_col.notna()
        & nci_col.apply(np.isfinite)
        & (n_valid_col >= cfg.adaptive_min_midday_points)
        & (rain_col < cfg.rain_threshold_mm)
        & (nci_col >= 0.5)
        & (nci_col <= 1.15)
    )
    rows = df[mask]

    # ---- Count rain/wash events inside window ----
    n_rain = _count_rain_events_in_window(rain_events, window_start, max_date)

    # ---- Check minimum clean days ----
    if len(rows) < cfg.adaptive_min_clean_days:
        return dict(value=None, reason="insufficient_clean_days",
                    n_used=int(len(rows)), source="reject_insufficient_data",
                    p50=None, p95=None, p99=None,
                    n_rain_events_in_window=n_rain)

    nci_vals = pd.to_numeric(rows["NCI_noon"], errors="coerce").dropna().values.astype(float)
    p50 = float(np.quantile(nci_vals, 0.50))
    p95 = float(np.quantile(nci_vals, 0.95))
    p99 = float(np.quantile(nci_vals, 0.99))

    # ---- Gate A: absolute floor ----
    if p95 < cfg.adaptive_min_p95:
        return dict(value=None, reason="p95_below_floor",
                    n_used=int(len(rows)), source="reject_floor_violated",
                    p50=p50, p95=p95, p99=p99,
                    n_rain_events_in_window=n_rain)

    # ---- Gate B: no rain anchor ----
    if n_rain == 0 and p95 < cfg.adaptive_no_rain_floor:
        return dict(value=None, reason="no_rain_anchor",
                    n_used=int(len(rows)), source="reject_no_rain_anchor",
                    p50=p50, p95=p95, p99=p99,
                    n_rain_events_in_window=n_rain)

    # ---- All gates passed ----
    return dict(value=float(p95), reason="ok",
                n_used=int(len(rows)), source="adaptive_string",
                p50=p50, p95=p95, p99=p99,
                n_rain_events_in_window=n_rain)


def _count_rain_events_in_window(
    rain_events: Any,
    window_start: pd.Timestamp,
    max_date: pd.Timestamp,
) -> int:
    """Count wash/rain events that fall inside [window_start, max_date]."""
    n = 0
    if rain_events is None:
        return 0
    if isinstance(rain_events, pd.DataFrame):
        if rain_events.empty or "event_date" not in rain_events.columns:
            return 0
        ev_dates = pd.to_datetime(rain_events["event_date"])
        return int(((ev_dates >= window_start) & (ev_dates <= max_date)).sum())
    # List of dicts
    for ev in rain_events:
        try:
            ed = pd.to_datetime(ev.get("event_date", ev.get("date")))
            if ed is not None and window_start <= ed <= max_date:
                n += 1
        except Exception:
            pass
    return n


# ---------------------------------------------------------------------------
# Layer 2 — cluster baseline
# ---------------------------------------------------------------------------

def estimate_cluster_clean_baseline(
    per_string_p95: Dict[str, Optional[float]],
    clusters: Dict[str, str],
) -> Dict[str, Optional[float]]:
    """Cluster-median of per-string P95 values (only from strings that passed Gates A+B).

    Parameters
    ----------
    per_string_p95 : {label: p95_float_or_None}
        Only strings with a non-None p95 (i.e. value != None after Gates A+B)
        contribute.  Strings rejected by Gate C are still None here because
        Gate C is applied later by apply_cross_string_gate.
    clusters : {label: cluster_id}
        Flat cluster-id mapping (e.g. full_cluster string).

    Returns
    -------
    {cluster_id: float_or_None}
        None when fewer than 2 finite contributors.
    """
    cluster_vals: Dict[str, List[float]] = defaultdict(list)
    for label, p95 in per_string_p95.items():
        cid = clusters.get(label)
        if cid is None:
            continue
        if p95 is not None and np.isfinite(float(p95)):
            cluster_vals[cid].append(float(p95))

    result: Dict[str, Optional[float]] = {}
    all_cluster_ids: set = set(clusters.values())
    for cid in all_cluster_ids:
        vals = cluster_vals.get(cid, [])
        result[cid] = float(np.median(vals)) if len(vals) >= 2 else None
    return result


# ---------------------------------------------------------------------------
# Gate C — cross-string check
# ---------------------------------------------------------------------------

def apply_cross_string_gate(
    per_string_estimate: Dict[str, dict],
    cluster_baseline: Dict[str, Optional[float]],
    clusters: Dict[str, str],
    cfg: PipelineConfig,
) -> Dict[str, dict]:
    """Gate C: reject strings whose P95 is far below their cluster median.

    Strings that already have ``value=None`` pass through unchanged.
    Rejection updates ``value``, ``source``, and ``reason`` in the estimate
    dict.

    Parameters
    ----------
    per_string_estimate : {label: estimate_dict}
        Estimates from estimate_string_clean_baseline (before Gate C).
    cluster_baseline : {cluster_id: float_or_None}
        From estimate_cluster_clean_baseline.
    clusters : {label: cluster_id}
    cfg : PipelineConfig

    Returns
    -------
    Updated copy of per_string_estimate.
    """
    result: Dict[str, dict] = {}
    for label, est in per_string_estimate.items():
        est = dict(est)  # defensive copy
        # Already rejected — pass through
        if est.get("value") is None:
            result[label] = est
            continue
        p95 = est.get("p95")
        if p95 is None or not np.isfinite(float(p95)):
            result[label] = est
            continue
        cid = clusters.get(label)
        if cid is None:
            result[label] = est
            continue
        cluster_med = cluster_baseline.get(cid)
        if cluster_med is None or not np.isfinite(float(cluster_med)):
            result[label] = est
            continue
        threshold = float(cluster_med) - cfg.adaptive_cluster_gate
        if float(p95) < threshold:
            est["value"] = None
            est["source"] = "reject_below_cluster"
            est["reason"] = (
                f"p95={float(p95):.3f} < cluster_med-{cfg.adaptive_cluster_gate}"
                f" = {threshold:.3f}"
            )
        result[label] = est
    return result


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

def resolve_clean_baseline(
    string_label: str,
    per_string_estimate: Dict[str, dict],
    cluster_baseline: Dict[str, Optional[float]],
    clusters: Dict[str, str],
    plate_age_baseline: float,
    last_rain_days_ago: float,
    cfg: PipelineConfig,
) -> AdaptiveBaselineResult:
    """Resolve the final clean NCI reference for one string with full provenance.

    Layer priority (first success wins):
      1 — per-string adaptive (value != None after all three gates)
      2 — cluster-median adaptive (with optional dry-season plate blend)
      3 — plate-age baseline (always succeeds; source="plate_only" or "plate_blended")

    Parameters
    ----------
    string_label : str
    per_string_estimate : {label: estimate_dict}  (after Gate C applied)
    cluster_baseline : {cluster_id: float_or_None}
    clusters : {label: cluster_id}
    plate_age_baseline : float
        Degradation-corrected plate baseline (from degradation_baseline()).
    last_rain_days_ago : float
        Days since the most recent rain/wash event; used for dry-season blend.
    cfg : PipelineConfig

    Returns
    -------
    AdaptiveBaselineResult
    """
    est = per_string_estimate.get(string_label, {})
    cid = clusters.get(string_label, "unknown")
    cluster_med_raw = cluster_baseline.get(cid)

    # ---- Layer 1: per-string adaptive ----
    if est.get("value") is not None:
        v = float(est["value"])
        expl = (
            f"Layer 1 {est.get('source', 'adaptive_string')}={v:.4f} "
            f"(n_used={est.get('n_used', '?')}, "
            f"p95={_fmt(est.get('p95'))}, "
            f"n_rain_events_in_window={est.get('n_rain_events_in_window', '?')})"
        )
        return AdaptiveBaselineResult(
            value=v,
            source=est.get("source", "adaptive_string"),
            layer=1,
            explainability=expl,
            p95=est.get("p95"),
            p50=est.get("p50"),
            n_used=int(est.get("n_used", 0)),
            n_rain_events_in_window=int(est.get("n_rain_events_in_window", 0)),
            cluster_id=cid,
        )

    # Save Layer 1 rejection reason for explainability
    l1_reason = est.get("reason", "unknown")
    l1_source = est.get("source", "unknown")

    # ---- Layer 2: cluster adaptive ----
    if cluster_med_raw is not None and np.isfinite(float(cluster_med_raw)):
        cluster_med = float(cluster_med_raw)
        if last_rain_days_ago > cfg.dry_season_threshold:
            weight_plate = float(np.clip(
                (last_rain_days_ago - cfg.dry_season_threshold)
                / cfg.dry_season_threshold,
                0.0, 0.7,
            ))
            v = (1.0 - weight_plate) * cluster_med + weight_plate * float(plate_age_baseline)
            src = "cluster_adaptive_blended"
            blend_note = (
                f" [dry-season blend: weight_plate={weight_plate:.2f}, "
                f"cluster={cluster_med:.4f}, plate={float(plate_age_baseline):.4f}]"
            )
        else:
            v = cluster_med
            src = "cluster_adaptive"
            blend_note = ""
        expl = (
            f"Layer 2 {src}={v:.4f}{blend_note} "
            f"(Layer 1 rejected: {l1_source}, {l1_reason})"
        )
        return AdaptiveBaselineResult(
            value=v,
            source=src,
            layer=2,
            explainability=expl,
            p95=est.get("p95"),
            p50=est.get("p50"),
            n_used=int(est.get("n_used", 0)),
            n_rain_events_in_window=int(est.get("n_rain_events_in_window", 0)),
            cluster_id=cid,
        )

    # ---- Layer 3: plate-based ----
    # When neither L1 nor L2 has a value, the dry-season blend term collapses
    # to plate_age_baseline regardless of weight_plate, so value == plate_age_baseline.
    v = float(plate_age_baseline)
    if last_rain_days_ago > cfg.dry_season_threshold:
        src = "plate_blended"
    else:
        src = "plate_only"
    expl = (
        f"Layer 3 {src}={v:.4f} "
        f"(Layer 1 rejected: {l1_source}, {l1_reason}; "
        f"Layer 2 rejected: no cluster median available)"
    )
    return AdaptiveBaselineResult(
        value=v,
        source=src,
        layer=3,
        explainability=expl,
        p95=est.get("p95"),
        p50=est.get("p50"),
        n_used=int(est.get("n_used", 0)),
        n_rain_events_in_window=int(est.get("n_rain_events_in_window", 0)),
        cluster_id=cid,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)
