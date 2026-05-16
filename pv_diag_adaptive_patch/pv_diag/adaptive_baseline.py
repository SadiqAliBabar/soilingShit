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
    peer_ladder_level: Optional[int] = None   # 1–4 from build_peer_groups; None when adaptive disabled
    reference_method: str = "unknown"          # "recovery_anchored" | "p95_fallback"
    peer_substituted: bool = False
    peer_substituted_delta: float = float("nan")
    peer_median_ref: Optional[float] = None
    n_recovery_events_used: int = 0            # how many post-wash plateaus contributed to clean_ref


# ---------------------------------------------------------------------------
# Layer 1 — per-string estimate
# ---------------------------------------------------------------------------

def estimate_string_clean_baseline(
    daily_df: pd.DataFrame,
    cfg: PipelineConfig,
    rain_events: Any,  # events_df DataFrame or list of dicts from detect_wash_events
) -> dict:
    """Compute the per-string clean-NCI estimate from recent high-quality days.

    Uses a two-path approach to avoid self-referential baseline drift:

    Path A — Recovery-anchored: if any Full/Partial recovery event falls in
    the adaptive window, measure NCI_noon on the D+1…D+plateau_days days after
    each event and take the maximum plateau median as the clean reference.  This
    anchors the baseline to a physically verified clean state rather than to the
    string's own recent history, which may be chronically depressed by soiling.

    Path B — P95 fallback: when no valid recovery events are found, fall back to
    P95 of the filtered distribution.  Gate B is stricter in this path — it
    requires the P95 to exceed cfg.adaptive_no_rain_floor regardless of whether
    rain events were counted, because none proved to be a reliable clean anchor.

    Gates A and C are applied to the resolved clean_ref in both paths.

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
        value                   – float clean_ref (or None if any gate fails)
        source                  – "adaptive_string" | "reject_*"
        reason                  – human-readable reason string
        reference_method        – "recovery_anchored" | "p95_fallback" | "unknown"
        n_recovery_events_used  – int, number of plateaus that contributed
        n_used                  – int, surviving rows count
        p50, p95, p99           – float quantiles of the full distribution
        n_rain_events_in_window – int
        peer_substituted        – False (updated later by apply_peer_cross_check)
        peer_substituted_delta  – nan  (updated later)
        peer_median_ref         – None (updated later)
    """
    _reject_extra = dict(
        reference_method="unknown", n_recovery_events_used=0,
        peer_substituted=False, peer_substituted_delta=float("nan"),
        peer_median_ref=None,
    )

    if daily_df is None or len(daily_df) == 0:
        return dict(value=None, reason="no_data", n_used=0,
                    source="reject_no_data", p50=None, p95=None, p99=None,
                    n_rain_events_in_window=0, **_reject_extra)

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

    # ---- Count rain/wash events inside window (kept for diagnostics) ----
    n_rain = _count_rain_events_in_window(rain_events, window_start, max_date)

    # ---- Check minimum clean days ----
    if len(rows) < cfg.adaptive_min_clean_days:
        return dict(value=None, reason="insufficient_clean_days",
                    n_used=int(len(rows)), source="reject_insufficient_data",
                    p50=None, p95=None, p99=None,
                    n_rain_events_in_window=n_rain, **_reject_extra)

    nci_vals = pd.to_numeric(rows["NCI_noon"], errors="coerce").dropna().values.astype(float)
    p50 = float(np.quantile(nci_vals, 0.50))
    p95 = float(np.quantile(nci_vals, 0.95))
    p99 = float(np.quantile(nci_vals, 0.99))

    # ---- Step 1: extract valid recovery events in window ----
    valid_recoveries = _extract_valid_recoveries(rain_events, window_start, max_date)

    # ---- Step 2: recovery-anchored reference (Full/Partial events only) ----
    plateaus: List[float] = []
    for ev_date in valid_recoveries:
        plateau_val = _compute_recovery_plateau(
            df, ev_date, cfg.recovery_plateau_days, cfg.adaptive_min_midday_points
        )
        if plateau_val is not None:
            plateaus.append(plateau_val)

    if plateaus:
        clean_ref = max(plateaus)
        reference_method = "recovery_anchored"
        n_recovery_events_used = len(plateaus)
    else:
        # ---- Step 3: P95 fallback when no valid recoveries ----
        clean_ref = p95
        reference_method = "p95_fallback"
        n_recovery_events_used = 0

    _pass_extra = dict(
        reference_method=reference_method,
        n_recovery_events_used=n_recovery_events_used,
        peer_substituted=False,
        peer_substituted_delta=float("nan"),
        peer_median_ref=None,
    )

    # ---- Gate A: absolute floor (applied to resolved clean_ref) ----
    if clean_ref < cfg.adaptive_min_p95:
        return dict(value=None, reason="p95_below_floor",
                    n_used=int(len(rows)), source="reject_floor_violated",
                    p50=p50, p95=p95, p99=p99,
                    n_rain_events_in_window=n_rain, **_pass_extra)

    # ---- Gate B: p95_fallback with no reliable anchor ----
    # Previously checked n_rain == 0; now checks reference_method == "p95_fallback"
    # so it fires whenever no Full/Partial recovery confirmed the clean state,
    # even if raw rain counts were > 0.
    if reference_method == "p95_fallback" and clean_ref < cfg.adaptive_no_rain_floor:
        return dict(value=None, reason="no_rain_anchor",
                    n_used=int(len(rows)), source="reject_no_rain_anchor",
                    p50=p50, p95=p95, p99=p99,
                    n_rain_events_in_window=n_rain, **_pass_extra)

    # ---- All gates passed ----
    return dict(value=float(clean_ref), reason="ok",
                n_used=int(len(rows)), source="adaptive_string",
                p50=p50, p95=p95, p99=p99,
                n_rain_events_in_window=n_rain, **_pass_extra)


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


def _extract_valid_recoveries(
    rain_events: Any,
    window_start: pd.Timestamp,
    max_date: pd.Timestamp,
) -> List[pd.Timestamp]:
    """Return event dates for Full/Partial recovery events inside the adaptive window.

    Minimal recovery events are excluded because their post-wash NCI plateau
    is unreliable — the string may still be partially soiled.  If events_df
    has no recovery_class column (e.g. from an older wash_detect version), we
    return [] and the caller falls back to P95.
    """
    valid: List[pd.Timestamp] = []
    if rain_events is None:
        return valid
    if isinstance(rain_events, pd.DataFrame):
        if rain_events.empty or "event_date" not in rain_events.columns:
            return valid
        if "recovery_class" not in rain_events.columns:
            return valid
        ev = rain_events.copy()
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        in_window = (ev["event_date"] >= window_start) & (ev["event_date"] <= max_date)
        ev = ev[in_window & ev["recovery_class"].isin(
            ["Full recovery", "Partial recovery"]
        )]
        return ev["event_date"].tolist()
    # List-of-dicts path
    for e in rain_events:
        try:
            ed = pd.to_datetime(e.get("event_date", e.get("date")))
            if ed is None or not (window_start <= ed <= max_date):
                continue
            rc = e.get("recovery_class", "")
            if rc in ("Full recovery", "Partial recovery"):
                valid.append(ed)
        except Exception:
            pass
    return valid


def _compute_recovery_plateau(
    df_in_window: pd.DataFrame,
    event_date: pd.Timestamp,
    plateau_days: int,
    min_midday_points: int,
) -> Optional[float]:
    """Median NCI_noon for the plateau window immediately after a wash event.

    Samples days D+1 through D+plateau_days after event_date, filtered to
    days where n_valid >= min_midday_points.  Returns None when there are
    insufficient high-quality days in the plateau window.
    """
    plateau_start = event_date + pd.Timedelta(days=1)
    plateau_end = event_date + pd.Timedelta(days=plateau_days)
    mask = (
        (df_in_window["date"] >= plateau_start)
        & (df_in_window["date"] <= plateau_end)
    )
    sub = df_in_window[mask].copy()
    if "n_valid" in sub.columns:
        nv = pd.to_numeric(sub["n_valid"], errors="coerce").fillna(0)
        sub = sub[nv >= min_midday_points]
    if sub.empty:
        return None
    nci = pd.to_numeric(sub["NCI_noon"], errors="coerce").dropna()
    nci = nci[(nci >= 0.5) & (nci <= 1.15)]
    if nci.empty:
        return None
    return float(np.median(nci))


# ---------------------------------------------------------------------------
# Layer 2 — cluster baseline
# ---------------------------------------------------------------------------

def estimate_cluster_clean_baseline(
    per_string_p95: Dict[str, Optional[float]],
    peer_groups: Dict[str, dict],
) -> Dict[str, Optional[float]]:
    """Per-string peer-group median of P95 values (only from strings that passed Gates A+B).

    Replaces the flat cluster-median approach, which returned None on plants
    where every MPPT port hosts exactly one string (unique full_cluster per string).
    peer_groups (from build_peer_groups) defines each string's candidate peers
    independently of inverter/MPPT identity, so orientation-matched strings
    across different inverters can form a valid group.

    Parameters
    ----------
    per_string_p95 : {label: p95_float_or_None}
        Only strings with non-None p95 (after Gates A+B) contribute as peers.
        Gate C is applied later by apply_cross_string_gate.
    peer_groups : {label: {"level": int, "peers": [labels]}}
        From build_peer_groups().  Level-4 entries produce None immediately.

    Returns
    -------
    {string_label: float_or_None}
        None when the string is level-4 or fewer than 2 valid P95 values exist
        in its peer group (including itself).
    """
    result: Dict[str, Optional[float]] = {}
    for label, pg in peer_groups.items():
        if pg.get("level", 4) == 4:
            result[label] = None
            continue
        # Include the string itself so it anchors its own peer-group median.
        all_members = [label] + list(pg.get("peers", []))
        vals = [
            float(per_string_p95[m])
            for m in all_members
            if m in per_string_p95
            and per_string_p95[m] is not None
            and np.isfinite(float(per_string_p95[m]))
        ]
        result[label] = float(np.median(vals)) if len(vals) >= 2 else None
    return result


# ---------------------------------------------------------------------------
# Gate C — cross-string check
# ---------------------------------------------------------------------------

def apply_cross_string_gate(
    per_string_estimate: Dict[str, dict],
    cluster_baseline: Dict[str, Optional[float]],
    peer_groups: Dict[str, dict],
    cfg: PipelineConfig,
) -> Dict[str, dict]:
    """Gate C: reject strings whose P95 is far below their per-string peer median.

    cluster_baseline now maps string_label → peer-group median (from the updated
    estimate_cluster_clean_baseline), so the cluster-ID indirection is gone and
    the lookup is a direct per-string comparison.  peer_groups is accepted for
    API symmetry with build_peer_groups callers; it is not used in the body.

    Strings that already have ``value=None`` pass through unchanged.
    Rejection updates ``value``, ``source``, and ``reason`` in the estimate
    dict.

    Parameters
    ----------
    per_string_estimate : {label: estimate_dict}
        Estimates from estimate_string_clean_baseline (before Gate C).
    cluster_baseline : {string_label: float_or_None}
        From estimate_cluster_clean_baseline; keyed by string label.
    peer_groups : {label: {"level": int, "peers": [labels]}}
        From build_peer_groups(); accepted for API symmetry.
    cfg : PipelineConfig

    Returns
    -------
    Updated copy of per_string_estimate.
    """
    result: Dict[str, dict] = {}
    for label, est in per_string_estimate.items():
        est = dict(est)  # defensive copy
        if est.get("value") is None:
            result[label] = est
            continue
        p95 = est.get("p95")
        if p95 is None or not np.isfinite(float(p95)):
            result[label] = est
            continue
        peer_med = cluster_baseline.get(label)
        if peer_med is None or not np.isfinite(float(peer_med)):
            result[label] = est
            continue
        threshold = float(peer_med) - cfg.adaptive_cluster_gate
        if float(p95) < threshold:
            est["value"] = None
            est["source"] = "reject_below_cluster"
            est["reason"] = (
                f"p95={float(p95):.3f} < peer_median-{cfg.adaptive_cluster_gate}"
                f" = {threshold:.3f}"
            )
        result[label] = est
    return result


# ---------------------------------------------------------------------------
# Peer cross-check (Part B) — runs after ALL strings finish Layer-1
# ---------------------------------------------------------------------------

def apply_peer_cross_check(
    per_string_est: Dict[str, dict],
    peer_groups: Dict[str, dict],
    cfg: PipelineConfig,
) -> Dict[str, dict]:
    """Detect self-referential baseline masking via recovery-anchored peer comparison.

    A string that has been chronically soiled or faulty throughout the adaptive
    window will produce a depressed P95 or a depressed recovery plateau.  If
    its recovery-anchored peers show a clearly higher clean state, the string's
    own reference is substituted with the peer median and flagged for physical
    inspection.

    Only recovery-anchored peers are used in the median — P95-fallback peers
    may also be biased and are excluded.

    Must be called AFTER all strings have completed Layer-1 estimation so that
    every string's reference_method is populated.  Reads from the original
    per_string_est snapshot (not the in-progress result) to avoid
    order-dependency between strings.

    Parameters
    ----------
    per_string_est : {label: estimate_dict}
        After apply_cross_string_gate; each dict must contain reference_method.
    peer_groups : {label: {"level": int, "peers": [labels]}}
        From build_peer_groups().
    cfg : PipelineConfig

    Returns
    -------
    Updated copy of per_string_est with peer_substituted / peer_median_ref set.
    """
    result: Dict[str, dict] = {lbl: dict(est) for lbl, est in per_string_est.items()}

    for label in result:
        est_orig = per_string_est[label]   # read-only original
        est_new  = result[label]           # mutable copy

        # Only cross-check strings that passed Layer-1 gates
        if est_orig.get("value") is None:
            est_new.setdefault("peer_substituted", False)
            est_new.setdefault("peer_substituted_delta", float("nan"))
            est_new.setdefault("peer_median_ref", None)
            continue

        peer_info = peer_groups.get(label, {"level": 4, "peers": []})
        if peer_info.get("level", 4) == 4:
            # Level-4 strings have no peers — cross-check impossible
            est_new.setdefault("peer_substituted", False)
            est_new.setdefault("peer_substituted_delta", float("nan"))
            est_new.setdefault("peer_median_ref", None)
            continue

        # Collect recovery-anchored clean_ref values from peers (not self)
        peers = list(peer_info.get("peers", []))
        anchored_vals: List[float] = []
        for m in peers:
            m_est = per_string_est.get(m, {})  # read from original snapshot
            if (m_est.get("reference_method") == "recovery_anchored"
                    and m_est.get("value") is not None
                    and np.isfinite(float(m_est["value"]))):
                anchored_vals.append(float(m_est["value"]))

        if len(anchored_vals) < cfg.peer_min_members:
            # Not enough anchored peers for a reliable cross-check
            est_new["peer_substituted"] = False
            est_new["peer_substituted_delta"] = float("nan")
            est_new["peer_median_ref"] = None
            continue

        peer_median = float(np.median(anchored_vals))
        est_new["peer_median_ref"] = peer_median

        string_val = float(est_orig["value"])
        delta = peer_median - string_val

        if delta > cfg.peer_disagreement_margin:
            # String's own clean state is not actually clean; use peer anchor
            est_new["value"] = peer_median
            est_new["source"] = "peer_substituted"
            est_new["peer_substituted"] = True
            est_new["peer_substituted_delta"] = delta
        else:
            est_new["peer_substituted"] = False
            est_new["peer_substituted_delta"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

def resolve_clean_baseline(
    string_label: str,
    per_string_estimate: Dict[str, dict],
    cluster_baseline: Dict[str, Optional[float]],
    peer_groups: Dict[str, dict],
    plate_age_baseline: float,
    last_rain_days_ago: float,
    cfg: PipelineConfig,
) -> AdaptiveBaselineResult:
    """Resolve the final clean NCI reference for one string with full provenance.

    Layer priority (first success wins):
      1 — per-string adaptive (value != None after all three gates)
      2 — peer-group median (with optional dry-season plate blend)
      3 — plate-age baseline (always succeeds; source="plate_only" or "plate_blended")

    cluster_baseline is now keyed by string_label (from the updated
    estimate_cluster_clean_baseline), not by cluster_id.  peer_groups supplies
    the structural ladder level recorded in peer_ladder_level for diagnostics.

    Parameters
    ----------
    string_label : str
    per_string_estimate : {label: estimate_dict}  (after Gate C applied)
    cluster_baseline : {string_label: float_or_None}
    peer_groups : {label: {"level": int, "peers": [labels]}}
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
    peer_info = peer_groups.get(string_label, {"level": 4, "peers": []})
    peer_ladder_level = peer_info.get("level", 4)
    cid = f"peer_lvl{peer_ladder_level}__{string_label}"
    cluster_med_raw = cluster_baseline.get(string_label)

    # ---- Layer 1: per-string adaptive ----
    if est.get("value") is not None:
        v = float(est["value"])
        _peer_note = (
            f", peer_sub_delta={est.get('peer_substituted_delta', float('nan')):.3f}"
            if est.get("peer_substituted") else ""
        )
        expl = (
            f"Layer 1 {est.get('source', 'adaptive_string')}={v:.4f} "
            f"(n_used={est.get('n_used', '?')}, "
            f"ref_method={est.get('reference_method', 'unknown')}, "
            f"n_recovery={est.get('n_recovery_events_used', 0)}, "
            f"p95={_fmt(est.get('p95'))}, "
            f"n_rain_events_in_window={est.get('n_rain_events_in_window', '?')}"
            f"{_peer_note})"
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
            peer_ladder_level=peer_ladder_level,
            reference_method=est.get("reference_method", "unknown"),
            peer_substituted=bool(est.get("peer_substituted", False)),
            peer_substituted_delta=float(est.get("peer_substituted_delta", float("nan"))),
            peer_median_ref=est.get("peer_median_ref"),
            n_recovery_events_used=int(est.get("n_recovery_events_used", 0)),
        )

    # Save Layer 1 rejection reason for explainability
    l1_reason = est.get("reason", "unknown")
    l1_source = est.get("source", "unknown")

    # ---- Layer 2: peer-group adaptive ----
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
            f"(peer_level={peer_ladder_level}, "
            f"Layer 1 rejected: {l1_source}, {l1_reason})"
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
            peer_ladder_level=peer_ladder_level,
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
        f"(peer_level={peer_ladder_level}, "
        f"Layer 1 rejected: {l1_source}, {l1_reason}; "
        f"Layer 2 rejected: no peer median available)"
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
        peer_ladder_level=peer_ladder_level,
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
