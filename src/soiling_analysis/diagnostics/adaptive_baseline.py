"""Adaptive per-string clean NCI baseline estimation.

Architecture mirrors NREL RdTools soiling_srr:
  Layer 1 — per-string P95 of recent high-quality NCI days (Gates A, B, C).
  Layer 2 — cluster-median fallback (with optional dry-season plate blend).
  Layer 3 — plate-age baseline (existing path, always succeeds).

Batch 7 extends the Layer-2/3 fallback into a five-tier monsoon/smog cascade:
  Tier 1  — string adaptive, normal window (= old Layer 1)
  Tier 1b — string adaptive, widened window (new; only when monsoon_fallback_enabled)
  Tier 2  — pure peer-group median, no blend (new; replaces old L2 when not dry)
  Tier 3  — hold-last-good from persistent state store (new)
  Tier 4  — dry-season blend decay: peer median + plate (= old L2 dry-season blend)
  Tier 5  — plate-age baseline, SDM refit suppressed (= old Layer 3)

When monsoon_fallback_enabled=False the old 3-layer cascade is preserved exactly.

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
    age_baseline: float = 1.0                  # string's degradation-corrected nameplate factor (Batch 4)
    age_source: str = "plant_default"          # "string_spec" | "plant_default" (Batch 4)
    # Batch 7 provenance fields
    held_from_date: Optional[str] = None      # ISO timestamp of the last-good hold source
    tier: int = 1                             # cascade tier: 1=adaptive, 2=peer, 3=hold-last-good, 4=dry-blend, 5=plate
    drought_flag: bool = False                # True when a plant-scope low-Kc drought was detected
    blend_weight: float = 0.0                 # weight on plate in the dry-season blend (tier 4)
    suppress_sdm_refit: bool = False          # True when SDM should not be refit (starved window)


# ---------------------------------------------------------------------------
# Layer 1 — per-string estimate
# ---------------------------------------------------------------------------

def estimate_string_clean_baseline(
    daily_df: pd.DataFrame,
    cfg: PipelineConfig,
    rain_events: Any,  # events_df DataFrame or list of dicts from detect_wash_events
    age_baseline: float = 1.0,
    window_days: Optional[int] = None,  # override cfg.adaptive_window_days (Batch 7 window-widen)
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
    _window = window_days if window_days is not None else cfg.adaptive_window_days
    window_start = max_date - pd.Timedelta(days=_window)

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

    # Effective floors scale with the string's age baseline when enabled.
    # A 10-yr string (baseline≈0.954) gets Gate-A = 0.92×0.954 = 0.878 and
    # Gate-B = 0.96×0.954 = 0.916 — accepting a genuinely clean old string
    # whose nameplate NCI is depressed by cumulative degradation.
    _bl = age_baseline if cfg.age_relative_gates_enabled else 1.0
    _gate_a_floor = cfg.adaptive_min_p95 * _bl
    _gate_b_floor = cfg.adaptive_no_rain_floor * _bl

    # ---- Gate A: floor (applied to resolved clean_ref) ----
    if clean_ref < _gate_a_floor:
        return dict(value=None, reason="p95_below_floor",
                    n_used=int(len(rows)), source="reject_floor_violated",
                    p50=p50, p95=p95, p99=p99,
                    n_rain_events_in_window=n_rain, **_pass_extra)

    # ---- Gate B: p95_fallback with no reliable anchor ----
    # Previously checked n_rain == 0; now checks reference_method == "p95_fallback"
    # so it fires whenever no Full/Partial recovery confirmed the clean state,
    # even if raw rain counts were > 0.
    if reference_method == "p95_fallback" and clean_ref < _gate_b_floor:
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
    per_string_age_baselines: Optional[Dict[str, float]] = None,
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
        _age_bl = (per_string_age_baselines.get(label, 1.0)
                   if per_string_age_baselines is not None else 1.0)
        _margin = cfg.adaptive_cluster_gate * (_age_bl if cfg.age_relative_gates_enabled else 1.0)
        threshold = float(peer_med) - _margin
        if float(p95) < threshold:
            est["value"] = None
            est["source"] = "reject_below_cluster"
            est["reason"] = (
                f"p95={float(p95):.3f} < peer_median-margin={threshold:.3f}"
                f" (margin={_margin:.4f})"
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
# Batch 7 — Plant-scope drought detection
# ---------------------------------------------------------------------------

def detect_plant_drought(
    per_string_estimate: Dict[str, dict],
    cfg: PipelineConfig,
) -> bool:
    """Return True if a plant-scope low-Kc / data drought is detected.

    A drought is plant-scope (weather-driven) when ≥ drought_min_string_frac
    of strings simultaneously have no valid L1 clean-day estimate (value=None).
    This distinguishes a monsoon/smog period from a string-local fault, which
    Batch 3 handles via STRING_UNDERPERFORM.

    When monsoon_fallback_enabled=False this always returns False.
    """
    if not cfg.monsoon_fallback_enabled:
        return False
    if not per_string_estimate:
        return False
    total = len(per_string_estimate)
    failed = sum(1 for est in per_string_estimate.values() if est.get("value") is None)
    return (failed / total) >= cfg.drought_min_string_frac


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
    age_source: str = "plant_default",
    # Batch 7 additions (all optional for back-compat):
    daily_df: Optional[pd.DataFrame] = None,
    rain_events: Any = None,
    last_good: Optional[Dict] = None,
    drought_flag: bool = False,
) -> AdaptiveBaselineResult:
    """Resolve the final clean NCI reference for one string with full provenance.

    When monsoon_fallback_enabled=True (default) the five-tier cascade fires:
      Tier 1  — per-string adaptive, normal window (layer=1)
      Tier 1b — per-string adaptive, widened window (layer=1, source=*_widened)
      Tier 2  — pure peer-group median, not blended (layer=2)
      Tier 3  — hold-last-good from state store (layer=2, suppress_sdm_refit=True)
      Tier 4  — dry-season blend decay: cluster_med + plate (layer=2)
      Tier 5  — plate-age baseline (layer=3, suppress_sdm_refit=drought_flag)

    When monsoon_fallback_enabled=False the legacy 3-layer cascade is used:
      Layer 1 → Layer 2 (peer + optional dry blend) → Layer 3 (plate)

    The ``tier`` field is always populated; ``layer`` preserves legacy values
    (1/2/3) for back-compat with existing callers.

    Parameters
    ----------
    string_label : str
    per_string_estimate : {label: estimate_dict}  (after Gate C applied)
    cluster_baseline : {string_label: float_or_None}
        From estimate_cluster_clean_baseline; keyed by string label.
    peer_groups : {label: {"level": int, "peers": [labels]}}
    plate_age_baseline : float
        Degradation-corrected plate baseline (from degradation_baseline()).
    last_rain_days_ago : float
        Days since the most recent rain/wash event; used for dry-season blend.
    cfg : PipelineConfig
    age_source : str
    daily_df : DataFrame or None   — required for the Tier-1b window-widen attempt.
    rain_events : Any or None      — passed to the Tier-1b estimation call.
    last_good : dict or None       — last-good record from StateStore.get_string().
    drought_flag : bool            — plant-scope drought detected by detect_plant_drought().

    Returns
    -------
    AdaptiveBaselineResult
    """
    import warnings as _warnings

    est = per_string_estimate.get(string_label, {})
    peer_info = peer_groups.get(string_label, {"level": 4, "peers": []})
    peer_ladder_level = peer_info.get("level", 4)
    cid = f"peer_lvl{peer_ladder_level}__{string_label}"
    cluster_med_raw = cluster_baseline.get(string_label)
    dry_season = last_rain_days_ago > cfg.dry_season_threshold

    # =========================================================
    # Tier 1 (Layer 1): per-string adaptive, normal window
    # =========================================================
    if est.get("value") is not None:
        v = float(est["value"])
        _peer_note = (
            f", peer_sub_delta={est.get('peer_substituted_delta', float('nan')):.3f}"
            if est.get("peer_substituted") else ""
        )
        expl = (
            f"Tier 1 {est.get('source', 'adaptive_string')}={v:.4f} "
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
            age_baseline=float(plate_age_baseline),
            age_source=age_source,
            tier=1,
            drought_flag=drought_flag,
        )

    l1_reason = est.get("reason", "unknown")
    l1_source = est.get("source", "unknown")

    # =========================================================
    # Batch 7: five-tier fallback cascade
    # =========================================================
    if cfg.monsoon_fallback_enabled:

        # ---------------------------------------------------------
        # Tier 1b: widen the look-back window
        # ---------------------------------------------------------
        if (daily_df is not None
                and cfg.window_widen_max_days > cfg.adaptive_window_days):
            try:
                wide_est = estimate_string_clean_baseline(
                    daily_df, cfg, rain_events,
                    age_baseline=float(plate_age_baseline),
                    window_days=cfg.window_widen_max_days,
                )
            except Exception as exc:
                _warnings.warn(
                    f"[B7] Tier-1b window-widen failed for {string_label}: {exc}"
                )
                wide_est = {}
            if wide_est.get("value") is not None:
                v = float(wide_est["value"])
                expl = (
                    f"Tier 1b adaptive_string_widened={v:.4f} "
                    f"(window={cfg.window_widen_max_days}d, "
                    f"n_used={wide_est.get('n_used', '?')}, "
                    f"ref_method={wide_est.get('reference_method', 'unknown')}; "
                    f"normal window rejected: {l1_source}, {l1_reason})"
                )
                return AdaptiveBaselineResult(
                    value=v,
                    source="adaptive_string_widened",
                    layer=1,
                    explainability=expl,
                    p95=wide_est.get("p95"),
                    p50=wide_est.get("p50"),
                    n_used=int(wide_est.get("n_used", 0)),
                    n_rain_events_in_window=int(wide_est.get("n_rain_events_in_window", 0)),
                    cluster_id=cid,
                    peer_ladder_level=peer_ladder_level,
                    reference_method=wide_est.get("reference_method", "unknown"),
                    peer_substituted=False,
                    peer_substituted_delta=float("nan"),
                    peer_median_ref=None,
                    n_recovery_events_used=int(wide_est.get("n_recovery_events_used", 0)),
                    age_baseline=float(plate_age_baseline),
                    age_source=age_source,
                    tier=1,
                    drought_flag=drought_flag,
                )

        # ---------------------------------------------------------
        # Tier 2: pure peer-group median (only when NOT dry season)
        # ---------------------------------------------------------
        if cluster_med_raw is not None and np.isfinite(float(cluster_med_raw)) and not dry_season:
            cluster_med = float(cluster_med_raw)
            expl = (
                f"Tier 2 cluster_adaptive={cluster_med:.4f} "
                f"(peer_level={peer_ladder_level}; "
                f"Tier 1 rejected: {l1_source}, {l1_reason})"
            )
            return AdaptiveBaselineResult(
                value=cluster_med,
                source="cluster_adaptive",
                layer=2,
                explainability=expl,
                p95=est.get("p95"),
                p50=est.get("p50"),
                n_used=int(est.get("n_used", 0)),
                n_rain_events_in_window=int(est.get("n_rain_events_in_window", 0)),
                cluster_id=cid,
                peer_ladder_level=peer_ladder_level,
                age_baseline=float(plate_age_baseline),
                age_source=age_source,
                tier=2,
                drought_flag=drought_flag,
            )

        # ---------------------------------------------------------
        # Tier 3: hold-last-good (from persistent state store)
        # ---------------------------------------------------------
        if cfg.hold_last_good_enabled and last_good is not None:
            lg_val = last_good.get("baseline")
            if lg_val is not None and np.isfinite(float(lg_val)):
                v = float(lg_val)
                held_from = str(last_good.get("timestamp", ""))
                expl = (
                    f"Tier 3 hold_last_good={v:.4f} "
                    f"(held_from={held_from}, drought_flag={drought_flag}; "
                    f"Tier 1 rejected: {l1_source}, {l1_reason}; "
                    f"Tier 2 unavailable: cluster_med={cluster_med_raw!r})"
                )
                return AdaptiveBaselineResult(
                    value=v,
                    source="hold_last_good",
                    layer=2,
                    explainability=expl,
                    p95=est.get("p95"),
                    p50=est.get("p50"),
                    n_used=int(est.get("n_used", 0)),
                    n_rain_events_in_window=int(est.get("n_rain_events_in_window", 0)),
                    cluster_id=cid,
                    peer_ladder_level=peer_ladder_level,
                    age_baseline=float(plate_age_baseline),
                    age_source=age_source,
                    held_from_date=held_from,
                    tier=3,
                    drought_flag=drought_flag,
                    suppress_sdm_refit=True,
                )

        # ---------------------------------------------------------
        # Tier 4: dry-season blend decay (peer + plate)
        # Applies when cluster_med is available AND dry season is active.
        # ---------------------------------------------------------
        if cluster_med_raw is not None and np.isfinite(float(cluster_med_raw)) and dry_season:
            cluster_med = float(cluster_med_raw)
            weight_plate = float(np.clip(
                (last_rain_days_ago - cfg.dry_season_threshold)
                / cfg.dry_season_threshold,
                0.0, 0.7,
            ))
            v = (1.0 - weight_plate) * cluster_med + weight_plate * float(plate_age_baseline)
            blend_note = (
                f" [dry-season blend: weight_plate={weight_plate:.2f}, "
                f"cluster={cluster_med:.4f}, plate={float(plate_age_baseline):.4f}]"
            )
            expl = (
                f"Tier 4 cluster_adaptive_blended={v:.4f}{blend_note} "
                f"(peer_level={peer_ladder_level}; "
                f"Tier 1 rejected: {l1_source}, {l1_reason})"
            )
            return AdaptiveBaselineResult(
                value=v,
                source="cluster_adaptive_blended",
                layer=2,
                explainability=expl,
                p95=est.get("p95"),
                p50=est.get("p50"),
                n_used=int(est.get("n_used", 0)),
                n_rain_events_in_window=int(est.get("n_rain_events_in_window", 0)),
                cluster_id=cid,
                peer_ladder_level=peer_ladder_level,
                age_baseline=float(plate_age_baseline),
                age_source=age_source,
                tier=4,
                drought_flag=drought_flag,
                blend_weight=weight_plate,
                suppress_sdm_refit=drought_flag,
            )

    else:
        # =========================================================
        # Legacy Layer 2 (monsoon_fallback_enabled=False)
        # Preserves original 3-layer behaviour exactly.
        # =========================================================
        if cluster_med_raw is not None and np.isfinite(float(cluster_med_raw)):
            cluster_med = float(cluster_med_raw)
            if dry_season:
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
                _tier = 4
                _blend_w = weight_plate
            else:
                v = cluster_med
                src = "cluster_adaptive"
                blend_note = ""
                _tier = 2
                _blend_w = 0.0
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
                age_baseline=float(plate_age_baseline),
                age_source=age_source,
                tier=_tier,
                drought_flag=drought_flag,
                blend_weight=_blend_w,
            )

    # =========================================================
    # Tier 5 (Layer 3): plate-age baseline — always succeeds.
    # SDM refit is suppressed when called during a drought.
    # =========================================================
    v = float(plate_age_baseline)
    src = "plate_blended" if dry_season else "plate_only"
    expl = (
        f"Tier 5 {src}={v:.4f} "
        f"(peer_level={peer_ladder_level}, "
        f"Tier 1 rejected: {l1_source}, {l1_reason}; "
        f"Tiers 2-4 unavailable)"
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
        age_baseline=float(plate_age_baseline),
        age_source=age_source,
        tier=5,
        drought_flag=drought_flag,
        suppress_sdm_refit=drought_flag,
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
