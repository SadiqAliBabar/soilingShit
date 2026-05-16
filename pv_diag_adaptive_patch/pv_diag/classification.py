"""Multi-axis verdict using the current (post-wash) segment.

Changes vs pre-patch:
  1. Column selection uses pick_nci_column() — prefers NCI_adaptive_noon.
  2. Confidence score: replaces the old ordinal n_days rule with a 5-dimension
     scalar score (0–1) computed by confidence.py.  The ordinal "confidence"
     key is retained for backwards-compatibility; confidence_score is the
     primary output.  Layer-3 notch is now encoded in D2 of the score.
  3. Disagreement flag: when both NCI_adaptive_noon and NCI_corrected_noon
     are present, compares soiling losses; if |delta| > 3 pp, sets
     baseline_disagreement_flag = True in axes and appends a WARNING to
     explainability.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .utils import pick_nci_column
from .soiling import has_recovery_signature
from .confidence import ConfidenceInputs, compute_confidence_score

_BAND_CLEAN = 0.97
_BAND_LT    = 0.93
_BAND_MOD   = 0.85


def classify_string(
    daily_df,
    wash_result,
    soiling_full,
    soiling_current,
    cfg: PipelineConfig,
    sdm_metrics=None,
    expected_asym: float = 0.0,
    sufficiency: str = "Good",
    adaptive_result=None,        # AdaptiveBaselineResult or None
) -> dict:
    """Classify a string and return verdict + axes + confidence score.

    The scalar confidence_score replaces the old ordinal n_days rule so that
    downstream systems can distinguish evidence quality within the same label
    bucket.  The "confidence" key remains the ordinal label for
    backwards-compatibility with the Excel export.

    Parameters
    ----------
    adaptive_result : AdaptiveBaselineResult or None
        Provenance of the clean reference used.  Layer and peer info are
        forwarded to compute_confidence_score() for D2/D4 scoring.
    """
    _no_conf = dict(confidence_score=0.0, confidence_sub_scores={},
                    confidence_explainability="insufficient data")

    out = dict(verdict="Insufficient", primary_axis="data",
               axes={}, explainability="", confidence="low",
               **_no_conf)

    if sufficiency == "Skipped":
        out["verdict"] = "Skipped"
        out["explainability"] = "Skipped by sufficiency gate"
        return out
    if daily_df is None or len(daily_df) == 0:
        out["explainability"] = "no daily data"
        return out

    use_cur = cfg.use_current_segment_verdict
    cur_df  = wash_result.get("current_segment_df", pd.DataFrame())
    base_df = cur_df if (use_cur and not cur_df.empty) else daily_df

    # ---- Column selection (adaptive > corrected > raw) ----
    col = pick_nci_column(base_df)
    nci_series = pd.to_numeric(base_df[col], errors="coerce").dropna()
    if len(nci_series) < 3:
        out["verdict"] = "Insufficient"
        out["explainability"] = f"only {len(nci_series)} valid days in {col}"
        return out

    mean_nci = float(nci_series.mean())
    p50_nci  = float(nci_series.median())
    me = wash_result.get("most_recent_event")
    has_full_recovery    = bool(me and me.get("recovery_class") == "Full recovery")
    has_partial_recovery = bool(me and me.get("recovery_class") == "Partial recovery")
    has_minimal_recovery = bool(me and me.get("recovery_class") == "Minimal recovery")

    if mean_nci >= _BAND_CLEAN:   band = "Clean"
    elif mean_nci >= _BAND_LT:    band = "Lt.Soiling"
    elif mean_nci >= _BAND_MOD:   band = "Mod.Soiling"
    else:                         band = "Hvy.Soiling"

    # ---- Soiling accumulation signature ----
    slope_sig = bool(soiling_current.get("any_segment_slope_significant", False))
    has_recov = has_recovery_signature(wash_result)
    soiling_signature = slope_sig or has_recov

    asym_col = "asym"
    obs_asym = (float(base_df[asym_col].dropna().median())
                if asym_col in base_df.columns
                and base_df[asym_col].dropna().size else 0.0)
    excess_asym = obs_asym - abs(expected_asym)
    has_shading = (excess_asym > 0.08 and len(nci_series) >= 5)

    has_degradation = False
    if sdm_metrics:
        voc_ratio = sdm_metrics.get("voc_stc_ratio", np.nan)
        isc_ratio = sdm_metrics.get("isc_stc_ratio", np.nan)
        ff_ratio  = sdm_metrics.get("ff_stc_ratio", np.nan)
        if (np.isfinite(voc_ratio) and np.isfinite(isc_ratio)
                and voc_ratio < 0.95 and isc_ratio > 0.97):
            has_degradation = True
        if np.isfinite(ff_ratio) and ff_ratio < 0.93 and band == "Clean":
            has_degradation = True

    # ---- Verdict requires BOTH below-clean AND accumulation signature ----
    if band == "Clean":
        verdict = band
        primary = "soiling"
        if has_full_recovery:
            verdict = "Clean (post-wash)"; primary = "post_wash_clean"
        elif has_degradation and not has_full_recovery:
            verdict = "Degradation"; primary = "degradation"
        elif has_shading:
            verdict = "Shading"; primary = "shading"
    elif soiling_signature:
        verdict = band
        primary = "soiling"
        if has_partial_recovery and band != "Hvy.Soiling":
            verdict = "Partial Recovery"; primary = "partial_recovery"
        elif has_minimal_recovery and band in ("Mod.Soiling", "Hvy.Soiling"):
            verdict = f"{band} (minimal recovery)"
            primary = "soiling_with_minimal_recovery"

        multi = sum([band != "Clean", has_shading, has_degradation])
        if multi >= 2 and not has_full_recovery and not has_partial_recovery:
            verdict = f"Mixed ({band}+{'shading' if has_shading else 'degradation'})"
            primary = "mixed"
    else:
        verdict = "Fault / degradation — investigate"
        primary = "flat_low_no_signature"

    n = len(nci_series)

    axes = dict(soiling_band=band, mean_nci_current=mean_nci,
                median_nci_current=p50_nci,
                nci_col_used=col,
                wash_event_recovery=(me.get("recovery_class") if me else None),
                wash_event_cause=(me.get("cause") if me else None),
                obs_asymmetry=obs_asym, expected_asymmetry=float(expected_asym),
                excess_asymmetry=float(excess_asym),
                has_shading_flag=has_shading, has_degradation_flag=has_degradation,
                srr_current_pct_per_day=soiling_current.get("srr_pct_per_day", np.nan),
                srr_full_window_pct_per_day=soiling_full.get("srr_pct_per_day", np.nan),
                n_days_current_segment=n,
                slope_significant=slope_sig,
                has_recovery_signature=has_recov,
                soiling_signature_present=soiling_signature,
                mean_nci_based_loss_pct=float(
                    soiling_current.get("weighted_soiling_loss_pct", np.nan)
                    if soiling_current.get("weighted_soiling_loss_pct") is not None
                    else np.nan),
                baseline_disagreement_flag=False,
                baseline_disagreement_pp=np.nan)

    # ---- Disagreement flag (runs before confidence scoring so D4 sees it) ----
    _disagreement_delta = _compute_disagreement(base_df)
    if _disagreement_delta is not None and _disagreement_delta > 3.0:
        axes["baseline_disagreement_flag"] = True
        axes["baseline_disagreement_pp"]   = float(_disagreement_delta)

    # ---- Confidence scoring ----
    _segs = soiling_current.get("segments") or []
    _slope_snr = (float(_segs[0].get("slope_snr", np.nan)) if _segs
                  else float(soiling_current.get("slope_snr", np.nan)))
    _n_base = len(base_df) if (base_df is not None and len(base_df) > 0) else max(n, 1)
    _avail_pct = float(n / _n_base * 100.0)

    _ci = ConfidenceInputs(
        n_days_current_segment=n,
        n_days_full_window=len(daily_df) if (daily_df is not None
                                             and len(daily_df) > 0) else n,
        baseline_layer=int(getattr(adaptive_result, "layer", 3))
                       if adaptive_result is not None else 3,
        reference_method=getattr(adaptive_result, "reference_method", "unknown")
                         if adaptive_result is not None else "unknown",
        peer_ladder_level=getattr(adaptive_result, "peer_ladder_level", None)
                          if adaptive_result is not None else None,
        peer_substituted=bool(getattr(adaptive_result, "peer_substituted", False))
                         if adaptive_result is not None else False,
        n_recovery_events_used=int(getattr(adaptive_result,
                                           "n_recovery_events_used", 0))
                               if adaptive_result is not None else 0,
        slope_snr=_slope_snr,
        slope_significant=slope_sig,
        any_segment_slope_significant=slope_sig,
        n_valid_segments=int(soiling_current.get("n_segments", 0)),
        n_total_segments=int(soiling_full.get("n_segments", 0)),
        baseline_disagreement_flag=bool(axes["baseline_disagreement_flag"]),
        baseline_disagreement_pp=float(axes["baseline_disagreement_pp"]),
        sufficiency=sufficiency,
        avail_pct=_avail_pct,
        has_recovery_signature=bool(axes.get("has_recovery_signature", False)),
        most_recent_recovery_class=me.get("recovery_class") if me else None,
    )
    _score_dict = compute_confidence_score(_ci, cfg)
    conf = _score_dict["confidence_label"]

    # ---- Explainability ----
    expl = [f"Verdict: {verdict} (axis={primary}, confidence={conf})",
            f"  current-segment mean {col} = {mean_nci:.3f} -> band={band}",
            f"  n days in current segment = {n}"]
    if me:
        expl.append(f"  most-recent wash event: {me['event_date']} {me['cause']} "
                    f"-> {me['recovery_class']} ({me['completeness']*100:.0f}%)")
    else:
        expl.append("  no wash/rain event detected")
    if has_shading:
        expl.append(f"  AM/PM asymmetry {obs_asym:.3f} > expected "
                    f"{abs(expected_asym):.3f}")
    if primary == "flat_low_no_signature":
        expl.append(
            f"  NCI below clean threshold (mean={mean_nci:.3f}) but no accumulation "
            f"signature detected (slope not significant, no recovery events). "
            f"Soiling loss NOT reported. Recommend physical inspection."
        )
    elif band != "Clean":
        expl.append(
            f"  Soiling signature: slope_significant={slope_sig}, "
            f"recovery_detected={has_recov}"
        )

    if axes["baseline_disagreement_flag"]:
        expl.append(
            f"WARNING: adaptive vs plate baseline disagree by "
            f"{axes['baseline_disagreement_pp']:.1f}pp — human review recommended"
        )

    return dict(verdict=verdict, primary_axis=primary, axes=axes,
                explainability="\n".join(expl), confidence=conf,
                confidence_score=_score_dict["confidence_score"],
                confidence_sub_scores=_score_dict["sub_scores"],
                confidence_explainability=_score_dict["confidence_explainability"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_disagreement(base_df: pd.DataFrame) -> Optional[float]:
    """Return |soil_loss_adaptive - soil_loss_plate| in pp, or None."""
    if "NCI_adaptive_noon" not in base_df.columns:
        return None
    if "NCI_corrected_noon" not in base_df.columns:
        return None
    nci_adap  = pd.to_numeric(base_df["NCI_adaptive_noon"],  errors="coerce").dropna()
    nci_plate = pd.to_numeric(base_df["NCI_corrected_noon"], errors="coerce").dropna()
    if len(nci_adap) < 3 or len(nci_plate) < 3:
        return None
    soil_adap  = (1.0 - float(nci_adap.mean()))  * 100.0
    soil_plate = (1.0 - float(nci_plate.mean())) * 100.0
    return abs(soil_adap - soil_plate)
