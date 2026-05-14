"""Multi-axis verdict using the current (post-wash) segment.

Changes vs pre-patch:
  1. Column selection uses pick_nci_column() — prefers NCI_adaptive_noon.
  2. Confidence notch: if adaptive_result.layer == 3 (plate-only fallback),
     confidence is reduced by one notch (high→medium, medium→low, low→low).
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

_BAND_CLEAN = 0.97
_BAND_LT    = 0.93
_BAND_MOD   = 0.85

# Confidence notch table (applied when baseline is Layer 3)
_CONF_NOTCH = {"high": "medium", "medium": "low", "low": "low"}


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
    """Classify a string and return verdict + axes + explainability.

    Parameters
    ----------
    adaptive_result : AdaptiveBaselineResult or None
        Provenance of the clean reference used.  When layer==3, confidence
        is reduced by one notch and explainability notes the fallback.
    """
    out = dict(verdict="Insufficient", primary_axis="data",
               axes={}, explainability="", confidence="low")

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

    verdict = band
    primary = "soiling"
    if band == "Clean" and has_full_recovery:
        verdict = "Clean (post-wash)"; primary = "post_wash_clean"
    elif has_partial_recovery and band != "Hvy.Soiling":
        verdict = "Partial Recovery"; primary = "partial_recovery"
    elif has_minimal_recovery and band in ("Mod.Soiling", "Hvy.Soiling"):
        verdict = f"{band} (minimal recovery)"
        primary = "soiling_with_minimal_recovery"

    if has_degradation and band == "Clean" and not has_full_recovery:
        verdict = "Degradation"; primary = "degradation"
    elif has_shading and band == "Clean":
        verdict = "Shading"; primary = "shading"

    multi = sum([band != "Clean", has_shading, has_degradation])
    if multi >= 2 and not has_full_recovery and not has_partial_recovery:
        verdict = f"Mixed ({band}+{'shading' if has_shading else 'degradation'})"
        primary = "mixed"

    n = len(nci_series)
    conf = "high" if n >= 20 else ("medium" if n >= 10 else "low")

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
                baseline_disagreement_flag=False,
                baseline_disagreement_pp=np.nan)

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

    # ---- Disagreement flag ----
    # When both NCI_adaptive_noon and NCI_corrected_noon exist in base_df,
    # compare their implied soiling losses.
    _disagreement_delta = _compute_disagreement(base_df)
    if _disagreement_delta is not None and _disagreement_delta > 3.0:
        axes["baseline_disagreement_flag"] = True
        axes["baseline_disagreement_pp"]   = float(_disagreement_delta)
        expl.append(
            f"WARNING: adaptive vs plate baseline disagree by "
            f"{_disagreement_delta:.1f}pp — human review recommended"
        )

    # ---- Confidence notch for Layer 3 baseline ----
    if adaptive_result is not None and getattr(adaptive_result, "layer", None) == 3:
        conf = _CONF_NOTCH.get(conf, conf)
        expl.append(
            f"baseline_layer=3 ({getattr(adaptive_result, 'source', 'plate-based')} "
            f"fallback) → confidence reduced"
        )

    return dict(verdict=verdict, primary_axis=primary, axes=axes,
                explainability="\n".join(expl), confidence=conf)


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
