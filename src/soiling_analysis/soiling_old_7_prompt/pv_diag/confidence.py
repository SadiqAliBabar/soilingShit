"""Confidence scoring for soiling verdicts (Prompt 7).

Replaces the three-level ordinal label (high/medium/low) with a scalar
0–1 score built from five independent evidence dimensions.  The ordinal
label is retained as a derived field for backwards-compatibility and
human readability.  classification.py calls into this module; no scoring
logic is inlined there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Inputs dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInputs:
    """All raw signals needed to score confidence in a soiling verdict."""
    # Data quantity
    n_days_current_segment: int
    n_days_full_window: int
    # Baseline quality
    baseline_layer: int              # 1, 2, or 3
    reference_method: str            # "recovery_anchored" | "p95_fallback"
    peer_ladder_level: Optional[int] # 1–4 or None
    peer_substituted: bool
    n_recovery_events_used: int      # how many plateaus anchored the ref
    # Slope / trend quality
    slope_snr: float                 # |slope| / se (may be NaN)
    slope_significant: bool
    any_segment_slope_significant: bool
    n_valid_segments: int
    n_total_segments: int
    # Cross-string agreement
    baseline_disagreement_flag: bool
    baseline_disagreement_pp: float  # nan if not computed
    # Sufficiency
    sufficiency: str                 # "Good" | "Limited" | "Poor" | "Skipped"
    avail_pct: float
    # Recovery evidence
    has_recovery_signature: bool
    most_recent_recovery_class: Optional[str]
        # "Full recovery" | "Partial recovery" | "Minimal recovery" | None


# ---------------------------------------------------------------------------
# Lookup tables (no magic numbers in logic)
# ---------------------------------------------------------------------------

_DAYS_BPS  = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
_SCORE_BPS = np.array([0.0, 0.3,  0.6, 0.85,  1.0])

_LAYER_BASE  = {1: 1.0, 2: 0.7, 3: 0.4}
_PEER_FACTOR = {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.55}
_SUFF_SCORE  = {"Good": 1.0, "Limited": 0.6, "Poor": 0.3, "Skipped": 0.0}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_confidence_score(inputs: ConfidenceInputs,
                             cfg: PipelineConfig) -> dict:
    """Compute a scalar confidence score (0.0–1.0) for a soiling verdict.

    Five independent evidence dimensions each produce a sub-score in [0, 1].
    Their weighted sum (weights from cfg) is the primary output.  The ordinal
    label is derived by comparing the score to cfg thresholds.  Separating
    scoring from classification keeps the formula fully testable without
    needing a full pipeline run.

    Returns
    -------
    dict with keys: confidence_score, confidence_label, sub_scores,
    confidence_explainability.
    """
    # ---- D1: data quantity ----
    d1 = float(np.interp(inputs.n_days_current_segment, _DAYS_BPS, _SCORE_BPS))
    if inputs.avail_pct < 60.0:
        d1 *= inputs.avail_pct / 60.0

    # ---- D2: baseline quality ----
    layer = inputs.baseline_layer if inputs.baseline_layer in (1, 2, 3) else 3
    d2 = _LAYER_BASE[layer]
    d2 *= 1.0 if inputs.reference_method == "recovery_anchored" else 0.7
    d2 *= _PEER_FACTOR.get(inputs.peer_ladder_level, 0.55)
    n_rec = inputs.n_recovery_events_used
    d2 *= 1.0 if n_rec >= 2 else (0.85 if n_rec == 1 else 0.65)
    if inputs.peer_substituted:
        d2 *= 0.7

    # ---- D3: trend quality ----
    snr_ok = np.isfinite(inputs.slope_snr) and inputs.slope_snr >= cfg.soiling_slope_snr
    if inputs.slope_significant and snr_ok:
        d3_base = min(inputs.slope_snr / 10.0, 1.0)
    elif inputs.has_recovery_signature:
        d3_base = 0.6
    else:
        d3_base = 0.2
    seg_quality = inputs.n_valid_segments / max(inputs.n_total_segments, 1)
    d3 = d3_base * seg_quality

    # ---- D4: cross-string agreement ----
    if inputs.baseline_disagreement_flag:
        pp = float(inputs.baseline_disagreement_pp) if np.isfinite(
            float(inputs.baseline_disagreement_pp)) else 0.0
        d4 = max(0.0, 1.0 - pp / 10.0)
    elif inputs.peer_ladder_level is not None and inputs.peer_ladder_level <= 2:
        d4 = 1.0
    else:
        d4 = 0.6

    # ---- D5: sufficiency ----
    d5 = _SUFF_SCORE.get(inputs.sufficiency, 0.0)

    raw_score = (cfg.conf_weight_data_quantity * d1
                 + cfg.conf_weight_baseline    * d2
                 + cfg.conf_weight_trend       * d3
                 + cfg.conf_weight_agreement   * d4
                 + cfg.conf_weight_sufficiency * d5)
    confidence_score = float(np.clip(raw_score, 0.0, 1.0))

    if confidence_score >= cfg.conf_high_threshold:
        label = "high"
    elif confidence_score >= cfg.conf_medium_threshold:
        label = "medium"
    else:
        label = "low"

    sub_scores = {
        "D1_data_quantity": float(np.clip(d1, 0.0, 1.0)),
        "D2_baseline":      float(np.clip(d2, 0.0, 1.0)),
        "D3_trend":         float(np.clip(d3, 0.0, 1.0)),
        "D4_agreement":     float(np.clip(d4, 0.0, 1.0)),
        "D5_sufficiency":   float(np.clip(d5, 0.0, 1.0)),
    }

    expl = format_confidence_explainability(
        {"confidence_score": confidence_score,
         "confidence_label": label,
         "sub_scores": sub_scores},
        inputs,
    )

    return dict(
        confidence_score=confidence_score,
        confidence_label=label,
        sub_scores=sub_scores,
        confidence_explainability=expl,
    )


# ---------------------------------------------------------------------------
# Explainability formatter
# ---------------------------------------------------------------------------

def format_confidence_explainability(score_dict: dict,
                                     inputs: ConfidenceInputs) -> str:
    """Produce a human-readable per-dimension confidence breakdown.

    Each line is kept under 80 characters so it fits cleanly in terminal
    output, Excel cells, and alert messages.
    """
    ss    = score_dict["sub_scores"]
    score = score_dict["confidence_score"]
    label = score_dict["confidence_label"]

    snr_str = (f"{inputs.slope_snr:.1f}"
               if np.isfinite(inputs.slope_snr) else "nan")
    sig_str = "significant" if inputs.slope_significant else "not significant"
    n_v = inputs.n_valid_segments
    n_t = max(inputs.n_total_segments, 1)
    peer_str = (f"L{inputs.peer_ladder_level}"
                if inputs.peer_ladder_level is not None else "none")
    dis_str = ("no disagreement"
               if not inputs.baseline_disagreement_flag
               else f"disagree={inputs.baseline_disagreement_pp:.1f}pp")

    lines = [
        f"Confidence: {score:.2f} ({label})",
        (f"  D1 data_quantity={ss['D1_data_quantity']:.2f}"
         f" [{inputs.n_days_current_segment} days,"
         f" avail={inputs.avail_pct:.0f}%]"),
        (f"  D2 baseline={ss['D2_baseline']:.2f}"
         f" [layer={inputs.baseline_layer},"
         f" {inputs.reference_method},"
         f" {inputs.n_recovery_events_used} anchors,"
         f" peers={peer_str}]"),
        (f"  D3 trend={ss['D3_trend']:.2f}"
         f" [slope_snr={snr_str}, {sig_str},"
         f" {n_v}/{n_t} valid segments]"),
        (f"  D4 agreement={ss['D4_agreement']:.2f}"
         f" [{dis_str}, peer_ladder={peer_str}]"),
        (f"  D5 sufficiency={ss['D5_sufficiency']:.2f}"
         f" [{inputs.sufficiency}]"),
    ]
    return "\n".join(lines)
