"""End-to-end pipeline orchestrator with optional joblib parallelism.

Two-pass adaptive baseline scheme (when cfg.adaptive_baseline_enabled=True):

  Pass 1 (plate-based):
    SDM fit, compute_daily_metrics(adaptive_clean_ref=None), wash_detect.

  Between passes:
    estimate_string_clean_baseline → estimate_cluster_clean_baseline
    → apply_cross_string_gate → resolve_clean_baseline  (per string)

  Pass 2 (adaptive):
    compute_daily_metrics(adaptive_clean_ref=resolved.value)
    + wash_detect, soiling, classification — all using pick_nci_column()
    which now prefers NCI_adaptive_noon.

When cfg.adaptive_baseline_enabled=False the pipeline is a single pass
identical to the pre-patch behaviour.
"""
from __future__ import annotations
import warnings
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .constants import QUALITY_FLAGS
from .ingestion import (load_plant_data, split_into_string_dfs,
                        extract_string_meta, apply_plant_meta_to_cfg)
from .quality import flag_data_quality
from .curtailment import (detect_curtailment, curtailment_summary,
                          quantify_curtailment_loss)
from .sufficiency import compute_data_availability, decide_sufficiency
from .plate import infer_plate_params
from .clustering import assign_clusters, cluster_summary
from .degradation import degradation_baseline, explain_baseline
from .sdm import fit_single_diode, iv_metrics_at_stc, has_pvlib
from .daily import compute_daily_metrics
from .wash_detect import detect_wash_events
from .soiling import extract_soiling_trend, extract_soiling_current_segment
from .transient import detect_transient_events
from .classification import classify_string
from .losses import quantify_string_losses, aggregate_plant_losses
from .orientation import expected_asymmetry
from .adaptive_baseline import (
    estimate_string_clean_baseline,
    estimate_cluster_clean_baseline,
    apply_cross_string_gate,
    resolve_clean_baseline,
    AdaptiveBaselineResult,
)


# ---------------------------------------------------------------------------
# Quality-day filter for SDM fitting
# ---------------------------------------------------------------------------

_CURT_BITS = QUALITY_FLAGS["CURT_STATE"] | QUALITY_FLAGS["CURT_STATISTICAL"]


def _select_quality_days(df: pd.DataFrame, cfg: PipelineConfig) -> set:
    """Return the set of calendar dates that pass all quality gates."""
    ts   = pd.to_datetime(df["ts"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    tmp  = df.assign(
        __hour=hour,
        __date=ts.dt.date,
        __midday=(hour >= 11.0) & (hour <= 13.0),
    )

    good_days: set = set()
    for day, grp in tmp.groupby("__date"):
        mid = grp[grp["__midday"]]
        if len(mid) == 0:
            continue
        poa = pd.to_numeric(mid["POA"], errors="coerce").dropna()
        if len(poa) == 0:
            continue
        if float(poa.max()) < 600.0:
            continue
        mean_poa = float(poa.mean())
        if mean_poa > 0.0:
            cv = float(poa.std()) / mean_poa
            if cv > 0.20:
                continue
        if "qflag" in mid.columns:
            qf = mid["qflag"].values.astype(np.int64)
            curt_frac = float(((qf & _CURT_BITS) > 0).sum()) / max(len(qf), 1)
            if curt_frac >= 0.30:
                continue
        if "rainfall" in grp.columns:
            rain = pd.to_numeric(grp["rainfall"], errors="coerce").fillna(0.0).sum()
            if float(rain) >= cfg.rain_threshold_mm:
                continue
        good_days.add(day)
    return good_days


# ---------------------------------------------------------------------------
# SDM fit helper (shared across passes)
# ---------------------------------------------------------------------------

def _fit_sdm(label: str, df: pd.DataFrame, plate, cfg: PipelineConfig):
    """Run SDM quality-day filter and fit; return (sdm, sdm_metrics)."""
    try:
        good_days = _select_quality_days(df, cfg)
        ts_dates  = pd.to_datetime(df["ts"]).dt.date
        df_for_sdm = df[ts_dates.isin(good_days)]
        if len(df_for_sdm) < 100:
            warnings.warn(
                f"[{label}] SDM quality-day filter: only {len(df_for_sdm)} rows "
                f"survive ({len(good_days)} good days); falling back to full df"
            )
            df_for_sdm = df
    except Exception as _qd_exc:
        warnings.warn(
            f"[{label}] quality-day filter failed ({_qd_exc}); "
            f"using full df for SDM fit"
        )
        df_for_sdm = df

    try:
        sdm = fit_single_diode(df_for_sdm, plate, cfg)
    except Exception as e:
        sdm = dict(success=False, reason=f"sdm_exception:{type(e).__name__}")
    sdm_metrics = (iv_metrics_at_stc(sdm, plate)
                   if sdm and sdm.get("success") else None)
    return sdm, sdm_metrics


# ---------------------------------------------------------------------------
# Pass-1 light processing (daily_df + wash only, no full analysis)
# ---------------------------------------------------------------------------

def _pass1_string(label: str, df: pd.DataFrame, plate, cfg: PipelineConfig,
                  baseline: float, freq_min: float):
    """SDM fit + plate-based daily_df + wash_detect.  Returns a compact dict."""
    sdm, sdm_metrics = _fit_sdm(label, df, plate, cfg)
    try:
        daily_df = compute_daily_metrics(
            df, plate, sdm, cfg, baseline, freq_min,
            adaptive_clean_ref=None,
        )
    except Exception as e:
        warnings.warn(f"[{label}] Pass-1 daily_df failed: {e}")
        daily_df = pd.DataFrame()
    try:
        wash = detect_wash_events(daily_df, cfg)
    except Exception as e:
        warnings.warn(f"[{label}] Pass-1 wash_detect failed: {e}")
        from .wash_detect import _empty as _wash_empty
        wash = _wash_empty()
    return dict(sdm=sdm, sdm_metrics=sdm_metrics, daily_df=daily_df, wash=wash)


# ---------------------------------------------------------------------------
# Full per-string processing (Pass 2 or single-pass)
# ---------------------------------------------------------------------------

def _process_one_string(
    label: str,
    df: pd.DataFrame,
    meta_one: dict,
    cluster_one: dict,
    plate,
    cfg: PipelineConfig,
    baseline: float,
    freq_min: float,
    adaptive_clean_ref: Optional[float] = None,
    adaptive_result: Optional[AdaptiveBaselineResult] = None,
    sdm_precomputed=None,
    sdm_metrics_precomputed=None,
):
    """Full per-string analysis.

    Parameters
    ----------
    adaptive_clean_ref : float or None
        When provided, compute_daily_metrics adds NCI_adaptive_noon.
    adaptive_result : AdaptiveBaselineResult or None
        Provenance forwarded to classify_string for confidence notch.
    sdm_precomputed, sdm_metrics_precomputed
        If provided (from Pass 1), skip SDM re-fit.
    """
    res = dict(label=label)
    try:
        # ---- SDM ----
        if sdm_precomputed is not None:
            sdm         = sdm_precomputed
            sdm_metrics = sdm_metrics_precomputed
        else:
            sdm, sdm_metrics = _fit_sdm(label, df, plate, cfg)
        res["sdm"]         = sdm
        res["sdm_metrics"] = sdm_metrics

        # ---- Daily metrics ----
        daily_df = compute_daily_metrics(
            df, plate, sdm, cfg, baseline, freq_min,
            adaptive_clean_ref=adaptive_clean_ref,
        )
        res["daily_df"] = daily_df

        # ---- Data quality / sufficiency ----
        dq = compute_data_availability(df, cfg, freq_min)
        verdict_suff, reason_suff = decide_sufficiency(dq, cfg)
        res["data_quality"]      = dq
        res["sufficiency"]       = verdict_suff
        res["sufficiency_reason"] = reason_suff

        res["curtailment_summary"] = curtailment_summary(df)
        res["curt_loss"] = quantify_curtailment_loss(df, cfg, freq_min)

        # ---- Wash detect ----
        wash = detect_wash_events(daily_df, cfg)
        res["wash"] = wash

        # ---- Soiling ----
        res["soiling_full"]    = extract_soiling_trend(daily_df, wash, cfg)
        res["soiling_current"] = extract_soiling_current_segment(daily_df, wash, cfg)

        # ---- Transients ----
        res["transients"] = detect_transient_events(daily_df, cfg)

        # ---- Orientation ----
        exp_asym = expected_asymmetry(
            meta_one.get("azimuth", cfg.plant.default_azimuth),
            meta_one.get("tilt",    cfg.plant.default_tilt),
            cfg.site.lat)
        res["expected_asymmetry"] = float(exp_asym)

        # ---- Classification ----
        clx = classify_string(
            daily_df, wash,
            res["soiling_full"], res["soiling_current"],
            cfg,
            sdm_metrics=sdm_metrics,
            expected_asym=exp_asym,
            sufficiency=verdict_suff,
            adaptive_result=adaptive_result,
        )
        res["classification"] = clx

        # ---- Losses ----
        if verdict_suff != "Skipped":
            res["losses"] = quantify_string_losses(
                df, daily_df, res["curt_loss"], cfg, freq_min
            )
        else:
            res["losses"] = dict(
                soiling_kwh=0.0, soiling_pkr=0.0,
                curtailment_kwh=0.0, curtailment_pkr=0.0,
                total_avoidable_kwh=0.0, total_avoidable_pkr=0.0,
                annualised_kwh=0.0, annualised_pkr=0.0,
                period_days=0,
                explainability="skipped: insufficient data")

        res["meta"]    = meta_one
        res["cluster"] = cluster_one

        # ---- Adaptive baseline provenance ----
        if adaptive_result is not None:
            res["adaptive_baseline"] = adaptive_result
        else:
            # Store a minimal sentinel so the Excel export always has the key
            res["adaptive_baseline"] = None

    except Exception as e:
        warnings.warn(f"[{label}] pipeline failure: {e}")
        import traceback
        res["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return res


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(xlsx_path: str, cfg: PipelineConfig | None = None,
                 cluster_method: str = "combined",
                 verbose: bool = True) -> dict:
    cfg = cfg or PipelineConfig()

    if verbose:
        print(f"[1/9] Loading {xlsx_path}...")
    long_df, plant_meta = load_plant_data(xlsx_path, cfg=cfg)
    cfg = apply_plant_meta_to_cfg(cfg, plant_meta)
    if verbose:
        print(f"      plant: {cfg.site.name}  lat={cfg.site.lat:.3f}, lon={cfg.site.lon:.3f}")
        print(f"      commissioning: {cfg.plant.commissioning_date}")
        print(f"      tariff: {cfg.site.tariff:.1f} {cfg.site.currency}/kWh")
        print(f"      defaults applied: {plant_meta['azimuth_filled_rows']} az / "
              f"{plant_meta['tilt_filled_rows']} tilt rows")

    freq_min = float(plant_meta.get("freq_min", 5.0))

    if verbose:
        print("[2/9] Quality + curtailment flagging...")
    long_df = flag_data_quality(long_df, cfg)
    long_df = detect_curtailment(long_df, cfg)

    string_dfs  = split_into_string_dfs(long_df)
    string_meta = extract_string_meta(string_dfs)
    if verbose:
        print(f"      {len(string_dfs)} strings")

    if verbose:
        print("[3/9] Plate inference...")
    plate_inferred = infer_plate_params({k: v for k, v in string_dfs.items()},
                                        cfg.module, cfg)
    plate = plate_inferred.get("plate", cfg.module)

    if verbose:
        print("[4/9] Clustering...")
    clusters    = assign_clusters(string_dfs, string_meta, cluster_method)
    cluster_tbl = cluster_summary(clusters, string_meta)

    if verbose:
        print("[5/9] Degradation baseline...")
    ref_date = pd.to_datetime(long_df["ts"].max()).date()
    baseline_info = degradation_baseline(
        cfg.plant.commissioning_date, ref_date, plate.technology,
        override_rate=cfg.annual_degradation_pct,
        override_lid=cfg.lid_loss_pct, floor=cfg.baseline_floor)
    baseline = baseline_info["baseline"] if cfg.apply_degradation_correction else 1.0
    if verbose:
        print(f"      {explain_baseline(baseline_info)}")

    labels = sorted(string_dfs.keys())

    if verbose:
        print(f"[6/9] Per-string analysis (n_jobs={cfg.n_jobs}, "
              f"pvlib={'on' if has_pvlib() else 'off'}, "
              f"adaptive={'on' if cfg.adaptive_baseline_enabled else 'off'})...")

    # ---------------------------------------------------------------
    # Flat cluster-id map  {label: full_cluster_string}
    # ---------------------------------------------------------------
    cluster_ids: Dict[str, str] = {
        lbl: c["full_cluster"] for lbl, c in clusters.items()
    }

    # ---------------------------------------------------------------
    # SINGLE-PASS (adaptive disabled) — original behaviour
    # ---------------------------------------------------------------
    if not cfg.adaptive_baseline_enabled:
        def _job_single(label):
            return label, _process_one_string(
                label, string_dfs[label], string_meta[label],
                clusters[label], plate, cfg, baseline, freq_min,
                adaptive_clean_ref=None,
                adaptive_result=None,
            )

        if cfg.n_jobs == 1 or len(labels) == 1:
            per_string_list = [_job_single(l) for l in labels]
        else:
            try:
                from joblib import Parallel, delayed
                per_string_list = Parallel(
                    n_jobs=cfg.n_jobs, prefer="threads", verbose=0
                )(delayed(_job_single)(l) for l in labels)
            except Exception as e:
                warnings.warn(f"joblib failure ({e}), falling back to serial")
                per_string_list = [_job_single(l) for l in labels]
        per_string = dict(per_string_list)
        adaptive_results_map: Dict[str, Any] = {}

    # ---------------------------------------------------------------
    # TWO-PASS (adaptive enabled)
    # ---------------------------------------------------------------
    else:
        # ---- Pass 1: SDM + plate daily_df + wash_detect ----
        if verbose:
            print("      [Pass 1] plate-based daily metrics + wash detect...")
        pass1: Dict[str, dict] = {}
        for label in labels:
            pass1[label] = _pass1_string(
                label, string_dfs[label], plate, cfg, baseline, freq_min
            )

        # ---- Between passes: estimate adaptive baselines ----
        if verbose:
            print("      [Adaptive] estimating per-string clean baselines...")
        per_string_est: Dict[str, dict] = {}
        for label in labels:
            rain_events = pass1[label]["wash"].get("events_df", pd.DataFrame())
            per_string_est[label] = estimate_string_clean_baseline(
                pass1[label]["daily_df"], cfg, rain_events
            )

        # p95 map for cluster estimation
        per_string_p95: Dict[str, Optional[float]] = {
            lbl: (est.get("p95") if est.get("value") is not None else None)
            for lbl, est in per_string_est.items()
        }

        cluster_bl = estimate_cluster_clean_baseline(per_string_p95, cluster_ids)
        per_string_est = apply_cross_string_gate(
            per_string_est, cluster_bl, cluster_ids, cfg
        )

        # Resolve final reference per string
        adaptive_results_map: Dict[str, AdaptiveBaselineResult] = {}
        for label in labels:
            most_recent = pass1[label]["wash"].get("most_recent_event")
            if most_recent:
                daily_df_p1 = pass1[label]["daily_df"]
                if len(daily_df_p1) > 0:
                    ref_ts = pd.to_datetime(str(daily_df_p1["date"].max()))
                    evt_ts = pd.to_datetime(str(most_recent["event_date"]))
                    last_rain_days_ago = float(max((ref_ts - evt_ts).days, 0))
                else:
                    last_rain_days_ago = float(cfg.adaptive_window_days)
            else:
                last_rain_days_ago = float(cfg.adaptive_window_days)

            adaptive_results_map[label] = resolve_clean_baseline(
                label, per_string_est, cluster_bl, cluster_ids,
                float(baseline), last_rain_days_ago, cfg,
            )

        if verbose:
            layer_counts = {}
            for r in adaptive_results_map.values():
                layer_counts[r.layer] = layer_counts.get(r.layer, 0) + 1
            print(f"      Adaptive layers resolved: {layer_counts}")

        # ---- Pass 2: full analysis with adaptive ref ----
        if verbose:
            print("      [Pass 2] adaptive daily metrics + full analysis...")

        def _job_pass2(label):
            ar = adaptive_results_map[label]
            return label, _process_one_string(
                label, string_dfs[label], string_meta[label],
                clusters[label], plate, cfg, baseline, freq_min,
                adaptive_clean_ref=float(ar.value),
                adaptive_result=ar,
                sdm_precomputed=pass1[label]["sdm"],
                sdm_metrics_precomputed=pass1[label]["sdm_metrics"],
            )

        if cfg.n_jobs == 1 or len(labels) == 1:
            per_string_list = [_job_pass2(l) for l in labels]
        else:
            try:
                from joblib import Parallel, delayed
                per_string_list = Parallel(
                    n_jobs=cfg.n_jobs, prefer="threads", verbose=0
                )(delayed(_job_pass2)(l) for l in labels)
            except Exception as e:
                warnings.warn(f"joblib failure ({e}), falling back to serial")
                per_string_list = [_job_pass2(l) for l in labels]
        per_string = dict(per_string_list)

    # ---------------------------------------------------------------
    # Plant-level aggregation (unchanged)
    # ---------------------------------------------------------------
    if verbose:
        print("[7/9] Aggregating plant losses...")
    loss_dicts  = {k: v.get("losses", {}) for k, v in per_string.items()}
    plant_losses = aggregate_plant_losses(loss_dicts, cfg)

    verdicts = pd.Series([v.get("classification", {}).get("verdict", "Unknown")
                          for v in per_string.values()]).value_counts().to_dict()
    if verbose:
        print(f"[8/9] Verdicts: {verdicts}")
        print(f"[9/9] Total avoidable: {plant_losses['total_avoidable_kwh']:,.0f} kWh "
              f"({cfg.site.currency} {plant_losses['total_avoidable_pkr']:,.0f}) "
              f"over {plant_losses['period_days']} days")

    return dict(
        cfg=cfg, plant_meta=plant_meta, long_df=long_df,
        plate=plate, plate_inferred=plate_inferred,
        clusters=clusters, cluster_table=cluster_tbl,
        baseline_info=baseline_info, baseline=baseline,
        freq_min=freq_min, per_string=per_string,
        plant_losses=plant_losses, verdict_counts=verdicts,
        string_meta=string_meta,
        adaptive_results=adaptive_results_map,
    )
