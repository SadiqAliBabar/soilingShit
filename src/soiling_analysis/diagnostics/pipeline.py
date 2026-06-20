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

from .config import PipelineConfig, ModuleConfig
from .constants import QUALITY_FLAGS
from .ingestion import (load_plant_data, split_into_string_dfs,
                        extract_string_meta, apply_plant_meta_to_cfg)
from .quality import flag_data_quality
from .curtailment import (detect_curtailment, curtailment_summary,
                          quantify_curtailment_loss)
from .sufficiency import compute_data_availability, decide_sufficiency
# plate.infer_plate_params removed from spine in Batch 8 (manufacturer specs used directly)
from .clustering import assign_clusters, cluster_summary, build_peer_groups
from .degradation import degradation_baseline, explain_baseline
from .sdm import fit_single_diode, iv_metrics_at_stc, has_pvlib
from .daily import compute_daily_metrics
from .wash_detect import detect_wash_events
from .soiling import extract_soiling_trend, extract_soiling_current_segment
from .transient import detect_transient_events
from .classification import classify_string
from .losses import (quantify_string_losses, aggregate_plant_losses,
                     cleaning_economics, aggregate_plant_economics)
from .orientation import expected_asymmetry
from .adaptive_baseline import (
    estimate_string_clean_baseline,
    estimate_cluster_clean_baseline,
    apply_cross_string_gate,
    apply_peer_cross_check,
    resolve_clean_baseline,
    AdaptiveBaselineResult,
)


# ---------------------------------------------------------------------------
# Quality-day filter for SDM fitting
# ---------------------------------------------------------------------------

_CURT_BITS = (QUALITY_FLAGS["CURT_STATE"]
              | QUALITY_FLAGS["CURT_STATISTICAL"]
              | QUALITY_FLAGS["CURT_EXPORT_LIMIT"])


def _select_quality_days(df: pd.DataFrame, cfg: PipelineConfig) -> set:
    """Return the set of calendar dates that pass all quality gates.

    Batch 6: when cfg.clearsky_quality_enabled=True, the fixed 600 W/m²
    midday-peak floor and POA-CV check are replaced by a Kc clearness +
    rolling-CV stability test.  This accepts clear winter days (POA < 600
    but Kc ≈ 1) and rejects cloudy summer days (POA > 600 but Kc = 0.6).
    The curtailment-fraction and rainfall gates are unchanged.
    """
    ts   = pd.to_datetime(df["ts"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    tmp  = df.assign(
        __hour=hour,
        __date=ts.dt.date,
        __midday=(hour >= 11.0) & (hour <= 13.0),
    )

    # Batch 6: Kc-quality day set (replaces fixed 600 W/m² floor)
    _kc_good_dates: Optional[set] = None
    if cfg.clearsky_quality_enabled:
        try:
            from .clearsky_quality import day_kc_quality_dates
            _kc_good_dates = day_kc_quality_dates(
                df, cfg,
                lat=cfg.site.lat, lon=cfg.site.lon,
                azimuth=cfg.plant.default_azimuth,
                tilt=cfg.plant.default_tilt,
                altitude=float(getattr(cfg.site, "altitude", 217.0)),
            )
        except Exception as _kc_exc:
            warnings.warn(f"[B6] Kc quality-day filter failed ({_kc_exc}); "
                          f"falling back to 600 W/m² floor")
            _kc_good_dates = None

    good_days: set = set()
    for day, grp in tmp.groupby("__date"):
        mid = grp[grp["__midday"]]
        if len(mid) == 0:
            continue
        poa = pd.to_numeric(mid["POA"], errors="coerce").dropna()
        if len(poa) == 0:
            continue

        if _kc_good_dates is not None:
            # Kc path: replaced fixed 600 W/m² + POA-CV by Kc clearness+stability
            if day not in _kc_good_dates:
                continue
        else:
            # Legacy path: fixed 600 W/m² peak floor + POA midday-CV check
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


def _select_clean_window_days(
    df: pd.DataFrame,
    quality_days: set,
    cfg: PipelineConfig,
    wash_events_df: Optional[pd.DataFrame] = None,
) -> tuple:
    """Refine quality_days to a clean+recent window for SDM fitting.

    Strategy (per playbook):
      1. Recovery-anchored + recent: days 1-7 after a confirmed wash event
         that also fall within sdm_recent_days.  These are "unsoiled + current."
      2. Recent-clean fallback (C5=FALSE / no rain): most recent sdm_recent_days
         quality days, flagged "recent_clean_no_anchor".
      3. Full quality days fallback: when neither yields >= adaptive_min_clean_days.

    Returns (filtered_days: set, source_label: str).
    """
    if not cfg.sdm_clean_window_enabled or len(quality_days) == 0:
        return quality_days, "all_quality_days"

    ts_dates  = pd.to_datetime(df["ts"]).dt.date
    max_date  = ts_dates.max()
    cutoff    = max_date - pd.Timedelta(days=cfg.sdm_recent_days)

    recent_quality = {d for d in quality_days if d >= cutoff}

    # Attempt 1: recovery-anchored days (shortly after a confirmed cleaning)
    if (wash_events_df is not None
            and len(wash_events_df) > 0
            and "event_date" in wash_events_df.columns):
        post_wash: set = set()
        for _, ev in wash_events_df.iterrows():
            ev_date = pd.to_datetime(ev["event_date"]).date()
            for d in quality_days:
                delta = (d - ev_date).days
                if 1 <= delta <= 7 and d >= cutoff:
                    post_wash.add(d)
        if len(post_wash) >= cfg.adaptive_min_clean_days:
            return post_wash, "recovery_anchored_recent"

    # Attempt 2: most recent quality days (no anchors — C5=FALSE fallback)
    if len(recent_quality) >= cfg.adaptive_min_clean_days:
        return recent_quality, "recent_clean_no_anchor"

    # Fallback: all quality days (monsoon/smog or short history)
    return quality_days, "all_quality_fallback"


# ---------------------------------------------------------------------------
# SDM fit helper (shared across passes)
# ---------------------------------------------------------------------------

def _fit_sdm(label: str, df: pd.DataFrame, plate, cfg: PipelineConfig,
             wash_events_df: Optional[pd.DataFrame] = None):
    """Run SDM quality-day filter and fit; return (sdm, sdm_metrics).

    Batch 6 additions:
    - _select_quality_days now uses Kc clearness+stability instead of the
      fixed 600 W/m² floor (when cfg.clearsky_quality_enabled=True).
    - When wash_events_df is provided and cfg.sdm_clean_window_enabled=True,
      further refines the training set to clean+recent windows.
    - The sdm result dict carries sdm_window_source for provenance.
    """
    sdm_window_source = "all_quality_days"
    try:
        good_days = _select_quality_days(df, cfg)

        # Batch 6: refine to clean+recent window when wash events are available
        if cfg.sdm_clean_window_enabled:
            good_days, sdm_window_source = _select_clean_window_days(
                df, good_days, cfg, wash_events_df
            )

        ts_dates   = pd.to_datetime(df["ts"]).dt.date
        df_for_sdm = df[ts_dates.isin(good_days)]

        # Safety net: widen fallback so a Kc-starved or clean-window-starved
        # set does not send an under-fitted SDM downstream.
        if len(df_for_sdm) < 100:
            warnings.warn(
                f"[{label}] SDM window ({sdm_window_source}): only "
                f"{len(df_for_sdm)} rows survive ({len(good_days)} good days); "
                f"falling back to full df"
            )
            df_for_sdm = df
            sdm_window_source = "full_df_fallback"
    except Exception as _qd_exc:
        warnings.warn(
            f"[{label}] quality-day filter failed ({_qd_exc}); "
            f"using full df for SDM fit"
        )
        df_for_sdm = df
        sdm_window_source = "full_df_fallback"

    try:
        sdm = fit_single_diode(df_for_sdm, plate, cfg)
        if sdm:
            sdm["sdm_window_source"] = sdm_window_source
    except Exception as e:
        sdm = dict(success=False, reason=f"sdm_exception:{type(e).__name__}",
                   sdm_window_source=sdm_window_source)
    sdm_metrics = (iv_metrics_at_stc(sdm, plate)
                   if sdm and sdm.get("success") else None)
    return sdm, sdm_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_string_plate(df: pd.DataFrame, base_plate: ModuleConfig) -> ModuleConfig:
    """Derive n_modules dynamically from pv_capacity column if present."""
    if "pv_capacity" not in df.columns:
        return base_plate
    try:
        # Get capacity (assuming it's filled consistently or taking first)
        cap_val = pd.to_numeric(df["pv_capacity"], errors="coerce").dropna()
        if cap_val.empty:
            return base_plate
        cap_kw = float(cap_val.iloc[0])
        if not np.isfinite(cap_kw) or cap_kw <= 0:
            return base_plate

        # Module capacity in kW from datasheet properties
        mod_cap_kw = (base_plate.vmp_stc * base_plate.imp_stc) / 1000.0
        n_modules = int(round(cap_kw / mod_cap_kw))
        
        if 5 <= n_modules <= 100: # Sanity check
            new_plate = ModuleConfig(**base_plate.__dict__)
            new_plate.n_modules = n_modules
            return new_plate
    except Exception:
        pass
    return base_plate


# ---------------------------------------------------------------------------
# Pass-1 light processing (daily_df + wash only, no full analysis)
# ---------------------------------------------------------------------------

def _pass1_string(label: str, df: pd.DataFrame, plate, cfg: PipelineConfig,
                  baseline: float, freq_min: float,
                  azimuth: Optional[float] = None,
                  tilt: Optional[float] = None):
    """SDM fit + plate-based daily_df + wash_detect.  Returns a compact dict."""
    plate = _get_string_plate(df, plate)
    sdm, sdm_metrics = _fit_sdm(label, df, plate, cfg)
    try:
        daily_df = compute_daily_metrics(
            df, plate, sdm, cfg, baseline, freq_min,
            adaptive_clean_ref=None,
            azimuth=azimuth, tilt=tilt,
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
    age_baseline: float = 1.0,
    age_source: str = "plant_default",
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
    plate = _get_string_plate(df, plate)
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
        # SCHEMA-DEP C1 (Batch 5): per-string azimuth/tilt wired into IAM.
        # meta_one carries per-string orientation surfaced by loader.py (Batch 1).
        daily_df = compute_daily_metrics(
            df, plate, sdm, cfg, baseline, freq_min,
            adaptive_clean_ref=adaptive_clean_ref,
            azimuth=meta_one.get("azimuth"),
            tilt=meta_one.get("tilt"),
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

        # ---- Transients (detect first — feeds transient_dates into wash + soiling) ----
        transients = detect_transient_events(daily_df, cfg)
        res["transients"] = transients

        # Batch 8: build transient date set when prefilter is enabled
        _transient_dates: set = set()
        if getattr(cfg, "transient_prefilter_enabled", True) and len(transients) > 0:
            _transient_dates = set(pd.to_datetime(transients["date"]).dt.date)

        # ---- Wash detect (transient feedback: suppress steps on transient days) ----
        wash = detect_wash_events(daily_df, cfg, transient_dates=_transient_dates)
        res["wash"] = wash

        # ---- Soiling (transient pre-filter: exclude transient days from fit) ----
        res["soiling_full"]    = extract_soiling_trend(
            daily_df, wash, cfg, transient_dates=_transient_dates)
        res["soiling_current"] = extract_soiling_current_segment(
            daily_df, wash, cfg, transient_dates=_transient_dates)

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
            age_baseline=age_baseline,
        )
        res["classification"] = clx
        res["age_baseline"] = age_baseline
        res["age_source"]   = age_source

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

        # ---- Cleaning economics (Batch 9) ----
        _pv_cap_kw = 0.0
        if "pv_capacity" in df.columns:
            _cap = pd.to_numeric(df["pv_capacity"], errors="coerce").dropna()
            if len(_cap) > 0:
                _pv_cap_kw = float(_cap.iloc[0])
        _last_wash = None
        _wash_ev = res.get("wash", {}).get("most_recent_event")
        if _wash_ev:
            _last_wash = _wash_ev.get("event_date")
        _ref_date = pd.to_datetime(df["ts"].max()).date() if len(df) > 0 else None
        res["cleaning_economics"] = cleaning_economics(
            label,
            res.get("soiling_full", {}),
            daily_df,
            cfg,
            pv_capacity_kw=_pv_cap_kw,
            last_wash_date=_last_wash,
            ref_date=_ref_date,
        )

        res["meta"]    = meta_one
        res["cluster"] = cluster_one

        # ---- Adaptive baseline provenance ----
        if adaptive_result is not None:
            res["adaptive_baseline"] = adaptive_result
            res["peer_ladder_level"] = adaptive_result.peer_ladder_level
        else:
            # Store sentinels so the Excel export always has both keys.
            res["adaptive_baseline"] = None
            res["peer_ladder_level"] = None

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
    return run_pipeline_from_frame(long_df, plant_meta, cfg, cluster_method, verbose)


def run_pipeline_from_frame(
    long_df: pd.DataFrame,
    plant_meta: dict,
    cfg: PipelineConfig | None = None,
    cluster_method: str = "combined",
    verbose: bool = True,
) -> dict:
    cfg = cfg or PipelineConfig()

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
    long_df = detect_curtailment(
        long_df, cfg, freq_min=freq_min,
        inverter_specs=plant_meta.get("inverter_specs"),
        inverter_ac_power=plant_meta.get("inverter_ac_power"),
    )

    string_dfs  = split_into_string_dfs(long_df)
    string_meta = extract_string_meta(string_dfs)
    if verbose:
        print(f"      {len(string_dfs)} strings")

    if verbose:
        print("[3/9] Plate params (manufacturer specs from workbook — plate.py deprecated)...")
    # Batch 8: infer_plate_params() removed from spine; use manufacturer specs
    # from loader/specs.py directly.  Per-string n_modules scaling is preserved
    # in _get_string_plate() via the pv_capacity column.
    plate = cfg.module
    plate_inferred = {"plate": plate,
                      "notes": "Batch 8: plate.py deprecated; using manufacturer specs",
                      "n_strings_used": 0}

    if verbose:
        print("[4/9] Clustering...")
    clusters    = assign_clusters(string_dfs, string_meta, cluster_method)
    cluster_tbl = cluster_summary(clusters, string_meta)

    # SCHEMA-DEP C1 (Batch 5): POA transposition — compute one transposed POA
    # per orientation cluster and add POA_transposed to each string's DataFrame.
    # Co-oriented strings share one result; identical WS/string orientations
    # return immediately without computation (fast exit in transpose_poa).
    if cfg.poa_transposition_enabled:
        try:
            from .transposition import compute_cluster_poa_map
            # Build cluster_key → (tilt, azimuth) from string_meta
            # clusters[label] is a dict; use full_cluster string as the hashable key.
            _cluster_orientations: Dict[str, tuple] = {}
            for _lbl, _cdict in clusters.items():
                _ckey = _cdict["full_cluster"] if isinstance(_cdict, dict) else _cdict
                if _ckey not in _cluster_orientations:
                    _sm = string_meta.get(_lbl, {})
                    _cluster_orientations[_ckey] = (
                        float(_sm.get("tilt",    cfg.plant.default_tilt)),
                        float(_sm.get("azimuth", cfg.plant.default_azimuth)),
                    )
            # Get unique timestamp→WS-POA mapping from long_df
            _ts_poa = (
                long_df.sort_values("ts")
                .groupby("ts", sort=True)["POA"]
                .first()
                .reset_index()
            )
            _ts_idx  = pd.DatetimeIndex(pd.to_datetime(_ts_poa["ts"]))
            _poa_ws  = pd.to_numeric(_ts_poa["POA"], errors="coerce").fillna(0.0).values

            _poa_cluster_map = compute_cluster_poa_map(
                _ts_idx, _poa_ws,
                float(cfg.plant.default_tilt),
                float(cfg.plant.default_azimuth),
                _cluster_orientations,
                cfg.site.lat, cfg.site.lon,
                cfg.site.albedo,
                cfg.transposition_model,
            )
            # Build per-cluster timestamp→POA_transposed lookup Series
            _poa_ts_series: Dict[str, Any] = {
                ckey: pd.Series(arr, index=_ts_poa["ts"].values)
                for ckey, arr in _poa_cluster_map.items()
            }
            # Write POA_transposed into each string df (keyed by timestamp)
            for _lbl, _sdf in string_dfs.items():
                _cdict  = clusters[_lbl]
                _ckey   = _cdict["full_cluster"] if isinstance(_cdict, dict) else _cdict
                _ts_ser = _poa_ts_series.get(_ckey)
                if _ts_ser is not None:
                    _mapped = pd.to_datetime(_sdf["ts"]).map(_ts_ser)
                    # Fall back to measured POA where the map produced NaN
                    string_dfs[_lbl]["POA_transposed"] = _mapped.where(
                        _mapped.notna(),
                        pd.to_numeric(_sdf["POA"], errors="coerce"),
                    )
            if verbose:
                _n_orient = len(_cluster_orientations)
                print(f"      [B5] POA transposition: {_n_orient} orientation "
                      f"cluster(s), model={cfg.transposition_model!r}")
        except Exception as _trans_exc:
            warnings.warn(
                f"[B5] POA transposition failed ({_trans_exc}); "
                f"all strings will use measured WS POA."
            )

    if verbose:
        print("[5/9] Degradation baseline (per-string)...")
    ref_date = pd.to_datetime(long_df["ts"].max()).date()

    labels = sorted(string_dfs.keys())

    # Plant-level baseline — used for logging and the top-level result dict.
    baseline_info = degradation_baseline(
        cfg.plant.commissioning_date, ref_date, plate.technology,
        override_rate=cfg.annual_degradation_pct,
        override_lid=cfg.lid_loss_pct, floor=cfg.baseline_floor)
    # Scalar kept for any legacy single-pass reference (used in return dict).
    baseline = baseline_info["baseline"] if cfg.apply_degradation_correction else 1.0

    # Per-string baselines from each string's commissioning_date (Batch 4).
    # Falls back to plant commissioning date per [C2] when string_specs is absent.
    _str_specs_b4 = plant_meta.get("string_specs", {})
    per_string_baseline_info: Dict[str, dict] = {}
    for _lbl in labels:
        _sp = _str_specs_b4.get(_lbl, {})
        _comm = _sp.get("commissioning_date") or cfg.plant.commissioning_date
        _age_src = _sp.get("age_source", "plant_default")
        _bl_info = degradation_baseline(
            _comm, ref_date, plate.technology,
            override_rate=cfg.annual_degradation_pct,
            override_lid=cfg.lid_loss_pct, floor=cfg.baseline_floor,
        )
        _bl_info["age_source"] = _age_src
        per_string_baseline_info[_lbl] = _bl_info

    # Build a convenience float dict once — avoids repeated dict lookups in hot loops.
    _str_baseline: Dict[str, float] = {
        lbl: (info["baseline"] if cfg.apply_degradation_correction else 1.0)
        for lbl, info in per_string_baseline_info.items()
    }

    if verbose:
        print(f"      {explain_baseline(baseline_info)}")
        _unique_bl = set(round(v, 4) for v in _str_baseline.values())
        print(f"      per-string: {len(_unique_bl)} unique baseline(s) "
              f"(range {min(_unique_bl):.4f}–{max(_unique_bl):.4f})")

    if verbose:
        print(f"[6/9] Per-string analysis (n_jobs={cfg.n_jobs}, "
              f"pvlib={'on' if has_pvlib() else 'off'}, "
              f"adaptive={'on' if cfg.adaptive_baseline_enabled else 'off'})...")

    # ---------------------------------------------------------------
    # SINGLE-PASS (adaptive disabled) — original behaviour
    # ---------------------------------------------------------------
    if not cfg.adaptive_baseline_enabled:
        def _job_single(label):
            _age_bl  = _str_baseline.get(label, baseline)
            _age_src = per_string_baseline_info.get(label, baseline_info).get("age_source", "plant_default")
            return label, _process_one_string(
                label, string_dfs[label], string_meta[label],
                clusters[label], plate, cfg, _age_bl, freq_min,
                adaptive_clean_ref=None,
                adaptive_result=None,
                age_baseline=_age_bl,
                age_source=_age_src,
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
                label, string_dfs[label], plate, cfg,
                _str_baseline.get(label, baseline), freq_min,
                azimuth=string_meta[label].get("azimuth"),
                tilt=string_meta[label].get("tilt"),
            )

        # ---- Batch 6: refit SDMs on clean+recent windows ----
        # Now that Pass 1 has produced wash events, we know which days are
        # post-wash (clean panel fingerprint).  Refit the SDM on those
        # recovery-anchored + recent windows before the adaptive baseline step.
        if cfg.sdm_clean_window_enabled:
            if verbose:
                print("      [B6] Refitting SDMs on clean+recent windows...")
            for label in labels:
                wash_ev = pass1[label]["wash"].get("events_df")
                sdm_r, sdm_met_r = _fit_sdm(
                    label, string_dfs[label], plate, cfg,
                    wash_events_df=wash_ev,
                )
                if sdm_r and sdm_r.get("success"):
                    pass1[label]["sdm"]         = sdm_r
                    pass1[label]["sdm_metrics"] = sdm_met_r
                # If refit failed, keep the original Pass-1 SDM

        # ---- Between passes: peer groups + adaptive baseline ----
        if verbose:
            print("      [Adaptive] building peer groups...")
        peer_groups_map = build_peer_groups(string_meta, string_dfs, cfg)

        if verbose:
            print("      [Adaptive] estimating per-string clean baselines...")
        per_string_est: Dict[str, dict] = {}
        for label in labels:
            rain_events = pass1[label]["wash"].get("events_df", pd.DataFrame())
            per_string_est[label] = estimate_string_clean_baseline(
                pass1[label]["daily_df"], cfg, rain_events,
                age_baseline=_str_baseline.get(label, baseline),
            )

        # p95 map for peer-group estimation
        per_string_p95: Dict[str, Optional[float]] = {
            lbl: (est.get("p95") if est.get("value") is not None else None)
            for lbl, est in per_string_est.items()
        }

        cluster_bl = estimate_cluster_clean_baseline(per_string_p95, peer_groups_map)
        per_string_est = apply_cross_string_gate(
            per_string_est, cluster_bl, peer_groups_map, cfg,
            per_string_age_baselines=_str_baseline,
        )

        # Peer cross-check: runs after ALL strings complete Layer-1 so every
        # string's reference_method is available.  Substitutes clean_ref with
        # the recovery-anchored peer median for strings that are far below their
        # peers (likely faulty, not merely soiled).
        per_string_est = apply_peer_cross_check(per_string_est, peer_groups_map, cfg)

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

            _age_bl  = _str_baseline.get(label, baseline)
            _age_src = per_string_baseline_info.get(label, baseline_info).get("age_source", "plant_default")
            adaptive_results_map[label] = resolve_clean_baseline(
                label, per_string_est, cluster_bl, peer_groups_map,
                float(_age_bl), last_rain_days_ago, cfg,
                age_source=_age_src,
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
            ar       = adaptive_results_map[label]
            _age_bl  = _str_baseline.get(label, baseline)
            _age_src = per_string_baseline_info.get(label, baseline_info).get("age_source", "plant_default")
            return label, _process_one_string(
                label, string_dfs[label], string_meta[label],
                clusters[label], plate, cfg, _age_bl, freq_min,
                adaptive_clean_ref=float(ar.value),
                adaptive_result=ar,
                sdm_precomputed=pass1[label]["sdm"],
                sdm_metrics_precomputed=pass1[label]["sdm_metrics"],
                age_baseline=_age_bl,
                age_source=_age_src,
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
    # Plant-level aggregation (Batch 9: add economics)
    # ---------------------------------------------------------------
    if verbose:
        print("[7/9] Aggregating plant losses + cleaning economics...")
    loss_dicts  = {k: v.get("losses", {}) for k, v in per_string.items()}
    plant_losses = aggregate_plant_losses(loss_dicts, cfg)

    econ_dicts = {k: v.get("cleaning_economics", {}) for k, v in per_string.items()}
    plant_economics = aggregate_plant_economics(econ_dicts)

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
        per_string_baseline_info=per_string_baseline_info,
        freq_min=freq_min, per_string=per_string,
        plant_losses=plant_losses, plant_economics=plant_economics,
        verdict_counts=verdicts,
        string_meta=string_meta,
        adaptive_results=adaptive_results_map,
    )
