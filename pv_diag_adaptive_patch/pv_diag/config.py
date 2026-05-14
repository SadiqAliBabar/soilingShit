"""Configuration dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

from .constants import (LAHORE_LAT, LAHORE_LON, LAHORE_TZ, LAHORE_ALT,
    DEFAULT_AZIMUTH_PK, DEFAULT_TILT_PK,
    DEFAULT_TARIFF_PKR_PER_KWH, DEFAULT_CURRENCY, TECH_DEFAULTS)


@dataclass
class ModuleConfig:
    voc_stc:    float = 49.5
    vmp_stc:    float = 41.7
    isc_stc:    float = 13.85
    imp_stc:    float = 12.95
    alpha_isc:  float = 0.00040
    beta_voc:   float = -0.00270
    gamma_pmp:  float = -0.00350
    n_modules:  int   = 22
    technology: str   = "mono-c-Si"
    cells_in_series: int = 144

    @property
    def voc_str_stc(self) -> float: return self.voc_stc * self.n_modules
    @property
    def vmp_str_stc(self) -> float: return self.vmp_stc * self.n_modules
    @property
    def pmp_str_stc(self) -> float: return self.vmp_str_stc * self.imp_stc


@dataclass
class SiteConfig:
    name:       str   = "Coca Cola Faisalabad"
    lat:        float = LAHORE_LAT
    lon:        float = LAHORE_LON
    tz:         str   = LAHORE_TZ
    altitude:   float = LAHORE_ALT
    albedo:     float = 0.20
    temp_model: str   = "sapm"
    racking:    str   = "open_rack_glass_glass"
    tariff:     float = DEFAULT_TARIFF_PKR_PER_KWH
    currency:   str   = DEFAULT_CURRENCY
    p_ac_max_kw:       float = 100.0
    n_strings_per_inv: int   = 6


@dataclass
class PlantConfig:
    commissioning_date: date = field(default_factory=lambda: date(2023, 1, 1))
    default_azimuth:   float = DEFAULT_AZIMUTH_PK
    default_tilt:      float = DEFAULT_TILT_PK
    lat: float = LAHORE_LAT
    lon: float = LAHORE_LON


@dataclass
class PipelineConfig:
    site:   SiteConfig    = field(default_factory=SiteConfig)
    module: ModuleConfig  = field(default_factory=ModuleConfig)
    plant:  PlantConfig   = field(default_factory=PlantConfig)

    min_peak_poa:    float = 700.0
    max_variability: float = 0.15
    min_midday_pts:  int   = 48
    midday_window:   tuple = (10.0, 14.0)

    clip_band_pct:         float = 0.015
    clip_min_dwell:        int   = 3
    expected_p_low_thresh: float = 0.85

    rain_threshold_mm:    float = 5.0
    soiling_recovery_pct: float = 0.7
    min_days_for_trend:   int   = 7
    soiling_loss_cap:     float = 0.50

    suff_good_avail_pct:    float = 60.0
    suff_good_curt_pct:     float = 15.0
    suff_limited_avail_pct: float = 35.0
    suff_limited_curt_pct:  float = 35.0
    suff_max_gap_days:      int   = 7
    suff_fault_pct_skip:    float = 60.0
    suff_avail_pct_skip:    float = 10.0

    transient_dip_threshold: float = 0.75
    transient_rolling_days:  int   = 7

    wash_step_thr:           float = 0.03
    wash_window_days:        int   = 2
    wash_full_recovery_pct:  float = 0.85
    wash_partial_recovery_pct: float = 0.50
    use_current_segment_verdict: bool = True

    apply_degradation_correction: bool = True
    annual_degradation_pct:       float = 0.005
    lid_loss_pct:                 float = 0.020
    baseline_floor:               float = 0.70

    confidence_z: float = 1.96
    n_jobs: int = 1

    # ------------------------------------------------------------------
    # Adaptive per-string clean baseline  (new in this patch)
    # ------------------------------------------------------------------
    # Master switch — set False to restore pre-patch behaviour exactly.
    adaptive_baseline_enabled: bool  = True

    # How many calendar days of history to look back when estimating P95.
    adaptive_window_days:      int   = 90

    # A string day is accepted into the P95 sample only when n_valid >= this.
    adaptive_min_midday_points: int  = 6

    # At least this many passing days must survive before we compute P95.
    adaptive_min_clean_days:   int   = 5

    # Gate A: if P95 < this we assume the string is chronically soiled and
    # reject it (use cluster or plate fallback instead).
    adaptive_min_p95:          float = 0.92

    # Gate B: when there are zero rain/wash events in the window, require a
    # higher P95 floor (more conservative because we cannot anchor to a
    # post-wash recovery).
    adaptive_no_rain_floor:    float = 0.96

    # Gate C: reject a string whose P95 is more than this far below its
    # cluster median (prevents a dirty string from dragging down others).
    adaptive_cluster_gate:     float = 0.05

    # If no rain event has occurred in more than this many days, start
    # blending the cluster/string adaptive ref with the plate baseline
    # (dry-season hedge).
    dry_season_threshold:      int   = 30


    # Plotting toggles
    plot_soiling_dashboard:   bool = True
    plot_iv_diagnostics:      bool = False
    plot_string_data_quality: bool = False
    plot_plant_data_quality:  bool = True

def technology_degradation(tech: str):
    rec = TECH_DEFAULTS.get(tech, TECH_DEFAULTS["mono-c-Si"])
    return float(rec["annual_degradation"]), float(rec["lid_loss"])
