"""Configuration dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

from .constants import (LAHORE_LAT, LAHORE_LON, LAHORE_TZ, LAHORE_ALT,
    DEFAULT_AZIMUTH_PK, DEFAULT_TILT_PK,
    DEFAULT_TARIFF_PKR_PER_KWH, DEFAULT_CURRENCY, TECH_DEFAULTS)


@dataclass
class ModuleConfig:
    voc_stc:    float = 51.80    # From JA Solar 585W spec
    vmp_stc:    float = 43.24    # From JA Solar 585W spec
    isc_stc:    float = 14.29    # From JA Solar 585W spec
    imp_stc:    float = 13.53    # From JA Solar 585W spec
    alpha_isc:  float = 0.00046  # +0.046%/C
    beta_voc:   float = -0.00260 # -0.260%/C
    gamma_pmp:  float = -0.00300 # -0.300%/C
    n_modules:  int   = 22       # PLEASE CONFIRM (depends on plant wiring)
    technology: str   = "mono-c-Si" # n-type bifacial
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

    clip_band_pct:   float = 0.05
    clip_min_dwell:  int   = 3
    suppression_poa_threshold:    float = 400.0  # W/m² — sun must be this bright
    suppression_power_ratio:      float = 0.20   # actual < 20% of expected = suppressed
    suppression_min_dwell:        int   = 2      # must persist for at least 2 rows (10 min)

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
    iam_b0: float = 0.05  # ASHRAE IAM model incidence angle modifier coefficient

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

    # Peer-group ladder: minimum group size (self + peers) to be considered valid.
    peer_min_members:          int   = 3
    # Peer-group ladder level 1: max fractional DC-capacity difference allowed.
    peer_capacity_tolerance:   float = 0.10

    # Days post-wash to sample the clean plateau for recovery-anchored baseline.
    recovery_plateau_days:     int   = 4
    # Peer median must exceed string's own ref by this margin (pp) to trigger substitution.
    peer_disagreement_margin:  float = 0.04

    # ------------------------------------------------------------------
    # Slope significance gate  (new in Prompt 4)
    # ------------------------------------------------------------------
    # Minimum absolute slope (NCI/day) to be operationally meaningful (~11%/year).
    soiling_slope_significance: float = 0.0003
    # Minimum slope-to-noise ratio (|slope| / se) to trust the trend direction.
    soiling_slope_snr:          float = 2.0

    # ------------------------------------------------------------------
    # Voltage-rise soft curtailment detector  (new in Prompt 5)
    # ------------------------------------------------------------------
    # These thresholds were set for 5-minute resolution data on a typical string
    # inverter.  For different resolutions or inverter types, curt_vr_vdc_rise_rate
    # and curt_vr_window_min may need tuning.  Calibration approach: manually
    # identify 3-5 known voltage-rise events in your data (look for midday Vdc
    # spikes coinciding with grid voltage measurements if available), then verify
    # the detector flags those rows and not the surrounding clean rows.
    curt_vr_min_poa: float = 200.0          # minimum POA (W/m2) to apply detection
    curt_vr_vdc_rise_rate: float = 0.5      # minimum Vdc rise rate (V/min)
    curt_vr_pdc_flat_threshold: float = 5.0 # max Pdc rise rate (W/min); tolerates noise
    curt_vr_poa_falling_threshold: float = -2.0  # min POA rate (W/m2/min); excludes clouds
    curt_vr_vdc_min_fraction: float = 0.75  # min Vdc / Voc_str_stc; excludes startup
    curt_vr_window_min: float = 15.0        # rolling window width in minutes

    # ------------------------------------------------------------------
    # Confidence score weights and thresholds  (new in Prompt 7)
    # ------------------------------------------------------------------
    conf_weight_data_quantity: float = 0.20  # D1: number of valid days in segment
    conf_weight_baseline: float = 0.25       # D2: baseline layer, method, peer quality
    conf_weight_trend: float = 0.25          # D3: slope significance and SNR
    conf_weight_agreement: float = 0.15      # D4: cross-string baseline disagreement
    conf_weight_sufficiency: float = 0.15    # D5: data availability sufficiency verdict
    conf_high_threshold: float = 0.72        # minimum score for "high" confidence label
    conf_medium_threshold: float = 0.45      # minimum score for "medium" confidence label

    # ------------------------------------------------------------------
    # Multi-day distributed wash/rain recovery detector  (new in Prompt 6)
    # ------------------------------------------------------------------
    wash_rain_lookback_days: int = 3
        # days before a single-day event to search for rain (drying-delay correction)
    wash_multiday_max_days: int = 4
        # maximum window width in days for distributed recovery search;
        # windows wider than this are unlikely to be a single event
    wash_multiday_step_thr: float = 0.03
        # minimum cumulative NCI rise over the window to trigger;
        # same total threshold as single-day wash_step_thr
    wash_multiday_rain_lookback_days: int = 3
        # days before window start to search for rain association;
        # captures post-rain drying delay patterns
    wash_multiday_monotone_tolerance: float = -0.005
        # how much a single day within the window is allowed to dip
        # before the monotone-non-declining condition fails;
        # small negative value tolerates measurement noise


    def __post_init__(self):
        total = (self.conf_weight_data_quantity + self.conf_weight_baseline
                 + self.conf_weight_trend + self.conf_weight_agreement
                 + self.conf_weight_sufficiency)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {total:.3f}"
            )


def technology_degradation(tech: str):
    rec = TECH_DEFAULTS.get(tech, TECH_DEFAULTS["mono-c-Si"])
    return float(rec["annual_degradation"]), float(rec["lid_loss"])
