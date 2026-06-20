"""soiling_test.py — Batch 8 pytest for soiling.py robust regression + transient prefilter.

Tests:
  (a) A single injected high-leverage dip tugs trimmed-OLS slope but not Theil-Sen.
  (b) A transient day is excluded from the fit and does NOT split a segment boundary.
  (c) Per-string n_modules is still correct after plate.py deprecation.
  (d) Zero soiling → cleaning_economics returns "No cleaning recommended".
  (e) Payback consistency: wash_cost / daily_loss == payback_days (within 1%).

Run from repo root:
    pytest src/soiling_analysis/soiling_old_7_prompt/pv_diag/soiling_test.py -v
"""
from __future__ import annotations
import pathlib, sys

_this = pathlib.Path(__file__).resolve().parent
for _p in (_this.parent, _this.parent.parent):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np
import pandas as pd
import pytest

from soiling_analysis.diagnostics.config import PipelineConfig, ModuleConfig
from soiling_analysis.diagnostics.soiling import _trimmed_lr, _robust_lr, extract_soiling_trend, _empty_soiling
from soiling_analysis.diagnostics.losses import cleaning_economics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(n_days: int = 60, slope_per_day: float = -0.001,
                   dip_day: int | None = None, dip_magnitude: float = 0.30) -> pd.DataFrame:
    """Synthetic daily_df with a linear NCI trend and an optional single-day dip."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    nci = 0.95 + slope_per_day * np.arange(n_days)
    if dip_day is not None and 0 <= dip_day < n_days:
        nci[dip_day] -= dip_magnitude
    return pd.DataFrame({
        "date": dates,
        "NCI_noon": nci,
        "NCI_corrected_noon": nci,
        "is_present": True,
        "n_valid": 6,
    })


def _empty_wash():
    return {"events_df": pd.DataFrame(), "current_segment_df": pd.DataFrame(),
            "most_recent_event": None}


# ---------------------------------------------------------------------------
# (a) Theil-Sen ignores high-leverage outlier that pulls trimmed-OLS
# ---------------------------------------------------------------------------

class TestRobustRegressionOutlier:
    """A single-day dip (outlier) moves trimmed-OLS slope but not Theil-Sen."""

    def test_outlier_moves_ols_not_theilsen(self):
        # Clean trend: slope = -0.002/day
        n = 60
        x = np.arange(n, dtype=float)
        y = 0.95 - 0.002 * x

        clean_ols = _trimmed_lr(x.copy(), y.copy())
        clean_ts  = _robust_lr(x.copy(), y.copy(), method="theilsen")

        # Inject a massive dip at day 30 (high-leverage outlier)
        y_dirty = y.copy()
        y_dirty[30] -= 0.30
        dirty_ols = _trimmed_lr(x.copy(), y_dirty)
        dirty_ts  = _robust_lr(x.copy(), y_dirty, method="theilsen")

        # Trimmed OLS slope shifts noticeably
        ols_shift = abs(dirty_ols["slope"] - clean_ols["slope"])
        # Theil-Sen slope barely changes
        ts_shift  = abs(dirty_ts["slope"] - clean_ts["slope"])

        assert ols_shift > ts_shift, (
            f"Expected Theil-Sen (shift={ts_shift:.6f}) more stable than "
            f"trimmed OLS (shift={ols_shift:.6f}) under outlier"
        )
        # Theil-Sen should stay within 20% of the clean slope
        assert ts_shift < abs(clean_ts["slope"]) * 0.20, (
            f"Theil-Sen shifted too much ({ts_shift:.6f}) for known outlier"
        )


# ---------------------------------------------------------------------------
# (b) Transient day excluded from fit; does not split segment
# ---------------------------------------------------------------------------

class TestTransientPrefilter:
    """Transient days are excluded from soiling regression and do not create segments."""

    def _cfg(self, regressor="theilsen", prefilter=True):
        c = PipelineConfig()
        c.robust_soiling_regression = regressor
        c.transient_prefilter_enabled = prefilter
        c.daily_grid_enabled = False  # keep density gate off for this test
        return c

    def test_transient_excluded_from_fit(self):
        # 60-day trend; day 30 is a transient dip
        df = _make_daily_df(60, slope_per_day=-0.001, dip_day=30, dip_magnitude=0.15)
        transient_date = df["date"].iloc[30].date()

        cfg_with    = self._cfg(prefilter=True)
        cfg_without = self._cfg(prefilter=False)

        res_with    = extract_soiling_trend(df, _empty_wash(), cfg_with,
                                            transient_dates={transient_date})
        res_without = extract_soiling_trend(df, _empty_wash(), cfg_without,
                                            transient_dates=None)

        # Both should detect soiling; with-prefilter should have cleaner (less noisy) slope
        assert np.isfinite(res_with["srr_pct_per_day"]), "prefilter: slope should be finite"
        assert np.isfinite(res_without["srr_pct_per_day"]), "no-prefilter: slope should be finite"
        assert res_with.get("transient_days_excluded", 0) == 1

    def test_transient_does_not_split_segment(self):
        # Wash event on day 20, transient on day 35
        # The transient is NOT a segment boundary; we should still have 2 segments (split on day 20)
        df = _make_daily_df(60, slope_per_day=-0.001)
        transient_date = df["date"].iloc[35].date()
        wash_date      = df["date"].iloc[20].date()

        wash_result = {"events_df": pd.DataFrame([{
            "event_date": wash_date, "cause": "Manual wash (suspected)",
            "delta_nci": 0.05, "pre_event_low": 0.90, "post_event_high": 0.95,
            "baseline_clean": 0.95, "completeness": 1.0, "recovery_class": "Full recovery",
            "rain_mm_today": 0.0, "detection_method": "single_day",
        }]), "current_segment_df": pd.DataFrame(), "most_recent_event": None}

        cfg = self._cfg(prefilter=True)
        res = extract_soiling_trend(df, wash_result, cfg,
                                    transient_dates={transient_date})
        # Should have 2 segments (split by wash at day 20), NOT 3
        assert res["n_segments"] == 2, (
            f"Expected 2 segments (wash split only), got {res['n_segments']}"
        )


# ---------------------------------------------------------------------------
# (c) Per-string n_modules preserved after plate.py deprecation
# ---------------------------------------------------------------------------

class TestStringPlateNModules:
    """_get_string_plate still computes correct n_modules from pv_capacity."""

    def test_n_modules_26_panel_string(self):
        from soiling_analysis.diagnostics.pipeline import _get_string_plate
        base = ModuleConfig()
        # 26 × (43.24 × 13.53 / 1000) kW = 26 × 0.585 ≈ 15.21 kW
        cap_kw = 26 * (base.vmp_stc * base.imp_stc / 1000.0)
        df = pd.DataFrame({"pv_capacity": [cap_kw] * 10})
        plate = _get_string_plate(df, base)
        assert plate.n_modules == 26, (
            f"Expected 26 modules, got {plate.n_modules} (cap={cap_kw:.3f} kW)"
        )

    def test_n_modules_29_panel_string(self):
        from soiling_analysis.diagnostics.pipeline import _get_string_plate
        base = ModuleConfig()
        cap_kw = 29 * (base.vmp_stc * base.imp_stc / 1000.0)
        df = pd.DataFrame({"pv_capacity": [cap_kw] * 10})
        plate = _get_string_plate(df, base)
        assert plate.n_modules == 29, (
            f"Expected 29 modules, got {plate.n_modules} (cap={cap_kw:.3f} kW)"
        )


# ---------------------------------------------------------------------------
# (d-e) Cleaning economics consistency
# ---------------------------------------------------------------------------

class TestCleaningEconomics:
    def _cfg(self, wash_cost=5000.0):
        c = PipelineConfig()
        c.wash_cost_per_string_pkr = wash_cost
        return c

    def test_zero_soiling_no_cleaning(self):
        cfg = self._cfg()
        soiling = {"srr_pct_per_day": 0.0}  # no soiling
        result = cleaning_economics("S1", soiling, None, cfg)
        assert result["recommendation"] == "No cleaning recommended"
        assert result["daily_loss_pkr"] == 0.0

    def test_positive_soiling_no_cleaning(self):
        cfg = self._cfg()
        soiling = {"srr_pct_per_day": +0.05}  # recovering (positive) — no cleaning
        result = cleaning_economics("S1", soiling, None, cfg)
        assert result["recommendation"] == "No cleaning recommended"

    def test_payback_consistency(self):
        """payback_days ≈ wash_cost / daily_loss_pkr (within 1%)."""
        wash_cost = 4000.0
        cfg = self._cfg(wash_cost=wash_cost)
        # Build a daily_df with known expected energy
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        daily_df = pd.DataFrame({
            "date": dates,
            "E_exp_kWh": [50.0] * 60,  # 50 kWh/day expected
        })
        soiling = {"srr_pct_per_day": -0.05}  # 0.05%/day soiling
        result = cleaning_economics("S1", soiling, daily_df, cfg)

        expected_daily_loss = (0.05 / 100.0) * 50.0 * cfg.site.tariff
        expected_payback    = wash_cost / expected_daily_loss

        assert np.isfinite(result["payback_days"]), "payback_days should be finite"
        assert abs(result["daily_loss_pkr"] - expected_daily_loss) < 0.01, (
            f"daily_loss_pkr mismatch: {result['daily_loss_pkr']:.4f} vs {expected_daily_loss:.4f}"
        )
        assert abs(result["payback_days"] - expected_payback) / expected_payback < 0.01, (
            f"payback_days mismatch: {result['payback_days']:.1f} vs {expected_payback:.1f}"
        )

    def test_no_wash_cost_still_returns_daily_loss(self):
        """When wash cost is not configured, daily_loss_pkr is still computed."""
        cfg = PipelineConfig()
        cfg.wash_cost_per_string_pkr = None
        cfg.wash_cost_per_kw_pkr = None
        daily_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "E_exp_kWh": [40.0] * 30,
        })
        soiling = {"srr_pct_per_day": -0.03}
        result = cleaning_economics("S1", soiling, daily_df, cfg)
        assert result["daily_loss_pkr"] > 0.0
        assert result["economics_inputs"] == "defaulted"
