"""clearsky_quality_test.py — Batch 6 unit tests for clearsky_quality.py.

Run from the repo root:
    pytest src/soiling_analysis/soiling_old_7_prompt/pv_diag/clearsky_quality_test.py -v
"""
from __future__ import annotations

import pathlib
import sys

_this_dir = pathlib.Path(__file__).resolve().parent
_pkg_root  = _this_dir.parent
for _p in (_pkg_root, _pkg_root.parent):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pandas as pd
import pytest

from soiling_analysis.diagnostics.clearsky_quality import (
    compute_kc,
    clearsky_quality_mask,
    day_kc_quality_dates,
)
from soiling_analysis.diagnostics.config import PipelineConfig
from soiling_analysis.diagnostics.orientation import compute_clearsky_poa

# Site constants
LAT = 31.4504   # Lahore
LON = 73.1350
AZ  = 180.0
TILT = 25.0


def _make_ts(date_str: str = "2024-01-15", n_hours: float = 10.0, freq_min: int = 5) -> pd.DatetimeIndex:
    """5-min timestamps centred on a clear winter day 07:00-17:00."""
    base = pd.Timestamp(date_str)
    periods = int(n_hours * 60 / freq_min) + 1
    return pd.date_range(base + pd.Timedelta(hours=7), periods=periods, freq=f"{freq_min}min")


def _clearsky_poa_arr(ts_idx: pd.DatetimeIndex) -> np.ndarray:
    """Return the model clear-sky POA for the given timestamps."""
    cs = compute_clearsky_poa(ts_idx, LAT, LON, AZ, TILT)
    return cs.values.astype(float)


# ---------------------------------------------------------------------------
# Tests for compute_kc
# ---------------------------------------------------------------------------

class TestComputeKc:
    def test_kc_unity_on_clear_sky(self):
        """When measured POA equals the clear-sky model, Kc ≈ 1.0 at bright times."""
        ts = _make_ts()
        cs = _clearsky_poa_arr(ts)
        kc = compute_kc(ts, cs, LAT, LON, AZ, TILT)
        # At bright midday samples (cs >= 50 W/m²), Kc must be 1.0
        bright = cs >= 50.0
        assert bright.sum() > 10, "expected at least 10 bright samples on a winter day"
        np.testing.assert_allclose(kc.values[bright], 1.0, atol=1e-6,
                                   err_msg="Kc should be 1.0 when measured = clearsky")

    def test_kc_nan_below_horizon(self):
        """Kc is NaN where clearsky_POA < 50 (near-horizon / night)."""
        ts = _make_ts()
        cs = _clearsky_poa_arr(ts)
        kc = compute_kc(ts, cs, LAT, LON, AZ, TILT)
        night_mask = cs < 50.0
        if night_mask.any():
            assert np.all(np.isnan(kc.values[night_mask])), \
                "Kc must be NaN where clearsky_POA < 50 W/m²"

    def test_kc_less_than_one_on_cloudy(self):
        """Cloudy day (measured = 50% of clear-sky) gives Kc ≈ 0.5."""
        ts = _make_ts()
        cs = _clearsky_poa_arr(ts)
        kc = compute_kc(ts, cs * 0.5, LAT, LON, AZ, TILT)
        bright = cs >= 50.0
        np.testing.assert_allclose(kc.values[bright], 0.5, atol=0.01)

    def test_empty_index(self):
        """Empty input returns an empty Series without error."""
        kc = compute_kc(pd.DatetimeIndex([]), np.array([]), LAT, LON, AZ, TILT)
        assert len(kc) == 0


# ---------------------------------------------------------------------------
# Tests for clearsky_quality_mask
# ---------------------------------------------------------------------------

class TestClearSkyQualityMask:
    def test_clear_day_passes(self):
        """A perfectly clear day (measured = clearsky) has >80% passing intervals."""
        ts  = _make_ts()
        cs  = _clearsky_poa_arr(ts)
        cfg = PipelineConfig()
        mask = clearsky_quality_mask(ts, cs, cfg, LAT, LON, AZ, TILT)
        bright = cs >= 50.0
        n_bright = int(bright.sum())
        n_pass   = int((mask & bright).sum())
        assert n_bright > 0
        assert n_pass / n_bright > 0.80, (
            f"Expect >80% of bright clear intervals to pass; got {n_pass}/{n_bright}"
        )

    def test_cloudy_day_mostly_fails(self):
        """A heavily cloudy day (measured = 40% of clearsky) mostly fails."""
        ts  = _make_ts()
        cs  = _clearsky_poa_arr(ts)
        cfg = PipelineConfig()
        # Kc = 0.40 → |Kc - 1| = 0.60 > kc_tolerance=0.10 → fails clearness
        mask = clearsky_quality_mask(ts, cs * 0.40, cfg, LAT, LON, AZ, TILT)
        bright = cs >= 50.0
        n_bright = int(bright.sum())
        n_pass   = int((mask & bright).sum())
        assert n_bright > 0
        assert n_pass / n_bright < 0.10, (
            f"Expect <10% of bright intervals to pass for Kc=0.40; got {n_pass}/{n_bright}"
        )

    def test_returns_bool_array(self):
        """Output must be a boolean numpy array of the correct length."""
        ts   = _make_ts()
        cs   = _clearsky_poa_arr(ts)
        cfg  = PipelineConfig()
        mask = clearsky_quality_mask(ts, cs, cfg, LAT, LON, AZ, TILT)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(ts)

    def test_disabled_does_not_crash(self):
        """Calling with clearsky_quality_enabled=False has no effect on the mask function itself."""
        ts   = _make_ts()
        cs   = _clearsky_poa_arr(ts)
        cfg  = PipelineConfig()
        cfg.clearsky_quality_enabled = False
        # The function itself doesn't check the flag; the caller (pipeline.py) does.
        # Confirm it still returns a valid array.
        mask = clearsky_quality_mask(ts, cs, cfg, LAT, LON, AZ, TILT)
        assert len(mask) == len(ts)


# ---------------------------------------------------------------------------
# Tests for day_kc_quality_dates
# ---------------------------------------------------------------------------

class TestDayKcQualityDates:
    def _make_day_df(self, date_str: str, poa_scale: float = 1.0) -> pd.DataFrame:
        """Build a single-day DataFrame with POA = poa_scale × clearsky."""
        ts = _make_ts(date_str)
        cs = _clearsky_poa_arr(ts)
        return pd.DataFrame({"ts": ts, "POA": cs * poa_scale})

    def test_clear_day_is_included(self):
        """A clear day (POA = clearsky) appears in the good-date set."""
        df  = self._make_day_df("2024-01-15", poa_scale=1.0)
        cfg = PipelineConfig()
        good = day_kc_quality_dates(df, cfg, LAT, LON, AZ, TILT)
        from datetime import date
        d = date(2024, 1, 15)
        assert d in good, f"Clear day {d} should be in Kc quality set"

    def test_cloudy_day_is_excluded(self):
        """A heavily cloudy day (POA = 40% clearsky) is excluded."""
        df  = self._make_day_df("2024-06-15", poa_scale=0.40)
        cfg = PipelineConfig()
        good = day_kc_quality_dates(df, cfg, LAT, LON, AZ, TILT)
        from datetime import date
        d = date(2024, 6, 15)
        assert d not in good, f"Cloudy day {d} (Kc=0.40) should NOT be in Kc quality set"

    def test_winter_day_low_poa_accepted(self):
        """Key Batch-6 test: a clear winter day with peak POA < 600 W/m² is accepted.

        The old 600 W/m² floor would reject it; the Kc gate accepts it because
        the sky is clear (Kc ≈ 1.0) even though winter peak irradiance is lower.
        """
        # Winter solstice — peak clearsky POA for Lahore at 25° tilt is ~550 W/m²
        ts  = _make_ts("2024-12-21")
        cs  = _clearsky_poa_arr(ts)
        peak_cs = float(cs.max())

        df  = pd.DataFrame({"ts": ts, "POA": cs})  # measured = clearsky (perfectly clear)
        cfg = PipelineConfig()
        good = day_kc_quality_dates(df, cfg, LAT, LON, AZ, TILT)
        from datetime import date
        d = date(2024, 12, 21)

        # Confirm the peak is actually below the old 600 W/m² floor (validates the test premise)
        # (If the model gives > 600 for this date, skip the premise check but still assert inclusion.)
        if peak_cs < 600.0:
            assert d in good, (
                f"Winter clear day {d} with peak clearsky POA={peak_cs:.0f} W/m² "
                f"(<600 W/m²) should be accepted by Kc gate but rejected by the old floor"
            )
        else:
            # Peak happened to exceed 600 — just verify it's still accepted
            assert d in good, (
                f"Clear day {d} should be in Kc quality set regardless of absolute POA"
            )

    def test_empty_df_returns_empty_set(self):
        """Empty DataFrame returns an empty set without error."""
        df   = pd.DataFrame({"ts": pd.Series([], dtype="datetime64[ns]"), "POA": []})
        cfg  = PipelineConfig()
        good = day_kc_quality_dates(df, cfg, LAT, LON, AZ, TILT)
        assert good == set()

    def test_select_quality_days_kc_vs_legacy(self):
        """Integration: _select_quality_days with Kc enabled accepts the winter day
        that the legacy 600 W/m² floor rejects."""
        from soiling_analysis.diagnostics.pipeline import _select_quality_days
        from datetime import date

        ts  = _make_ts("2024-12-21")
        cs  = _clearsky_poa_arr(ts)
        peak_cs = float(cs.max())

        if peak_cs >= 600.0:
            pytest.skip(f"Peak clearsky POA {peak_cs:.0f} ≥ 600 — premise not met for this location/season")

        # Build minimal DataFrame compatible with _select_quality_days
        df = pd.DataFrame({
            "ts":    ts,
            "POA":   cs,
            "qflag": np.zeros(len(ts), dtype=np.int64),
        })

        cfg_kc = PipelineConfig(clearsky_quality_enabled=True)
        cfg_old = PipelineConfig(clearsky_quality_enabled=False)

        good_kc  = _select_quality_days(df, cfg_kc)
        good_old = _select_quality_days(df, cfg_old)

        d = date(2024, 12, 21)
        assert d in good_kc,  "Kc gate should accept clear winter day with peak < 600 W/m²"
        assert d not in good_old, "Legacy 600 W/m² floor should reject the same day"
