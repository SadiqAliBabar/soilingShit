"""sdm_test.py — Self-contained pytest for the patched pv_diag SDM module.

Generates synthetic string DataFrames from first principles (numpy/pandas +
pvlib for physics — no imports from either diagnostics package for data
generation) and exercises fit_single_diode / iv_metrics_at_stc.

Run from the repository root:
    pytest pv_diag/sdm_test.py -v

Or from inside pv_diag_package/:
    pytest pv_diag/sdm_test.py -v
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# Ensure pv_diag_package is importable regardless of CWD / install state
# ---------------------------------------------------------------------------
_this_dir = pathlib.Path(__file__).resolve().parent          # pv_diag/
_pkg_root  = _this_dir.parent                                # pv_diag_package/
for _p in (_pkg_root, _pkg_root.parent):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Standard library / third-party imports (no pv_diag_package for data gen)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import pvlib

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from pv_diag.sdm import fit_single_diode, iv_metrics_at_stc
from pv_diag.config import ModuleConfig, PipelineConfig

# ---------------------------------------------------------------------------
# Nameplate: mono-Si 550 W × 22 modules
# ---------------------------------------------------------------------------
N_MODULES   = 22
CELLS       = 144
P_STC_MOD   = 550.0       # W per module

VMP_STC     = 41.7        # V per module at STC
IMP_STC     = P_STC_MOD / VMP_STC          # ≈ 13.19 A
ISC_STC     = IMP_STC / 0.945              # ≈ 13.96 A
VOC_STC     = VMP_STC / 0.842              # ≈ 49.5  V per module

ALPHA_ISC   = 0.00040     # A/A/K (fractional)
BETA_VOC    = -0.00270    # 1/K
GAMMA_PMP   = -0.00350    # 1/K

# True SDM parameters used for synthetic data generation
_KT_Q_25    = 0.02585     # V at 25 °C
TRUE_A_REF  = 1.5 * _KT_Q_25 * CELLS      # ≈ 5.586 V
TRUE_IL_REF = ISC_STC                      # ≈ 13.96 A
TRUE_IO_REF = 1e-10                        # A
TRUE_RS     = 0.30                         # Ω
TRUE_RSH    = 350.0                        # Ω


# ---------------------------------------------------------------------------
# Helper: build the plate and cfg objects
# ---------------------------------------------------------------------------

def _make_plate() -> ModuleConfig:
    return ModuleConfig(
        voc_stc=VOC_STC,
        vmp_stc=VMP_STC,
        isc_stc=ISC_STC,
        imp_stc=IMP_STC,
        alpha_isc=ALPHA_ISC,
        beta_voc=BETA_VOC,
        gamma_pmp=GAMMA_PMP,
        n_modules=N_MODULES,
        technology="mono-c-Si",
        cells_in_series=CELLS,
    )


def _make_cfg() -> PipelineConfig:
    return PipelineConfig()


# ---------------------------------------------------------------------------
# Helper: generate synthetic string DataFrame
# ---------------------------------------------------------------------------

def _make_string_df(
    n_days: int = 30,
    nci: float = 0.97,
    isc_scale: float = 1.0,
    voc_scale: float = 1.0,
    noise_seed: int = 42,
) -> pd.DataFrame:
    """Generate a 30-day × 5-min synthetic string DataFrame.

    The irradiance profile follows a Lahore-style Gaussian bell (peak ~950
    W/m² at solar noon, FWHM ≈ 7 hours).  Cell temperature is estimated via
    a simple NOCT model.  IV operating points are computed exactly using the
    pvlib De Soto / singlediode model with the TRUE_* parameters above.

    Parameters
    ----------
    nci : float
        Soiling factor applied to effective irradiance (1.0 = clean).
        Also interpreted as the normalised current index of the generated
        data; 0.97 corresponds to a lightly soiled string.
    isc_scale : float
        Additional multiplicative scale on I_L_ref (for Isc-suppression
        scenario, e.g. 0.80 for heavy soiling / cell mismatch).
    voc_scale : float
        Multiplicative scale applied directly to the generated string
        voltage (for voltage-degradation scenario, e.g. 0.93).
    noise_seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(noise_seed)

    # Build timestamp index: 5-min intervals 07:00–17:55 (local time)
    base = pd.Timestamp("2024-01-15", tz="Asia/Karachi")  # mid-winter Lahore
    records = []

    for day_i in range(n_days):
        day_start = base + pd.Timedelta(days=day_i)
        for m in range(0, 600 + 5, 5):          # 07:00 to 17:00 inclusive
            ts = day_start + pd.Timedelta(hours=7, minutes=m)
            hour_frac = 7.0 + m / 60.0
            # Gaussian bell centred at 12:30 local solar noon, σ≈2.5 h
            angle = (hour_frac - 12.5)
            poa = 950.0 * float(np.exp(-0.5 * (angle / 2.5) ** 2))
            if poa < 30.0:
                continue
            records.append((ts, poa))

    ts_arr   = [r[0] for r in records]
    poa_arr  = np.array([r[1] for r in records])

    # NOCT-style cell temperature: T_c = T_amb + (G/800) * (NOCT - T_amb_ref)
    # Use T_amb = 25 °C, NOCT = 47 °C → T_c ≈ 25 + 0.028 * G
    tc_arr = 25.0 + 0.028 * poa_arr

    # Effective irradiance with soiling + optional Isc suppression
    eff_g = poa_arr * nci

    # Generate operating-point (v_mp, i_mp) via pvlib De Soto + singlediode
    IL, Io, Rs, Rsh, nVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=eff_g,
        temp_cell=tc_arr,
        alpha_sc=ALPHA_ISC * ISC_STC * isc_scale,   # scale alpha by isc_scale
        a_ref=TRUE_A_REF,
        I_L_ref=TRUE_IL_REF * isc_scale,
        I_o_ref=TRUE_IO_REF,
        R_sh_ref=TRUE_RSH,
        R_s=TRUE_RS,
    )
    result = pvlib.pvsystem.singlediode(IL, Io, Rs, Rsh, nVth)

    v_mp = np.array(result["v_mp"], dtype=float)   # per-module
    i_mp = np.array(result["i_mp"], dtype=float)

    # String-level voltage, apply voc_scale for voltage-degradation scenario
    v_str = v_mp * N_MODULES * voc_scale

    # Add realistic measurement noise
    v_noise = rng.normal(0.0, 0.5, len(v_str))    # ±0.5 V on ~900 V string
    i_noise = rng.normal(0.0, 0.05, len(i_mp))    # ±50 mA
    v_str = np.maximum(v_str + v_noise, 0.1)
    i_meas = np.maximum(i_mp + i_noise, 0.01)

    # Assemble DataFrame (qflag=0 → all rows are good; ts present for anchoring)
    df = pd.DataFrame({
        "ts":       ts_arr,
        "POA":      poa_arr,
        "T_module": tc_arr,
        "V":        v_str,
        "I":        i_meas,
        "qflag":    np.zeros(len(ts_arr), dtype=np.int64),
    })
    return df


# ===========================================================================
# Tests
# ===========================================================================

class TestFitSingleDiode:
    """Tests for fit_single_diode."""

    def test_b_clean_fit_success_and_confidence(self):
        """(b) Clean NCI=0.97 data: fit should succeed and confidence > 0.40."""
        plate = _make_plate()
        cfg   = _make_cfg()
        df    = _make_string_df(n_days=30, nci=0.97)

        result = fit_single_diode(df, plate, cfg)

        assert result["success"] is True, (
            f"Expected success=True; got reason={result.get('reason')}"
        )
        assert result["fit_confidence"] > 0.40, (
            f"Expected fit_confidence > 0.40; got {result['fit_confidence']:.3f}"
        )

    def test_c_iv_metrics_at_stc_clean(self):
        """(c) iv_metrics_at_stc on clean data: voc/isc ratios in [0.90,1.05], FF in [0.65,0.85]."""
        plate = _make_plate()
        cfg   = _make_cfg()
        df    = _make_string_df(n_days=30, nci=0.97)

        sdm     = fit_single_diode(df, plate, cfg)
        metrics = iv_metrics_at_stc(sdm, plate)

        assert metrics is not None, "iv_metrics_at_stc returned None on a successful fit"

        voc_ratio = metrics["voc_stc_ratio"]
        isc_ratio = metrics["isc_stc_ratio"]
        ff        = metrics["ff"]

        assert 0.90 <= voc_ratio <= 1.05, (
            f"voc_stc_ratio={voc_ratio:.3f} not in [0.90, 1.05]"
        )
        assert 0.90 <= isc_ratio <= 1.05, (
            f"isc_stc_ratio={isc_ratio:.3f} not in [0.90, 1.05]"
        )
        assert 0.65 <= ff <= 0.85, (
            f"FF={ff:.3f} not in [0.65, 0.85]"
        )

    def test_d_soiling_degraded_isc_ratio(self):
        """(d) NCI=0.80 / Isc suppressed 20%: isc_stc_ratio < 0.93."""
        plate = _make_plate()
        cfg   = _make_cfg()
        # Effective irradiance = 80% of measured → I_L_ref ≈ 0.80 × ISC_STC
        df    = _make_string_df(n_days=30, nci=0.80, isc_scale=0.80)

        sdm     = fit_single_diode(df, plate, cfg)
        assert sdm["success"] is True, (
            f"Fit failed on soiling-degraded data: {sdm.get('reason')}"
        )
        metrics = iv_metrics_at_stc(sdm, plate)
        assert metrics is not None

        isc_ratio = metrics["isc_stc_ratio"]
        assert isc_ratio < 0.93, (
            f"Expected isc_stc_ratio < 0.93 for NCI=0.80 data; got {isc_ratio:.3f}"
        )

    def test_e_voltage_degraded_voc_ratio(self):
        """(e) Voc suppressed 7%: voc_stc_ratio < 0.95."""
        plate = _make_plate()
        cfg   = _make_cfg()
        # String voltage uniformly scaled down by 7%
        df    = _make_string_df(n_days=30, nci=0.97, voc_scale=0.93)

        sdm     = fit_single_diode(df, plate, cfg)
        assert sdm["success"] is True, (
            f"Fit failed on voltage-degraded data: {sdm.get('reason')}"
        )
        metrics = iv_metrics_at_stc(sdm, plate)
        assert metrics is not None

        voc_ratio = metrics["voc_stc_ratio"]
        assert voc_ratio < 0.95, (
            f"Expected voc_stc_ratio < 0.95 for 7%-suppressed Voc; got {voc_ratio:.3f}"
        )


class TestIvMetricsAtStc:
    """Additional unit tests for iv_metrics_at_stc return-dict contract."""

    def test_keys_present(self):
        """All mandatory keys must be present in a successful metrics dict."""
        plate = _make_plate()
        cfg   = _make_cfg()
        df    = _make_string_df(n_days=30, nci=0.97)
        sdm   = fit_single_diode(df, plate, cfg)
        m     = iv_metrics_at_stc(sdm, plate)
        assert m is not None
        for key in ("voc_stc", "isc_stc", "pmp_stc", "ff",
                    "voc_stc_ratio", "isc_stc_ratio", "ff_stc_ratio"):
            assert key in m, f"Key '{key}' missing from iv_metrics_at_stc result"

    def test_none_on_failed_fit(self):
        """iv_metrics_at_stc must return None when success=False."""
        plate  = _make_plate()
        failed = dict(success=False, reason="test")
        assert iv_metrics_at_stc(failed, plate) is None

    def test_none_on_empty_dict(self):
        """iv_metrics_at_stc must return None when params is empty."""
        plate = _make_plate()
        assert iv_metrics_at_stc({}, plate) is None

    def test_pmp_stc_positive(self):
        """String Pmp at STC should be strictly positive."""
        plate = _make_plate()
        cfg   = _make_cfg()
        df    = _make_string_df(n_days=30, nci=0.97)
        sdm   = fit_single_diode(df, plate, cfg)
        m     = iv_metrics_at_stc(sdm, plate)
        assert m is not None
        assert m["pmp_stc"] > 0.0, f"pmp_stc={m['pmp_stc']:.1f} should be > 0"

    def test_voc_str_consistent(self):
        """voc_stc (string) should equal Voc_mod * n_modules."""
        plate = _make_plate()
        cfg   = _make_cfg()
        df    = _make_string_df(n_days=30, nci=0.97)
        sdm   = fit_single_diode(df, plate, cfg)
        m     = iv_metrics_at_stc(sdm, plate)
        assert m is not None
        assert abs(m["voc_stc"] - m["Voc_mod"] * N_MODULES) < 0.5, (
            "voc_stc (string) != Voc_mod * N_MODULES"
        )


class TestFitReturnDictContract:
    """Regression tests: the return dict from fit_single_diode must have every
    expected key so downstream consumers (daily.py, classify, losses) won't
    KeyError on a successful fit."""

    _REQUIRED_KEYS = (
        "success", "reason",
        "I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref",
        "rmse_v", "rmse_i",
        "n_pts", "n_voc_anchors",
        "bounds_hit", "fit_confidence",
    )

    def test_success_keys(self):
        plate = _make_plate()
        cfg   = _make_cfg()
        df    = _make_string_df(n_days=30, nci=0.97)
        result = fit_single_diode(df, plate, cfg)
        assert result["success"] is True
        for k in self._REQUIRED_KEYS:
            assert k in result, f"Key '{k}' missing from successful fit dict"

    def test_failure_keys(self):
        """Even a failing fit must include success, reason, n_pts, fit_confidence."""
        plate = _make_plate()
        cfg   = _make_cfg()
        # Pass an empty DataFrame to trigger the insufficient-points failure path
        empty = pd.DataFrame({"ts": pd.Series([], dtype="datetime64[ns]"),
                               "POA": [], "V": [], "I": [], "T_module": [],
                               "qflag": []})
        result = fit_single_diode(empty, plate, cfg)
        assert result["success"] is False
        assert "reason" in result
        assert "n_pts" in result
        assert "fit_confidence" in result
