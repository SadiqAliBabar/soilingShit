"""Single-diode model fit (pvlib-aware, graceful fallback).

Implements a real 5-parameter De Soto fit using:
  - POA-stratified sampling (10 bins, n_target=1500)
  - Voc-anchor extraction (top-2% per day by high-V/low-I score, weight 0.5)
  - scipy.optimize.least_squares with TRF bounds
  - pvlib.pvsystem.calcparams_desoto + singlediode in the residual function
  - iv_metrics_at_stc returns absolute values + ratio keys for classification.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import ModuleConfig, PipelineConfig
from .utils import _is_ok

try:
    import pvlib
    _HAS_PVLIB = True
except ImportError:
    _HAS_PVLIB = False


# ---------------------------------------------------------------------------
# Public guard
# ---------------------------------------------------------------------------

def has_pvlib() -> bool:
    """Return True if pvlib is importable."""
    return _HAS_PVLIB


# ---------------------------------------------------------------------------
# POA-stratified sampling (10 bins, n_target=1500 — from v3 physics.py)
# ---------------------------------------------------------------------------

def _stratified_sample(df: pd.DataFrame, n_target: int = 1500,
                       n_bins: int = 10) -> pd.DataFrame:
    """Sample evenly across POA bins to avoid bias toward common irradiance.

    Uses 10 bins spanning [POA_min, POA_max].  At most n_target / n_bins
    rows are drawn from each bin (random_state=0 for reproducibility).
    """
    if len(df) <= n_target:
        return df
    poa_vals = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    lo, hi = poa_vals.min(), poa_vals.max()
    if hi <= lo:
        return df.sample(min(n_target, len(df)), random_state=0)
    edges = np.linspace(lo, hi, n_bins + 1)
    parts = []
    per_bin = max(1, n_target // n_bins)
    for i in range(n_bins):
        edge_lo = edges[i]
        edge_hi = edges[i + 1]
        if i == n_bins - 1:
            m = (poa_vals >= edge_lo) & (poa_vals <= edge_hi)
        else:
            m = (poa_vals >= edge_lo) & (poa_vals < edge_hi)
        if m.sum() == 0:
            continue
        b = df.loc[m]
        parts.append(b.sample(min(per_bin, len(b)), random_state=0))
    if not parts:
        return df
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Voc-anchor extraction (from v3 physics.py, adapted for pv_diag conventions)
# ---------------------------------------------------------------------------

def extract_voc_anchors(df: pd.DataFrame, plate: ModuleConfig,
                        top_pct: float = 0.02) -> pd.DataFrame:
    """Per-day top top_pct samples by Voc-proxy score (high V, low I).

    The Voc-proxy score is:
        score = V / Voc_str_stc  -  0.6 * I / I_mp_expected(G, Tc)

    Rows near open-circuit get high scores.  Returned rows are used as
    anchor points weighted at 0.5 in the residuals vector so the fit
    anchors to open-circuit behaviour.
    """
    df = df.copy()

    # Attach calendar date for per-day grouping
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"])
        tz = getattr(ts.dt, "tz", None)
        if tz is not None:
            ts_naive = ts.dt.tz_convert(None)
        else:
            ts_naive = ts
        df["__date"] = ts_naive.dt.date

    # Eligibility mask
    elig = (df["POA"] > 50) & df["V"].gt(0) & df["I"].gt(0)
    if "qflag" in df.columns:
        elig = elig & _is_ok(df["qflag"].values)
    if not elig.any():
        return df.iloc[0:0]

    sub = df.loc[elig].copy()

    # Expected Imp(G, Tc) for score normalisation
    Gn = pd.to_numeric(sub["POA"], errors="coerce").fillna(0).values / 1000.0
    if "T_module" in sub.columns:
        dT = sub["T_module"].fillna(25.0).values - 25.0
    else:
        dT = np.zeros(len(sub))
    imp_exp = plate.imp_stc * Gn * (1.0 + plate.alpha_isc * dT)
    imp_exp = np.where(imp_exp > 0.05, imp_exp, 1.0)

    voc_str_stc = plate.voc_stc * plate.n_modules  # string open-circuit
    sub["__voc_score"] = (
        pd.to_numeric(sub["V"], errors="coerce").fillna(0) / voc_str_stc
        - 0.6 * pd.to_numeric(sub["I"], errors="coerce").fillna(0) / imp_exp
    )

    # Select top top_pct per day (or globally if no date column)
    if "__date" in sub.columns:
        anchors = sub.groupby("__date", group_keys=False).apply(
            lambda g: g.nlargest(max(1, int(len(g) * top_pct)), "__voc_score")
        )
    else:
        anchors = sub.nlargest(max(1, int(len(sub) * top_pct)), "__voc_score")

    drop_cols = [c for c in ("__Imp_exp", "__voc_score", "__date")
                 if c in anchors.columns]
    return anchors.drop(columns=drop_cols)


# ---------------------------------------------------------------------------
# Main fitter
# ---------------------------------------------------------------------------

def fit_single_diode(df: pd.DataFrame, plate: ModuleConfig,
                     cfg: PipelineConfig) -> dict:
    """Fit the 5-parameter De Soto single-diode model.

    Parameters
    ----------
    df : DataFrame
        String-level data.  Required columns: POA, V, I.
        Optional but strongly recommended: T_module, qflag, ts.
    plate : ModuleConfig
        Module nameplate parameters.
    cfg : PipelineConfig
        Pipeline configuration (not actively used inside the fit, kept for
        API compatibility).

    Returns
    -------
    dict with keys:
        success, reason, I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref,
        rmse_v, rmse_i, n_pts, n_voc_anchors, bounds_hit, fit_confidence
    """
    if not _HAS_PVLIB:
        return dict(success=False, reason="pvlib_unavailable",
                    n_pts=0, fit_confidence=0.0)

    # ------------------------------------------------------------------
    # 1.  Build the clean working subset
    # ------------------------------------------------------------------
    min_pts = 200
    max_pts = 1500

    poa_num = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values

    if "qflag" in df.columns:
        ok_mask = _is_ok(df["qflag"].values) & (poa_num > 100)
    else:
        ok_mask = poa_num > 100

    # Ensure T_module exists (synthesise if absent)
    work = df.copy()
    if "T_module" not in work.columns:
        work["T_module"] = 25.0 + (
            pd.to_numeric(work["POA"], errors="coerce").fillna(0) / 1000.0 * 30.0
        )

    sub = work.loc[ok_mask, ["POA", "V", "I", "T_module"]].copy()
    sub = sub.dropna(subset=["POA", "V", "I"])
    sub["T_module"] = sub["T_module"].fillna(25.0)
    sub = sub[(pd.to_numeric(sub["V"], errors="coerce") > 0) &
              (pd.to_numeric(sub["I"], errors="coerce") > 0)]
    sub["POA"] = pd.to_numeric(sub["POA"], errors="coerce")
    sub["V"] = pd.to_numeric(sub["V"], errors="coerce")
    sub["I"] = pd.to_numeric(sub["I"], errors="coerce")

    n_avail = len(sub)
    if n_avail < min_pts:
        return dict(success=False,
                    reason=f"insufficient_points ({n_avail}/{min_pts} required)",
                    n_pts=n_avail, fit_confidence=0.0)

    # POA-stratified sample (10 bins, target 1500)
    sub = _stratified_sample(sub, n_target=max_pts, n_bins=10)

    # ------------------------------------------------------------------
    # 2.  Voc anchors (top-2% per day by high-V/low-I score, weight 0.5)
    # ------------------------------------------------------------------
    n_anchors = 0
    try:
        adf = extract_voc_anchors(work, plate, top_pct=0.02)
        if len(adf) > 0:
            need = ["POA", "V", "I", "T_module"]
            a_sub = adf[[c for c in need if c in adf.columns]].copy()
            if "T_module" not in a_sub.columns:
                a_sub["T_module"] = 25.0
            a_sub = a_sub.dropna(subset=["POA", "V", "I"])
            a_sub["T_module"] = a_sub["T_module"].fillna(25.0)
            for col in ("POA", "V", "I"):
                a_sub[col] = pd.to_numeric(a_sub[col], errors="coerce")
            a_sub = a_sub[(a_sub["V"] > 0) & (a_sub["I"] > 0)]
            if len(a_sub) > 0:
                a_sub = a_sub.copy()
                a_sub["_a"] = True
                sub["_a"] = False
                sub = pd.concat([sub, a_sub], ignore_index=True)
                n_anchors = len(a_sub)
            else:
                sub["_a"] = False
        else:
            sub["_a"] = False
    except Exception:
        sub["_a"] = False

    # ------------------------------------------------------------------
    # 3.  Extract arrays for the residual function
    # ------------------------------------------------------------------
    G = sub["POA"].values.astype(float)
    V = sub["V"].values.astype(float) / plate.n_modules   # per-module voltage
    I = sub["I"].values.astype(float)
    Tc = sub["T_module"].values.astype(float)
    is_anchor = sub["_a"].values.astype(bool) if "_a" in sub.columns else np.zeros(len(G), dtype=bool)

    # Noise floor → relative sigma, minimum 50 mV / 50 mA
    sigma_V = np.maximum(0.005 * np.abs(V), 0.05)
    sigma_I = np.maximum(0.010 * np.abs(I), 0.05)
    aw = np.where(is_anchor, 0.5, 1.0)   # anchors weighted at 0.5

    isc_stc = plate.isc_stc
    a0 = 1.5 * 0.0257 * plate.cells_in_series  # initial a_ref (nNsVth at STC)
    I_L0 = isc_stc * 1.005
    I_o0 = 1e-10
    Rs0, Rsh0 = 0.30, 350.0

    alpha_sc = plate.alpha_isc * plate.isc_stc  # absolute temperature coeff [A/K]

    # ------------------------------------------------------------------
    # 4.  Residual function
    # ------------------------------------------------------------------
    def residuals(p: np.ndarray) -> np.ndarray:
        I_L, log_Io, R_s, R_sh, a = p
        I_o = np.exp(log_Io)
        try:
            IL, Io, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
                effective_irradiance=G,
                temp_cell=Tc,
                alpha_sc=alpha_sc,
                a_ref=a,
                I_L_ref=I_L,
                I_o_ref=I_o,
                R_sh_ref=R_sh,
                R_s=R_s,
            )
            res = pvlib.pvsystem.singlediode(IL, Io, Rs, Rsh, nNsVth)
        except Exception:
            return np.full(len(G) * 2, 1e6)
        r_v = (res["v_mp"] - V) / sigma_V * aw
        r_i = (res["i_mp"] - I) / sigma_I * aw
        return np.concatenate([r_v, r_i])

    # ------------------------------------------------------------------
    # 5.  Bounds and initial point
    # ------------------------------------------------------------------
    p0 = [I_L0,          np.log(I_o0), Rs0,   Rsh0,   a0      ]
    bl = [I_L0 * 0.7,   -30.0,          0.05,    50.0, a0 * 0.5]
    bh = [I_L0 * 1.3,   -15.0,          2.00, 2000.0, a0 * 2.0]

    # ------------------------------------------------------------------
    # 6.  Solve
    # ------------------------------------------------------------------
    try:
        result = least_squares(
            residuals, p0,
            bounds=(bl, bh),
            max_nfev=300,
            method="trf",
        )
    except Exception as exc:
        return dict(success=False, reason=f"solver_failed: {exc}",
                    n_pts=len(G), fit_confidence=0.0)

    I_L, log_Io, R_s, R_sh, a = result.x
    I_o = np.exp(log_Io)

    # ------------------------------------------------------------------
    # 7.  Post-fit diagnostics
    # ------------------------------------------------------------------
    try:
        IL_p, Io_p, Rs_p, Rsh_p, nVth_p = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=G,
            temp_cell=Tc,
            alpha_sc=alpha_sc,
            a_ref=a,
            I_L_ref=I_L,
            I_o_ref=I_o,
            R_sh_ref=R_sh,
            R_s=R_s,
        )
        sd = pvlib.pvsystem.singlediode(IL_p, Io_p, Rs_p, Rsh_p, nVth_p)
        rmse_v = float(np.sqrt(np.mean((sd["v_mp"] - V) ** 2)))
        rmse_i = float(np.sqrt(np.mean((sd["i_mp"] - I) ** 2)))
    except Exception:
        rmse_v = rmse_i = float("nan")

    bounds_hit: list[str] = []
    for nm, val, lo, hi in zip(
        ["I_L", "log_Io", "R_s", "R_sh", "a"],
        result.x, bl, bh,
    ):
        if val <= lo * 1.001 or val >= hi * 0.999:
            bounds_hit.append(nm)

    conf_rmse = float(np.clip(1.0 - rmse_i / (0.10 * isc_stc), 0.0, 1.0))
    conf_npts = float(np.clip(len(G) / 500.0, 0.4, 1.0))
    conf_bound = 0.6 if bounds_hit else 1.0
    fit_conf = conf_rmse * conf_npts * conf_bound

    return dict(
        success=True,
        reason="ok" if not bounds_hit else f"bounds_hit:{bounds_hit}",
        I_L_ref=float(I_L),
        I_o_ref=float(I_o),
        R_s=float(R_s),
        R_sh_ref=float(R_sh),
        a_ref=float(a),
        rmse_v=rmse_v,
        rmse_i=rmse_i,
        n_pts=len(G),
        n_voc_anchors=n_anchors,
        bounds_hit=bounds_hit,
        fit_confidence=fit_conf,
    )


# ---------------------------------------------------------------------------
# IV curve from a fitted SDM
# ---------------------------------------------------------------------------

def iv_curve_from_sdm(params: dict, plate: ModuleConfig,
                      n: int = 100,
                      G: float = 1000.0,
                      Tc: float = 25.0) -> pd.DataFrame:
    """Generate full IV curve at (G, Tc) using pvlib singlediode.

    Returns a DataFrame with columns V (string-level, V) and I (A).
    On any failure returns an empty DataFrame with the same columns.

    Parameters
    ----------
    params : dict
        Result dict from fit_single_diode (must have success=True).
    plate : ModuleConfig
        Module nameplate.
    n : int
        Number of points on the IV curve (default 100).
    G : float
        Irradiance in W/m² (default 1000, i.e. STC).
    Tc : float
        Cell temperature in °C (default 25, i.e. STC).
    """
    if not params or not params.get("success", False):
        return pd.DataFrame({"V": [], "I": []})
    if not _HAS_PVLIB:
        return pd.DataFrame({"V": [], "I": []})
    try:
        IL, Io, Rs, Rsh, nVth = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=np.array([float(G)]),
            temp_cell=np.array([float(Tc)]),
            alpha_sc=plate.alpha_isc * plate.isc_stc,
            a_ref=params["a_ref"],
            I_L_ref=params["I_L_ref"],
            I_o_ref=params["I_o_ref"],
            R_sh_ref=params["R_sh_ref"],
            R_s=params["R_s"],
        )
        pts = pvlib.pvsystem.singlediode(IL, Io, Rs, Rsh, nVth)
        voc_mod = float(np.atleast_1d(pts["v_oc"])[0])
        Vm = np.linspace(0.0, voc_mod * 1.01, n)
        Im = pvlib.pvsystem.i_from_v(
            Vm,
            float(np.atleast_1d(IL)[0]),
            float(np.atleast_1d(Io)[0]),
            float(Rs),
            float(np.atleast_1d(Rsh)[0]),
            float(np.atleast_1d(nVth)[0]),
        )
        Im = np.maximum(Im, 0.0)
        return pd.DataFrame({"V": Vm * plate.n_modules, "I": Im})
    except Exception:
        return pd.DataFrame({"V": [], "I": []})


# ---------------------------------------------------------------------------
# STC fingerprint from a fitted SDM
# ---------------------------------------------------------------------------

def iv_metrics_at_stc(params: dict, plate: ModuleConfig) -> dict | None:
    """Compute STC metrics by running pvlib singlediode at G=1000, Tc=25.

    Returns a dict with both absolute values and ratio keys so that
    classification.py (which reads voc_stc_ratio, isc_stc_ratio, ff_stc_ratio)
    works without modification.

    Ratio definitions
    -----------------
    voc_stc_ratio = Voc_per_module / plate.voc_stc
    isc_stc_ratio = Isc            / plate.isc_stc
    ff_stc_ratio  = FF             / 0.78

    Returns None on any failure or if params is not a successful fit.
    """
    if not params or not params.get("success", False):
        return None
    if not _HAS_PVLIB:
        return None
    try:
        IL, Io, Rs, Rsh, nVth = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=np.array([1000.0]),
            temp_cell=np.array([25.0]),
            alpha_sc=plate.alpha_isc * plate.isc_stc,
            a_ref=params["a_ref"],
            I_L_ref=params["I_L_ref"],
            I_o_ref=params["I_o_ref"],
            R_sh_ref=params["R_sh_ref"],
            R_s=params["R_s"],
        )
        r = pvlib.pvsystem.singlediode(IL, Io, Rs, Rsh, nVth)

        voc_mod = float(np.atleast_1d(r["v_oc"])[0])   # per-module Voc [V]
        isc = float(np.atleast_1d(r["i_sc"])[0])        # current [A]
        vmp_mod = float(np.atleast_1d(r["v_mp"])[0])    # per-module Vmp [V]
        imp = float(np.atleast_1d(r["i_mp"])[0])        # current [A]
        pmp_mod = vmp_mod * imp                          # per-module Pmp [W]
        ff = pmp_mod / (voc_mod * isc) if (voc_mod * isc) > 0 else 0.0

        voc_stc_ratio = (voc_mod / plate.voc_stc
                         if plate.voc_stc > 0 else float("nan"))
        isc_stc_ratio = (isc / plate.isc_stc
                         if plate.isc_stc > 0 else float("nan"))
        ff_stc_ratio = (ff / 0.78 if not (np.isnan(ff) or ff == 0.0)
                        else float("nan"))

        return dict(
            # Absolute values (string-level where relevant)
            voc_stc=voc_mod * plate.n_modules,       # string Voc [V]
            isc_stc=isc,                              # current [A]
            pmp_stc=pmp_mod * plate.n_modules,        # string Pmp [W]
            ff=ff,
            # Per-module / per-module fields for downstream use
            Voc_mod=voc_mod,
            Vmp_mod=vmp_mod,
            Imp=imp,
            Voc_str=voc_mod * plate.n_modules,
            Vmp_str=vmp_mod * plate.n_modules,
            Pmp_str=pmp_mod * plate.n_modules,
            # Ratio keys — consumed by classification.py without any changes
            voc_stc_ratio=voc_stc_ratio,
            isc_stc_ratio=isc_stc_ratio,
            ff_stc_ratio=ff_stc_ratio,
        )
    except Exception:
        return None
