"""Curtailment detection (state + statistical + voltage-rise) + loss quantification.

Batch 3: when curtailment_inverter_level_enabled=True all detection runs at the
inverter level on AC power with cross-string consensus for suppression and VR.
When False the pre-Batch-3 per-string behaviour is restored exactly.

New flags added in Batch 3:
  CURT_EXPORT_LIMIT  — clipping at a POI/export setpoint (disqualifying, recoverable revenue)
  STRING_UNDERPERFORM — lone low string that is NOT inverter-level suppression;
                        intentionally excluded from DISQUALIFYING so the fault
                        classifier sees those rows.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .constants import QUALITY_FLAGS, STATE_NAME

# Bitmask for already-detected curtailment types (used to avoid double-flagging).
# Includes CURT_EXPORT_LIMIT so it is also skipped by the VR detector.
_ALREADY_CURT = (QUALITY_FLAGS["CURT_STATE"]
                 | QUALITY_FLAGS["CURT_STATISTICAL"]
                 | QUALITY_FLAGS["CURT_EXPORT_LIMIT"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dwell_filter(mask: np.ndarray, min_dwell: int) -> np.ndarray:
    """Return copy of boolean mask with runs shorter than min_dwell cleared."""
    out = np.zeros(len(mask), dtype=bool)
    if not mask.any():
        return out
    diffs = np.diff(mask.astype(np.int8), prepend=0, append=0)
    for s, e in zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]):
        if (e - s) >= min_dwell:
            out[s:e] = True
    return out


def _reconstruct_inverter_ac(
    long_df: pd.DataFrame,
    inverter_specs: dict | None,
) -> pd.Series:
    """Build inverter AC power (kW) from Σ(string DC) × efficiency.

    Used when measured inverter AC is absent (C4 false).  Returns a Series
    with MultiIndex (ts, inverter_id).
    """
    if "P" not in long_df.columns or "inverter_id" not in long_df.columns:
        return pd.Series(dtype=float)
    P_w = pd.to_numeric(long_df["P"], errors="coerce").fillna(0.0)
    inv_dc_w = P_w.groupby([long_df["ts"], long_df["inverter_id"]]).sum()

    def _eff(inv_id: str) -> float:
        spec = (inverter_specs or {}).get(str(inv_id), {})
        eff = spec.get("max_efficiency_pct")
        if eff is not None:
            try:
                eff_f = float(eff)
                if 50.0 < eff_f <= 100.0:
                    return eff_f / 100.0
            except (ValueError, TypeError):
                pass
        return 0.97

    idx = inv_dc_w.index
    eff_arr = np.array([_eff(inv_id) for _, inv_id in idx])
    return pd.Series((inv_dc_w.values * eff_arr) / 1000.0, index=idx, name="ac_power_kw")


def _detect_clip_inverter_level(
    inv_ac: pd.Series,
    cfg: PipelineConfig,
    freq_min: float,
) -> tuple[set, set]:
    """Adaptive plateau clipping detection per inverter on AC power.

    The plateau level is the 90th-percentile of daily-max AC values for each
    inverter.  A run is clipping when AC stays within cfg.clip_band_rel of
    that level, rolling CV < cfg.clip_max_cv, and the run dwells for at
    least cfg.clip_min_dwell consecutive intervals.

    Returns:
        stat_pairs   — set of (ts, inv_id) to flag CURT_STATISTICAL
        export_pairs — subset where ALL inverters clip simultaneously → CURT_EXPORT_LIMIT
    """
    if inv_ac is None or len(inv_ac) == 0:
        return set(), set()

    inv_ac_df = inv_ac.reset_index()
    inv_ac_df.columns = ["ts", "inverter_id", "ac_power_kw"]
    inv_ac_df["ts"] = pd.to_datetime(inv_ac_df["ts"])
    inv_ac_df["date"] = inv_ac_df["ts"].dt.date
    inv_ac_df["ac_power_kw"] = pd.to_numeric(inv_ac_df["ac_power_kw"], errors="coerce").fillna(0.0)

    inv_ids = inv_ac_df["inverter_id"].unique()
    n_inv = len(inv_ids)
    stat_pairs: set = set()
    clip_ts_inv: dict = {}  # ts -> set of inv_ids currently clipping
    inv_plateau: dict = {}  # inv_id -> its detected plateau level (kW)

    win = max(2, int(round(15.0 / max(freq_min, 0.1))))

    for inv_id in inv_ids:
        sub = inv_ac_df[inv_ac_df["inverter_id"] == inv_id].sort_values("ts")
        if len(sub) < cfg.clip_repeat_days:
            continue

        daily_max = sub.groupby("date")["ac_power_kw"].max()
        arr_max = daily_max.values.astype(float)
        if arr_max.max() <= 0:
            continue

        plateau = float(np.percentile(arr_max, 90))
        if plateau <= 0:
            continue

        near_days = np.abs(arr_max - plateau) / plateau <= cfg.clip_band_rel
        if int(near_days.sum()) < cfg.clip_repeat_days:
            continue

        ac_vals = sub["ac_power_kw"].values.astype(float)
        ts_vals = sub["ts"].values

        near_plateau = np.abs(ac_vals - plateau) / max(plateau, 1e-6) <= cfg.clip_band_rel

        s_ac = pd.Series(ac_vals)
        roll_mean = s_ac.rolling(win, min_periods=2).mean()
        roll_std  = s_ac.rolling(win, min_periods=2).std()
        cv = (roll_std / roll_mean.replace(0.0, np.nan)).fillna(1.0).values
        low_cv = cv <= cfg.clip_max_cv

        clip_mask = _dwell_filter(near_plateau & low_cv, cfg.clip_min_dwell)
        inv_plateau[inv_id] = plateau

        for i, ts_val in enumerate(ts_vals):
            if clip_mask[i]:
                pair = (ts_val, inv_id)
                stat_pairs.add(pair)
                if ts_val not in clip_ts_inv:
                    clip_ts_inv[ts_val] = set()
                clip_ts_inv[ts_val].add(inv_id)

    # CURT_EXPORT_LIMIT: all inverters clip simultaneously AND at similar plateau levels.
    # Different plateau levels (e.g. 80 kW vs 40 kW) indicate individual DC/AC-ratio
    # constraints, not a shared point-of-interconnection export setpoint.
    export_ts: set = set()
    if n_inv > 1 and len(inv_plateau) == n_inv:
        all_levels = [inv_plateau[iid] for iid in inv_ids]
        med = float(np.median(all_levels))
        if med > 0:
            # ±10% relative tolerance: inverters at the same export setpoint
            # will cluster tightly; individually-constrained inverters will not.
            similar = all(abs(lv - med) / med <= 0.10 for lv in all_levels)
            if similar:
                export_ts = {ts for ts, invs in clip_ts_inv.items()
                             if len(invs) == n_inv}
    export_pairs = {p for p in stat_pairs if p[0] in export_ts}

    return stat_pairs, export_pairs


def _apply_clip_flags(
    df: pd.DataFrame,
    q: np.ndarray,
    stat_pairs: set,
    export_pairs: set,
) -> np.ndarray:
    """Map (ts, inv_id) clipping pairs back onto q (qflag array for df)."""
    if not stat_pairs:
        return q

    STAT_F   = QUALITY_FLAGS["CURT_STATISTICAL"]
    EXPORT_F = QUALITY_FLAGS["CURT_EXPORT_LIMIT"]

    clip_list = [
        {"ts": ts, "inverter_id": inv_id,
         "_f": EXPORT_F if (ts, inv_id) in export_pairs else STAT_F}
        for ts, inv_id in stat_pairs
    ]
    clip_df = pd.DataFrame(clip_list)
    clip_df["ts"] = pd.to_datetime(clip_df["ts"])

    join = df[["ts", "inverter_id"]].copy().reset_index()
    join["ts"] = pd.to_datetime(join["ts"])
    merged = join.merge(clip_df, on=["ts", "inverter_id"], how="left")
    flags = merged.set_index("index")["_f"].reindex(df.index).fillna(0).values.astype(np.int64)
    q |= flags
    return q


def _detect_suppression_consensus(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppression with cross-string consensus.

    Returns two boolean arrays aligned to df:
        suppressed   — flag CURT_SUPPRESSED (inverter-level, consensus met + dwell)
        underperform — flag STRING_UNDERPERFORM (lone low string, not inverter fault)
    """
    n = len(df)
    zero = np.zeros(n, dtype=bool)
    if n == 0 or "POA" not in df.columns or "P" not in df.columns:
        return zero, zero
    if "inverter_id" not in df.columns:
        return zero, zero

    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0.0).values
    P   = pd.to_numeric(df["P"],   errors="coerce").fillna(0.0).values
    bright = poa > cfg.suppression_poa_threshold

    if "pv_capacity" in df.columns:
        cap_kw = pd.to_numeric(df["pv_capacity"], errors="coerce").fillna(0.0).values
        pmp_exp = cap_kw * 1000.0 * (poa / 1000.0)
    else:
        n_str = max(cfg.site.n_strings_per_inv, 1)
        pmp_exp = np.full(n, cfg.site.p_ac_max_kw * 1000.0 / n_str)

    ratio   = np.where(pmp_exp > 10, P / pmp_exp, 1.0)
    low_str = ratio < cfg.suppression_power_ratio

    wk = pd.DataFrame({
        "ts":          df["ts"].values,
        "inverter_id": df["inverter_id"].values,
        "_bright":     bright,
        "_low":        low_str,
    }, index=df.index)

    grp = wk.groupby(["ts", "inverter_id"])
    wk["_low_frac"] = grp["_low"].transform("mean")
    wk["_n_in_grp"] = grp["_low"].transform("count")

    # Use strict > (not >=) so that exactly 50% of strings being low (e.g. 1 of 2)
    # does NOT count as a majority — a lone dead string on a 2-string inverter
    # correctly falls through to STRING_UNDERPERFORM instead of CURT_SUPPRESSED.
    consensus_met = (
        wk["_bright"].values
        & (wk["_low_frac"].values > cfg.suppression_consensus_frac)
        & (wk["_n_in_grp"].values > 1)
    )

    # Dwell filter at the (ts, inverter_id) level — group rows first
    inv_ts_df = (
        wk.groupby(["ts", "inverter_id"])[["_bright", "_low_frac", "_n_in_grp"]]
        .first()
        .reset_index()
    )
    inv_ts_df["_candidate"] = (
        inv_ts_df["_bright"]
        & (inv_ts_df["_low_frac"] > cfg.suppression_consensus_frac)
        & (inv_ts_df["_n_in_grp"] > 1)
    )
    inv_ts_df = inv_ts_df.sort_values(["inverter_id", "ts"])

    dwell_records = []
    for inv_id, grp_df in inv_ts_df.groupby("inverter_id"):
        mask   = grp_df["_candidate"].values.astype(bool)
        ts_arr = grp_df["ts"].values
        dwelled = _dwell_filter(mask, cfg.suppression_min_dwell)
        for i, ts_val in enumerate(ts_arr):
            if dwelled[i]:
                dwell_records.append({"ts": ts_val, "inverter_id": inv_id, "_s": True})

    if dwell_records:
        dwell_df = pd.DataFrame(dwell_records)
        dwell_df["ts"] = pd.to_datetime(dwell_df["ts"])
        check = df[["ts", "inverter_id"]].copy().reset_index()
        check["ts"] = pd.to_datetime(check["ts"])
        merged = check.merge(dwell_df, on=["ts", "inverter_id"], how="left")
        suppressed = (
            merged.set_index("index")["_s"]
            .reindex(df.index)
            .fillna(False)
            .values.astype(bool)
        )
    else:
        suppressed = zero.copy()

    # Lone underperform: string is individually low AND inverter consensus NOT met
    lone_low = bright & low_str & ~consensus_met
    underperform = lone_low & ~suppressed

    return suppressed, underperform


def _apply_vr_consensus(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Enforce cross-string consensus on CURT_VOLTAGE_RISE.

    If fewer than cfg.vr_consensus_min_strings strings on the same inverter
    show VR at a given timestamp, downgrade CURT_VOLTAGE_RISE to STRING_UNDERPERFORM.
    Strings on inverters with only one string in the data skip consensus (treated
    as STRING_UNDERPERFORM since there are no peers to confirm).
    """
    VR_FLAG = QUALITY_FLAGS["CURT_VOLTAGE_RISE"]
    SU_FLAG = QUALITY_FLAGS["STRING_UNDERPERFORM"]

    if "inverter_id" not in df.columns:
        return df

    qf     = df["qflag"].values.astype(np.int64).copy()
    vr_row = (qf & VR_FLAG) > 0
    if not vr_row.any():
        return df

    wk = pd.DataFrame({
        "ts":          df["ts"].values,
        "inverter_id": df["inverter_id"].values,
        "_vr":         vr_row.astype(np.int8),
        "_n_total":    1,
    }, index=df.index)

    grp = wk.groupby(["ts", "inverter_id"])
    vr_count    = grp["_vr"].transform("sum")
    string_count = grp["_n_total"].transform("sum")

    # No consensus: VR fires on fewer strings than required, or only 1 string on inverter
    no_consensus = (
        vr_row
        & ((vr_count.values < cfg.vr_consensus_min_strings)
           | (string_count.values <= 1))
    )

    qf[no_consensus] &= ~VR_FLAG
    qf[no_consensus] |= SU_FLAG

    df = df.copy()
    df["qflag"] = qf
    return df


# ---------------------------------------------------------------------------
# Per-string voltage-rise detector (unchanged from pre-Batch-3)
# ---------------------------------------------------------------------------

def detect_voltage_rise_curtailment(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    freq_min: float = 5.0,
) -> pd.DataFrame:
    """Detect soft curtailment caused by grid voltage rise (per-string).

    All five conditions must be simultaneously true:
      1. POA >= cfg.curt_vr_min_poa
      2. dV/dt >= cfg.curt_vr_vdc_rise_rate
      3. dP/dt <= cfg.curt_vr_pdc_flat_threshold
      4. dG/dt >= cfg.curt_vr_poa_falling_threshold
      5. V >= cfg.curt_vr_vdc_min_fraction * Voc_str_stc

    Does NOT flag rows that already carry CURT_STATE, CURT_STATISTICAL, or
    CURT_EXPORT_LIMIT.  Returns a copy; does not modify df in place.
    """
    df = df.copy()
    n  = len(df)
    if n == 0:
        return df

    qf  = df["qflag"].values.astype(np.int64).copy() if "qflag" in df.columns \
        else np.zeros(n, dtype=np.int64)

    poa = pd.to_numeric(df.get("POA", pd.Series([np.nan]*n, index=df.index)),
                        errors="coerce").fillna(0.0)
    V   = pd.to_numeric(df.get("V",   pd.Series([np.nan]*n, index=df.index)),
                        errors="coerce").fillna(0.0)
    P   = pd.to_numeric(df.get("P",   pd.Series([np.nan]*n, index=df.index)),
                        errors="coerce").fillna(0.0)

    win = max(2, int(round(cfg.curt_vr_window_min / freq_min)))

    dV_dt = V.diff().rolling(win, min_periods=2).mean() / freq_min
    dP_dt = P.diff().rolling(win, min_periods=2).mean() / freq_min
    dG_dt = poa.diff().rolling(win, min_periods=2).mean() / freq_min

    voc_str = cfg.module.voc_str_stc
    vdc_min = cfg.curt_vr_vdc_min_fraction * voc_str

    c1 = poa.values  >= cfg.curt_vr_min_poa
    c2 = dV_dt.values >= cfg.curt_vr_vdc_rise_rate
    c3 = dP_dt.values <= cfg.curt_vr_pdc_flat_threshold
    c4 = dG_dt.values >= cfg.curt_vr_poa_falling_threshold
    c5 = V.values    >= vdc_min

    not_already = (qf & _ALREADY_CURT) == 0
    vr_flag = c1 & c2 & c3 & c4 & c5 & not_already
    qf[vr_flag] |= QUALITY_FLAGS["CURT_VOLTAGE_RISE"]

    df["qflag"] = qf
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_curtailment(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    freq_min: float = 5.0,
    inverter_specs: dict | None = None,
    inverter_ac_power: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Add curtailment quality flags to df.

    When cfg.curtailment_inverter_level_enabled=True (default):
      - Statistical clipping detected on inverter AC power (adaptive plateau).
      - Suppression uses cross-string consensus; lone low strings → STRING_UNDERPERFORM.
      - Voltage-rise requires cross-string consensus on the same inverter.
      - If no measured inverter AC (C4 false), reconstructs from Σ(string DC)×efficiency.

    When cfg.curtailment_inverter_level_enabled=False:
      - Restores the pre-Batch-3 per-string behaviour exactly (kept for back-compat).

    Parameters
    ----------
    inverter_specs : dict | None
        From plant_meta["inverter_specs"]; keyed by inverter_id.
    inverter_ac_power : pd.Series | None
        From plant_meta["inverter_ac_power"]; MultiIndex (ts, inverter_id), kW.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df
    q = (df["qflag"].values.astype(np.int64).copy() if "qflag" in df.columns
         else np.zeros(n, dtype=np.int64))

    if not cfg.curtailment_inverter_level_enabled:
        # ---- LEGACY per-string path (unchanged from pre-Batch-3) ----
        poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
        P   = pd.to_numeric(df["P"],   errors="coerce").fillna(0).values

        p_cap = float(cfg.site.p_ac_max_kw) * 1000.0 / max(cfg.site.n_strings_per_inv, 1)
        high_poa = poa > 800
        near_cap = P > (1 - cfg.clip_band_pct) * p_cap
        statistical = high_poa & near_cap
        if statistical.any():
            diffs  = np.diff(statistical.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends   = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                if (e - s) >= cfg.clip_min_dwell:
                    q[s:e] |= QUALITY_FLAGS["CURT_STATISTICAL"]

        if "Pmp_exp" in df.columns:
            Pmp_exp = pd.to_numeric(df["Pmp_exp"], errors="coerce").fillna(0).values
        else:
            p_str_stc = float(cfg.site.p_ac_max_kw) * 1000.0 / max(cfg.site.n_strings_per_inv, 1)
            Pmp_exp   = p_str_stc * (poa / 1000.0)

        bright_sun  = poa > cfg.suppression_poa_threshold
        power_ratio = np.where(Pmp_exp > 10, P / Pmp_exp, 1.0)
        suppressed  = bright_sun & (power_ratio < cfg.suppression_power_ratio)
        if suppressed.any():
            diffs  = np.diff(suppressed.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends   = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                if (e - s) >= cfg.suppression_min_dwell:
                    q[s:e] |= QUALITY_FLAGS["CURT_SUPPRESSED"]

        df["qflag"] = q
        df = detect_voltage_rise_curtailment(df, cfg, freq_min=freq_min)
        return df

    # ---- NEW inverter-level path (Batch 3) ----

    # 1. Resolve inverter AC source
    inv_ac = inverter_ac_power
    inv_ac_source = "measured"
    if inv_ac is None or (hasattr(inv_ac, "__len__") and len(inv_ac) == 0):
        # SCHEMA-DEP C4: reconstruct from string DC × efficiency
        try:
            inv_ac = _reconstruct_inverter_ac(df, inverter_specs)
            inv_ac_source = "reconstructed_dc"
            if len(inv_ac) > 0:
                warnings.warn(
                    "Curtailment: no measured inverter AC (C4 false); "
                    "reconstructed from Σ(string DC)×efficiency — "
                    "clipping confidence is lower.",
                    stacklevel=3,
                )
        except Exception as exc:
            warnings.warn(f"Curtailment: AC reconstruction failed ({exc}); "
                          "skipping statistical clip detection.", stacklevel=3)
            inv_ac = pd.Series(dtype=float)

    # 2. Statistical clipping (adaptive plateau on inverter AC)
    try:
        stat_pairs, export_pairs = _detect_clip_inverter_level(inv_ac, cfg, freq_min)
        q = _apply_clip_flags(df, q, stat_pairs, export_pairs)
    except Exception as exc:
        warnings.warn(f"Curtailment: inverter-level clip detection failed ({exc}); "
                      "skipping statistical flags.", stacklevel=3)

    # 3. Suppression with cross-string consensus
    try:
        suppressed_arr, underperform_arr = _detect_suppression_consensus(df, cfg)
        q[suppressed_arr]   |= QUALITY_FLAGS["CURT_SUPPRESSED"]
        q[underperform_arr] |= QUALITY_FLAGS["STRING_UNDERPERFORM"]
    except Exception as exc:
        warnings.warn(f"Curtailment: suppression consensus failed ({exc}); "
                      "skipping suppression flags.", stacklevel=3)

    df["qflag"] = q

    # 4. Voltage-rise per string, then enforce consensus
    df = detect_voltage_rise_curtailment(df, cfg, freq_min=freq_min)
    try:
        df = _apply_vr_consensus(df, cfg)
    except Exception as exc:
        warnings.warn(f"Curtailment: VR consensus step failed ({exc}).", stacklevel=3)

    return df


# ---------------------------------------------------------------------------
# Summary + loss quantification (updated for new flags)
# ---------------------------------------------------------------------------

def curtailment_summary(df: pd.DataFrame, freq_min: float = 5.0) -> dict:
    """Summarise curtailment counts and estimated energy loss by type."""
    if len(df) == 0 or "qflag" not in df.columns:
        return dict(
            n_curt_state=0, n_curt_stat=0, n_curt_export_limit=0,
            n_curt_voltage_rise=0, n_string_underperform=0,
            n_curt_total=0, curt_pct=0.0,
            curt_hours_state=0.0, curt_hours_stat=0.0,
            curt_voltage_rise_pct=0.0, curt_voltage_rise_kwh=0.0,
            top_state_codes="",
        )

    qf  = df["qflag"].values.astype(np.int64)
    poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
    day = poa > 50
    day_n = max(int(day.sum()), 1)

    cs  = int(((qf & QUALITY_FLAGS["CURT_STATE"])        > 0).sum())
    ck  = int(((qf & QUALITY_FLAGS["CURT_STATISTICAL"])  > 0).sum())
    cel = int(((qf & QUALITY_FLAGS["CURT_EXPORT_LIMIT"]) > 0).sum())
    vr  = int(((qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0).sum())
    su  = int(((qf & QUALITY_FLAGS["STRING_UNDERPERFORM"])> 0).sum())

    _all_curt = (QUALITY_FLAGS["CURT_STATE"]
                 | QUALITY_FLAGS["CURT_STATISTICAL"]
                 | QUALITY_FLAGS["CURT_EXPORT_LIMIT"]
                 | QUALITY_FLAGS["CURT_VOLTAGE_RISE"])
    tot = int((((qf & _all_curt) > 0) & day).sum())

    state_codes = []
    if "inverter_state" in df.columns:
        curt_mask = (qf & QUALITY_FLAGS["CURT_STATE"]) > 0
        if curt_mask.any():
            sc = pd.Series(df.loc[curt_mask, "inverter_state"]).value_counts().head(3)
            state_codes = [f"{int(k)}:{STATE_NAME.get(int(k),'?')}({v})"
                           for k, v in sc.items()]

    h = freq_min / 60.0

    vr_kwh = 0.0
    vr_mask = (qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0
    if vr_mask.any() and "P" in df.columns:
        P_obs = pd.to_numeric(df["P"], errors="coerce").fillna(0).values
        P_exp = (pd.to_numeric(df["Pmp_exp"], errors="coerce").fillna(0).values
                 if "Pmp_exp" in df.columns else P_obs)
        vr_kwh = float((np.maximum(P_exp - P_obs, 0.0)[vr_mask] / 1000.0 * h).sum())

    return dict(
        n_curt_state=cs, n_curt_stat=ck, n_curt_export_limit=cel,
        n_curt_voltage_rise=vr, n_string_underperform=su,
        n_curt_total=tot,
        curt_pct=100.0 * tot / day_n,
        curt_hours_state=cs * h, curt_hours_stat=ck * h,
        curt_voltage_rise_pct=100.0 * vr / day_n,
        curt_voltage_rise_kwh=vr_kwh,
        top_state_codes=", ".join(state_codes),
    )


def quantify_curtailment_loss(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    freq_min: float = 5.0,
) -> dict:
    """Compute lost kWh and PKR due to curtailment, split by type.

    STRING_UNDERPERFORM is excluded from curtailment loss — it goes to the
    fault classifier's loss accounting, not here.
    The legacy key ``total_curt_kwh`` is kept as an alias.
    """
    n = len(df)
    if n == 0:
        return _empty_curt_loss(cfg)

    qf  = df["qflag"].values.astype(np.int64)
    poa = df["POA"].fillna(0).values

    if "Pmp_exp" not in df.columns:
        plate    = cfg.module
        Gn       = poa / 1000.0
        Tc       = df.get("T_module", pd.Series(25.0, index=df.index)).fillna(25).values
        Pmp_exp_w = plate.pmp_str_stc * Gn * (1 + plate.gamma_pmp * (Tc - 25))
    else:
        Pmp_exp_w = df["Pmp_exp"].fillna(0).values

    P_obs_w = df["P"].fillna(0).values
    dt_h    = freq_min / 60.0

    def _loss_kwh(mask: np.ndarray) -> tuple[pd.Series, float]:
        daylight = poa > 100
        active   = mask & daylight
        dP_w  = np.where(active, np.maximum(Pmp_exp_w - P_obs_w, 0.0), 0.0)
        dP_kw = pd.Series(dP_w / 1000.0, index=df.index)
        return dP_kw, float(dP_kw.sum() * dt_h)

    mask_state  = (qf & QUALITY_FLAGS["CURT_STATE"])        > 0
    mask_stat   = (qf & QUALITY_FLAGS["CURT_STATISTICAL"])  > 0
    mask_export = (qf & QUALITY_FLAGS["CURT_EXPORT_LIMIT"]) > 0
    mask_vr     = (qf & QUALITY_FLAGS["CURT_VOLTAGE_RISE"]) > 0
    mask_all    = mask_state | mask_stat | mask_export | mask_vr

    dP_state,  kwh_state  = _loss_kwh(mask_state)
    _,         kwh_stat   = _loss_kwh(mask_stat)
    _,         kwh_export = _loss_kwh(mask_export)
    _,         kwh_vr     = _loss_kwh(mask_vr)
    dP_all,    kwh_total  = _loss_kwh(mask_all)

    revenue_pkr = kwh_total * cfg.site.tariff

    ts       = pd.to_datetime(df["ts"])
    ts_naive = ts.dt.tz_convert(None) if getattr(ts.dt, "tz", None) else ts
    dates    = ts_naive.dt.date
    daily_kwh = (dP_all * dt_h).groupby(dates).sum()

    period_days    = max((ts_naive.max() - ts_naive.min()).days, 1)
    annualised_kwh = kwh_total / period_days * 365.0
    annualised_pkr = revenue_pkr / period_days * 365.0

    n_curt = int(mask_all.sum())
    expl = (f"{n_curt} curtailed daylight samples -> "
            f"{kwh_total:.2f} kWh deficit in {period_days} days -> "
            f"{cfg.site.currency} {revenue_pkr:,.0f} "
            f"[state={kwh_state:.2f} stat={kwh_stat:.2f} "
            f"export={kwh_export:.2f} vr={kwh_vr:.2f}]")

    return dict(
        per_row_dP_kw=dP_all,
        curtailment_loss_state_kwh=float(kwh_state),
        curtailment_loss_statistical_kwh=float(kwh_stat),
        curtailment_loss_export_limit_kwh=float(kwh_export),
        curtailment_loss_voltage_rise_kwh=float(kwh_vr),
        curtailment_loss_total_kwh=float(kwh_total),
        # Legacy alias
        total_curt_kwh=float(kwh_total),
        total_curt_pkr=float(revenue_pkr),
        n_curt_intervals=n_curt,
        daily_curt_kwh=daily_kwh,
        period_days=int(period_days),
        annualised_kwh=float(annualised_kwh),
        annualised_pkr=float(annualised_pkr),
        method="plate_pmp_minus_observed",
        explainability=expl,
    )


def _empty_curt_loss(cfg):
    return dict(
        per_row_dP_kw=pd.Series(dtype=float),
        curtailment_loss_state_kwh=0.0,
        curtailment_loss_statistical_kwh=0.0,
        curtailment_loss_export_limit_kwh=0.0,
        curtailment_loss_voltage_rise_kwh=0.0,
        curtailment_loss_total_kwh=0.0,
        total_curt_kwh=0.0,
        total_curt_pkr=0.0,
        n_curt_intervals=0,
        daily_curt_kwh=pd.Series(dtype=float),
        period_days=0,
        annualised_kwh=0.0,
        annualised_pkr=0.0,
        method="plate_pmp_minus_observed",
        explainability="no data",
    )
