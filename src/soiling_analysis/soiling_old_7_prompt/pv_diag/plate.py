"""Plate-parameter inference — conservative.

CRITICAL: plate values represent NEW-MODULE NAMEPLATE at STC.  We do NOT
override Imp/Isc from observations (would force NCI≈1 regardless of soil).
We also do NOT override Vmp from observations (operating-temperature Vmp
is below STC Vmp by beta_voc × dT, and treating it as STC mis-scales Imp).

Strategy:
- If `pv_capacity` column present (preferred): trust it as Pmp_str_stc;
  derive Imp_stc = Pmp / (Vmp_stc_default × n_modules).
- Otherwise keep cfg.module defaults entirely.
- Voc_stc may be lightly updated from observed P99 V (voltage degrades
  slowly and is largely insensitive to soiling).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import ModuleConfig, PipelineConfig
from .utils import _is_ok


def infer_plate_params(plant_data, default_plate: ModuleConfig,
                       cfg: PipelineConfig) -> dict:
    plate = ModuleConfig(**default_plate.__dict__)
    notes = []
    cap_vals, voc_obs = [], []

    for label, df in plant_data.items():
        if "pv_capacity" in df.columns:
            v = pd.to_numeric(df["pv_capacity"], errors="coerce").dropna()
            if len(v) > 0:
                cap_vals.append(float(v.iloc[0]))     # kW per string
        if "qflag" not in df.columns: continue
        ok = _is_ok(df["qflag"].values)
        poa = pd.to_numeric(df["POA"], errors="coerce").fillna(0).values
        clean = ok & (poa > 800)
        if clean.sum() < 30: continue
        sub = df.loc[clean]
        if "V" in sub.columns:
            voc_obs.append(np.percentile(sub["V"].dropna(), 99))

    # DISABLED: Global Imp inference based on fixed n_modules breaks
    # sites with variable string lengths (like Coca Cola Faisalabad).
    # We will derive n_modules per-string in the pipeline instead.
    # if cap_vals:
    #     pmp_str_kw = float(np.median(cap_vals))
    #     # Use DEFAULT Vmp to back out Imp; don't trust observed Vmp because
    #     # it's at operating temperature (below STC).
    #     plate.imp_stc = pmp_str_kw * 1000.0 / (plate.vmp_stc * plate.n_modules)
    #     plate.isc_stc = plate.imp_stc / 0.945
    #     notes.append(f"pv_capacity={pmp_str_kw:.2f}kW -> Imp_stc={plate.imp_stc:.2f}A "
    #                  f"(Vmp_stc kept at default {plate.vmp_stc:.2f}V)")


    if voc_obs:
        voc_str_obs = float(np.median(voc_obs))
        # Convert operating-T Voc to STC Voc (rough). Voc at T_op ≈ Voc_stc*(1 + beta*dT)
        # Assume average operating Tc ≈ 45°C, dT=20, beta=-0.0027 → factor 0.946
        voc_stc_estimated = voc_str_obs / 0.946 / plate.n_modules
        if voc_stc_estimated > plate.voc_stc * 0.85:    # sanity
            plate.voc_stc = voc_stc_estimated
            notes.append(f"voc_stc from observed P99/temp-corrected -> {plate.voc_stc:.2f}V/module")

    return dict(plate=plate, notes="; ".join(notes) or "defaults retained",
                n_strings_used=len(voc_obs))


def estimate_cells_in_series(long_df: pd.DataFrame,
                             plate: ModuleConfig) -> int:
    if "V" not in long_df.columns: return plate.cells_in_series
    v98 = float(np.nanpercentile(pd.to_numeric(long_df["V"], errors="coerce"), 98))
    return int(v98 / 0.6) if v98 > 5 else plate.cells_in_series
