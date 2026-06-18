"""Per-row quality flagging."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .constants import (QUALITY_FLAGS, FAULT_STATES, STANDBY_STATES,
    IV_SCAN_STATES, TRANSIENT_STATES, CURTAILED_STATES)


def flag_data_quality(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    q = np.zeros(n, dtype=np.int64)

    poa = pd.to_numeric(df.get("POA", pd.Series([np.nan]*n)), errors="coerce").values
    V   = pd.to_numeric(df.get("V",   pd.Series([np.nan]*n)), errors="coerce").values
    I   = pd.to_numeric(df.get("I",   pd.Series([np.nan]*n)), errors="coerce").values
    P   = pd.to_numeric(df.get("P",   pd.Series([np.nan]*n)), errors="coerce").values
    st  = pd.to_numeric(df.get("inverter_state", pd.Series([-1]*n)),
                        errors="coerce").fillna(-1).astype(int).values

    plate = cfg.module
    v_min, v_max = 0.5 * plate.voc_str_stc, 1.15 * plate.voc_str_stc
    i_max = 1.20 * plate.isc_stc

    q |= np.where(poa < 50, QUALITY_FLAGS["NIGHT"], 0).astype(np.int64)
    q |= np.where(poa < 100, QUALITY_FLAGS["G_LOW"], 0).astype(np.int64)
    q |= np.where(np.isnan(V) | np.isnan(I) | np.isnan(P),
                  QUALITY_FLAGS["COMMS_GAP"], 0).astype(np.int64)
    with np.errstate(invalid='ignore'):
        v_oor = (~np.isnan(V)) & ((V < v_min) | (V > v_max))
        i_oor = (~np.isnan(I)) & ((I < 0) | (I > i_max))
        p_neg = (~np.isnan(P)) & (P < -10)
    q |= np.where(v_oor, QUALITY_FLAGS["V_OUT_OF_RANGE"], 0).astype(np.int64)
    q |= np.where(i_oor, QUALITY_FLAGS["I_OUT_OF_RANGE"], 0).astype(np.int64)
    q |= np.where(p_neg, QUALITY_FLAGS["P_NEG"], 0).astype(np.int64)

    is_fault = np.isin(st, list(FAULT_STATES))
    is_stby  = np.isin(st, list(STANDBY_STATES))
    is_iv    = np.isin(st, list(IV_SCAN_STATES))
    is_tr    = np.isin(st, list(TRANSIENT_STATES))
    is_curt_st = np.isin(st, list(CURTAILED_STATES))
    q |= np.where(is_fault, QUALITY_FLAGS["INVERTER_FAULT"], 0).astype(np.int64)
    q |= np.where(is_stby,  QUALITY_FLAGS["STANDBY"],        0).astype(np.int64)
    q |= np.where(is_iv,    QUALITY_FLAGS["IV_SCAN"],        0).astype(np.int64)
    q |= np.where(is_tr,    QUALITY_FLAGS["TRANSIENT"],      0).astype(np.int64)
    q |= np.where(is_curt_st, QUALITY_FLAGS["CURT_STATE"],   0).astype(np.int64)

    df["qflag"] = q
    return df
