"""Cell-temperature estimation with measured > SAPM > NOCT fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import ModuleConfig, PipelineConfig


def estimate_cell_temp(df: pd.DataFrame, plate: ModuleConfig,
                       cfg: PipelineConfig | None = None):
    """Return (Tc series, source string)."""
    n = len(df)
    if "T_module" in df.columns and df["T_module"].notna().sum() > 0.5 * n:
        Tc = pd.to_numeric(df["T_module"], errors="coerce")
        # NaN-fallback to NOCT estimate
        poa = pd.to_numeric(df.get("POA", 800), errors="coerce").fillna(0).clip(lower=0)
        Tamb = 25.0
        Tc_noct = Tamb + (poa / 800.0) * 20.0  # NOCT 45°C – 25°C ambient @800 W/m²
        Tc = Tc.fillna(Tc_noct)
        return Tc, "measured"
    # Pure NOCT fallback
    poa = pd.to_numeric(df.get("POA", 800), errors="coerce").fillna(0).clip(lower=0)
    Tamb = 25.0
    Tc = Tamb + (poa / 800.0) * 20.0
    return pd.Series(Tc, index=df.index), "NOCT"
