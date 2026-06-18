"""Detect single-day anomalous dips (not soiling)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig


def detect_transient_events(daily_df, cfg: PipelineConfig) -> pd.DataFrame:
    if daily_df is None or len(daily_df) == 0:
        return pd.DataFrame(columns=["date","NCI_corrected_noon",
                "rolling_median","rolling_iqr","z_score","cause"])
    col = ("NCI_corrected_noon" if "NCI_corrected_noon" in daily_df.columns
           else "NCI_noon")
    d = daily_df.copy().sort_values("date").reset_index(drop=True)
    s = pd.to_numeric(d[col], errors="coerce")
    win = max(int(cfg.transient_rolling_days), 3)
    rmed = s.rolling(win, min_periods=3, center=True).median()
    rq1 = s.rolling(win, min_periods=3, center=True).quantile(0.25)
    rq3 = s.rolling(win, min_periods=3, center=True).quantile(0.75)
    iqr = (rq3 - rq1).abs()

    dip_thr = float(cfg.transient_dip_threshold)
    mask = ((s < dip_thr * rmed) & (s < (rmed - 2 * iqr))).fillna(False)

    out = d.loc[mask, ["date"]].copy()
    out["NCI_corrected_noon"] = s[mask].values
    out["rolling_median"]     = rmed[mask].values
    out["rolling_iqr"]        = iqr[mask].values
    out["z_score"] = ((s[mask] - rmed[mask]) / iqr[mask].replace(0, np.nan)).values
    out["cause"] = np.where((s[mask] < 0.4 * rmed[mask]).values,
                            "Severe transient", "Moderate transient")
    return out.reset_index(drop=True)
