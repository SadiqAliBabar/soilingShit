"""Small helpers: safe IDs, OK mask, scalar coercion, NCI column picker."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .constants import DISQUALIFYING


def _safe_id(s) -> str:
    s = str(s)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "x"


def _is_ok(qflag) -> np.ndarray:
    q = np.asarray(qflag, dtype=np.int64)
    return (q & DISQUALIFYING) == 0


def _short_label(label: str, max_len: int = 24) -> str:
    return label if len(label) <= max_len else label[:max_len-2] + ".."


def _scalar(v):
    """Coerce any value to something openpyxl can write."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    if isinstance(v, np.generic):
        x = v.item()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x
    if isinstance(v, (pd.Timestamp,)):
        return v.to_pydatetime()
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except Exception:
        return str(v)


def safe_pct(num, den, default=0.0) -> float:
    try:
        return float(num) / float(den) * 100.0 if float(den) > 0 else default
    except Exception:
        return default


def coerce_date(x, fallback=None):
    """Coerce to datetime.date, falling back to `fallback` on failure."""
    from datetime import date, datetime
    if x is None:
        return fallback
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return fallback


def pick_nci_column(df: pd.DataFrame) -> str:
    """Return the preferred NCI noon column present in *df*.

    Priority (highest to lowest):
        NCI_adaptive_noon  — adaptive per-string / cluster reference (Pass 2)
        NCI_relative_noon  — physics IAM-corrected reference (Pass 1 fallback, Prompt 3)
        NCI_corrected_noon — plate-age-corrected reference
        NCI_noon           — raw nameplate reference (always present)

    A column is accepted only when it contains at least one finite value;
    otherwise the next candidate is tried.
    """
    for col in ("NCI_adaptive_noon", "NCI_relative_noon",
                "NCI_corrected_noon", "NCI_noon"):
        if col in df.columns:
            n_finite = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if n_finite >= 1:
                return col
    # Ultimate fallback — return NCI_noon even if all-NaN
    return "NCI_noon"
