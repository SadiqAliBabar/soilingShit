"""Calendar-aware daily-series reindexing (Batch 2).

Exposes one public function: ``to_calendar_grid``.

Design rules (from playbook):
- Insert NaN rows for missing days; do NOT forward-fill NCI.
- Mark every row with ``is_present`` (True = real data, False = gap day).
- ``n_valid`` → 0 and ``rain_mm`` → 0.0 for gap days so downstream
  arithmetic is safe without further NaN guards.
- All other metric columns (NCI_noon, etc.) stay NaN for gap days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_calendar_grid(daily_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Reindex *daily_df* onto a continuous calendar grid.

    Parameters
    ----------
    daily_df : DataFrame
        Output of ``compute_daily_metrics`` — one row per date that has data.
        Must have a ``date`` column (Python date or datetime-like).
    freq : str
        Pandas frequency string for the grid; ``"D"`` (daily) is the only
        supported value for now.

    Returns
    -------
    DataFrame
        One row per calendar day from min(date) to max(date), with
        ``is_present=True`` for original rows and ``is_present=False`` for
        inserted gap rows.  Column order and dtypes are preserved for the
        original columns; gap rows have NaN for every metric except
        ``n_valid`` (0) and ``rain_mm`` (0.0).
    """
    if daily_df is None or len(daily_df) == 0:
        return daily_df if daily_df is not None else pd.DataFrame()

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_present"] = True

    df = df.set_index("date").sort_index()

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.index.name = "date"

    # Gap rows: is_present=False, safe zero-fills for counters
    # Cast to numpy bool so that ~ works correctly (Python bool in object-dtype columns
    # returns bitwise int NOT which is -2/-1, not True/False).
    df["is_present"] = df["is_present"].fillna(False).astype(bool)
    df["n_valid"]    = df["n_valid"].fillna(0).astype(int)
    df["rain_mm"]    = df["rain_mm"].fillna(0.0)
    # NCI metric columns intentionally left as NaN — do NOT forward-fill

    df = df.reset_index()
    # Keep date as Python date objects (consistent with the non-grid path)
    df["date"] = df["date"].dt.date
    return df


def valid_day_density(
    is_present: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    """Return present_days / window_days for a closed [start, end] slice."""
    window_days = end_idx - start_idx + 1
    if window_days <= 0:
        return 0.0
    n_present = int(np.sum(is_present[start_idx: end_idx + 1]))
    return n_present / window_days
