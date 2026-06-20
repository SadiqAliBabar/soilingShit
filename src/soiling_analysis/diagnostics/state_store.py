"""Persistent last-good state store (Batch 7).

A small JSON store keyed by string_uid holding the last-good clean baseline
value, SDM parameters, and a timestamp.  Read at the start of a run, written
after successful completion.

Design principles
-----------------
- Never blocks a run: missing / corrupt store → warn and continue silently.
- Path is configurable via PipelineConfig.state_store_path ("" disables disk
  persistence; the store still works in-memory for within-run tier-3 usage).
- Only tier-1 and tier-2 results are written back (trust only trustworthy refs).
- Format: JSON dict with a top-level ``strings`` key keyed by string_uid.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_SCHEMA_VERSION = 1


class StateStore:
    """Read/write the last-good JSON store.

    Usage
    -----
    >>> store = StateStore(cfg.state_store_path)
    >>> store.load()
    >>> rec = store.get_string("INV01_STR01")   # None if first run
    >>> store.update_if_better("INV01_STR01", baseline=0.975, tier=1)
    >>> store.save()
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load from disk.  Silently skips if missing or corrupt."""
        if self._path is None:
            return
        try:
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    self._data = raw.get("strings", {})
        except Exception as exc:
            warnings.warn(f"[B7] StateStore.load failed ({exc}); starting with empty store.")
            self._data = {}

    def save(self) -> None:
        """Write to disk.  Silently skips if no path is configured."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "saved_at": datetime.utcnow().isoformat(),
                "strings": self._data,
            }
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
        except Exception as exc:
            warnings.warn(
                f"[B7] StateStore.save failed ({exc}); last-good values not persisted."
            )

    def get_string(self, uid: str) -> Optional[Dict[str, Any]]:
        """Return the last-good record for *uid*, or None if not yet stored."""
        return self._data.get(uid)

    def set_string(self, uid: str, record: Dict[str, Any]) -> None:
        """Unconditionally replace the record for *uid*."""
        self._data[uid] = record

    def update_if_better(
        self,
        uid: str,
        baseline: float,
        sdm_params: Optional[Dict] = None,
        tier: int = 1,
        timestamp: Optional[str] = None,
    ) -> None:
        """Persist a new last-good record only from a high-quality tier (1 or 2).

        Tier-3+ values are held values themselves and must never overwrite the
        stored record, to avoid a circular reference (held → stored → held...).
        """
        if tier > 2:
            return
        self._data[uid] = {
            "baseline": float(baseline),
            "sdm_params": sdm_params or {},
            "tier": tier,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """True when a disk path is configured (persistence is enabled)."""
        return self._path is not None

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._data)
        return f"StateStore(path={self._path!r}, n_strings={n})"
