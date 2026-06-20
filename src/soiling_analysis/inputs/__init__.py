"""Input layer for the soiling analysis.

Two inputs feed the analysis:

1. ``RequireInputData.xlsx`` — the static spec workbook (Plant / Inverter /
   Panel sheets). Authored by hand in Excel, but normalized *once* into a
   typed, nested dict (the "JSON internally" layer) by :mod:`specs`.
2. The measured time-series CSV exported from MongoDB (all levels in one file,
   tagged by a ``level`` column).

:func:`specs.load_specs` is the entry point for layer 1.
"""

from .specs import load_specs, specs_to_json

__all__ = [
    "load_specs",
    "specs_to_json",
]
