"""Input layer for the soiling analysis.

Two inputs feed the analysis:

1. ``RequireInputData.xlsx`` — the static spec workbook (Plant / Inverter /
   Panel sheets). Authored by hand in Excel, but normalized *once* into a
   typed, nested dict (the "JSON internally" layer) by :mod:`specs`.
2. Two CSVs exported from MongoDB by ``soiling-fetch``:
   - ``*_string.csv``   — string-level time-series including irradiance and pv_temperature
   - ``*_inverter.csv`` — inverter-level time-series for state and AC power

:func:`specs.load_specs` is the entry point for layer 1.
:func:`~soiling_analysis.loader.load_from_sources` is the entry point for layer 2.
"""

from .specs import load_specs, specs_to_json

__all__ = [
    "load_specs",
    "specs_to_json",
]
