"""Age-based NCI baseline (NREL Jordan 2016 + LID first-year ramp)."""
from __future__ import annotations
from datetime import date
import pandas as pd
from .constants import TECH_DEFAULTS


def degradation_baseline(commissioning_date, reference_date,
                         technology: str = "mono-c-Si",
                         override_rate: float | None = None,
                         override_lid:  float | None = None,
                         floor: float = 0.70) -> dict:
    """Return baseline NCI factor (≤1) at reference_date.
    LID ramps up over the first year; then linear annual rate."""
    if commissioning_date is None or reference_date is None:
        return dict(baseline=1.0, years=0.0, lid_applied=0.0,
                    deg_applied=0.0, technology=technology,
                    annual_rate=override_rate, lid_rate=override_lid,
                    floor=floor, commissioning_date=str(commissioning_date),
                    reference_date=str(reference_date),
                    note="no commissioning date")
    d0 = pd.to_datetime(commissioning_date).date()
    d1 = pd.to_datetime(reference_date).date()
    years = (d1 - d0).days / 365.25
    if years < 0: years = 0

    rec = TECH_DEFAULTS.get(technology, TECH_DEFAULTS["mono-c-Si"])
    annual = float(override_rate if override_rate is not None else rec["annual_degradation"])
    lid    = float(override_lid  if override_lid  is not None else rec["lid_loss"])

    lid_applied = lid * min(years, 1.0)
    deg_applied = annual * max(years - 1.0, 0.0) if years > 1.0 else 0.0
    baseline = max(1.0 - lid_applied - deg_applied, floor)

    return dict(baseline=float(baseline), years=float(years),
                lid_applied=float(lid_applied), deg_applied=float(deg_applied),
                technology=technology, annual_rate=annual, lid_rate=lid,
                floor=floor, commissioning_date=str(d0),
                reference_date=str(d1),
                note=f"{technology}: {years:.2f} yr, LID {lid_applied*100:.2f}%, "
                     f"linear deg {deg_applied*100:.2f}%, baseline {baseline:.3f}")


def explain_baseline(b: dict) -> str:
    return (f"age={b['years']:.2f} yr, LID={b['lid_applied']*100:.2f}%, "
            f"deg={b['deg_applied']*100:.2f}%, baseline NCI={b['baseline']:.3f}")
