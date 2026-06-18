# CLAUDE.md — pv_diag Adaptive Patch

## Project Overview

**Project**: PV String Diagnostics Pipeline
**Type**: Data analysis / Solar analytics Python package
**Purpose**: End-to-end diagnostics for utility/C&I solar plants using inverter-string telemetry (Huawei state-code aware). Detects and quantifies curtailment loss, soiling, degradation, and classifies string health.

## Key Dependencies

- **pandas** (>=3.0.2) — Data manipulation
- **numpy** (>=2.4.4) — Numerical computing
- **scipy** (>=1.17.1) — Scientific computing (trimmed linear regression for soiling)
- **scikit-learn** (>=1.8.0) — Clustering (MPPT × orientation)
- **matplotlib** (>=3.10.9) — Visualization
- **openpyxl** (>=3.1.5) — Excel export
- **pvlib** (>=0.15.1) — Single-diode model fitting (optional, graceful fallback if unavailable)
- **pymongo** (>=4.17.0) — MongoDB integration

## Project Structure

```
pv_diag_adaptive_patch/
├── run.py                    # CLI entry point
├── pv_diag/                  # Main package
│   ├── __init__.py
│   ├── config.py             # Dataclasses: ModuleConfig, SiteConfig, PlantConfig, PipelineConfig
│   ├── constants.py          # Huawei state codes, quality bit-mask, Pakistan defaults
│   ├── utils.py              # _safe_id, _is_ok, _scalar (Excel-safe)
│   ├── ingestion.py          # load_plant_data → long_df + plant_meta
│   ├── quality.py            # Per-row quality bit-mask
│   ├── curtailment.py       # Detect + quantify curtailment loss (kWh + PKR)
│   ├── sufficiency.py        # Good / Limited / Poor / Skipped gate
│   ├── celltemp.py           # Measured > NOCT fallback chain
│   ├── plate.py              # Conservative nameplate inference
│   ├── orientation.py        # Solar position, clear-sky POA, az/tilt clustering
│   ├── degradation.py       # NREL Jordan 2016 + LID first-year ramp baseline
│   ├── sdm.py               # Single-diode fit (pvlib-aware, graceful fallback)
│   ├── daily.py              # PR, NCI, NCI_corrected, AM/PM asymmetry
│   ├── wash_detect.py       # Rain/wash step detection w/ Full/Partial/Minimal recovery
│   ├── soiling.py            # Segment-aware trimmed-LR slope (%/day)
│   ├── transient.py          # Single-day dip detector
│   ├── classification.py     # Multi-axis verdict (current-segment aware)
│   ├── losses.py             # Soiling + curtailment loss aggregation
│   ├── clustering.py         # MPPT × (azimuth, tilt) combined clusters
│   ├── pipeline.py           # End-to-end orchestrator + joblib parallelism
│   ├── plotting.py           # Soiling dashboard, IV diagnostics, data-quality, overview
│   ├── excel_export.py      # 16-sheet xlsx report
│   └── adaptive_baseline.py # NEW: Adaptive baseline for soiling detection
├── data/                     # Demo data and test datasets
└── outputs/                  # Generated reports and figures
```

## Running the Pipeline

```bash
# Basic run
python run.py data/demo_data.xlsx --out-dir outputs

# With options
python run.py data/demo_data.xlsx --out-dir outputs --cluster-method combined --n-jobs 4
python run.py data/demo_data.xlsx --out-dir outputs --no-figures --quiet
```

## Input Format

**Main "Data" sheet** — columns from DataTemplate.xlsx:
```
timestamp, plant, inverter_id, mppt_id, string_id, irradiance KW/m2,
voltage_u, current_i, power kw, pv_temperature, pv_Capacity,
inverter_state, azimuth, tilt
```

**Optional "Metadata" sheet** (key/value pairs):
- plant_name, latitude, longitude, tariff_pkr_kwh, commissioning_date, p_ac_max_kw, technology

Missing values fall back to Pakistan defaults (Lahore: lat 31.4504, lon 73.1350, azimuth 180, tilt 25, tariff 38 PKR/kWh).

## Output

**Excel**: `<input_stem>_diagnostics.xlsx` — 16 sheets:
- 00_Run_Summary, 01_Plant_Metadata, 02_Data_Sufficiency, 03_Curtailment_Detection,
- 04_Curtailment_Loss, 05_Plate_Inference, 06_SDM_Fits, 07_Daily_Metrics,
- 08_Degradation_Baseline, 09_Wash_Events, 10_Soiling_Trends, 11_Orientation_Clusters,
- 12_Classification, 13_Losses, 14_Transient_Events, 15_Per_String_Detail

**Figures** (in `figures/` folder):
- `soiling_dashboard__<string>.png` — Daily NCI with wash markers + segments
- `iv_diagnostics__<string>.png` — Measured I-V cloud + SDM @STC
- `data_quality__<string>.png` — Stacked daily quality bars
- `plant_overview.png` — Verdict pie + per-string loss bars

## Key Concepts

### Soiling Detection
- Uses trimmed linear regression on daily Normalized Capacity Index (NCI)
- Segment-aware (splits data around wash events)
- Output: slope in %/day soiling loss

### Curtailment Detection
- Huawei state code analysis (state codes 1-4, 247, 65535)
- Quantifies kWh and PKR lost per string

### Classification
- Multi-axis verdict: Soiling (Mod/Heavy) / Curtailment / Degradation / Healthy
- Current-segment aware (uses recent days for wash-aware verdicts)

### Adaptive Baseline (NEW)
- `adaptive_baseline.py` — Adaptive baseline for soiling detection
- Handles changing operating conditions over time

## Development Notes

- pvlib is optional. If unavailable, SDM fits marked `pvlib_unavailable`, degradation defaults to soiling diagnosis
- Uses joblib for parallel processing (`cfg.n_jobs`)
- Tariff defaults to 38 PKR/kWh (Pakistan industrial B-3)
- Commissioning date defaults to 2023-01-01 if not provided