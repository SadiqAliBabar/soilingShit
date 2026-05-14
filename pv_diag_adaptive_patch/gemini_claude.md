# PV String Diagnostics Pipeline (pv_diag)

## 1. Project Overview

**Project Name**: PV String Diagnostics Pipeline
**Module Name**: `pv_diag` (Adaptive Patch)
**Type**: Data Analysis / Solar Analytics Python Package
**Domain**: Renewable Energy / Solar PV Analytics

The `pv_diag` pipeline provides end-to-end, automated diagnostics for utility-scale and Commercial & Industrial (C&I) solar photovoltaic (PV) plants. Leveraging inverter-string telemetry—specifically tailored for Huawei inverter state codes—the system detects, quantifies, and classifies string health. Key capabilities include detecting soiling losses, evaluating module degradation, identifying physical curtailment, and generating comprehensive actionable reports.

## 2. Architecture & Data Flow

The system is designed as a modular, parallelizable batch-processing pipeline.

1. **Ingestion & Preprocessing**: Raw telemetry (typically 5-minute interval data) and optional metadata are loaded. Missing geographic or configuration data falls back to regional defaults (e.g., Pakistan standard settings).
2. **Quality Assurance**: Data is filtered using a rigorous per-row quality bit-mask, excluding invalid measurements (e.g., negative irradiance, offline states).
3. **Physical Modeling (SDM)**: Utilizes the Single-Diode Model (via `pvlib` when available) to establish the theoretical baseline performance under Standard Test Conditions (STC).
4. **Metric Calculation**: Computes the daily Normalized Capacity Index (NCI) and Performance Ratio (PR) with AM/PM asymmetry corrections.
5. **Event Detection**: 
    - **Curtailment**: Analyzes inverter state codes to quantify energy (kWh) and revenue (PKR) lost to grid curtailment.
    - **Wash Detection**: Identifies rain/wash events and categorizes recovery as Full, Partial, or Minimal.
    - **Soiling**: Applies segment-aware, trimmed linear regression on the NCI to calculate soiling degradation slopes (%/day).
    - **Degradation**: Applies NREL Jordan 2016 baseline and Light-Induced Degradation (LID) first-year ramp corrections.
6. **Classification & Aggregation**: Strings are clustered (by MPPT and orientation) and assigned a multi-axis health verdict (e.g., Healthy, Mod. Soiling, Curtailment).
7. **Reporting**: Generates a 16-sheet Excel diagnostic report and detailed string-level visualization dashboards.

## 3. Technology Stack & Dependencies

The project relies on a robust scientific Python stack, managed via `uv`.

- **Core Data Processing**: 
  - `pandas` (>=3.0.2): Fast, flexible data manipulation.
  - `numpy` (>=2.4.4): Core numerical computing.
- **Scientific Computing & Machine Learning**: 
  - `scipy` (>=1.17.1): Statistical functions (e.g., trimmed linear regression).
  - `scikit-learn` (>=1.8.0): Unsupervised clustering (MPPT × orientation grouping).
- **Domain-Specific**:
  - `pvlib` (>=0.15.1): Advanced PV modeling (Single-Diode Model fitting). *Note: The pipeline degrades gracefully if `pvlib` is unavailable.*
- **Visualization & Export**:
  - `matplotlib` (>=3.10.9): Plotting and dashboard generation.
  - `openpyxl` (>=3.1.5): Multi-sheet Excel workbook creation.
- **Database**:
  - `pymongo` (>=4.17.0): MongoDB integration for scalable data persistence.

## 4. Directory Structure

```text
pv_diag_adaptive_patch/
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Deterministic dependency lockfile (managed by uv)
├── run.py                    # CLI entry point for executing the pipeline
├── pv_diag/                  # Core library module
│   ├── __init__.py
│   ├── config.py             # Strongly-typed configuration (Dataclasses)
│   ├── constants.py          # State codes, quality bit-masks, regional defaults
│   ├── utils.py              # Helper functions (safe IDs, Excel sanitization)
│   ├── ingestion.py          # Data loaders and metadata resolution
│   ├── quality.py            # Data quality gating and masking
│   ├── curtailment.py        # Curtailment detection and financial loss quantification
│   ├── sufficiency.py        # Data volume gating (Good/Limited/Poor)
│   ├── celltemp.py           # Module temperature estimation and NOCT fallbacks
│   ├── plate.py              # Nameplate capacity inference
│   ├── orientation.py        # Solar geometry, clear-sky modeling, orientation clustering
│   ├── degradation.py        # Expected baseline degradation modeling
│   ├── sdm.py                # Single-Diode Model fitting and IV-curve analysis
│   ├── daily.py              # Daily aggregation (PR, NCI, asymmetry)
│   ├── wash_detect.py        # Wash event isolation and recovery classification
│   ├── soiling.py            # Trend analysis for soiling rates
│   ├── transient.py          # Short-term anomaly/dip detection
│   ├── classification.py     # Final multi-axis string health classification
│   ├── losses.py             # Financial and energy loss aggregation
│   ├── clustering.py         # Statistical clustering of behavioral profiles
│   ├── pipeline.py           # Main orchestrator implementing parallel processing
│   ├── plotting.py           # Report and dashboard figure generation
│   ├── excel_export.py       # Excel diagnostic workbook compiler
│   └── adaptive_baseline.py  # Advanced adaptive baselining for changing conditions
├── data/                     # Demo datasets and preprocessing utilities
└── outputs/                  # Pipeline execution artifacts (Excel, figures)
```

## 5. Input Data Specifications

The pipeline expects data provided via Excel or CSV formats, mapping closely to the `DataTemplate.xlsx` structure.

**Primary Telemetry (`Data` sheet/table)**:
Required columns include: `timestamp`, `plant`, `inverter_id`, `mppt_id`, `string_id`, `irradiance KW/m2`, `voltage_u`, `current_i`, `power kw`, `pv_temperature`, `pv_Capacity`, `inverter_state`, `azimuth`, `tilt`.
*(Optional: `rainfall` in mm per interval to improve wash-cause attribution).*

**Plant Metadata (`Metadata` sheet/table)**:
Optional key/value pairs allowing contextual overrides:
- `plant_name` (e.g., Coca Cola Faisalabad)
- `latitude`, `longitude` (Defaults to Lahore coordinates)
- `tariff_pkr_kwh` (Defaults to Pakistan industrial B-3: 38.0 PKR/kWh)
- `commissioning_date` (Defaults to ~2 years prior to data max date)
- `p_ac_max_kw`
- `technology` (e.g., mono-c-Si)

## 6. Execution & Usage

The application is executed via the command-line interface. Processing is parallelized by default using `joblib`.

```bash
# Ensure you are using the virtual environment managed by uv
# Basic execution with default parameters
python run.py data/demo_data.xlsx --out-dir outputs

# Execution with custom clustering and explicit thread count
python run.py data/demo_data.xlsx --out-dir outputs --cluster-method combined --n-jobs 4

# Headless execution (useful for CI/CD or batch processing servers)
python run.py data/demo_data.xlsx --out-dir outputs --no-figures --quiet
```

## 7. Output Artifacts

The pipeline generates comprehensive, multi-modal diagnostics:

### 7.1. Diagnostic Workbook (`*_diagnostics.xlsx`)
Contains 16 detailed sheets providing traceability from summary to granular string-level data:
1. `00_Run_Summary`: High-level execution metrics.
2. `01_Plant_Metadata`: Contextual configuration used.
3. `02_Data_Sufficiency`: Data volume and quality gates.
4. `03_Curtailment_Detection`: Raw curtailment events.
5. `04_Curtailment_Loss`: Quantified kWh and PKR loss.
6. `05_Plate_Inference`: Derived string characteristics.
7. `06_SDM_Fits`: Modeling parameters and STC estimates.
8. `07_Daily_Metrics`: Aggregated daily NCI and PR.
9. `08_Degradation_Baseline`: Expected degradation trajectory.
10. `09_Wash_Events`: Detected cleaning events and recovery rates.
11. `10_Soiling_Trends`: Calculated soiling slopes.
12. `11_Orientation_Clusters`: String groupings by physical layout.
13. `12_Classification`: Final categorical verdict per string.
14. `13_Losses`: Total aggregated avoidable financial and energy losses.
15. `14_Transient_Events`: Short-duration anomalies.
16. `15_Per_String_Detail`: Comprehensive per-string master table.

### 7.2. Visual Dashboards (`figures/`)
- `soiling_dashboard__<string>.png`: Daily NCI overlay with wash event markers and segmented trends.
- `iv_diagnostics__<string>.png`: Scatter density plot of measured I-V points against the SDM curve at STC.
- `data_quality__<string>.png`: Stacked bar charts indicating data validity per day.
- `plant_overview.png`: Executive summary including verdict pie charts and top loss contributors.

## 8. Development & Extension

- **Environment Management**: The project strictly utilizes `uv` for dependency management. Ensure `uv.lock` is updated when modifying `pyproject.toml`.
- **Modularity**: New analytical steps should be implemented as standalone modules within `pv_diag/` and integrated into the orchestrator inside `pipeline.py`.
- **Graceful Fallbacks**: Features relying on external packages like `pvlib` must include fallback execution paths to ensure the pipeline completes successfully even if dependencies are missing.

## 9. Known Limitations

- **Missing `pvlib` Context**: If `pvlib` is unavailable, degradation analysis cannot utilize STC fingerprinting (low Voc, normal Isc) and falls back to classifying anomalies generically as `Mod.Soiling`.
- **Compound Faults**: Strings experiencing overwhelming physical curtailment will be correctly quantified for curtailment loss; however, underlying soiling classifications may be skewed due to lack of representative uncurtailed data. The `Per_String_Detail` sheet explicitly shows both metrics to mitigate misinterpretation.