# Walkthrough: Wiring Old Pipeline to Real Data Sources

This document details all the changes made to transition the `pv_diag` pipeline from using hardcoded assumptions and defaults to running directly from real data sources (the `RequireInputData.xlsx` workbook and the measured MongoDB CSV).

## What Was Accomplished

1. **Created a Centralised Loader (`loader.py`)**
   - Built [loader.py](file:///d:/Nexalyze/Solarflux/SoilingNewShit/src/soiling_analysis/loader.py) as the new data ingestion bridge.
   - It reads the specs workbook via `load_specs()` and extracts all necessary parameters for modules, site, and plant.
   - It parses the raw CSV (with `plant`, `inverter`, and `string` levels) and builds the `long_df`.
   - **Broadcasting**: It extracts plant-level `irradiance_wm2` and inverter-level `inverter_state` and broadcasts them down to the string-level rows.
   - **Data Types**: Enforced string types for all identifiers (`plant`, `inverter_id`, `mppt_id`, `string_id`) to prevent sorting and merging errors (e.g., `TypeError` when comparing strings to `NaN` floats).
   - **Derived Variables**: Calculated `POA`, `P` (in Watts), and constructed the standard `string_label` format required by the pipeline.
   - Fully populated the `PipelineConfig` instance purely from the workbook, ensuring no hardcoded parameters (like default location or degradation) leak through.

2. **Refactored the Pipeline Entry Point (`pipeline.py`)**
   - Modified [pipeline.py](file:///d:/Nexalyze/Solarflux/SoilingNewShit/src/soiling_analysis/soiling_old_7_prompt/pv_diag/pipeline.py).
   - Extracted the core pipeline execution logic into a new function: `run_pipeline_from_frame(long_df, plant_meta, cfg, ...)`.
   - This allows us to feed pre-loaded DataFrames and configuration objects directly into the analysis without touching the disk or legacy ingestion routines.
   - Retained the original `run_pipeline()` function as a wrapper for backward compatibility.

3. **Created a New CLI Runner (`run.py`)**
   - Replaced the old entry point with a clean, root-level [run.py](file:///d:/Nexalyze/Solarflux/SoilingNewShit/run.py).
   - Setup `sys.path` to correctly resolve imports across both `src/` and the nested `soiling_old_7_prompt/pv_diag` structure.
   - Parses command-line arguments to accept `--csv`, `--specs`, `--out-dir`, and performance flags like `--n-jobs` and `--no-figures`.
   - Drives the full workflow: `load_from_sources` -> `run_pipeline_from_frame` -> `export_results_to_excel` -> `make_all_figures`.

4. **Cleaned Up Obsolete Files**
   - Deleted the outdated `run_pipeline.py`.
   - Deleted `src/soiling_analysis/inputs/assemble.py`.
   - Deleted the entire `src/soiling_analysis/baseline/` directory (including `baseline.py` and `__init__.py`).
   - Removed obsolete exports from `src/soiling_analysis/inputs/__init__.py`.

5. **Resolved Dependencies**
   - Installed missing mathematical and scientific packages (`scipy`, `joblib`, `scikit-learn`, `pvlib`) needed to run the pipeline logic.

## Verification

An end-to-end smoke test was run using the provided dataset (`Coca_Cola_Faisalabad_20260218_to_20260617.csv` and `RequireInputData.xlsx`). 

> [!NOTE]
> All configuration constraints (site location, module specs, commissioning date) were confirmed to be successfully ingested from the provided input files rather than defaulting to hardcoded fallbacks like Lahore coordinates.
