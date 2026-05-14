# pv_diag — Modular PV String Diagnostics

End-to-end diagnostics pipeline for utility / C&I solar plants using
inverter-string telemetry (Huawei state-code aware). Outputs a multi-sheet
Excel report and per-string figures.

## Quick start

```bash
# 1. Generate a demo dataset (Coca Cola Faisalabad, 2 inv × 4 strings, 5-min, Oct 2025)
python generate_demo_data.py

# 2. Run the full pipeline on it
python run.py /mnt/user-data/outputs/demo_plant_data.xlsx \
              --out-dir /mnt/user-data/outputs
```

## Package layout

```
pv_diag/
├── constants.py        Huawei state codes, quality bit-mask, Pakistan defaults
├── config.py           Dataclasses: ModuleConfig, SiteConfig, PlantConfig,
│                       PipelineConfig
├── utils.py            _safe_id, _is_ok, _scalar (Excel-safe)
├── ingestion.py        load_plant_data → long_df + plant_meta
├── quality.py          per-row quality bit-mask
├── curtailment.py      detect + quantify curtailment loss (kWh + PKR)
├── sufficiency.py      Good / Limited / Poor / Skipped gate
├── celltemp.py         measured > NOCT fallback chain
├── plate.py            conservative nameplate inference (no over-fitting)
├── orientation.py      solar position, clear-sky POA, az/tilt clustering,
│                       expected_asymmetry (geometric shading discount)
├── degradation.py      NREL Jordan 2016 + LID first-year ramp baseline
├── sdm.py              single-diode fit (pvlib-aware; graceful fallback)
├── daily.py            PR, NCI, NCI_corrected, AM/PM asymmetry
├── wash_detect.py      rain/wash step detection w/ Full/Partial/Minimal recovery
├── soiling.py          segment-aware trimmed-LR slope (%/day)
├── transient.py        single-day dip detector
├── classification.py   multi-axis verdict (current-segment aware)
├── losses.py           soiling + curtailment loss aggregation
├── clustering.py       MPPT × (azimuth, tilt) combined clusters
├── pipeline.py         end-to-end orchestrator + joblib parallelism
├── plotting.py         soiling dashboard, IV diagnostics, data-quality, overview
└── excel_export.py     16-sheet xlsx report
```

## Spec compliance

| Spec item | Where addressed |
|---|---|
| (a) Modular + parallel | `pipeline.run_pipeline` uses `joblib.Parallel` with `cfg.n_jobs` |
| (b) Curtailment power & PKR loss | `curtailment.quantify_curtailment_loss` → sheet 04 |
| (c) Azimuth/tilt with PK defaults if blank | `ingestion.load_plant_data` fills NaN with `DEFAULT_AZIMUTH_PK=180`, `DEFAULT_TILT_PK=25` |
| (d) Commissioning, lat/lon, Lahore defaults | `ingestion._resolve_plant_meta` reads `Metadata` sheet, defaults to Lahore |
| (e) Wash-aware verdict ("Clean (post-wash)" / "Partial Recovery") | `wash_detect.detect_wash_events` + `classification.classify_string` with `use_current_segment_verdict` |
| (f) Plate efficiency degrades with time | `degradation.degradation_baseline` (LID ramp year 1, then linear 0.5%/yr mono-Si) — applied as `NCI_corrected = NCI / baseline` |
| (g) Charts + per-module Excel sheets | `plotting.make_all_figures` + `excel_export` 16 sheets |

## Input format

Main "Data" sheet — exactly the columns from the user's `DataTemplate.xlsx`:

```
timestamp, plant, inverter_id, mppt_id, string_id, irradiance KW/m2,
voltage_u, current_i, power kw, pv_temperature, pv_Capacity,
inverter_state, azimuth, tilt
```

Optional `rainfall` column (mm per interval) improves wash-cause attribution.

Optional `Metadata` sheet (key/value pairs):
```
plant_name | Coca Cola Faisalabad
latitude   | 31.4504
longitude  | 73.1350
tariff_pkr_kwh | 38.0
commissioning_date | 2023-06-01
p_ac_max_kw | 100.0
technology | mono-c-Si
```

Anything missing falls back to Pakistan defaults (see `constants.py`).

## Output

`<input_stem>_diagnostics.xlsx`:

```
00_Run_Summary
01_Plant_Metadata
02_Data_Sufficiency
03_Curtailment_Detection
04_Curtailment_Loss          ← (b) kWh + PKR per string
05_Plate_Inference
06_SDM_Fits
07_Daily_Metrics
08_Degradation_Baseline      ← (f) age-correction details
09_Wash_Events               ← (e) full/partial/minimal recovery
10_Soiling_Trends
11_Orientation_Clusters      ← (c)
12_Classification            ← verdict per string (current-segment aware)
13_Losses                    ← total avoidable energy + PKR
14_Transient_Events
15_Per_String_Detail
```

Plus PNG figures in `figures/`:
- `soiling_dashboard__<string>.png` — daily NCI with wash markers + segments
- `iv_diagnostics__<string>.png`    — measured I-V cloud + SDM @STC
- `data_quality__<string>.png`      — stacked daily quality bars
- `plant_overview.png`              — verdict pie + per-string loss bars

## Notes

- `pvlib` is optional. If present, SDM fits run and degradation can be
  detected via STC fingerprint (Voc-low + Isc-normal). If not, SDM is
  marked `pvlib_unavailable` and degradation defaults to soiling diagnosis.
- Tariff defaults to 38 PKR/kWh (industrial B-3). Override via Metadata sheet.
- Commissioning date defaults to 2023-01-01 (≈2 years) if not provided.

## Known limitations

- Without `pvlib`, degradation strings are classified as `Mod.Soiling`
  (mean NCI in that band). The Voc-low fingerprint can't fire.
- Curtailment-dominated strings are reported with their curtailment loss
  (sheets 03/04) but still classified by their soiling state; the
  `Per_String_Detail` sheet shows both axes side by side.
