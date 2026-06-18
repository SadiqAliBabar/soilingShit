# Implementation Plan: Wire Old Pipeline to Real Data Sources

## Goal

Make `pv_diag` (the pipeline under `soiling_old_7_prompt/pv_diag/`) run directly from the
two real input files — `RequireInputData.xlsx` and the measured CSV from MongoDB — with
**zero hardcoded assumptions**. Every physics constant, site parameter, and module spec
must come from the workbook or the CSV. Nothing from `config.py` defaults.

---

## Source-of-Truth Mapping

Before any code, here is the exact field-by-field mapping from the two files into what the
pipeline needs.

### A. Measured CSV → long_df columns

The CSV is a multi-level long format (`level` column = `plant` / `inverter` / `mppt` / `string`).
Each level only fills its own columns; the rest are NaN.

| long_df column  | CSV source                                    | Notes                                      |
|-----------------|-----------------------------------------------|--------------------------------------------|
| `ts`            | `timestamp` (string rows)                     | Parse to datetime                          |
| `plant`         | `plant_name` (string rows)                    |                                            |
| `inverter_id`   | `inverter_id` (string rows)                   |                                            |
| `mppt_id`       | `mppt_id` (string rows)                       | e.g. "MPPT1"                               |
| `string_id`     | `string_id` (string rows)                     | e.g. "pv1"                                 |
| `POA_kw`        | `irradiance_wm2` (plant rows, broadcast)      | Mislabeled — values ARE kW/m², not W/m²    |
| `V`             | `string_voltage_v` (string rows)              |                                            |
| `I`             | `string_current_a` (string rows)              |                                            |
| `P_kw`          | `string_power_kw` (string rows)               |                                            |
| `T_module`      | `pv_temperature` (string rows)                |                                            |
| `inverter_state`| `inverter_state` (inverter rows, broadcast)   | Must broadcast by (timestamp, inverter_id) |
| `pv_capacity`   | `string_capacity_kwp` (string rows)           | kW; used by `_get_string_plate()` for n_modules per string |
| `azimuth`       | Plant sheet `azimuth_deg`                     | Broadcast uniformly (one value, whole plant) |
| `tilt`          | Plant sheet `tilt_deg`                        | Broadcast uniformly                        |
| `rainfall`      | Not in CSV                                    | Leave absent; pipeline treats missing as 0 |

### B. RequireInputData.xlsx → PipelineConfig

#### ModuleConfig (Panel sheet — take first string's row; verify all same model)

| ModuleConfig field     | Panel sheet column              | Conversion                     |
|------------------------|---------------------------------|--------------------------------|
| `voc_stc`              | `voc_v`                         | Direct (V per module)          |
| `vmp_stc`              | `vmp_v`                         | Direct                         |
| `isc_stc`              | `isc_a`                         | Direct                         |
| `imp_stc`              | `imp_a`                         | Direct                         |
| `alpha_isc`            | `alpha_isc_pct_per_c`           | ÷ 100  (e.g. 0.046 → 0.00046) |
| `beta_voc`             | `beta_voc_pct_per_c`            | ÷ 100  (e.g. −0.260 → −0.00260)|
| `gamma_pmp`            | `gamma_pmax_pct_per_c`          | ÷ 100                          |
| `n_modules`            | `num_panels`                    | int; per-string override via `pv_capacity` already handled by `_get_string_plate()` |
| `technology`           | `technology`                    | str                            |
| `cells_in_series`      | `num_cells`                     | int                            |

#### SiteConfig (Plant sheet)

| SiteConfig field  | Plant sheet column    | Notes                                              |
|-------------------|-----------------------|----------------------------------------------------|
| `name`            | `site_name`           |                                                    |
| `lat`             | `latitude`            |                                                    |
| `lon`             | `longitude`           |                                                    |
| `altitude`        | `altitude_m`          |                                                    |
| `p_ac_max_kw`     | `size_kw_ac`          |                                                    |
| `tz`              | —                     | Keep "Asia/Karachi" (not in workbook; physical fact)|
| `albedo`          | —                     | Keep 0.20 default (not in workbook)                |
| `temp_model`      | —                     | Keep "sapm" (not in workbook)                      |
| `racking`         | —                     | Keep "open_rack_glass_glass"                       |
| `tariff`          | —                     | Keep DEFAULT_TARIFF_PKR_PER_KWH (not in workbook)  |
| `currency`        | —                     | Keep "PKR"                                         |

#### PlantConfig (Plant sheet)

| PlantConfig field    | Plant sheet column    | Notes                    |
|----------------------|-----------------------|--------------------------|
| `commissioning_date` | `commissioning_date`  | Parse to `date`          |
| `default_azimuth`    | `azimuth_deg`         | Already N-referenced (0=N, 180=S) |
| `default_tilt`       | `tilt_deg`            |                          |
| `lat`                | `latitude`            |                          |
| `lon`                | `longitude`           |                          |

#### PipelineConfig degradation params (Panel sheet)

| PipelineConfig field      | Panel sheet column          | Conversion |
|---------------------------|-----------------------------|------------|
| `annual_degradation_pct`  | `annual_degradation_pct`    | ÷ 100      |
| `lid_loss_pct`            | `first_year_degradation_pct`| ÷ 100      |

---

## Files Involved

### Files to CREATE
1. `src/soiling_analysis/loader.py` — reads both source files, produces long_df + plant_meta + PipelineConfig
2. `run.py` (root level) — new CLI entry point; replaces `run_pipeline.py`

### Files to MODIFY
3. `src/soiling_analysis/soiling_old_7_prompt/pv_diag/pipeline.py` — add `run_pipeline_from_frame()` that accepts pre-loaded data instead of a file path

### Files to DELETE (obsolete)
4. `run_pipeline.py` — the wrong entry point
5. `src/soiling_analysis/inputs/assemble.py` — not needed
6. `src/soiling_analysis/baseline/baseline.py` and `src/soiling_analysis/baseline/__init__.py` — not needed

### Files to KEEP (reuse)
7. `src/soiling_analysis/inputs/specs.py` — already reads RequireInputData.xlsx correctly; use it
8. `src/soiling_analysis/inputs/measured.py` — use as reference only; loader.py will supersede it with richer extraction (inverter_state broadcast)
9. All of `pv_diag/` — untouched except pipeline.py

---

## Step-by-Step Implementation

---

### Step 1 — Add `run_pipeline_from_frame()` to `pipeline.py`

**File:** `src/soiling_analysis/soiling_old_7_prompt/pv_diag/pipeline.py`

**What:** The existing `run_pipeline(xlsx_path, cfg, ...)` always calls `load_plant_data()` to load a file. We need a version that accepts an already-built `long_df` and `plant_meta` directly so the loader can inject data without touching a file.

**How:** Extract the body of `run_pipeline()` starting after the `load_plant_data()` call into a new function `run_pipeline_from_frame(long_df, plant_meta, cfg, cluster_method, verbose)`. Then make the original `run_pipeline()` call the new function after loading.

```python
def run_pipeline_from_frame(
    long_df: pd.DataFrame,
    plant_meta: dict,
    cfg: PipelineConfig | None = None,
    cluster_method: str = "combined",
    verbose: bool = True,
) -> dict:
    cfg = cfg or PipelineConfig()
    # Everything that was previously below load_plant_data() goes here:
    # apply_plant_meta_to_cfg, flag_data_quality, split_into_string_dfs, ...
    ...

def run_pipeline(xlsx_path: str, cfg=None, cluster_method="combined", verbose=True):
    cfg = cfg or PipelineConfig()
    if verbose:
        print(f"[1/9] Loading {xlsx_path}...")
    long_df, plant_meta = load_plant_data(xlsx_path, cfg=cfg)
    return run_pipeline_from_frame(long_df, plant_meta, cfg, cluster_method, verbose)
```

**Why this approach:** Zero changes to ingestion.py, config.py, or any analysis module. The original `run_pipeline()` still works for anyone who has a legacy xlsx.

---

### Step 2 — Create `src/soiling_analysis/loader.py`

This is the only new logic file. It reads both source files and produces the three things the pipeline needs.

#### 2a. Read the spec workbook

```python
from soiling_analysis.inputs.specs import load_specs
specs = load_specs(specs_path)   # already handles Plant / Inverter / Panel sheets
```

The `load_specs()` function already exists and works. Use it directly.

#### 2b. Read the raw CSV

```python
raw = pd.read_csv(csv_path, low_memory=False)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
```

#### 2c. Extract and broadcast irradiance

Plant rows have `irradiance_wm2` (unit is actually kW/m²). Build a series keyed by timestamp.

```python
plant_rows = raw[raw["level"] == "plant"]
irr = (
    plant_rows.dropna(subset=["irradiance_wm2"])
    .drop_duplicates("timestamp")
    .set_index("timestamp")["irradiance_wm2"]
)
```

#### 2d. Extract and broadcast inverter_state

Inverter rows have `inverter_state`. Build a series keyed by (timestamp, inverter_id).

```python
inv_rows = raw[raw["level"] == "inverter"]
inv_state = (
    inv_rows.dropna(subset=["inverter_state"])
    .drop_duplicates(["timestamp","inverter_id"])
    .set_index(["timestamp","inverter_id"])["inverter_state"]
)
```

Then on string rows, broadcast by (timestamp, inverter_id) lookup.

#### 2e. Filter to string rows and rename columns

```python
strings = raw[raw["level"] == "string"].copy()
```

Rename:
```python
strings = strings.rename(columns={
    "timestamp":          "ts",
    "plant_name":         "plant",
    "string_voltage_v":   "V",
    "string_current_a":   "I",
    "string_power_kw":    "P_kw",
    "pv_temperature":     "T_module",
    "string_capacity_kwp":"pv_capacity",
    "irradiance_wm2":     "POA_kw",   # will be filled from broadcast below
})
```

#### 2f. Merge irradiance and inverter_state onto string rows

```python
strings["POA_kw"] = strings["ts"].map(irr)
strings["inverter_state"] = (
    pd.MultiIndex.from_arrays([strings["ts"], strings["inverter_id"]])
    .map(inv_state)
)
strings["inverter_state"] = strings["inverter_state"].fillna(-1).astype(int)
```

#### 2g. Add azimuth and tilt from plant spec

```python
plant_spec = specs["plant"]
strings["azimuth"] = plant_spec["azimuth_deg"]
strings["tilt"]    = plant_spec["tilt_deg"]
```

#### 2h. Validate required columns are present

```python
REQUIRED = {"ts","plant","inverter_id","mppt_id","string_id",
            "POA_kw","V","I","P_kw","T_module","inverter_state"}
missing = REQUIRED - set(strings.columns)
if missing:
    raise ValueError(f"Loader produced missing columns: {sorted(missing)}")
```

#### 2i. Build plant_meta dict

Compute the same shape that `ingestion.py` returns so `apply_plant_meta_to_cfg()` and
the rest of the pipeline can use it without modification.

```python
plant_meta = dict(
    plants=sorted(strings["plant"].unique().tolist()),
    inverters=sorted(strings["inverter_id"].unique().tolist()),
    mppts_per_inv=strings.groupby("inverter_id")["mppt_id"].nunique().to_dict(),
    total_strings=int(strings.groupby(["plant","inverter_id","mppt_id","string_id"]).ngroups),
    ts_min=strings["ts"].min(),
    ts_max=strings["ts"].max(),
    freq_min=_detect_freq_min(strings),
    n_intervals=int(strings["ts"].nunique()),
    plant_resolved=dict(
        plant_name=plant_spec["site_name"],
        lat=plant_spec["latitude"],
        lon=plant_spec["longitude"],
        tariff=DEFAULT_TARIFF_PKR_PER_KWH,   # not in workbook
        commissioning_date=plant_spec["commissioning_date"],
        default_azimuth=plant_spec["azimuth_deg"],
        default_tilt=plant_spec["tilt_deg"],
        p_ac_max_kw=plant_spec["size_kw_ac"],
        technology=_dominant_technology(specs),
        substitution_notes=[],
    ),
    substitution_notes=[],
    azimuth_filled_rows=0,
    tilt_filled_rows=0,
)
```

`_detect_freq_min`: compute median interval in minutes from sorted unique timestamps.
`_dominant_technology`: take the most common technology string across all panel rows in the specs.

#### 2j. Build PipelineConfig from workbook

```python
from pv_diag.config import PipelineConfig, ModuleConfig, SiteConfig, PlantConfig
import datetime

panel = _representative_panel(specs)   # pick first string's panel; warn if multiple models

cfg = PipelineConfig(
    site=SiteConfig(
        name=plant_spec["site_name"],
        lat=plant_spec["latitude"],
        lon=plant_spec["longitude"],
        altitude=plant_spec["altitude_m"],
        p_ac_max_kw=plant_spec["size_kw_ac"],
        # keep tz, albedo, temp_model, racking, tariff, currency at defaults
    ),
    module=ModuleConfig(
        voc_stc=panel["voc_v"],
        vmp_stc=panel["vmp_v"],
        isc_stc=panel["isc_a"],
        imp_stc=panel["imp_a"],
        alpha_isc=panel["alpha_isc_pct_per_c"] / 100.0,
        beta_voc=panel["beta_voc_pct_per_c"] / 100.0,
        gamma_pmp=panel["gamma_pmax_pct_per_c"] / 100.0,
        n_modules=panel["num_panels"],
        technology=panel["technology"],
        cells_in_series=panel["num_cells"],
    ),
    plant=PlantConfig(
        commissioning_date=datetime.date.fromisoformat(plant_spec["commissioning_date"]),
        default_azimuth=plant_spec["azimuth_deg"],
        default_tilt=plant_spec["tilt_deg"],
        lat=plant_spec["latitude"],
        lon=plant_spec["longitude"],
    ),
)
# Overwrite degradation from workbook
cfg.annual_degradation_pct = panel["annual_degradation_pct"] / 100.0
cfg.lid_loss_pct = panel["first_year_degradation_pct"] / 100.0
```

#### 2k. Public function signature

```python
def load_from_sources(
    csv_path: str | Path,
    specs_path: str | Path = DEFAULT_REQUIRE_INPUT,
) -> tuple[pd.DataFrame, dict, PipelineConfig]:
    """Return (long_df, plant_meta, cfg) ready for run_pipeline_from_frame()."""
    ...
```

---

### Step 3 — Create `run.py` (root level, new CLI entry point)

Replace `run_pipeline.py` with this clean entry point.

```
Usage:
  python run.py --csv <measured.csv>
                [--specs <RequireInputData.xlsx>]
                [--out-dir <output_dir>]
                [--cluster-method combined|mppt|orient]
                [--n-jobs N]
                [--no-figures]
                [--quiet]
```

Logic:
1. Parse args
2. Call `load_from_sources(csv, specs)` → `(long_df, plant_meta, cfg)`
3. Apply `--n-jobs` to `cfg.n_jobs`
4. Call `run_pipeline_from_frame(long_df, plant_meta, cfg, cluster_method, verbose)`
5. Call `export_results_to_excel(results, out_path, verbose=verbose)`
6. Call `make_all_figures(results, fig_dir, verbose=verbose)` unless `--no-figures`

The sys.path setup must add both the repo `src/` and the `soiling_old_7_prompt/` directory so imports resolve:
```python
sys.path.insert(0, str(Path(__file__).resolve() / "src"))
sys.path.insert(0, str(Path(__file__).resolve() / "src/soiling_analysis/soiling_old_7_prompt"))
```

---

### Step 4 — Handle edge cases in the loader

#### 4a. Multiple panel models on the same plant

If `specs["inverters"]` contains strings with different `panel_model` values:
- Log a warning listing which inverter has which model
- Use the modal (most common) panel for the base `ModuleConfig`
- Per-string n_modules is still handled correctly by `_get_string_plate()` via the `pv_capacity` column

#### 4b. Missing irradiance coverage

Some timestamps may have no plant-level irradiance row. `strings["POA_kw"]` will be NaN for those rows.
The pipeline already handles NaN POA (quality flagged as night/low-G). No special treatment needed.

#### 4c. Missing inverter_state

If `inverter_state` is NaN for a row (no matching inverter row), fill with `-1`. The pipeline
treats unknown states as non-curtailed. This is the same fallback `ingestion.py` uses.

#### 4d. Timezone

`ingestion.py` does not parse timezone; timestamps are kept naive. The loader should also keep
them naive (strip tz info if present after `pd.to_datetime()`).

#### 4e. String ID case

The CSV has `string_id = "pv1"` (lowercase) while the Panel sheet has `"PV1"` (uppercase).
The pipeline's `string_label` column is built from raw IDs — case-sensitivity does not matter
there as long as IDs are consistent within the long_df. The loader should not alter the case
of `string_id`; leave it as-is from the CSV. The specs.py join (used by the loader only for
building the config) already does case-insensitive matching.

---

### Step 5 — Delete obsolete files

In order:
1. Delete `run_pipeline.py`
2. Delete `src/soiling_analysis/inputs/assemble.py`
3. Delete `src/soiling_analysis/baseline/baseline.py`
4. Delete `src/soiling_analysis/baseline/__init__.py`
5. Remove `src/soiling_analysis/baseline/` directory

`src/soiling_analysis/inputs/specs.py` and `src/soiling_analysis/inputs/measured.py` can stay —
they are utility readers. `loader.py` uses `specs.py`; `measured.py` is now superseded by the
richer extraction in `loader.py` but is harmless to leave.

---

### Step 6 — Verify no hardcoded defaults leak through

After implementation, grep the pipeline source for any remaining hardcoded fallbacks and
confirm they cannot trigger when data is provided from the workbook:

```
grep -rn "LAHORE_LAT\|LAHORE_LON\|DEFAULT_AZIMUTH\|DEFAULT_TILT" pv_diag/
```

Expected hits: only in `constants.py` (definitions) and `ingestion.py` (fallback for missing
Metadata sheet). Neither fires when `run_pipeline_from_frame()` is used because
`apply_plant_meta_to_cfg()` receives a fully populated `plant_resolved` dict.

Also verify the degradation params:
```
grep -n "annual_degradation_pct\|lid_loss_pct" pv_diag/config.py
```
These are still defined in `PipelineConfig` but overwritten by the loader in Step 2j. The
defaults in `config.py` only fire if someone constructs `PipelineConfig()` without the loader.

---

### Step 7 — End-to-end smoke test

After all code changes, run:

```bash
uv run python run.py \
  --csv "src/soiling_analysis/data_gather_from_mongo_db/output/csv/Coca_Cola_Faisalabad_20260218_to_20260617.csv" \
  --specs "src/soiling_analysis/RequireInputData/RequireInputData.xlsx" \
  --out-dir output/ \
  --quiet
```

Check:
- [ ] No `ValueError` on missing columns
- [ ] `cfg.site.lat` / `cfg.site.lon` match plant workbook values, not Lahore defaults
- [ ] `cfg.module.voc_stc` matches the JA Solar spec in the Panel sheet
- [ ] `cfg.plant.commissioning_date` matches workbook
- [ ] `cfg.annual_degradation_pct` matches Panel sheet value ÷ 100
- [ ] Output xlsx written to `output/`
- [ ] All 147 configured strings appear in per-string results (or explain why any are missing)

---

## What Does NOT Change

- All physics logic in `pv_diag/` — untouched
- `ingestion.py` — untouched (still works for legacy xlsx)
- `config.py` — untouched (defaults stay, but the loader always overwrites them)
- `constants.py` — untouched
- `run.py` inside `soiling_old_7_prompt/` — untouched (still works as before)
- The 16-sheet Excel export format
- All figure generation

---

## Summary of New Call Chain

```
run.py (new root CLI)
  └─ loader.load_from_sources(csv, specs)
       ├─ inputs/specs.py  → specs dict
       ├─ raw CSV          → long_df (string rows, with irradiance + inverter_state broadcast)
       ├─ plant_meta dict  (same shape as ingestion.py output)
       └─ PipelineConfig   (all fields from workbook, no defaults for plant/module/degradation)
  └─ pipeline.run_pipeline_from_frame(long_df, plant_meta, cfg)
       ├─ apply_plant_meta_to_cfg
       ├─ flag_data_quality → detect_curtailment
       ├─ split_into_string_dfs → extract_string_meta
       ├─ infer_plate_params
       ├─ assign_clusters
       ├─ degradation_baseline
       ├─ per-string analysis (Pass 1 + adaptive baseline + Pass 2)
       ├─ aggregate_plant_losses
       └─ returns results dict
  └─ export_results_to_excel(results, out_path)
  └─ make_all_figures(results, fig_dir)
```
