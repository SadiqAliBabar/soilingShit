# Soiling Pipeline — Sequential Implementation Playbook

**Audience:** Claude Sonnet, implementing one batch per session.
**Codebase:** `pv_diag` at `src/soiling_analysis/soiling_old_7_prompt/pv_diag/`, loader at `src/soiling_analysis/loader.py`, spec reader at `src/soiling_analysis/inputs/specs.py`, CLI at `run.py`.
**Author's role:** This file was written after reading every module the changes touch. File/function/threshold references below are real; quote them back to the user when explaining a change.

---

## 0. How to use this document

1. **Do exactly one batch per session.** Batches are ordered by dependency, not by importance. Do not skip ahead. Do not bundle two batches "while you're in the file."
2. **Read the batch's "Touches" list first**, open every file in it, and confirm the "Current behaviour" description still matches the code before writing anything. If the code has drifted, stop and report the drift rather than implementing against a stale description.
3. **Honour the [Global rules](#3-global-rules-every-batch-must-obey).** This codebase has hard conventions (never rename columns, keep legacy aliases, feature-flag every behaviour change, all thresholds in `PipelineConfig`). Violating them breaks downstream silently.
4. **Run the batch's tests** (and the smoke run in §5) before declaring the batch done. Each module has a sibling `*_test.py`.
5. **Stop at the [Validation Gate](#validation-gate-after-batch-4-not-a-prompt)** after Batch 4. Do not start Batch 5 until the gate's checklist passes on the real CCI Faisalabad data.

A batch is **done** only when its acceptance criteria pass *and* the full pipeline still runs end-to-end on the sample data with `--n-jobs 1`.

---

## 1. Assessment of the user's plan

The plan in the Word document is sound and I am adopting its ordering. The three reorderings it calls out are correct:

- **Curtailment belongs in the spine, upstream of the baseline.** Confirmed in code: `run_pipeline_from_frame()` calls `detect_curtailment(long_df, …)` at step `[2/9]`, and the curtailment flags it writes (`CURT_SUPPRESSED`, `CURT_VOLTAGE_RISE`, `CURT_STATISTICAL`) sit in the `DISQUALIFYING` bitmask in `constants.py`, which `utils._is_ok()` uses to mask rows *before* the adaptive baseline reads `NCI_noon`. Wrong flags poison the baseline. Fix flags first.
- **Calendar consolidation is the backbone.** Confirmed: `wash_detect`, `transient`, and the pre-slope in `soiling` all operate on `daily_df` rows that exist only for dates with data, and they `np.diff()` adjacent rows — a multi-day gap is silently treated as one day. The adaptive baseline and sufficiency are already calendar-aware, so the inconsistency is real.
- **A validation gate after the spine.** Correct — the spine reshapes the numbers, so refinements built on unvalidated spine output risk a redo.

**Three code-grounded refinements to the plan you should be aware of:**

1. **Batch 1 is a hard prerequisite, not just "economics inputs."** The plan references Batch 1 only as the source of economics inputs and schema TODOs. But I found that `inputs/specs.py::_read_panels()` **does not read** the `String Azimuth`, `String Tilt`, or `String Comissioning Date` columns at all, and `loader.py` broadcasts the *plant* azimuth/tilt to every string (`strings["azimuth"] = plant_spec["azimuth_deg"]`). So **per-string orientation (Batch 5) and per-string age (Batch 4) have no data to consume until Batch 1 surfaces it.** Batch 1 is therefore the true foundation of the spine. Do it first.
2. **Rain is unavailable for this plant.** The Plant sheet has `Rain Available = No`, and there is no `rainfall` column in the measured CSV. The recovery-anchored baseline (Batch 4) and wash-cause attribution (Batch 2) both lean on rain; with no rain signal they will fall back to the P95 path and peer cross-check almost always. This makes **Batch 7 (monsoon/smog fallback + hold-last-good) more load-bearing than it looks**, and means "verified recovery anchor" will be rare. Plan accordingly; do not assume recovery anchors exist.
3. **Inverter-level AC power is needed by Batch 3 but is currently dropped.** `loader.py` keeps only `level == "string"` rows (plus broadcast `inverter_state`). The CSV's `level == "inverter"` rows carry inverter AC output, which Batch 3 needs (clipping is an AC phenomenon at the inverter). Surfacing it is a **Batch 1 schema task** flagged by [Check 4](#2-the-six-preflight-checks).

Everything else in the plan stands. Batch numbering below matches the plan (1 = schema foundation; 2–4 = spine; gate; 5–9 = refinements).

---

## 2. The Six Preflight Checks

The plan asks to "mark batches 1 and 9's schema-dependent sections as TODOs keyed to your six checks." These are the six checks. **Batch 1 must implement them as a small `preflight_schema(specs, raw_csv) -> dict` function** that returns a capability dict; every schema-dependent section downstream branches on it (`do X if check passes, else fallback Y`). Run preflight once at load time and log the result.

| # | Check | How to test | If TRUE | If FALSE (fallback) |
|---|-------|-------------|---------|---------------------|
| **C1** | **Per-string orientation present** | `String Azimuth` & `String Tilt` columns exist in Panel sheet and are non-null & numeric for every string | Use per-string az/tilt (Batches 4-peer, 5) | Fall back to plant `azimuth_deg`/`tilt_deg`; set `orientation_source="plant_default"` |
| **C2** | **Per-string commissioning date present** | `String Comissioning Date` column exists (note the single-m typo) and parses to a date for every string | Per-string age baseline (Batch 4) | Fall back to plant `commissioning_date` for all strings; set `age_source="plant_default"` |
| **C3** | **Rear-side irradiance for bifacial** | Module `Bifacial == Yes` AND a rear/GHI irradiance column exists in CSV | Measured bifacial gain (Batch 5) | Modeled bifacial gain from albedo + bifaciality, clearly flagged `bifacial_gain_source="modeled"`; or skip if `Bifacial == No` |
| **C4** | **Inverter AC capacity + measured inverter AC power** | Inverter sheet has `Max AC Active Power`/`capacity_kw_ac` per inverter AND CSV `level=="inverter"` rows carry AC power | Inverter-level clipping & suppression on AC power (Batch 3) | Per-inverter capacity from specs only; reconstruct inverter AC as sum of string DC × efficiency, flagged `inverter_ac_source="reconstructed_dc"` |
| **C5** | **Rain available** | Plant sheet `Rain Available == Yes` AND CSV has a rainfall column | Rain-anchored wash cause + recovery anchors (Batches 2, 4) | Wash cause = "suspected"; recovery anchors rare → lean on P95 + peer cross-check + Batch 7 hold-last-good |
| **C6** | **Economics inputs present** | A tariff (PKR/kWh) and a wash cost are resolvable | Cleaning-economics recommendation (Batch 9) | Use `DEFAULT_TARIFF_PKR_PER_KWH=38.0` and a configurable default wash cost; mark recommendation `economics_inputs="defaulted"` and surface the assumption in the report |

For the supplied CCI Faisalabad data today: C1 ✅ (Panel sheet has String Azimuth/Tilt), C2 ✅ (String Comissioning Date present), C3 ⚠️ (`Bifacial=Yes` but **no rear-irradiance column** → modeled gain or skip), C4 ⚠️ (inverter AC capacity present; verify CSV inverter rows carry AC power), C5 ❌ (`Rain Available=No`), C6 ⚠️ (tariff defaulted, wash cost must be added).

> If the user's "six checks" differ from these, ask them to confirm before Batch 1 — but these are the concrete data-availability gates the codebase actually needs.

---

## 3. Global rules (every batch must obey)

These are non-negotiable conventions already established in the codebase. Breaking them causes silent downstream failures.

1. **Never remove or rename an existing column.** Downstream selects columns via `utils.pick_nci_column()` and direct names. *Add* new columns; leave old ones populated. (Example: `NCI`, `NCI_corrected`, `NCI_relative`, `NCI_adaptive` all coexist.)
2. **Keep legacy aliases.** When you compute a new split metric, keep the old key as an alias (e.g., `curtailment.py` keeps `total_curt_kwh` aliasing `curtailment_loss_total_kwh`).
3. **Feature-flag every behaviour change** with a `PipelineConfig` boolean that defaults to the new behaviour but can restore the old one — mirror the existing `adaptive_baseline_enabled` pattern. Name them clearly (`curtailment_inverter_level_enabled`, `age_relative_gates_enabled`, `clearsky_quality_enabled`, `robust_sdm_loss_enabled`, `robust_soiling_regression_enabled`, …).
4. **No magic numbers in modules.** Every threshold lives in `PipelineConfig` with a one-line comment explaining it (see `config.py` for the house style). The [reference table](#6-recommended-thresholds-reference-all-new-config-keys) lists every new key with a recommended default and rationale.
5. **Graceful degradation.** Wrap optional-dependency and data-dependent paths in `try/except` that `warnings.warn(...)` and fall back, exactly like `_fit_sdm()` and `fit_single_diode()` do. The pipeline must never crash on one bad string.
6. **Extend provenance, don't hide decisions.** Any held, blended, substituted, or defaulted value must be recorded in a provenance field (extend `AdaptiveBaselineResult` and the per-string `res` dict) and surfaced in the Excel export. No silent substitution.
7. **Tests travel with code.** For every module you touch, extend its sibling `*_test.py` (e.g., `curtailment_test.py`, `wash_detect_test.py`, `sdm_test.py`, `adaptive_baseline_test.py`). Add at least one test that would fail under the old behaviour and pass under the new.
8. **Preserve the two-pass structure.** Pass 1 (plate-based daily + wash) → between-pass adaptive baseline → Pass 2 (adaptive daily + full analysis). Do not collapse it.
9. **Recurring rule from the clear-sky discussion (keep visible):** *the clear-sky index (Kc) gates estimation quality; absolute irradiance (W/m²) gates physical validity and energy accounting.* Apply Kc only where you're deciding whether a sample is good enough to fit/estimate on; keep W/m² where you're flagging data or counting energy.

---

# THE SPINE — fix what makes the numbers wrong, in dependency order

---

## Batch 1 — Schema & inputs foundation

**Goal.** Surface every per-entity value the rest of the work depends on, change the string identifier, and add economics inputs. Nothing here changes a number yet; it changes what data is *available*. This unblocks Batches 3, 4, 5, 9.

**Touches.**
- `src/soiling_analysis/inputs/specs.py` — `_read_panels()`, `_read_plant()`, add a per-string-orientation/age/bifacial reader.
- `src/soiling_analysis/loader.py` — `load_from_sources()`, `_representative_panel()`, plant_meta construction.
- `pv_diag/ingestion.py` — `extract_string_meta()` (add fields), `_tolerant_rename()` (rear-irradiance/rain columns).
- `pv_diag/config.py` — new `PipelineConfig` keys for economics + per-entity stores.
- `pv_diag/utils.py` — string-label construction helper.
- New: `preflight_schema()` (put in `loader.py` or a new `inputs/preflight.py`).

**Current behaviour (verified).**
- `_read_panels()` reads capacity, panel specs, degradation, `bifacial`, `num_cells` — but **omits** `String Azimuth(°C)`, `String Tilt (°C)`, `String Comissioning Date`. `bifacial` is read but never consumed.
- `loader.py` sets `strings["azimuth"]/["tilt"]` from the **plant** spec and builds `ModuleConfig` from one `_representative_panel(specs)` (modal/first panel). Per-inverter AC capacity and inverter-level AC power are not surfaced.
- `string_label` = `plant__inverter__mppt__string` (built in both `loader.py` and `ingestion.py`).
- `extract_string_meta()` returns only `plant, inverter_id, mppt_id, string_id, azimuth, tilt`.
- Tariff comes from `DEFAULT_TARIFF_PKR_PER_KWH=38.0`; there is no wash-cost input anywhere.

**Change.**

1. **String identifier → `<Inverter ID>_<String ID>`.** The user wants the *human-facing* string identifier to read as `<InverterID>_<StringID>`. Implement carefully:
   - Keep the internal `string_label` unique key intact (it must stay unique across MPPTs, so keep `plant__inverter__mppt__string` as the grouping key) **but add a new display column `string_uid = f"{inverter_id}_{string_id}"`** and use it everywhere the report shows a string name. If `<inverter>_<string>` is not unique because the same `string_id` repeats across MPPTs on one inverter, fall back to `{inverter_id}_{mppt_id}_{string_id}` and warn. Confirm uniqueness in preflight.
   - Do **not** silently drop the old label; downstream `groupby("string_label")` depends on it.

2. **Read per-string orientation, age, bifacial in `specs.py`.** Add to `_read_panels()` fields:
   - `string_azimuth_deg` from `_col(df, "string", "azimuth")` (the column header is `String Azimuth(°C)` — the `(°C)` unit is a typo; the value is degrees). Convert to the analysis N-referenced convention the same way the plant does: `azimuth_N = 180.0 + raw_south_referenced`. Confirm sign convention against the plant value (plant WS azimuth `-13` → `167`; string azimuth `-13` → `167`).
   - `string_tilt_deg` from `_col(df, "string", "tilt")`.
   - `string_commissioning_date` from `_col(df, "comissioning")` — **note the single-m spelling in the workbook**; match on `"comissioning"`. Values are Excel serial dates (e.g., `45544`); parse with `pd.to_datetime(value, unit="D", origin="1899-12-30")` for serials, or `pd.to_datetime` if already a date. Store ISO date string + derived `commissioning_year`.
   - Surface `bifacial` (already read) into the per-string spec consumed downstream.

3. **Surface per-inverter AC capacity + inverter-level AC power.** In `loader.py`, build an `inverter_ac` structure: per-inverter `max_ac_power_kw`/`capacity_kw_ac` from `specs["inverters"]`, and (per [C4](#2-the-six-preflight-checks)) extract measured inverter AC power from the CSV `level=="inverter"` rows keyed by `(timestamp, inverter_id)`. Add it to `plant_meta` (e.g., `plant_meta["inverter_ac_power"]` series) and `plant_meta["inverter_specs"]` dict. Batch 3 consumes these.

4. **Per-string config, not one representative panel.** Keep `ModuleConfig` (base/representative) for back-compat, but build a `plant_meta["string_specs"]` dict keyed by `string_label` carrying that string's `azimuth, tilt, commissioning_date, commissioning_year, bifacial, pv_capacity_kw, panel specs`. `_get_string_plate()` in `pipeline.py` already derives per-string `n_modules` from `pv_capacity`; keep that.

5. **Propagate to `extract_string_meta()`.** Add `commissioning_date`, `commissioning_year`, `pv_capacity`, `bifacial`, and per-string `azimuth`/`tilt` (already present once the loader stops broadcasting plant defaults — see Batch 5; in Batch 1 just make sure the columns flow through `split_into_string_dfs`). `clustering.build_peer_groups()` already reads `meta["commissioning_year"]` and the per-string `pv_capacity` column — wiring `commissioning_year` here **activates age-aware peer grouping for free**.

6. **Economics inputs.** Add to `SiteConfig`/`PipelineConfig`: `tariff` (already there, keep workbook/CLI override), `wash_cost_per_string_pkr` (or `wash_cost_per_kw_pkr` and/or `wash_cost_per_inverter_pkr`), and an optional `module_area_m2`/`gcr` if available. Energy-based economics (rate × tariff × kWh) is more robust than area-based; make area optional. Add a `--tariff` and `--wash-cost` CLI flag in `run.py`.

7. **`preflight_schema()`.** Implement the six checks; return a dict; log it at load. Store on `plant_meta["preflight"]`.

**New/changed config keys.** See [reference table](#6-recommended-thresholds-reference-all-new-config-keys): `wash_cost_per_string_pkr`, `wash_cost_per_kw_pkr`, `module_area_m2` (optional).

**Back-compat & guardrails.** No behaviour change to numbers in this batch. Keep `string_label`. Keep `ModuleConfig` representative panel. Everything new is additive. If any per-string field is missing for a given string, fall back to the plant default and record the fallback in `plant_meta["string_specs"][label]["_source"]`.

**Tests.** Add `inputs/specs_test.py` (or extend) asserting the three new Panel columns are read and az conversion is correct; assert preflight returns the expected dict for the sample workbook (C1✅ C2✅ C3⚠️ C4⚠️ C5❌ C6⚠️). Assert `string_uid` uniqueness logic.

**Done when.** `load_from_sources()` returns a `long_df` with per-string `azimuth`/`tilt`/`commissioning_date`/`bifacial`/`pv_capacity` columns *(values may still equal plant defaults until Batch 5 wires orientation, but the columns and `string_specs` exist)*, `plant_meta` carries `inverter_specs`, `inverter_ac_power`, `string_specs`, and `preflight`, and the pipeline still runs unchanged. **TODO markers** (per the plan): tag the orientation-consumption and bifacial-gain code paths `# SCHEMA-DEP C1/C3` and the inverter-AC path `# SCHEMA-DEP C4`.

---

## Batch 2 — Calendar-aware daily-series consolidation (the backbone)

**Goal.** Put every day-to-day mechanism on one shared, continuous calendar index with explicit missing-day handling and a minimum valid-day density, removing the duplicated and subtly inconsistent indexing logic. This is the single highest-value robustness fix and the backbone for Batches 7-8.

**Touches.** `pv_diag/daily.py` (the `daily_df` producer), `pv_diag/wash_detect.py`, `pv_diag/transient.py`, `pv_diag/soiling.py` (pre-slope), `pv_diag/pipeline.py` (so the shared series is built once and passed through). New small module suggested: `pv_diag/calendar_grid.py`.

**Current behaviour (verified).** `compute_daily_metrics()` returns one row per date *that has data* (`for date, sub in df.groupby("date")`). `wash_detect.detect_wash_events()`, `detect_distributed_recovery()`, `transient.detect_transient_events()`, and `soiling._trimmed_lr`/pre-slope all consume that gappy `daily_df` and difference adjacent rows (`np.diff(raw)`), so an N-day gap reads as a single 1-day step. The adaptive baseline (`adaptive_baseline.py`) and `sufficiency.py` are already calendar-window based — they are the model to match.

**Change.**
1. Add a function `to_calendar_grid(daily_df, freq="D") -> daily_df_reindexed` that reindexes the daily frame onto a continuous `pd.date_range(min_date, max_date, freq="D")`, inserting rows with `NaN` metrics and an explicit `is_present`/`n_valid=0` marker for missing days. Do **not** forward-fill NCI.
2. Migrate `wash_detect`, `transient`, and the `soiling` pre-slope to consume the gridded frame. Step/slope logic must treat a `NaN` day as a break, not a one-step transition: when differencing, compute deltas only between *present* days and explicitly suppress steps that span a gap longer than `max_step_gap_days` (new config; default 2). The 14-row look-back in `wash_detect` (`raw[i-14:i]`) becomes a **14-calendar-day** look-back on the gridded index (this also sets up Batch's wash-baseline fix in Batch 8/§ below — but here just make the window calendar-correct).
3. Add a **minimum valid-day density** gate: a calendar window is usable only if `present_days / window_days >= min_valid_day_density` (new config; default 0.4). Apply it where windows are formed (wash look-back, segment fits).
4. **Make it parallel/quick (the "make code parallel" ask folds in here).** Build the gridded series **once** per string in Pass 1 and pass it into Pass 2 and into wash/transient/soiling rather than recomputing the grid in each. Keep the existing joblib `Parallel(prefer="threads")` per-string parallelism; do not add nested parallelism.

**New/changed config keys.** `daily_grid_enabled: bool=True`, `max_step_gap_days: int=2`, `min_valid_day_density: float=0.4`. (Reference table.)

**Back-compat & guardrails.** Flag with `daily_grid_enabled`. When `False`, return the old gappy frame and old indexing. Keep all existing columns; add `is_present` and keep `n_valid`. Wash/transient/soiling outputs keep their existing schemas.

**Tests.** Extend `daily_test.py`, `wash_detect_test.py`, `transient_test.py`: construct a daily series with a deliberate 5-day gap straddling an NCI jump and assert (a) old code reports a spurious single-day step, (b) new code does not, (c) the 14-day look-back spans calendar days not rows.

**Done when.** Wash, transient, and pre-slope all read from the shared gridded series; a synthetic gap no longer creates a phantom one-day step; density gate suppresses sparse windows; pipeline runtime is not worse than before.

---

## Batch 3 — Curtailment rework

**Goal.** Detect each curtailment type at the level where the physical constraint lives, then propagate the flag down to string rows for NCI/soiling exclusion — and close the dead-string-as-suppression loophole. Correct flags must exist before the baseline reads them.

**Touches.** `pv_diag/curtailment.py` (all three detectors + summaries + loss), `pv_diag/constants.py` (a new non-disqualifying flag), `pv_diag/pipeline.py` (run detection at inverter level before splitting), `pv_diag/config.py`. Consumes Batch 1's `inverter_specs` + `inverter_ac_power`.

**Current behaviour (verified — all confirmed defects).**
- `detect_curtailment()` runs on the concatenated `long_df` (all strings) and uses `np.diff` on the whole frame — plateaus can cross string boundaries.
- **Statistical clipping is per-string and hard-coded:** `p_cap = p_ac_max_kw*1000/n_strings_per_inv`, `high_poa = poa > 800`, `near_cap = P > (1-clip_band_pct)*p_cap` (i.e. > 95%). It uses string **DC** power `P`, assumes equal power-sharing across strings, and misses every below-nameplate setpoint.
- **Suppression is per-string:** `poa > 400 & power_ratio < 0.20` → `CURT_SUPPRESSED`, which is in `DISQUALIFYING`. A dead/blown/disconnected string is observationally identical and gets its evidence excluded as "curtailment"; the fault classifier never sees it.
- **Voltage-rise** is a 5-condition per-string detector → `CURT_VOLTAGE_RISE`, also in `DISQUALIFYING`.

**Change.**

1. **Statistical clipping → inverter level, AC power, adaptive plateau.**
   - Operate on **inverter AC active power** (Batch 1's `inverter_ac_power`), not string DC `P`. (Per the user: clipping is an AC phenomenon.)
   - **Detect the plateau adaptively**, not against 95% of nameplate: a flat top is a *low-variance* segment sitting at a *repeated daily maximum* value, wherever that value sits. Concretely: per inverter per day, find the daily max AC; flag a contiguous run as clipping when AC stays within `clip_band_rel` (relative, e.g. ±1-2%) of a value that recurs as the daily max on `clip_repeat_days` (≥ N days) and the run's rolling CV < `clip_max_cv`, with dwell ≥ `clip_min_dwell`. This catches below-nameplate export setpoints that the fixed 95% gate misses entirely.
   - **Distinguish export-limit vs DC/AC-ratio clipping for reporting:** if *all* inverters plateau simultaneously and the plant total is flat at a point-of-interconnection limit, label it `CURT_EXPORT_LIMIT` (contractual/grid — recoverable revenue); if a single inverter clips at its own sizing limit, label it inverter `CURT_STATISTICAL` (DC/AC-ratio design artifact — not recoverable). Keep both under the curtailment umbrella but report them separately.
   - **Per-inverter capacity:** use each inverter's own `max_ac_power_kw` from specs; strings on one inverter may differ in capacity, so do **not** divide plant nameplate by a fixed strings-per-inverter.
   - **Propagate down:** once an inverter interval is flagged, mark every string on that inverter `CURT_STATISTICAL` (or `CURT_EXPORT_LIMIT`) for those timestamps. Different strings can be different capacities — propagate the flag, not a per-string power threshold.

2. **Suppression → inverter level + dead-string fix (the critical loophole).**
   - Decide suppression at the **inverter** level: bright sun (`poa > suppression_poa_threshold`) AND the **inverter's** AC ≤ `suppression_power_ratio` × expected inverter AC, with dwell. Grid-commanded suppression hits the whole inverter.
   - **Do not flag a single low string as `CURT_SUPPRESSED`.** If one string is low while its co-oriented peers on the same inverter are producing normally, that is a string-local fault (blown fuse, disconnect, failed connector, severe local shading), **not** grid suppression. Add **cross-string consensus**: suppression requires the majority of an inverter's strings to be low together.
   - Add a new **non-disqualifying** flag `STRING_UNDERPERFORM` (a new bit in `QUALITY_FLAGS`, **excluded from `DISQUALIFYING`**) for the lone-low-string case, so the fault classifier (`classification.py`) *does* see those rows instead of having them silently dropped as curtailment. This is the misattribution fix the user is most concerned about.

3. **Voltage-rise → keep string-level signature + inverter-level causal confirmation + cross-string consensus.** The user is right that the signature is genuinely string-level. Keep the existing 5-condition detector but **require corroboration** before flagging `CURT_VOLTAGE_RISE`: (a) the same inverter shows a consistent grid-voltage / multi-string Vdc-rise pattern (causal confirmation), and (b) cross-string consensus (more than one string on the inverter shows it). A single string showing Vdc rise alone is an IV-curve/module artifact — flag it `STRING_UNDERPERFORM` for the classifier, not curtailment.

4. **Run detection grouped by inverter** in `pipeline.py` (group `long_df` by `inverter_id` before the plateau/dwell logic) so `np.diff` never crosses inverter or string boundaries.

**New/changed config keys.** `curtailment_inverter_level_enabled: bool=True`, `clip_band_rel: float=0.02`, `clip_repeat_days: int=3`, `clip_max_cv: float=0.03`, `clip_min_dwell: int=3` (reuse), `suppression_consensus_frac: float=0.5`, `vr_consensus_min_strings: int=2`. New flags `CURT_EXPORT_LIMIT`, `STRING_UNDERPERFORM` in `constants.py` (the latter **must not** be added to `DISQUALIFYING`).

**Back-compat & guardrails.** Flag with `curtailment_inverter_level_enabled`; `False` restores the per-string detectors exactly. Keep `total_curt_kwh` alias and the existing summary keys; add new split keys (`curt_export_limit_kwh`, etc.). If [C4](#2-the-six-preflight-checks) is FALSE (no measured inverter AC), reconstruct inverter AC as Σ(string DC)×efficiency and set `inverter_ac_source="reconstructed_dc"` — and lower confidence on those flags.

**Tests.** Extend `curtailment_test.py`: (a) a below-nameplate export setpoint plateau is now caught (old code missed it); (b) a single dead string under bright sun is flagged `STRING_UNDERPERFORM` (non-disqualifying) and **not** `CURT_SUPPRESSED`, and survives into classification; (c) simultaneous all-inverter plateau is labelled `CURT_EXPORT_LIMIT`; (d) plateau detection does not bleed across inverter boundaries.

**Done when.** Clipping/suppression are inverter-level on AC power with adaptive (not 95%) plateaus and per-inverter capacity; a lone dead string is no longer absorbed as curtailment; export-limit vs sizing clipping are reported separately; the spine's downstream baseline now reads clean flags.

---

## Batch 4 — String-wise age baseline + age-relative gates/bands

**Goal.** Compute the degradation baseline **per string** from its own commissioning date, then re-express the acceptance gates and band cuts relative to each string's age baseline so older plants/strings are not pushed off the adaptive layer. Runs after curtailment so peer sets are built on clean flags.

**Touches.** `pv_diag/degradation.py`, `pv_diag/pipeline.py` (step `[5/9]` baseline + per-string baseline plumbing), `pv_diag/adaptive_baseline.py` (Gates A/B/C), `pv_diag/classification.py` (band cuts), `pv_diag/config.py`. Consumes Batch 1's per-string `commissioning_date`.

**Current behaviour (verified).**
- `pipeline.py` computes **one** plant-wise baseline at `[5/9]`: `degradation_baseline(cfg.plant.commissioning_date, ref_date, …)` and passes that single float `baseline` to every string's `compute_daily_metrics` and every gate. `degradation_baseline()` already accepts a `commissioning_date` argument — it is simply only ever called once.
- Gate A: reject if `clean_ref < cfg.adaptive_min_p95` (**0.92**, absolute/nameplate). Gate B: reject if `reference_method=="p95_fallback"` and `clean_ref < cfg.adaptive_no_rain_floor` (**0.96**, absolute). Gate C: reject if `p95 < peer_median - cfg.adaptive_cluster_gate` (**0.05** margin).
- Classification bands (`classification.py`): `_BAND_CLEAN=0.97`, `_BAND_LT=0.93`, `_BAND_MOD=0.85`, compared against `mean_nci` from `pick_nci_column()`.

**Why this matters (worked example).** The age baseline is `1 − LID·min(yr,1) − annual·max(yr−1,0)`. With this plant's workbook values (`first_year_degradation=1.0%`, `annual=0.4%`):
- **CCI Faisalabad (~1.75 yr):** baseline ≈ `1 − 0.010 − 0.004·0.75 ≈ 0.987`. Age-scaling barely moves the gates here.
- **A 10-yr string:** baseline ≈ `1 − 0.010 − 0.004·9 ≈ 0.954`. Its *true clean* nameplate-referenced NCI is ≈ 0.95 — already **below** the 0.96 no-rain floor and brushing the 0.97 clean band. The fixed gates would wrongly push a clean 10-yr string off Layer 1 onto the plate fallback and could mislabel it. **Age-scaling fixes exactly this.**

**Change.**

1. **Per-string baseline.** In `pipeline.py`, compute a baseline per string from `string_specs[label]["commissioning_date"]` (fallback to plant date per [C2](#2-the-six-preflight-checks)). Pass each string its own `baseline` into `compute_daily_metrics` and into the gates. Keep `degradation_baseline()` as-is (it already supports per-date calls); just call it per string. Per [C2](#2-the-six-preflight-checks) fallback, set `age_source` provenance per string.

2. **The data-driven reference itself needs no age input.** Do not change how `clean_ref`/P95 is *measured* — only how it is *gated*.

3. **Age-relative gates** (in `adaptive_baseline.py`). Make the floors a fraction of the string's age baseline:
   - Gate A effective floor = `adaptive_min_p95 × age_baseline` (i.e. `0.92 × age_baseline`).
   - Gate B no-rain floor = `adaptive_no_rain_floor × age_baseline` (i.e. `0.96 × age_baseline`).
   - Gate C margin: the cluster gate compares two *same-age* peers' P95s, so the absolute 0.05 *difference* is already roughly age-neutral; still, scale it as `adaptive_cluster_gate × age_baseline` for consistency with the user's request (the spread compresses slightly for older strings, which is physically reasonable).
   - **Are 0.92 / 0.96 still reasonable?** Yes, as *coefficients on the age baseline*. 0.92 means "accept a clean reference down to 8% below same-age-clean" and 0.96 means "with no verified wash anchor, require within 4% of same-age-clean." Those tolerances are sensible industrial values; what was wrong was applying them against nameplate. Keep the coefficients; multiply by `age_baseline`.

4. **Age-relative band cuts** (in `classification.py`) — **but only when the active NCI column is nameplate-referenced.** Important subtlety: `pick_nci_column()` prefers `NCI_adaptive_noon` (÷ adaptive clean ref — already age-normalised) > `NCI_relative_noon` (IAM-corrected but **nameplate**) > `NCI_corrected_noon` (÷ age baseline — already age-normalised) > `NCI_noon` (**nameplate**). So:
   - If the chosen column is `NCI_noon` or `NCI_relative_noon` (nameplate-referenced), scale the bands: clean = `0.97 × age_baseline`, lt = `0.93 × age_baseline`, mod = `0.85 × age_baseline`.
   - If the chosen column is `NCI_adaptive_noon` or `NCI_corrected_noon` (already age-normalised), leave the bands at 0.97/0.93/0.85 — scaling twice would double-count age. Detect which column is active (it's already returned as `nci_col_used`) and branch.

**New/changed config keys.** `age_relative_gates_enabled: bool=True`, `age_relative_bands_enabled: bool=True`. Existing `adaptive_min_p95`, `adaptive_no_rain_floor`, `adaptive_cluster_gate` now act as *coefficients*; keep their values. (Reference table.)

**Back-compat & guardrails.** Flags restore absolute gates/bands. Record per-string `age_baseline` and `age_source` in `AdaptiveBaselineResult` and in classification axes. `build_peer_groups()` already consumes `commissioning_year` — confirm Batch 1 populated it so peers are age-bracketed.

**Tests.** Extend `adaptive_baseline_test.py` and add a classification test: a synthetic 10-yr clean string with nameplate NCI≈0.95 is **accepted** on Layer 1 under age-relative gates (rejected under absolute), and is **not** mislabelled as soiled when the nameplate column is active; a young string is unaffected.

**Done when.** Each string has its own baseline; gates and (nameplate-only) bands scale by it; the 10-yr-string failure mode is gone; young plants behave as before.

---

## VALIDATION GATE (after Batch 4 — not a prompt)

**Do not start Batch 5 until this passes.** The spine reshapes the numbers; confirm verdicts move as expected on real strings before building refinements, or you risk redoing Batches 5-9.

Run on the real plant:

```bash
uv run python run.py \
  --csv "<measured CCI Faisalabad CSV>" \
  --specs "src/soiling_analysis/RequireInputData/RequireInputData.xlsx" \
  --out-dir output/ --n-jobs 1
```

Checklist:
- [ ] Pipeline completes with no per-string `error` keys for the bulk of the 147 strings (investigate any that error).
- [ ] Per-string `string_uid` displays as `<inverter>_<string>` in the report.
- [ ] Per-string `age_baseline` appears and differs across strings only if their commissioning dates differ (for this plant they're identical → all ≈0.987; that's expected).
- [ ] A spot-checked inverter shows clipping flagged on **AC** power with an adaptive plateau; a deliberately-low string is `STRING_UNDERPERFORM`, not `CURT_SUPPRESSED`.
- [ ] Adaptive-layer resolution counts (`layer_counts`) did **not** collapse to mostly plate fallback (Layer 3) — if they did, the age-relative gates or the clean-flag fix may be mis-wired.
- [ ] Headline soiling and curtailment kWh are physically plausible vs the prior run; explain any large swing.

Record before/after verdict counts and the `layer_counts` dict in a short note. Only then proceed.

---

# REFINEMENTS — make a trustworthy output sharper

---

## Batch 5 — POA transposition + per-string orientation wiring + bifacial gain

**Goal.** Stop using the weather-station-plane irradiance for differently-oriented strings. Compute one transposed POA per orientation cluster, shared among co-oriented strings, and wire per-string azimuth/tilt into the IAM. Add bifacial gain where data allows.

**Touches.** New `pv_diag/transposition.py`; `pv_diag/orientation.py` (it already has `_solar_position`, `_clearsky_ghi`, `_ghi_to_poa`, `compute_clearsky_poa`, `poa_health_check`); `pv_diag/daily.py` (`compute_daily_metrics` already accepts `azimuth`/`tilt` — see below); `pv_diag/pipeline.py` (pass per-string orientation); `loader.py` (stop broadcasting plant az/tilt). Consumes Batch 1 ([C1], [C3]).

**Current behaviour (verified).**
- The measured plant irradiance is taken at the **weather-station** orientation (`tilt=15°, azimuth=−13°` for this plant) and used directly for every string. `loader.py` writes the **plant** az/tilt onto all string rows.
- `compute_daily_metrics()` **already accepts `azimuth`/`tilt` parameters** for the IAM, **but `pipeline.py` never passes them** (both `_pass1_string` and `_process_one_string` call it without orientation), so the IAM uses `cfg.plant.default_azimuth/tilt`. The cosine correction therefore ignores the very mismatch in question — exactly as the user states.
- `orientation.py::_ghi_to_poa()` is a crude daily-averaged-AOI transposition; `compute_clearsky_poa()` exists but is only used in `poa_health_check`.

**Change.**

1. **Dedicated transposition module, one POA per orientation cluster.** Build `transposition.py` that, for each orientation cluster (the existing `cluster_by_azimuth_tilt` bins), produces a transposed POA series and shares it among all co-oriented strings:
   - **Preferred (robust, industrial):** use **pvlib** (already a dependency). Decompose the measured weather-station POA to GHI/DNI/DHI (e.g., invert with the known WS tilt/azimuth + `pvlib` `erbs`/`disc` on the implied GHI), then re-transpose to each cluster's tilt/azimuth with `pvlib.irradiance.get_total_irradiance` (Hay-Davies or Perez sky-diffuse model; Perez is the standard for accuracy). This is the IEC 61724 / NREL-style approach.
   - **If decomposition is unreliable** (POA-only, no GHI): apply a geometric POA-to-POA transposition between the WS plane and the string plane using `_solar_position` + AOI ratio for the beam component and a tilt view-factor ratio for diffuse — an upgrade of the existing `_ghi_to_poa`, but using the **real per-sample solar azimuth** (not the daily-averaged approximation).
   - For this plant the WS and strings are co-oriented (both `15°/−13°`), so the transposed POA ≈ measured POA — but the module must be correct for the general multi-orientation plant.

2. **Wire per-string orientation into the IAM.** In `pipeline.py`, pass `azimuth=string_meta[label]["azimuth"]`, `tilt=string_meta[label]["tilt"]` into both `compute_daily_metrics` calls. In `loader.py`, stop broadcasting plant az/tilt — use the per-string values surfaced in Batch 1 (fallback to plant default per [C1]). This is mostly a **wiring fix**; `compute_daily_metrics` already supports it.

3. **Bifacial gain (data-gated, [C3]).**
   - If a **rear/GHI irradiance** column exists: compute effective irradiance = front POA + bifaciality × rear POA (bifaciality from the panel datasheet if present, else a default ≈0.70), and use the effective irradiance in the expected-current/power computation.
   - If **no rear-irradiance** (this plant's case): apply a **modeled** bifacial gain — either `pvlib.bifacial` (needs GCR/row geometry, likely unavailable) or a single conservative gain factor `bifacial_gain_default` (e.g. 1.04-1.08) applied to the expected POA, **clearly flagged** `bifacial_gain_source="modeled"`. If even that is too speculative, **drop bifacial gain from this batch** and leave a TODO — the plan explicitly allows dropping it when rear data is absent. Do not invent rear irradiance.

**New/changed config keys.** `poa_transposition_enabled: bool=True`, `transposition_model: str="perez"` (`"haydavies"`/`"isotropic"` fallbacks), `bifacial_gain_enabled: bool` (gated on [C3]), `bifaciality: float=0.70`, `bifacial_gain_default: float=1.05`. (Reference table.)

**Back-compat & guardrails.** Flags restore measured-POA-for-all + plant-default IAM. Keep measured POA in a column (`POA`) and add `POA_transposed` rather than overwriting, so you can compare. Per-cluster caching: compute each cluster's transposed POA once, not per string.

**Tests.** New `transposition_test.py`: a synthetic east-facing and west-facing string share the WS GHI but get distinct transposed POA with the correct AM/PM tilt; IAM now uses per-string orientation (assert `compute_daily_metrics` produces different `NCI_relative_noon` for two orientations given identical raw current). Assert co-oriented case ≈ measured POA.

**Done when.** Each orientation cluster has its own transposed POA; the IAM uses per-string az/tilt; bifacial gain is applied only where data supports it and is flagged as measured vs modeled.

---

## Batch 6 — Clear-sky service + clean-window SDM fitting + robust SDM loss

**Goal.** Replace absolute-W/m² *estimation* gates with a clear-sky-index + stability test where it matters; fit the SDM on clean, recently-washed, low-variability windows; and make the SDM optimiser robust to outliers. Ships paired with Batch 7 (do not merge Batch 6 without Batch 7's empty-set policy — see note).

**Touches.** New `pv_diag/clearsky_quality.py`; `pv_diag/pipeline.py` (`_select_quality_days`, `_fit_sdm`, and the NCI_noon interval mask in `compute_daily_metrics`); `pv_diag/sdm.py` (`fit_single_diode` loss); `pv_diag/config.py`. Builds on `orientation.compute_clearsky_poa`.

**Current behaviour (verified).**
- `_select_quality_days()` (in `pipeline.py`) requires `midday POA.max() ≥ 600` and `cv ≤ 0.20` and `curt_frac < 0.30` and rain below threshold. The **fixed 600 W/m²** floor starves the SDM in winter exactly when a winter refit would happen. Elsewhere there are fixed gates: `min_peak_poa=700`, statistical-clip `poa>800`, suppression `poa>400`, `daily.py` masks `POA>100`, quality.py `poa<50/<100`.
- The SDM training set is quality-filtered but **soiled days are not removed** — a heavily soiled but clear, stable day passes every filter, depressing the fitted Isc ratio and biasing the degradation verdict.
- `fit_single_diode()` calls `scipy.optimize.least_squares(residuals, p0, bounds, max_nfev=300, method="trf")` with sigma-weighting and 0.5-weighted Voc anchors but the **default linear loss** — not robust to high-leverage outliers (a curtailed sample or transient that slipped the filter still tilts the solution).

**Change.**

1. **One `clear_sky_quality` service.** Build `clearsky_quality.py` exposing a per-interval quality at the *estimation* level:
   - `Kc = measured_POA / clearsky_POA` (use `orientation.compute_clearsky_poa`, or upgrade to `pvlib` Ineichen/Perez-Ineichen clear-sky with Linke turbidity for accuracy).
   - Stability = rolling coefficient-of-variation of `Kc` over a short window (generalise the CV gate the SDM already uses — the user correctly identifies this as the right philosophy).
   - A sample passes when `Kc` is within a tolerance of 1.0 (clearness) **and** rolling-CV is low (stability). Stability-weight the samples.
2. **Wire it into exactly two sites** (per the plan): `_select_quality_days()` (replace the fixed 600 W/m² midday floor with the Kc+stability test) and the **NCI_noon interval mask** in `compute_daily_metrics`. **Leave `quality.py`, `losses.py`, `sufficiency.py`, and the curtailment brightness floors on absolute W/m²** — those gate physical validity and energy accounting, not estimation. (This is the recurring rule: *Kc gates estimation; absolute irradiance gates physical validity and energy accounting.*) Answer to the user's "everywhere or specific modules?": **specific modules only** (the two estimation sites); the rest stay on W/m².
3. **Clean-window SDM fitting (recovery-anchored, clean *and* recent).** Filter the SDM training df to clean, high-clear-sky, low-variability windows shortly after confirmed cleanings — the same recovery-anchored idea the baseline already uses, applied to the SDM. **Answer to "cleanest vs most recent?":** use windows that are **both** — prefer the *most recent* clean window after a confirmed cleaning, with recency as the tie-breaker among clean windows. Rationale: pure-recent risks fitting on a soiled recent stretch (biasing degradation); pure-cleanest risks fitting on a stale window that predates real degradation. Recovery-anchored-and-recent gives an unsoiled *and* current fingerprint. When [C5](#2-the-six-preflight-checks) is FALSE (no rain anchors, this plant), fall back to the cleanest high-Kc low-CV window in the most recent N days, flagged `sdm_window_source="recent_clean_no_anchor"`.
4. **Robust SDM loss.** Change the `least_squares` call to a robust loss: `loss="soft_l1"` (or `"huber"`/`"cauchy"`) with an `f_scale` set to a robust scale of the residuals (e.g. `f_scale ≈ 1.0` in the sigma-normalised residual space, or 1.4826·MAD of an initial linear-loss residual). This down-weights outliers automatically — the cheapest high-value robustness fix, as the user notes. Keep bounds, sigma-weighting, and anchor weighting.

**New/changed config keys.** `clearsky_quality_enabled: bool=True`, `kc_tolerance: float=0.10` (|Kc−1| ≤ 0.10... tune), `kc_stability_cv_max: float=0.10`, `kc_window_min: float=30.0`, `sdm_clean_window_enabled: bool=True`, `sdm_recent_days: int=45`, `robust_sdm_loss: str="soft_l1"`, `robust_sdm_f_scale: float=1.0`. (Reference table.)

**Back-compat & guardrails.** Each sub-change is independently flagged. `clearsky_quality_enabled=False` restores the 600 W/m² gate; `robust_sdm_loss="linear"` restores the old loss; `sdm_clean_window_enabled=False` restores full-quality-day fitting. Keep the existing `df_for_sdm < 100 rows → fall back to full df` safety in `_fit_sdm`, and **widen** that safety net so a winter Kc filter that starves the set falls back gracefully (ties into Batch 7).

**Tests.** Extend `sdm_test.py`: (a) injected high-leverage outliers move the linear-loss fit but not the soft-L1 fit; (b) a soiled-but-clear day is excluded from the clean-window training set; (c) the Kc filter accepts a clear winter day that the 600 W/m² floor rejected. Add `clearsky_quality_test.py`.

**Done when.** Estimation gates are Kc+stability at the two sites only; SDM trains on clean+recent windows; the optimiser uses a robust loss; physical/energy gates remain on W/m².

> **Pairing note (from the plan):** shipping the clear-sky filter **without** Batch 7's empty-set/monsoon policy is a regression during a Lahore monsoon (the Kc filter can return an empty set for days). Implement Batch 7 immediately after, or in the same change set.

---

## Batch 7 — Monsoon/smog fallback policy + last-good state store

**Goal.** A graceful cascade for extended low-quality periods (monsoon and smog routinely last days/weeks in Lahore), plus a small persistent store so the system can hold the last-good SDM/baseline instead of silently degrading. Ships with Batch 6. Depends on Batches 2, 4, 6.

**Touches.** `pv_diag/adaptive_baseline.py` (the layer cascade in `resolve_clean_baseline`), `pv_diag/sufficiency.py` (a "Limited" tier for drought), new `pv_diag/state_store.py` (persistent last-good store), `pv_diag/clearsky_quality.py` (empty-set policy), `pv_diag/config.py`.

**Current behaviour (verified).** The adaptive baseline has a **3-layer** cascade: Layer 1 per-string adaptive → Layer 2 peer-group median (with a `dry_season_threshold=30`-day plate blend: `weight_plate = clip((days−30)/30, 0, 0.7)`) → Layer 3 plate-age baseline. `sufficiency.py` already has Good/Limited/Poor/Skipped. There is **no persistence** between runs and **no explicit empty-Kc-set policy**.

**Change.**

1. **Five-tier cascade.** Extend the existing layers into the cascade the user describes, in order: (1) **widen the window** (extend `adaptive_window_days` when the normal window is too sparse) → (2) **peer borrow** (existing Layer 2 peer-group median) → (3) **hold-last-good** (NEW layer, inserted *between* peer and plate: read the last-good SDM + baseline from the state store) → (4) **dry-season-blend decay** (existing plate blend, but as a decaying fallback) → (5) **suppress SDM refit + sufficiency "Limited"** (do not refit SDM on a starved window; mark the verdict "Limited"). Record which tier produced the value in `AdaptiveBaselineResult.source`.
2. **Drought detection at plant scope.** Decide "we are in an extended low-quality period" at the **plant** level (weather-driven: most strings simultaneously sparse/low-Kc), distinguished from a **string-specific** problem (one string sparse while peers are fine → that's a fault, handled by Batch 3's `STRING_UNDERPERFORM`, not a weather fallback). Use peers to make this distinction.
3. **Persistent last-good store.** Build `state_store.py` — a small JSON/parquet keyed by `string_uid` holding the last-good SDM params, last-good clean baseline, and timestamps. Read at start, write at end of a successful run. This is the substrate for tier 3 and is **used wherever last-good values are held** (keep it visible, not buried). Make the path configurable; degrade gracefully if absent (first run has no last-good → skip tier 3).
4. **Kc empty-set policy.** In `clearsky_quality.py`, when the Kc+stability filter returns an empty (or below-density) set for a window — the monsoon/smog case — do **not** fail: trigger the cascade (widen window → peer → hold-last-good), and **suppress the SDM refit** rather than fitting on garbage. **Answer to "how are smog/monsoon days adjusted?":** they are not force-fit; estimation is suspended and the system holds last-good, blends toward plate as the drought lengthens, and reports sufficiency "Limited" with provenance — so a multi-day monsoon does not silently corrupt the fingerprint.
5. **Provenance fields.** Extend `AdaptiveBaselineResult` with `held_from_date`, `tier`, `drought_flag`, `blend_weight`, so no held/blended value is silent (ties to the cross-cutting provenance rule).

**New/changed config keys.** `monsoon_fallback_enabled: bool=True`, `window_widen_max_days: int=180`, `hold_last_good_enabled: bool=True`, `state_store_path: str` (configurable), `drought_min_string_frac: float=0.6`, `drought_min_days: int=5`. Reuse `dry_season_threshold`. (Reference table.)

**Back-compat & guardrails.** Flags restore the 3-layer cascade and disable persistence. First run with no store → tier 3 is skipped silently. Never block a run because the store is missing/locked. Keep all existing `source`/`layer` values; add the new tiers as new `source` strings.

**Tests.** Extend `adaptive_baseline_test.py`: a simulated 10-day plant-wide low-Kc drought triggers widen→peer→hold-last-good in order, suppresses SDM refit, and yields sufficiency "Limited" with provenance; a single sparse string (peers fine) does **not** trigger the weather fallback.

**Done when.** Extended low-quality periods degrade gracefully through the five tiers, last-good values persist across runs, SDM is not force-fit on monsoon/smog windows, and every fallback is logged.

---

## Batch 8 — Robust soiling regression + transient handling (+ deprecate plate.py)

**Goal.** Replace trimmed-OLS with a true robust regressor, remove transients *before* fitting and feed them back into wash/segmentation, and deprecate `plate.py` now that manufacturer specs are used directly. Sits on Batch 2's consolidated calendar series.

**Touches.** `pv_diag/soiling.py` (`_trimmed_lr` → robust; segmentation), `pv_diag/transient.py` (feed-back wiring), `pv_diag/pipeline.py` (use transients before soiling/wash; remove `infer_plate_params` from the spine), `pv_diag/plate.py` (deprecate), `pv_diag/config.py`.

**Current behaviour (verified).**
- `soiling._trimmed_lr()` is `np.polyfit` + drop the worst 10% residuals + refit — **not** a true robust regressor (low breakdown point). Segments are split on **wash `event_dates` only**; transients are **not** excluded, so single-day dips still pull the regression.
- `transient.detect_transient_events()` detects dips but the pipeline only stores `res["transients"]`; they are **never** fed into wash detection, segmentation, or slope fitting. Reporting only.
- `plate.infer_plate_params()` has its Imp inference **already disabled** (commented out); it only lightly updates Voc. `_get_string_plate()` in `pipeline.py` derives per-string `n_modules` from `pv_capacity` — **this per-string scaling must be preserved** when plate.py is deprecated.

**Change.**
1. **Robust regression.** Replace `_trimmed_lr` with **Theil-Sen** (`scipy.stats.theilslopes`, or `sklearn.linear_model.TheilSenRegressor`) as the default, with **Huber** (`sklearn.linear_model.HuberRegressor`) as an alternative — both have far higher breakdown points and are the standard for slope estimation on noisy operational series (RdTools uses Theil-Sen for SRR). Keep the segment-weighted aggregation, the slope cap for loss accounting, the CI, and the `is_slope_significant` gate. Report slope SE consistently (Theil-Sen gives a confidence interval directly).
2. **Pre-fit transient removal + feedback.** Detect transients (Batch 2's gridded series), then (a) **exclude** transient days before the soiling regression, (b) **prevent** a transient from being mistaken for a segment boundary, and (c) **feed transients into wash detection** so a one-day dip+rebound is not read as a wash step. A one-day anomalous dip must no longer sit inside a segment and tug the regression.
3. **Deprecate `plate.py`.** Use the manufacturer specs surfaced in Batch 1 (`string_specs` / `ModuleConfig`) as the nameplate directly. Remove `infer_plate_params` from the spine in `pipeline.py` (step `[3/9]`), **but keep the per-string `n_modules`-from-`pv_capacity` logic in `_get_string_plate()`** (or move it into the loader/Batch-1 `string_specs`). Leave `plate.py` in the tree but mark it deprecated and unused, or fold the surviving `estimate_cells_in_series` helper elsewhere — do not delete in the same batch that removes its caller; deprecate first, delete in a follow-up once tests are green.

**New/changed config keys.** `robust_soiling_regression: str="theilsen"` (`"huber"`/`"trimmed_ols"` to restore old), `transient_prefilter_enabled: bool=True`. (Reference table.)

**Back-compat & guardrails.** `robust_soiling_regression="trimmed_ols"` restores `_trimmed_lr`. `transient_prefilter_enabled=False` restores the old "detect-only" behaviour. Keep `soiling`'s output schema (`srr_pct_per_day`, `segments`, `weighted_soiling_loss_pct`, …) unchanged. Removing `infer_plate_params` from the spine must not change `n_modules` handling — verify against the sample plant's variable string lengths (16965/16380/15210 W strings).

**Tests.** Extend `sdm_test.py`/add `soiling_test.py` + `transient_test.py`: (a) injected single-day dip tugs trimmed-OLS slope but not Theil-Sen; (b) a transient day is excluded from the fit and does not split a segment; (c) per-string `n_modules` is still correct after `plate.py` deprecation for a 26-panel vs 29-panel string.

**Done when.** Soiling slope uses Theil-Sen/Huber, transients are removed before fitting and inform wash/segmentation, `plate.py` is deprecated without breaking per-string scaling.

---

## Batch 9 — Reporting: cleaning economics + operational view + roll-ups + flags

**Goal.** Turn a trustworthy estimate into an actionable decision: a cleaning-economics recommendation, a short-horizon operational view, per-MPPT/inverter roll-ups, and at-a-glance contamination flags. Needs Batch 1's economics inputs.

**Touches.** `pv_diag/losses.py` (economics), `pv_diag/excel_export.py` (new sheets), `pv_diag/classification.py` (surface flags — most already exist), `pv_diag/config.py`. The export currently writes 16 sheets (`00_Run_Summary` … `12_Classification`, losses, etc.).

**Current behaviour (verified).** `losses.quantify_string_losses()` computes `soiling_kwh`, `soiling_pkr` (= kWh × tariff), `total_avoidable_*`, `annualised_*`, and an `unattributed_loss_kwh` for fault verdicts; `aggregate_plant_losses()` sums across strings. There is **no** cleaning-economics recommendation (no recommended date, recoverable energy, payback) and **no** explicit last-7/30-day operational view or per-MPPT/inverter roll-up sheet. `classification.py` already produces contamination flags (`has_shading_flag`, `has_degradation_flag`, `baseline_disagreement_flag`).

**Change.**
1. **Cleaning-economics recommendation.** Add a function computing, per string and per inverter: current soiling rate (%/day, from Batch 8) × tariff × energy (or area) → daily PKR loss; compare to **wash cost** (Batch 1's `wash_cost_*`) to yield: **recommended cleaning date** (when cumulative recoverable PKR since last clean exceeds wash cost, i.e. payback ≤ horizon), **projected recoverable kWh and PKR**, and **payback period**. Standard soiling-economics formula:
   - `daily_loss_pkr = soiling_rate_per_day × expected_daily_energy_kwh × tariff`
   - `days_to_payback = wash_cost_pkr / daily_loss_pkr`
   - recommend cleaning when `days_since_last_clean × daily_loss_pkr ≥ wash_cost_pkr` (or when projected loss before next opportunity exceeds the wash cost). Surface assumptions (tariff source, wash-cost source) per [C6](#2-the-six-preflight-checks).
2. **Operational view.** Add "last 7 days" and "last 30 days" summaries per string (soiling rate, mean NCI, availability, curtailment, any new flags) so a reviewer sees the short-horizon state, not just the full-window verdict.
3. **Roll-ups.** Add per-MPPT and per-inverter aggregations (mean/median NCI, soiling rate, curtailment %, count of strings by verdict). These let a reviewer find a bad inverter at a glance.
4. **Contamination flags surfaced.** Put the existing flags (orientation/shading, curtailment, degradation, baseline disagreement, `STRING_UNDERPERFORM`, age/orientation/bifacial provenance, fallback tier) into a compact per-string flag column block so each verdict can be trusted or discounted at a glance. No verdict should be shown without its provenance.
5. New export sheets: `13_Cleaning_Economics`, `14_Operational_View`, `15_Inverter_MPPT_Rollup` (renumber consistently; keep existing sheets and names).

**New/changed config keys.** Economics keys from Batch 1; `operational_view_windows: tuple=(7,30)`. (Reference table.)

**Back-compat & guardrails.** All additive — new sheets, new keys, new columns. Existing 16 sheets and their schemas are unchanged. If economics inputs are defaulted ([C6]), the recommendation still renders but is clearly labelled "inputs defaulted." Use `utils._scalar()` for every cell (the export sanitiser) to keep openpyxl happy.

**Tests.** Add an export smoke test asserting the new sheets exist and the economics recommendation is internally consistent (payback = wash_cost / daily_loss); a string with zero soiling yields "no cleaning recommended."

**Done when.** The report carries a defensible cleaning recommendation with payback, a 7/30-day operational view, inverter/MPPT roll-ups, and a per-string flag block — and degrades cleanly when economics inputs are defaulted.

---

## 4. Mapping: Word-doc flaw → batch

| Word-doc section | Batch |
|---|---|
| String Identifier Modification | 1 |
| Input Data Completeness (per-entity capacity, bifacial surfacing) | 1 (gain in 5) |
| Continuous Days vs. Valid Samples + "make code parallel/quick" | 2 |
| Curtailment Detection & Loss Accounting (statistical / suppression / voltage-rise) | 3 |
| Baseline Calculation (string-wise) | 4 |
| Dynamic Thresholding (age-relative gates/bands) | 4 |
| Orientation Mismatch (POA transposition + IAM wiring) | 5 |
| Irradiance Threshold vs. Clear-Sky Index | 6 |
| Impact of Soiled String Data on SDM (clean-window fitting) | 6 |
| Outlier Treatment Prior to SDM (robust loss) | 6 |
| (Monsoon/smog fallback — paired with clear-sky) | 7 |
| Sensitivity of Soiling-Rate Estimation (Theil-Sen/Huber) | 8 |
| Redundant & Unused Logic (transient feedback; deprecate plate.py) | 8 |
| Wash Detection Baseline Validity (P90-P95, gap-aware) | 2 (calendar) + 8 (baseline statistic — see below) |
| Verdict & Reporting Structure (cleaning economics, operational view, roll-ups, flags) | 9 |

> **Wash-baseline P90-P95 note (user's question answered):** Yes — replace the 14-row **max** with a **high percentile (P90-P95)** over a longer, **gap-aware calendar** window (the calendar part is Batch 2; the max→percentile change is a small, well-scoped edit — do it in Batch 2 alongside the calendar migration of `wash_detect`, or as the first step of Batch 8). Cross-check the percentile against the **most-recent recovery plateau** (already the adaptive design) and the **orientation-matched peer clean reference** (`build_peer_groups`). A percentile is robust to a soiled max; the cross-checks catch a window that is entirely soiled.

---

## 5. Cross-cutting (keep visible, not buried)

- **Persistent state store** (built in Batch 7) is used wherever last-good values are held — SDM params and clean baseline. Keep its read/write at the run boundary and its provenance fields populated.
- **Provenance logging** — extend `AdaptiveBaselineResult` (and the per-string `res` dict / classification axes) so **no held, blended, substituted, or defaulted value is silent**: `age_source`, `orientation_source`, `bifacial_gain_source`, `inverter_ac_source`, `sdm_window_source`, fallback `tier`, `drought_flag`, `blend_weight`, `economics_inputs`.
- **The recurring clear-sky rule:** Kc gates *estimation*; absolute irradiance gates physical *validity* and energy *accounting*. Re-state it in code comments at every Kc site so a future maintainer doesn't "clear-sky everything."

---

## 6. Recommended thresholds reference (all new config keys)

Every value below is a **recommended starting default** with a one-line rationale — tune against real data. Add each to `PipelineConfig` with the comment, per the house style.

| Key | Default | Rationale |
|---|---|---|
| `daily_grid_enabled` | `True` | master switch for calendar consolidation (B2) |
| `max_step_gap_days` | `2` | a step spanning a longer gap is not a one-day transition (B2) |
| `min_valid_day_density` | `0.4` | a window needs ≥40% present days to be trustworthy (B2) |
| `curtailment_inverter_level_enabled` | `True` | master switch for inverter-level curtailment (B3) |
| `clip_band_rel` | `0.02` | flat-top within ±2% of the recurring daily max = clipping (B3) |
| `clip_repeat_days` | `3` | the plateau value must recur as daily max on ≥3 days (B3) |
| `clip_max_cv` | `0.03` | clipping is low-variance; CV must be below this (B3) |
| `clip_min_dwell` | `3` | plateau must persist ≥3 intervals (reuse existing) |
| `suppression_consensus_frac` | `0.5` | suppression needs ≥50% of an inverter's strings low together (B3) |
| `vr_consensus_min_strings` | `2` | voltage-rise needs ≥2 strings + causal confirmation (B3) |
| `age_relative_gates_enabled` | `True` | scale Gates A/B/C by age baseline (B4) |
| `age_relative_bands_enabled` | `True` | scale clean/lt/mod bands by age baseline *only on nameplate columns* (B4) |
| `adaptive_min_p95` | `0.92` *(coefficient)* | now a coefficient × age_baseline (B4) |
| `adaptive_no_rain_floor` | `0.96` *(coefficient)* | now a coefficient × age_baseline (B4) |
| `adaptive_cluster_gate` | `0.05` *(coefficient)* | peer-margin coefficient × age_baseline (B4) |
| `poa_transposition_enabled` | `True` | master switch for per-orientation POA (B5) |
| `transposition_model` | `"perez"` | Perez sky-diffuse is the accuracy standard; haydavies/isotropic fallback (B5) |
| `bifacial_gain_enabled` | gated on [C3] | only when rear data or a justified model exists (B5) |
| `bifaciality` | `0.70` | datasheet bifaciality fallback (B5) |
| `bifacial_gain_default` | `1.05` | conservative modeled gain when no rear irradiance (B5) |
| `clearsky_quality_enabled` | `True` | Kc+stability at the two estimation sites (B6) |
| `kc_tolerance` | `0.10` | |Kc−1| ≤ 0.10 for clearness (tune) (B6) |
| `kc_stability_cv_max` | `0.10` | rolling-CV of Kc must be below this (B6) |
| `kc_window_min` | `30.0` | stability window in minutes (B6) |
| `sdm_clean_window_enabled` | `True` | fit SDM on clean+recent windows (B6) |
| `sdm_recent_days` | `45` | recency horizon for clean-window selection (B6) |
| `robust_sdm_loss` | `"soft_l1"` | robust optimiser loss; `"linear"` restores old (B6) |
| `robust_sdm_f_scale` | `1.0` | residual scale in sigma-normalised space (B6) |
| `monsoon_fallback_enabled` | `True` | master switch for the 5-tier cascade (B7) |
| `window_widen_max_days` | `180` | cap on window-widening tier (B7) |
| `hold_last_good_enabled` | `True` | enable tier-3 last-good hold (B7) |
| `state_store_path` | configurable | path to the persistent last-good store (B7) |
| `drought_min_string_frac` | `0.6` | ≥60% of strings sparse = plant-scope drought (B7) |
| `drought_min_days` | `5` | drought must persist ≥5 days (B7) |
| `robust_soiling_regression` | `"theilsen"` | Theil-Sen default; `"huber"`/`"trimmed_ols"` alts (B8) |
| `transient_prefilter_enabled` | `True` | remove transients before fit + feed back (B8) |
| `wash_baseline_percentile` | `0.92` | replace 14-row max with P92 over gap-aware window (B2/B8) |
| `wash_cost_per_string_pkr` | site-specific | economics input; ask the user for the real figure (B1/B9) |
| `wash_cost_per_kw_pkr` | site-specific | alternative economics basis (B1/B9) |
| `module_area_m2` | optional | only if area-based economics is wanted; energy-based preferred (B1/B9) |
| `operational_view_windows` | `(7, 30)` | short-horizon report windows in days (B9) |

---

## 7. Consolidated answers to the open questions in the Word document

- **"Is 0.92 / 0.96 still reasonable for a string with age factor?"** Yes — as *coefficients on the string's age baseline*, not as absolute nameplate floors. Use `0.92 × age_baseline` and `0.96 × age_baseline`. They were only wrong because they were applied against nameplate; a clean 10-yr string sits ≈0.95 vs nameplate and was being rejected. (Batch 4, with the worked example there.)
- **`Age_Baseline` string-wise?** Yes — computed per string from `String Comissioning Date` (Batch 1 surfaces it; Batch 4 uses it). For this plant all strings share one date, so all baselines ≈0.987; the machinery still must be per-string for the general multi-phase plant.
- **Bifacial mechanism?** Effective irradiance with measured rear POA when available; a flagged modeled gain otherwise; drop it if neither is defensible (this plant has `Bifacial=Yes` but no rear-irradiance column). (Batch 5, [C3].)
- **Per-entity capacities?** Use the actual values per inverter and per string from the workbook, not a single representative panel or `nameplate/n_strings`. (Batch 1; consumed in Batch 3.)
- **Orientation mismatch / POA transposition method?** Dedicated transposition module, one transposed POA per orientation cluster shared among co-oriented strings, plus per-string az/tilt wired into the IAM. Prefer pvlib decompose→re-transpose (Perez); geometric POA-to-POA with real per-sample solar azimuth as fallback. (Batch 5.)
- **Clear-sky index everywhere or specific modules?** Specific modules only — the two *estimation* sites (`_select_quality_days`, the NCI_noon interval mask). Everything else (quality flags, losses, sufficiency, curtailment brightness floors) stays on absolute W/m², because those gate physical validity and energy accounting. (Batch 6.)
- **SDM: most-recent vs cleanest points?** Both — recovery-anchored clean windows that are also the most recent, with recency as the tie-breaker among clean windows. Pure-recent risks soiled data; pure-cleanest risks staleness. (Batch 6.)
- **Smog/monsoon (multi-day) fallback?** Five-tier cascade (widen window → peer borrow → hold-last-good → dry-season blend decay → suppress SDM refit + sufficiency "Limited"), drought detected at plant scope, with a persistent last-good store and an empty-Kc-set policy that suspends fitting rather than fitting on garbage. (Batch 7.)
- **Wash baseline: P90-P95 over a gap-aware window?** Yes; replace the 14-row max with a P90-P95 over a continuous calendar window, cross-checked against the most-recent recovery plateau and the orientation-matched peer clean reference. (Batch 2 calendar + the percentile edit; cross-checks already exist.)
- **Cleaning-economics recommendation?** Yes — `daily_loss_pkr = soiling_rate × expected_daily_energy × tariff`; `payback_days = wash_cost / daily_loss_pkr`; recommend a clean when cumulative recoverable PKR since last clean exceeds wash cost; report recoverable kWh/PKR and payback, with inputs' provenance. (Batch 9.)
- **Robust regression?** Theil-Sen (default) or Huber, with pre-fit transient removal and transient feedback into wash/segmentation; trimmed-OLS is not a true robust regressor. (Batch 8.)
- **Statistical curtailment level/method?** Inverter level, AC power, adaptive plateau (flat-top low-variance at a repeated daily max — catches below-nameplate setpoints), export-limit vs DC/AC-ratio split, per-inverter capacity, flag propagated to that inverter's strings. (Batch 3.)
- **Suppression dead-string loophole?** Decide suppression at inverter level with cross-string consensus; a lone low string becomes a non-disqualifying `STRING_UNDERPERFORM` so the fault classifier sees it, instead of being silently excluded as `CURT_SUPPRESSED`. (Batch 3.)
- **Voltage-rise level/method?** Keep the string-level signature but require inverter-level causal confirmation + cross-string consensus; a single string's Vdc rise alone is a module/IV artifact, flagged `STRING_UNDERPERFORM`, not curtailment. (Batch 3.)
- **Transients fed back?** Yes — detect on the consolidated calendar series, exclude before the fit, prevent false segment boundaries, and inform wash detection. (Batches 2 + 8.)
- **`plate.py` redundant?** Yes — deprecate it and use manufacturer specs directly, but preserve the per-string `n_modules`-from-`pv_capacity` scaling that lives in `_get_string_plate()`. (Batch 8.)

---

*End of playbook. Implement Batch 1 first; stop at the Validation Gate after Batch 4; pair Batch 6 with Batch 7.*
