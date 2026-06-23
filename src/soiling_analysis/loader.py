import warnings
from pathlib import Path
import pandas as pd
import datetime

from soiling_analysis.inputs.specs import load_specs, DEFAULT_REQUIRE_INPUT, string_spec
from soiling_analysis.inputs.preflight import preflight_schema
from soiling_analysis.diagnostics.config import PipelineConfig, ModuleConfig, SiteConfig, PlantConfig
from soiling_analysis.diagnostics.constants import DEFAULT_TARIFF_PKR_PER_KWH

def _dominant_technology(specs: dict) -> str:
    techs = []
    for inv in specs.get("inverters", {}).values():
        for st in inv.get("strings", {}).values():
            if "technology" in st and st["technology"]:
                techs.append(st["technology"])
    if not techs:
        return "Mono-c-Si"
    return max(set(techs), key=techs.count)

def _representative_panel(specs: dict) -> dict:
    panels = []
    for inv in specs.get("inverters", {}).values():
        for st in inv.get("strings", {}).values():
            panels.append(st)
    if not panels:
        raise ValueError("No panel specifications found in workbook.")

    models = {p.get("panel_model") for p in panels if p.get("panel_model")}
    if len(models) > 1:
        warnings.warn(f"Multiple panel models found in specs: {models}. Using the first string's panel as representative. Module capacities per string are handled correctly via pv_capacity.")

    return panels[0]

def _detect_freq_min(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 5.0
    ts = pd.to_datetime(df["ts"]).sort_values().unique()
    if len(ts) < 2:
        return 5.0
    diffs = pd.Series(ts).diff().dropna()
    return float(diffs.median().total_seconds() / 60.0)

def _cells_in_series(num_cells: int, voc_stc: float) -> int:
    """Return the electrical cells-in-series count.

    Modern half-cell modules report the physical cell count (e.g. 144) but have
    only half as many cells in electrical series (72).  The giveaway is a low
    Voc-per-cell ratio: below 0.45 V/cell indicates the count is physical, so
    halve it.
    """
    if num_cells > 0 and voc_stc / num_cells < 0.45:
        return num_cells // 2
    return num_cells


def load_from_sources(
    string_csv: str | Path,
    inverter_csv: str | Path,
    specs_path: str | Path = DEFAULT_REQUIRE_INPUT,
) -> tuple[pd.DataFrame, dict, PipelineConfig]:
    """Return (long_df, plant_meta, cfg) ready for run_pipeline_from_frame().

    Parameters
    ----------
    string_csv   : path to the ``*_string.csv`` exported by ``soiling-fetch``
    inverter_csv : path to the ``*_inverter.csv`` exported by ``soiling-fetch``
    specs_path   : path to ``RequireInputData.xlsx`` spec workbook
    """

    print(f"\n[Loader] ── Starting data load ──────────────────────────────────────")
    print(f"[Loader]   string CSV : {string_csv}")
    print(f"[Loader]   inverter CSV: {inverter_csv}")
    print(f"[Loader]   specs      : {specs_path}")

    # 2a. Read the spec workbook
    print(f"\n[Loader] [1/6] Reading spec workbook...")
    specs = load_specs(specs_path)
    plant_spec = specs.get("plant", {})
    if not plant_spec:
        raise ValueError("Missing plant specifications in workbook.")
    _n_inv = len(specs.get("inverters", {}))
    _n_str = sum(
        len(inv.get("strings", {}))
        for inv in specs.get("inverters", {}).values()
    )
    print(f"[Loader]   plant: {plant_spec.get('site_name', '?')}  "
          f"| inverters: {_n_inv}  | configured strings: {_n_str}")

    # 2b. Read the string CSV in chunks (can be 1 M+ rows for large plants / long ranges).
    # irradiance_wm2 is already present in every string row — no plant-level broadcast needed.
    print(f"\n[Loader] [2/6] Reading string CSV (chunked)...")
    _STRING_COLS = {
        "timestamp", "plant_name", "irradiance_wm2",
        "inverter_id", "mppt_id", "string_id",
        "string_capacity_kwp", "string_power_kw",
        "string_current_a", "string_voltage_v",
        "pv_temperature", "rainfall",
    }
    _str_header = pd.read_csv(string_csv, nrows=0).columns.tolist()
    _str_use = [c for c in _str_header if c in _STRING_COLS]
    _missing_str_cols = _STRING_COLS - set(_str_header)
    if _missing_str_cols:
        print(f"[Loader]   ⚠ Optional string columns not in CSV: {sorted(_missing_str_cols)}")

    _str_chunks: list[pd.DataFrame] = []
    for _chunk in pd.read_csv(string_csv, chunksize=100_000, usecols=_str_use, low_memory=False):
        _str_chunks.append(_chunk)
    strings = pd.concat(_str_chunks, ignore_index=True)
    del _str_chunks
    strings["timestamp"] = pd.to_datetime(strings["timestamp"]).dt.tz_localize(None)
    print(f"[Loader]   rows: {len(strings):,}  | columns used: {_str_use}")

    # 2c. Read the inverter CSV (lightweight — only a handful of columns needed).
    # AC power  → inverter_active_power_kw  (from MongoDB: active_power)
    # DC power  → mppt_total_power_kw       (from MongoDB: mppt_power, total MPPT input)
    print(f"\n[Loader] [3/6] Reading inverter CSV...")
    _INV_COLS = {
        "timestamp", "inverter_id", "inverter_state",
        "inverter_active_power_kw", "mppt_total_power_kw",
    }
    _inv_header = pd.read_csv(inverter_csv, nrows=0).columns.tolist()
    _inv_use = [c for c in _inv_header if c in _INV_COLS]
    _missing_inv_cols = _INV_COLS - set(_inv_header)
    if _missing_inv_cols:
        print(f"[Loader]   ⚠ Expected inverter columns not in CSV: {sorted(_missing_inv_cols)}")

    inv_raw = pd.read_csv(inverter_csv, usecols=_inv_use, low_memory=False)
    inv_raw["timestamp"] = pd.to_datetime(inv_raw["timestamp"]).dt.tz_localize(None)
    print(f"[Loader]   rows: {len(inv_raw):,}  | columns used: {_inv_use}")

    # 2d. Build inverter_state lookup, AC power, and DC power series (SCHEMA-DEP C4)
    inv_state = (
        inv_raw.dropna(subset=["inverter_state"])
        .drop_duplicates(["timestamp", "inverter_id"])
        .set_index(["timestamp", "inverter_id"])["inverter_state"]
    )

    # ── AC power: inverter_active_power_kw (from MongoDB: active_power) ──────
    print(f"\n[Loader] [4/6] Resolving inverter AC power (inverter_active_power_kw)...")
    _inv_ac_power = pd.Series(dtype=float, name="ac_power_kw")
    if "inverter_active_power_kw" in inv_raw.columns:
        _ac_valid = int(inv_raw["inverter_active_power_kw"].notna().sum())
        _ac_total = len(inv_raw)
        if _ac_valid > 0:
            _inv_ac_power = (
                inv_raw.dropna(subset=["timestamp", "inverter_id"])
                .drop_duplicates(["timestamp", "inverter_id"])
                .set_index(["timestamp", "inverter_id"])["inverter_active_power_kw"]
                .rename("ac_power_kw")
            )
            print(f"[Loader]   ✓ AC power: inverter_active_power_kw "
                  f"— {_ac_valid:,}/{_ac_total:,} non-null readings "
                  f"({100*_ac_valid/_ac_total:.1f}%)")
        else:
            warnings.warn(
                "Loader: 'inverter_active_power_kw' column is present but entirely null. "
                "AC power will be reconstructed from Σ(string DC) × inverter efficiency "
                "during curtailment detection (C4 fallback).",
                stacklevel=2,
            )
            print(f"[Loader]   ⚠ FALLBACK: inverter_active_power_kw is all-null "
                  f"→ AC will be reconstructed from Σ(string DC) × efficiency at curtailment step")
    else:
        warnings.warn(
            "Loader: 'inverter_active_power_kw' not found in inverter CSV. "
            "Re-run soiling-fetch to export this column. "
            "AC power will be reconstructed from Σ(string DC) × inverter efficiency "
            "during curtailment detection (C4 fallback).",
            stacklevel=2,
        )
        print(f"[Loader]   ⚠ FALLBACK: inverter_active_power_kw column missing from inverter CSV "
              f"→ AC will be reconstructed from Σ(string DC) × efficiency at curtailment step")

    # ── DC power: mppt_total_power_kw (from MongoDB: mppt_power, Σ MPPT inputs) ──
    print(f"\n[Loader] [5/6] Resolving inverter DC power (mppt_total_power_kw)...")
    _inv_dc_power = pd.Series(dtype=float, name="dc_power_kw")
    if "mppt_total_power_kw" in inv_raw.columns:
        _dc_valid = int(inv_raw["mppt_total_power_kw"].notna().sum())
        _dc_total = len(inv_raw)
        if _dc_valid > 0:
            _inv_dc_power = (
                inv_raw.dropna(subset=["timestamp", "inverter_id"])
                .drop_duplicates(["timestamp", "inverter_id"])
                .set_index(["timestamp", "inverter_id"])["mppt_total_power_kw"]
                .rename("dc_power_kw")
            )
            print(f"[Loader]   ✓ DC power: mppt_total_power_kw "
                  f"— {_dc_valid:,}/{_dc_total:,} non-null readings "
                  f"({100*_dc_valid/_dc_total:.1f}%)")
        else:
            warnings.warn(
                "Loader: 'mppt_total_power_kw' column is present but entirely null. "
                "DC power unavailable — inverter efficiency cannot be derived from measured data.",
                stacklevel=2,
            )
            print(f"[Loader]   ⚠ FALLBACK: mppt_total_power_kw is all-null → DC power unavailable")
    else:
        warnings.warn(
            "Loader: 'mppt_total_power_kw' not found in inverter CSV. "
            "Re-run soiling-fetch to export this column. DC power unavailable.",
            stacklevel=2,
        )
        print(f"[Loader]   ⚠ FALLBACK: mppt_total_power_kw column missing from inverter CSV "
              f"→ DC power unavailable")

    # 2e. Rename string columns to pipeline-standard names
    print(f"\n[Loader] [6/6] Preparing string DataFrame...")
    strings = strings.rename(columns={
        "timestamp":           "ts",
        "plant_name":          "plant",
        "string_voltage_v":    "V",
        "string_current_a":    "I",
        "string_power_kw":     "P_kw",
        "pv_temperature":      "T_module",
        "string_capacity_kwp": "pv_capacity",
        "irradiance_wm2":      "POA_kw",
    })

    # Cast identifier columns to string to prevent mixed-type sorting errors
    for col in ["plant", "inverter_id", "mppt_id", "string_id"]:
        strings[col] = strings[col].fillna("Unknown").astype(str)

    if "pv_capacity" in strings.columns:
        strings["pv_capacity"] = pd.to_numeric(strings["pv_capacity"], errors="coerce")

    # ── Clip pv_temperature to physical plausible range ──────────────────────
    _T_MIN, _T_MAX = -5.0, 85.0
    if "T_module" in strings.columns:
        strings["T_module"] = pd.to_numeric(strings["T_module"], errors="coerce")
        _t_total = int(strings["T_module"].notna().sum())
        _t_out_mask = strings["T_module"].notna() & ~strings["T_module"].between(_T_MIN, _T_MAX)
        _t_clipped = int(_t_out_mask.sum())
        if _t_clipped > 0:
            strings.loc[_t_out_mask, "T_module"] = pd.NA
            warnings.warn(
                f"Loader: {_t_clipped:,} pv_temperature readings outside "
                f"[{_T_MIN}°C, {_T_MAX}°C] were set to NaN (sensor noise/errors). "
                "The SDM temperature model will use the ambient-temperature fallback for those rows.",
                stacklevel=2,
            )
            print(f"[Loader]   pv_temperature: clipped {_t_clipped:,}/{_t_total:,} out-of-range "
                  f"readings (outside {_T_MIN}°C – {_T_MAX}°C) → NaN "
                  f"({100*_t_clipped/_t_total:.2f}% of non-null)")
        else:
            print(f"[Loader]   pv_temperature: all {_t_total:,} readings within "
                  f"{_T_MIN}°C – {_T_MAX}°C ✓")
    else:
        print(f"[Loader]   pv_temperature: column not present in string CSV — T_module will be NaN")

    # Filter to only workbook-configured strings (skip unconfigured extra strings in CSV)
    _wb_keys: set = set()
    for _inv_sn, _inv_data in specs.get("inverters", {}).items():
        for _sk in _inv_data.get("strings", {}).keys():
            _wb_keys.add(f"{_inv_sn.strip()}||{_sk}")  # "ES23C0014748||MPPT1|PV1"

    _comb_key = (strings["inverter_id"].str.strip() + "||" +
                 strings["mppt_id"].str.strip().str.upper() + "|" +
                 strings["string_id"].str.strip().str.upper())
    _in_wb = _comb_key.isin(_wb_keys)
    _n_before_u = strings.groupby(["inverter_id", "mppt_id", "string_id"]).ngroups
    if not _in_wb.all():
        strings = strings[_in_wb].copy()
        _n_after_u = strings.groupby(["inverter_id", "mppt_id", "string_id"]).ngroups
        _skipped = _n_before_u - _n_after_u
        warnings.warn(
            f"Loader: skipped {_skipped} strings not in workbook "
            f"({_n_before_u} → {_n_after_u} configured strings retained)",
            stacklevel=2,
        )
        print(f"[Loader]   string filter: {_skipped} strings not in workbook skipped "
              f"({_n_before_u} → {_n_after_u} retained)")
    else:
        print(f"[Loader]   string filter: all {_n_before_u} strings match workbook ✓")

    # 2f. Merge inverter_state onto string rows; irradiance already present as POA_kw
    idx = pd.MultiIndex.from_arrays([strings["ts"], strings["inverter_id"]])
    strings["inverter_state"] = idx.map(inv_state)
    strings["inverter_state"] = strings["inverter_state"].fillna(-1).astype(int)

    # Add derived columns that the pipeline expects
    strings["POA"] = pd.to_numeric(strings["POA_kw"], errors="coerce") * 1000.0
    strings["P"]   = pd.to_numeric(strings["P_kw"], errors="coerce") * 1000.0

    from soiling_analysis.diagnostics.utils import _safe_id
    strings["string_label"] = (strings["plant"].apply(_safe_id) + "__" +
                               strings["inverter_id"].apply(_safe_id) + "__" +
                               strings["mppt_id"].apply(_safe_id) + "__" +
                               strings["string_id"].apply(_safe_id))

    # Human-facing display ID: <InverterID>_<StringID> — unique because string IDs
    # never repeat across MPPTs on the same inverter for this plant.
    strings["string_uid"] = (strings["inverter_id"].apply(_safe_id) + "_" +
                             strings["string_id"].apply(_safe_id))

    # 2g. Per-string orientation, age, bifacial (SCHEMA-DEP C1, C2, C3)
    _plant_az   = float(plant_spec.get("azimuth_deg", 180.0))
    _plant_tilt = float(plant_spec.get("tilt_deg", 20.0))
    _plant_comm = plant_spec.get("commissioning_date", "2000-01-01")

    _az_map:        dict = {}
    _tilt_map:      dict = {}
    _comm_map:      dict = {}
    _comm_year_map: dict = {}
    _bifacial_map:  dict = {}
    _str_specs:     dict = {}

    for _, _row in strings.drop_duplicates("string_label").iterrows():
        _lbl    = _row["string_label"]
        _sp     = string_spec(specs, _row["inverter_id"], _row["mppt_id"], _row["string_id"]) or {}

        _az = _sp.get("string_azimuth_deg")
        _az_src = "string_spec" if _az is not None else "plant_default"
        _az = float(_az) if _az is not None else _plant_az

        _tilt = _sp.get("string_tilt_deg")
        _tilt_src = "string_spec" if _tilt is not None else "plant_default"
        _tilt = float(_tilt) if _tilt is not None else _plant_tilt

        _comm = _sp.get("string_commissioning_date")
        _age_src = "string_spec" if _comm is not None else "plant_default"
        _comm = _comm if _comm is not None else _plant_comm

        _comm_year = int(_comm[:4]) if _comm and len(_comm) >= 4 else None

        _cap_val = _row.get("pv_capacity")
        _cap_kw  = (
            float(_cap_val)
            if _cap_val is not None and pd.notna(_cap_val)
            else float(_sp.get("string_capacity_w") or 0) / 1000.0
        )

        _az_map[_lbl]        = _az
        _tilt_map[_lbl]      = _tilt
        _comm_map[_lbl]      = _comm
        _comm_year_map[_lbl] = _comm_year
        _bifacial_map[_lbl]  = bool(_sp.get("bifacial", False))

        _str_specs[_lbl] = {
            **_sp,
            "azimuth":            _az,
            "tilt":               _tilt,
            "orientation_source": _az_src,
            "commissioning_date": _comm,
            "commissioning_year": _comm_year,
            "age_source":         _age_src,
            "bifacial":           bool(_sp.get("bifacial", False)),
            "pv_capacity_kw":     _cap_kw,
        }

    strings["azimuth"]            = strings["string_label"].map(_az_map).fillna(_plant_az)
    strings["tilt"]               = strings["string_label"].map(_tilt_map).fillna(_plant_tilt)
    strings["commissioning_date"] = strings["string_label"].map(_comm_map).fillna(_plant_comm)
    strings["commissioning_year"] = strings["string_label"].map(_comm_year_map)
    strings["bifacial"]           = strings["string_label"].map(_bifacial_map).fillna(False)

    # 2h. Validate required columns are present
    REQUIRED = {"ts","plant","inverter_id","mppt_id","string_id",
                "POA_kw","V","I","P_kw","T_module","inverter_state",
                "POA", "P", "string_label", "string_uid",
                "azimuth", "tilt", "commissioning_date", "bifacial"}
    missing = REQUIRED - set(strings.columns)
    if missing:
        raise ValueError(f"Loader produced missing columns: {sorted(missing)}")

    # 2i. Build plant_meta dict
    plant_name = plant_spec.get("site_name", "Unknown Plant")
    plant_lat = float(plant_spec.get("latitude", 0.0))
    plant_lon = float(plant_spec.get("longitude", 0.0))
    comm_date_str = str(plant_spec.get("commissioning_date", "2000-01-01"))
    comm_date = datetime.date.fromisoformat(comm_date_str.split("T")[0] if "T" in comm_date_str else comm_date_str.split(" ")[0])
    azimuth_deg = float(plant_spec.get("azimuth_deg", 180.0))
    tilt_deg = float(plant_spec.get("tilt_deg", 20.0))
    size_kw = float(plant_spec.get("size_kw_ac", 1000.0))

    # Run preflight — pass the inverter DataFrame directly for the C4 check
    _preflight = preflight_schema(specs, strings, inverter_df=inv_raw)

    _az_plant_fallback = sum(
        1 for sp in _str_specs.values() if sp.get("orientation_source") == "plant_default"
    )

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
            plant_name=plant_name,
            lat=plant_lat,
            lon=plant_lon,
            tariff=DEFAULT_TARIFF_PKR_PER_KWH,
            commissioning_date=comm_date_str,
            default_azimuth=azimuth_deg,
            default_tilt=tilt_deg,
            p_ac_max_kw=size_kw,
            technology=_dominant_technology(specs),
            substitution_notes=[],
            wash_cost_per_string_pkr=None,
            wash_cost_per_kw_pkr=None,
        ),
        substitution_notes=[],
        azimuth_filled_rows=_az_plant_fallback,
        tilt_filled_rows=_az_plant_fallback,
        string_specs=_str_specs,
        inverter_specs=specs.get("inverters", {}),
        inverter_ac_power=_inv_ac_power,          # SCHEMA-DEP C4 — inverter_active_power_kw
        inverter_dc_power=_inv_dc_power,          # mppt_total_power_kw (Σ MPPT DC inputs)
        preflight=_preflight,                      # C1-C6 capability flags
    )

    # 2j. Build PipelineConfig from workbook
    panel = _representative_panel(specs)

    cfg = PipelineConfig(
        site=SiteConfig(
            name=plant_name,
            lat=plant_lat,
            lon=plant_lon,
            altitude=float(plant_spec.get("altitude_m", 0.0) or 0.0),
            p_ac_max_kw=size_kw,
        ),
        module=ModuleConfig(
            voc_stc=float(panel.get("voc_v", 40.0) or 40.0),
            vmp_stc=float(panel.get("vmp_v", 32.0) or 32.0),
            isc_stc=float(panel.get("isc_a", 10.0) or 10.0),
            imp_stc=float(panel.get("imp_a", 9.0) or 9.0),
            alpha_isc=float(panel.get("alpha_isc_pct_per_c", 0.04) or 0.04) / 100.0,
            beta_voc=float(panel.get("beta_voc_pct_per_c", -0.28) or -0.28) / 100.0,
            gamma_pmp=float(panel.get("gamma_pmax_pct_per_c", -0.38) or -0.38) / 100.0,
            n_modules=int(panel.get("num_panels", 20) or 20),
            technology=str(panel.get("technology", "Mono-c-Si") or "Mono-c-Si"),
            cells_in_series=_cells_in_series(
                int(panel.get("num_cells", 72) or 72),
                float(panel.get("voc_v", 40.0) or 40.0),
            ),
        ),
        plant=PlantConfig(
            commissioning_date=comm_date,
            default_azimuth=azimuth_deg,
            default_tilt=tilt_deg,
            lat=plant_lat,
            lon=plant_lon,
        ),
    )

    cfg.annual_degradation_pct = float(panel.get("annual_degradation_pct", 0.5) or 0.5) / 100.0
    cfg.lid_loss_pct = float(panel.get("first_year_degradation_pct", 2.0) or 2.0) / 100.0

    _ac_ok = len(_inv_ac_power) > 0
    _dc_ok = len(_inv_dc_power) > 0
    print(f"\n[Loader] ── Load complete ───────────────────────────────────────────")
    print(f"[Loader]   strings loaded : {len(strings):,} rows  "
          f"| {plant_meta['total_strings']} strings  "
          f"| {plant_meta['n_intervals']} timestamps  "
          f"| freq ≈ {plant_meta['freq_min']:.0f} min")
    print(f"[Loader]   date range     : {plant_meta['ts_min']}  →  {plant_meta['ts_max']}")
    print(f"[Loader]   AC power       : {'✓ inverter_active_power_kw' if _ac_ok else '⚠ MISSING — will reconstruct from string DC × efficiency'}")
    print(f"[Loader]   DC power       : {'✓ mppt_total_power_kw' if _dc_ok else '⚠ MISSING — not available'}")
    _pf = plant_meta["preflight"]
    print(f"[Loader]   preflight      : "
          + "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in _pf.items() if k != "details"))
    print(f"[Loader] ────────────────────────────────────────────────────────────\n")

    return strings, plant_meta, cfg
