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
    
    # Check if there are multiple panel models
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

def load_from_sources(
    csv_path: str | Path,
    specs_path: str | Path = DEFAULT_REQUIRE_INPUT,
) -> tuple[pd.DataFrame, dict, PipelineConfig]:
    """Return (long_df, plant_meta, cfg) ready for run_pipeline_from_frame()."""
    
    # 2a. Read the spec workbook
    specs = load_specs(specs_path)
    plant_spec = specs.get("plant", {})
    if not plant_spec:
        raise ValueError("Missing plant specifications in workbook.")
    
    # 2b. Read the raw CSV in chunks to handle large files (1 GB+).
    # The exported CSV has 60+ columns; the C parser runs out of memory
    # tokenizing the whole file at once. Chunked reading processes 100k rows
    # at a time, filtering to only the ~15 columns the pipeline actually uses.
    # Note: using usecols as a callable with chunksize hits a pandas IndexError;
    # read the header first and build a concrete list instead.
    _NEEDED_COLS = {
        "level", "timestamp",
        "plant_name", "irradiance_wm2",
        "inverter_id", "inverter_state",
        "inverter_active_power_kw", "inverter_power_kw", "power_kw",
        "mppt_id", "string_id",
        "string_capacity_kwp", "string_power_kw",
        "string_current_a", "string_voltage_v",
        "pv_temperature", "rainfall",
    }
    _header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    _use_cols = [c for c in _header if c in _NEEDED_COLS]

    _raw_chunks: list[pd.DataFrame] = []
    for _chunk in pd.read_csv(
        csv_path,
        chunksize=100_000,
        usecols=_use_cols,
        low_memory=False,
    ):
        _raw_chunks.append(_chunk)
    raw = pd.concat(_raw_chunks, ignore_index=True)
    del _raw_chunks
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)  # Make naive
    
    # 2c. Extract and broadcast irradiance
    plant_rows = raw[raw["level"] == "plant"]
    irr = (
        plant_rows.dropna(subset=["irradiance_wm2"])
        .drop_duplicates("timestamp")
        .set_index("timestamp")["irradiance_wm2"]
    )
    
    # 2d. Extract inverter_state and surface inverter-level AC power (C4)
    inv_rows = raw[raw["level"] == "inverter"]
    inv_state = (
        inv_rows.dropna(subset=["inverter_state"])
        .drop_duplicates(["timestamp", "inverter_id"])
        .set_index(["timestamp", "inverter_id"])["inverter_state"]
    )

    # SCHEMA-DEP C4: inverter AC power for clipping detection (Batch 3)
    _inv_ac_power = pd.Series(dtype=float, name="ac_power_kw")
    for _pow_col in ["inverter_active_power_kw", "inverter_power_kw", "string_power_kw", "power_kw"]:
        if _pow_col in inv_rows.columns and inv_rows[_pow_col].notna().any():
            _inv_ac_power = (
                inv_rows.dropna(subset=["timestamp", "inverter_id"])
                .drop_duplicates(["timestamp", "inverter_id"])
                .set_index(["timestamp", "inverter_id"])[_pow_col]
                .rename("ac_power_kw")
            )
            break

    # 2e. Filter to string rows and rename columns
    strings = raw[raw["level"] == "string"].copy()
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

    # Cast identifier columns to string to prevent mixed-type sorting errors
    for col in ["plant", "inverter_id", "mppt_id", "string_id"]:
        strings[col] = strings[col].fillna("Unknown").astype(str)

    if "pv_capacity" in strings.columns:
        strings["pv_capacity"] = pd.to_numeric(strings["pv_capacity"], errors="coerce")

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
        warnings.warn(
            f"Loader: skipped {_n_before_u - _n_after_u} strings not in workbook "
            f"({_n_before_u} → {_n_after_u} configured strings retained)"
        )

    # 2f. Merge irradiance and inverter_state onto string rows
    strings["POA_kw"] = strings["ts"].map(irr)

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
    # One spec lookup per unique string; fall back to plant defaults per preflight check.
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

    # Run preflight once — downstream branches on this dict
    _preflight = preflight_schema(specs, raw)

    # Count strings that fell back to plant-level defaults
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
            wash_cost_per_string_pkr=None,   # supplied via --wash-cost CLI flag
            wash_cost_per_kw_pkr=None,
        ),
        substitution_notes=[],
        azimuth_filled_rows=_az_plant_fallback,
        tilt_filled_rows=_az_plant_fallback,
        # Batch 1 additions — consumed by Batches 3-5, 9
        string_specs=_str_specs,                          # per-string specs dict
        inverter_specs=specs.get("inverters", {}),        # per-inverter capacity etc.
        inverter_ac_power=_inv_ac_power,                  # SCHEMA-DEP C4 (Batch 3)
        preflight=_preflight,                             # C1-C6 capability flags
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
            cells_in_series=int(panel.get("num_cells", 72) or 72),
        ),
        plant=PlantConfig(
            commissioning_date=comm_date,
            default_azimuth=azimuth_deg,
            default_tilt=tilt_deg,
            lat=plant_lat,
            lon=plant_lon,
        ),
    )
    
    # Overwrite degradation from workbook
    cfg.annual_degradation_pct = float(panel.get("annual_degradation_pct", 0.5) or 0.5) / 100.0
    cfg.lid_loss_pct = float(panel.get("first_year_degradation_pct", 2.0) or 2.0) / 100.0

    return strings, plant_meta, cfg
