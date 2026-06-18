import warnings
from pathlib import Path
import pandas as pd
import datetime

from soiling_analysis.inputs.specs import load_specs, DEFAULT_REQUIRE_INPUT
from pv_diag.config import PipelineConfig, ModuleConfig, SiteConfig, PlantConfig
from pv_diag.constants import DEFAULT_TARIFF_PKR_PER_KWH

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
    
    # 2b. Read the raw CSV
    raw = pd.read_csv(csv_path, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None) # Make naive
    
    # 2c. Extract and broadcast irradiance
    plant_rows = raw[raw["level"] == "plant"]
    irr = (
        plant_rows.dropna(subset=["irradiance_wm2"])
        .drop_duplicates("timestamp")
        .set_index("timestamp")["irradiance_wm2"]
    )
    
    # 2d. Extract and broadcast inverter_state
    inv_rows = raw[raw["level"] == "inverter"]
    inv_state = (
        inv_rows.dropna(subset=["inverter_state"])
        .drop_duplicates(["timestamp", "inverter_id"])
        .set_index(["timestamp", "inverter_id"])["inverter_state"]
    )
    
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
    
    # Cast identifier columns to string to prevent mixed-type sorting errors (like NaN as float mixed with str)
    for col in ["plant", "inverter_id", "mppt_id", "string_id"]:
        strings[col] = strings[col].fillna("Unknown").astype(str)
    
    # 2f. Merge irradiance and inverter_state onto string rows
    strings["POA_kw"] = strings["ts"].map(irr)
    
    idx = pd.MultiIndex.from_arrays([strings["ts"], strings["inverter_id"]])
    strings["inverter_state"] = idx.map(inv_state)
    strings["inverter_state"] = strings["inverter_state"].fillna(-1).astype(int)
    
    # 2g. Add azimuth and tilt from plant spec
    strings["azimuth"] = plant_spec.get("azimuth_deg", 180.0)
    strings["tilt"]    = plant_spec.get("tilt_deg", 20.0)
    
    # Add derived columns that the pipeline expects
    strings["POA"] = pd.to_numeric(strings["POA_kw"], errors="coerce") * 1000.0
    strings["P"]   = pd.to_numeric(strings["P_kw"], errors="coerce") * 1000.0
    
    from pv_diag.utils import _safe_id
    strings["string_label"] = (strings["plant"].apply(_safe_id) + "__" +
                               strings["inverter_id"].apply(_safe_id) + "__" +
                               strings["mppt_id"].apply(_safe_id) + "__" +
                               strings["string_id"].apply(_safe_id))
    
    # 2h. Validate required columns are present
    REQUIRED = {"ts","plant","inverter_id","mppt_id","string_id",
                "POA_kw","V","I","P_kw","T_module","inverter_state",
                "POA", "P", "string_label"}
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
        ),
        substitution_notes=[],
        azimuth_filled_rows=0,
        tilt_filled_rows=0,
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
