"""Load customer xlsx into long_df + plant_meta with PK defaults."""
from __future__ import annotations
import re
from datetime import date
from typing import Any
import numpy as np
import pandas as pd
from .config import PipelineConfig, PlantConfig, SiteConfig
from .constants import (LAHORE_LAT, LAHORE_LON, DEFAULT_AZIMUTH_PK,
    DEFAULT_TILT_PK, DEFAULT_TARIFF_PKR_PER_KWH)
from .utils import _safe_id, coerce_date


REQUIRED = {"ts","plant","inverter_id","mppt_id","string_id",
            "POA_kw","V","I","P_kw","T_module","inverter_state"}


def _tolerant_rename(raw: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in raw.columns:
        cs = str(c).strip().lower()
        if cs in ("timestamp","time","datetime","date_time","ts"):
            rename[c] = "ts"
        elif cs == "plant":
            rename[c] = "plant"
        elif cs in ("inverter_id","inverter","inv_id","inv"):
            rename[c] = "inverter_id"
        elif cs in ("mppt_id","mppt"):
            rename[c] = "mppt_id"
        elif cs in ("string_id","string","str_id","pv","pv_id"):
            rename[c] = "string_id"
        elif "irradiance" in cs or "poa" in cs or cs.startswith("kw/m2"):
            rename[c] = "POA_kw"
        elif cs in ("voltage_u","voltage","u","v","v_dc","vdc","voltage_v"):
            rename[c] = "V"
        elif cs in ("current_i","current","i","i_dc","idc","current_a"):
            rename[c] = "I"
        elif cs == "power" or cs == "p_kw" or cs == "pkw" or cs == "p(kw)" or "power" in cs and "kw" in cs:
            rename[c] = "P_kw"
        elif "pv_temperature" in cs or cs in ("t_module","tmod","temp","temperature"):
            rename[c] = "T_module"
        elif cs in ("pv_capacity","capacity","plate","plate_kw"):
            rename[c] = "pv_capacity"
        elif cs == "inverter_state" or cs == "state":
            rename[c] = "inverter_state"
        elif cs == "azimuth":
            rename[c] = "azimuth"
        elif cs == "tilt":
            rename[c] = "tilt"
        elif "rain" in cs:
            rename[c] = "rainfall"
    return raw.rename(columns=rename)


def _read_metadata_sheet(path: str) -> dict:
    if path.lower().endswith(".csv"):
        return {}
    try:
        meta = pd.read_excel(path, sheet_name="Metadata", header=None)
        kv = {}
        for _, row in meta.iterrows():
            if pd.isna(row.iloc[0]): continue
            k = str(row.iloc[0]).strip().lower()
            v = row.iloc[1] if len(row) > 1 else None
            kv[k] = v
        return kv
    except Exception:
        return {}


def _resolve_plant_meta(meta_kv: dict, df: pd.DataFrame) -> dict:
    notes = []
    plant_name = meta_kv.get("plant_name") or meta_kv.get("plant")
    if plant_name is None and "plant" in df.columns and len(df) > 0:
        plant_name = df["plant"].dropna().iloc[0] if df["plant"].notna().any() else "Unknown Plant"
        if plant_name == "Unknown Plant": notes.append("plant_name defaulted: 'Unknown Plant'")

    lat = meta_kv.get("latitude") or meta_kv.get("lat")
    lon = meta_kv.get("longitude") or meta_kv.get("lon")
    if lat is None or (isinstance(lat, float) and np.isnan(lat)):
        lat = LAHORE_LAT
        notes.append(f"latitude defaulted to Lahore ({LAHORE_LAT})")
    if lon is None or (isinstance(lon, float) and np.isnan(lon)):
        lon = LAHORE_LON
        notes.append(f"longitude defaulted to Lahore ({LAHORE_LON})")

    tariff = meta_kv.get("tariff_pkr_kwh") or meta_kv.get("tariff")
    if tariff is None or (isinstance(tariff, float) and np.isnan(tariff)):
        tariff = DEFAULT_TARIFF_PKR_PER_KWH
        notes.append(f"tariff defaulted to {DEFAULT_TARIFF_PKR_PER_KWH} PKR/kWh")

    cdate_raw = meta_kv.get("commissioning_date") or meta_kv.get("commissioning")
    cdate = coerce_date(cdate_raw, fallback=date(2023, 1, 1))
    if cdate_raw is None:
        notes.append(f"commissioning_date defaulted to 2023-01-01")

    p_ac = meta_kv.get("p_ac_max_kw") or 100.0
    tech = meta_kv.get("technology") or "mono-c-Si"

    return dict(plant_name=str(plant_name), lat=float(lat), lon=float(lon),
                tariff=float(tariff), commissioning_date=cdate,
                default_azimuth=DEFAULT_AZIMUTH_PK,
                default_tilt=DEFAULT_TILT_PK,
                p_ac_max_kw=float(p_ac), technology=str(tech),
                substitution_notes=notes)


def load_plant_data(path: str, sheet_name=0, cfg=None):
    """Load main sheet + Metadata sheet. Returns (long_df, plant_meta)."""
    if path.lower().endswith(".csv"):
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path, sheet_name=sheet_name)
    df = _tolerant_rename(raw)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}\n"
                         f"Found: {list(df.columns)}")

    df["ts"] = pd.to_datetime(df["ts"])
    df["POA"] = pd.to_numeric(df["POA_kw"], errors="coerce") * 1000.0
    df["P"]   = pd.to_numeric(df["P_kw"],   errors="coerce") * 1000.0
    for c in ("V","I","T_module"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pv_capacity" in df.columns:
        df["pv_capacity"] = pd.to_numeric(df["pv_capacity"], errors="coerce")
    df["inverter_state"] = pd.to_numeric(df["inverter_state"], errors="coerce").fillna(-1).astype(int)

    plant_kv = _read_metadata_sheet(path)
    resolved = _resolve_plant_meta(plant_kv, df)

    if "azimuth" not in df.columns: df["azimuth"] = np.nan
    if "tilt"    not in df.columns: df["tilt"] = np.nan
    df["azimuth"] = pd.to_numeric(df["azimuth"], errors="coerce")
    df["tilt"]    = pd.to_numeric(df["tilt"],    errors="coerce")
    az_missing = int(df["azimuth"].isna().sum())
    til_missing = int(df["tilt"].isna().sum())
    df["azimuth"] = df["azimuth"].fillna(resolved["default_azimuth"])
    df["tilt"]    = df["tilt"].fillna(resolved["default_tilt"])

    df["string_label"] = (df["plant"].apply(_safe_id) + "__" +
                          df["inverter_id"].apply(_safe_id) + "__" +
                          df["mppt_id"].apply(_safe_id) + "__" +
                          df["string_id"].apply(_safe_id))

    diffs = (df.sort_values("ts")["ts"].drop_duplicates()
               .diff().dt.total_seconds().dropna())
    freq_min = float(diffs.median() / 60.0) if len(diffs) else 5.0

    plant_meta = dict(
        plants=sorted(df["plant"].dropna().astype(str).unique().tolist()),
        inverters=sorted(df["inverter_id"].dropna().astype(str).unique().tolist()),
        mppts_per_inv=df.groupby("inverter_id")["mppt_id"].nunique().to_dict(),
        total_strings=int(df.groupby(["plant","inverter_id","mppt_id","string_id"]).ngroups),
        ts_min=df["ts"].min(), ts_max=df["ts"].max(),
        freq_min=freq_min,
        n_intervals=int(df["ts"].nunique()),
        plant_resolved=resolved,
        substitution_notes=resolved["substitution_notes"],
        azimuth_filled_rows=az_missing,
        tilt_filled_rows=til_missing,
    )
    return df, plant_meta


def split_into_string_dfs(long_df):
    out = {}
    for label, g in long_df.groupby("string_label", sort=True):
        g = g.sort_values("ts").reset_index(drop=True).copy()
        if "rainfall" not in g.columns:
            g["rainfall"] = 0.0
        out[label] = g
    return out


def extract_string_meta(string_dfs):
    meta = {}
    for label, g in string_dfs.items():
        if len(g) == 0: continue
        meta[label] = dict(
            plant=str(g["plant"].iloc[0]),
            inverter_id=str(g["inverter_id"].iloc[0]),
            mppt_id=str(g["mppt_id"].iloc[0]),
            string_id=str(g["string_id"].iloc[0]),
            azimuth=float(g["azimuth"].iloc[0]),
            tilt=float(g["tilt"].iloc[0]),
        )
    return meta


def apply_plant_meta_to_cfg(cfg: PipelineConfig, plant_meta: dict) -> PipelineConfig:
    """Return a new cfg with plant_meta resolved values."""
    r = plant_meta.get("plant_resolved", {})
    # Update site
    cfg.site.name = r.get("plant_name", cfg.site.name)
    cfg.site.lat  = r.get("lat", cfg.site.lat)
    cfg.site.lon  = r.get("lon", cfg.site.lon)
    cfg.site.tariff = r.get("tariff", cfg.site.tariff)
    cfg.site.p_ac_max_kw = r.get("p_ac_max_kw", cfg.site.p_ac_max_kw)
    cdate = r.get("commissioning_date")
    if cdate is not None: cfg.plant.commissioning_date = cdate
    cfg.plant.default_azimuth = r.get("default_azimuth", cfg.plant.default_azimuth)
    cfg.plant.default_tilt = r.get("default_tilt", cfg.plant.default_tilt)
    cfg.plant.lat = r.get("lat", cfg.plant.lat)
    cfg.plant.lon = r.get("lon", cfg.plant.lon)
    if "technology" in r and r["technology"]:
        cfg.module.technology = r["technology"]
    return cfg
