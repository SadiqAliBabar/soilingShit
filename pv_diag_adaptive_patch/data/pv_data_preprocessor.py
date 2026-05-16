"""
================================================================================
  PV DATA PREPROCESSOR — MongoDB/JSON nested PV data → flat soiling pipeline file
================================================================================

WHAT THIS SCRIPT DOES:
  1. Reads nested PV documents from MongoDB or JSON files
  2. Flattens inverter → MPPT → PV-string data into one row per string per timestamp
  3. Outputs columns that match the soiling pipeline demo schema exactly:

     timestamp, plant, inverter_id, mppt_id, string_id,
     irradiance KW/m2, voltage_u, current_i, power kw,
     pv_temperature, pv_Capacity, inverter_state,
     azimuth, tilt, rainfall

  4. Adds plant geometry from PLANT_CONFIG or CLI overrides:
       CCI/Coca Cola Faisalabad → tilt=20, azimuth=167
       because plant sheet says Tilt/Azimuth = 20 / -13° and compass azimuth = 180 + (-13)

  5. Runs basic data-quality checks and saves:
       - clean flat data file
       - quality report file

USAGE:
  # Last 30 days from MongoDB, daytime/high-irradiance docs only:
  python pv_data_preprocessor_updated.py \
      --mongo "mongodb://localhost:27017" \
      --db shams_Coca_Cola_Faisalabad \
      --collection FM_AL_HIS_ANALYSIS_WITH_TEMPERATURE \
      --output clean_soiling_data.xlsx

  # Custom date range:
  python pv_data_preprocessor_updated.py --mongo "mongodb://localhost:27017" \
      --db shams_Coca_Cola_Faisalabad \
      --collection FM_AL_HIS_ANALYSIS_WITH_TEMPERATURE \
      --date-from 2026-02-01 --date-to 2026-02-28 \
      --output clean_soiling_data.xlsx

  # Use exact noon-hour filter, like the older script:
  python pv_data_preprocessor_updated.py --mongo ... --hour-filter noon

  # Use all rows in date range, no hour/irradiance filter at MongoDB level:
  python pv_data_preprocessor_updated.py --mongo ... --hour-filter all

  # Override plant geometry manually:
  python pv_data_preprocessor_updated.py --mongo ... --tilt 20 --azimuth 167 --rainfall 0

  # From JSON file/folder:
  python pv_data_preprocessor_updated.py --input ./data.json --output clean_soiling_data.xlsx

  # Quick test:
  python pv_data_preprocessor_updated.py --test
================================================================================
"""

import os
import sys
import json
import glob
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PLANT CONFIG — extracted from your plant info sheet
# ─────────────────────────────────────────────────────────────────────────────

PLANT_CONFIG = {
    # MongoDB value usually uses this name
    "Coca Cola Faisalabad": {
        "site_name": "CCI Faisalabad",
        "system_size_kwp_dc": 2439.45,
        "system_size_kw_ac": 2310,
        "tilt": 20.0,
        "azimuth_raw": -13.0,      # from sheet: Tilt/Azimuth = 20 / -13°
        "azimuth": 167.0,          # compass style: 180 + (-13) = 167
        "latitude": 31.62,
        "longitude": 73.17,
        "altitude_m": 184,
        "pv_panel_brand_model": "JAM72D40 585/LB/1500V",
        "pv_panel_wattage_w": 585,
        "pv_panel_quantity": 4170,
        "inverter_brand_model": "SUN2000-330KTL-H2",
        "inverter_wattage_ktl": 330,
        "inverter_quantity": 7,
        "weather_station": True,
        "rainfall": 0.0,
    },
    # Sheet value / alias
    "CCI Faisalabad": {
        "site_name": "CCI Faisalabad",
        "system_size_kwp_dc": 2439.45,
        "system_size_kw_ac": 2310,
        "tilt": 20.0,
        "azimuth_raw": -13.0,
        "azimuth": 167.0,
        "latitude": 31.62,
        "longitude": 73.17,
        "altitude_m": 184,
        "pv_panel_brand_model": "JAM72D40 585/LB/1500V",
        "pv_panel_wattage_w": 585,
        "pv_panel_quantity": 4170,
        "inverter_brand_model": "SUN2000-330KTL-H2",
        "inverter_wattage_ktl": 330,
        "inverter_quantity": 7,
        "weather_station": True,
        "rainfall": 0.0,
    },
}

DEFAULT_PLANT_META = {
    "tilt": 20.0,
    "azimuth": 167.0,
    "rainfall": 0.0,
}

SOILING_OUTPUT_COLUMNS = [
    "timestamp",
    "plant",
    "inverter_id",
    "mppt_id",
    "string_id",
    "irradiance KW/m2",
    "voltage_u",
    "current_i",
    "power kw",
    "pv_temperature",
    "pv_Capacity",
    "inverter_state",
    "azimuth",
    "tilt",
    "rainfall",
    "Plant_P_abd",
    "Plant_capacity",
    "PR_PL",
    "Deviation_PL",
    "Inverter_P_abd",
    "Inverter_capacity",
]


# ─────────────────────────────────────────────────────────────────────────────
#  SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = np.nan) -> float:
    """Convert value to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def parse_timestamp(raw_ts: Any) -> pd.Timestamp:
    """Parse MongoDB timestamp formats including {'$date': ...}."""
    if isinstance(raw_ts, dict):
        raw_ts = raw_ts.get("$date", raw_ts)
    try:
        return pd.to_datetime(raw_ts)
    except Exception:
        return pd.NaT


def get_plant_meta(plant: str,
                   azimuth_override: float | None = None,
                   tilt_override: float | None = None,
                   rainfall_override: float | None = None) -> dict:
    """Return plant geometry/weather metadata from config, with CLI override support."""
    meta = dict(DEFAULT_PLANT_META)
    meta.update(PLANT_CONFIG.get(plant, {}))

    if azimuth_override is not None:
        meta["azimuth"] = float(azimuth_override)
    if tilt_override is not None:
        meta["tilt"] = float(tilt_override)
    if rainfall_override is not None:
        meta["rainfall"] = float(rainfall_override)

    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — FLATTEN ONE MONGODB DOCUMENT INTO ROWS
# ─────────────────────────────────────────────────────────────────────────────

def flatten_document(doc: dict,
                     azimuth_override: float | None = None,
                     tilt_override: float | None = None,
                     rainfall_override: float | None = None) -> list[dict]:
    """
    Takes one MongoDB document and returns flat rows — one per PV string.

    Output columns match the soiling pipeline demo schema:
      timestamp, plant, inverter_id, mppt_id, string_id,
      irradiance KW/m2, voltage_u, current_i, power kw,
      pv_temperature, pv_Capacity, inverter_state, azimuth, tilt, rainfall
    """
    rows = []

    ts = parse_timestamp(doc.get("timestamp"))
    plant = doc.get("Plant", doc.get("plant", ""))
    meta = get_plant_meta(
        plant=plant,
        azimuth_override=azimuth_override,
        tilt_override=tilt_override,
        rainfall_override=rainfall_override,
    )

    # Your MongoDB radiation_intensity is already in kW/m².
    # Keep it as kW/m² because the demo schema column is "irradiance KW/m2".
    irradiance_kw_m2 = safe_float(doc.get("radiation_intensity", np.nan))

    plant_p_abd = safe_float(doc.get("Plant_P_abd", np.nan))
    plant_capacity_doc = safe_float(doc.get("Plant_capacity", np.nan))
    pr_pl = safe_float(doc.get("PR_PL", np.nan))
    deviation_pl = safe_float(doc.get("Deviation_PL", np.nan))

    for sn in doc.get("sns", []):
        inverter_id = sn.get("snid", "")
        inverter_state = safe_int(sn.get("inverter_state", 512), default=512)
        inverter_temp = safe_float(sn.get("temperature", np.nan))
        inverter_capacity = safe_float(sn.get("Inverter_capacity", sn.get("inverter_capacity", np.nan)))
        inverter_p_abd = safe_float(sn.get("Inverter_P_abd", np.nan))

        for mppt in sn.get("mppts", []):
            mppt_id = mppt.get("mpptId", "")
            mppt_capacity = safe_float(mppt.get("mppt_Capacity", mppt.get("mpptCapacity", np.nan)))

            for pv in mppt.get("pvs", []):
                pv_id = pv.get("pvId", "")

                current_raw = safe_float(pv.get("i", np.nan))
                current_clean = max(current_raw, 0.0) if not pd.isna(current_raw) else np.nan

                voltage = safe_float(pv.get("u", np.nan))
                power_kw = safe_float(pv.get("pv_P_abd", np.nan))
                capacity = safe_float(pv.get("pv_Capacity", np.nan))

                # Prefer PV-level temperature from the new collection.
                # Fallback to inverter temperature if PV temperature is missing.
                pv_temperature = safe_float(pv.get("pv_temperature", inverter_temp))

                current_per_kwp = current_clean / capacity if capacity > 0.1 else np.nan

                rows.append({
                    # exact demo / soiling pipeline schema
                    "timestamp": ts,
                    "plant": plant,
                    "inverter_id": inverter_id,
                    "mppt_id": mppt_id,
                    "string_id": pv_id,
                    "irradiance KW/m2": irradiance_kw_m2,
                    "voltage_u": voltage,
                    "current_i": current_clean,
                    "power kw": power_kw,
                    "pv_temperature": pv_temperature,
                    "pv_Capacity": capacity,
                    "inverter_state": inverter_state,
                    "azimuth": safe_float(meta.get("azimuth", np.nan)),
                    "tilt": safe_float(meta.get("tilt", np.nan)),
                    "rainfall": safe_float(meta.get("rainfall", 0.0), default=0.0),
                    "Plant_P_abd": plant_p_abd,
                    "Plant_capacity": plant_capacity_doc,
                    "PR_PL": pr_pl,
                    "Deviation_PL": deviation_pl,
                    "Inverter_P_abd": inverter_p_abd,
                    "Inverter_capacity": inverter_capacity,

                    # extra internal/context columns for QC report and debugging
                    "string_label": f"{inverter_id}__{mppt_id}__{pv_id}",
                    "inverter_capacity": inverter_capacity,
                    "mppt_capacity": mppt_capacity,
                    "current_per_kwp": current_per_kwp,
                    "plant_latitude": meta.get("latitude", np.nan),
                    "plant_longitude": meta.get("longitude", np.nan),
                    "plant_altitude_m": meta.get("altitude_m", np.nan),
                    "azimuth_raw": meta.get("azimuth_raw", np.nan),
                })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — LOAD DATA  (JSON files or MongoDB)
# ─────────────────────────────────────────────────────────────────────────────

def load_from_json(path: str,
                   azimuth_override: float | None = None,
                   tilt_override: float | None = None,
                   rainfall_override: float | None = None) -> pd.DataFrame:
    """Load from a folder of .json files or a single .json file."""
    all_rows = []

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not files:
            print(f"[ERROR] No .json files found in {path}")
            sys.exit(1)
        print(f"[INFO] Found {len(files)} JSON files in {path}")
    else:
        files = [path]

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                content = fp.read().strip()
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
            for doc in data:
                all_rows.extend(flatten_document(
                    doc,
                    azimuth_override=azimuth_override,
                    tilt_override=tilt_override,
                    rainfall_override=rainfall_override,
                ))
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")

    if not all_rows:
        print("[ERROR] No data rows extracted — check your JSON files.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"[INFO] Extracted {len(df):,} rows from {len(files)} file(s)")
    return df


def load_from_mongodb(mongo_uri: str, db_name: str, collection: str,
                      days: int = 30,
                      date_from: str | None = None,
                      date_to: str | None = None,
                      hour_filter: str = "daylight",
                      min_irradiance_kw_m2: float = 0.1,
                      azimuth_override: float | None = None,
                      tilt_override: float | None = None,
                      rainfall_override: float | None = None) -> pd.DataFrame:
    """
    Load from MongoDB.

    hour_filter:
      daylight → date range + radiation_intensity >= min_irradiance_kw_m2
      noon     → date range + Day_Hour ends with 11/12/13
      all      → date range only
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        print("[ERROR] pymongo not installed. Run: pip install pymongo")
        sys.exit(1)

    if date_to:
        dt_to = datetime.strptime(date_to, "%Y-%m-%d")
    else:
        dt_to = datetime.utcnow()

    if date_from:
        dt_from = datetime.strptime(date_from, "%Y-%m-%d")
    else:
        dt_from = dt_to - timedelta(days=days)

    query = {
        "timestamp": {
            "$gte": dt_from,
            "$lte": dt_to,
        }
    }

    if hour_filter == "noon":
        query["Day_Hour"] = {"$regex": r" 11$| 12$| 13$"}
        filter_msg = "noon hours only: 11, 12, 13"
    elif hour_filter == "daylight":
        query["radiation_intensity"] = {"$gte": min_irradiance_kw_m2}
        filter_msg = f"daylight/high-irradiance only: radiation_intensity >= {min_irradiance_kw_m2} kW/m²"
    elif hour_filter == "all":
        filter_msg = "all rows in date range"
    else:
        print("[ERROR] --hour-filter must be one of: daylight, noon, all")
        sys.exit(1)

    print(f"[INFO] Date range   : {dt_from.date()} → {dt_to.date()}  ({days} days if --date-from not used)")
    print(f"[INFO] Data filter  : {filter_msg}")

    client = MongoClient(mongo_uri)
    col = client[db_name][collection]

    total_in_db = col.count_documents({})
    matched_docs = col.count_documents(query)
    print(f"[INFO] Total docs in DB     : {total_in_db:,}")
    print(f"[INFO] Docs after filter    : {matched_docs:,} ({matched_docs / max(total_in_db, 1) * 100:.1f}% of DB)")

    if matched_docs == 0:
        print("\n[WARN] No documents matched the filter!")
        print("       Try one of these:")
        print("       1. Use a wider --date-from / --date-to range")
        print("       2. Use --hour-filter all")
        print("       3. Lower --min-irradiance-kw-m2, e.g. 0.05")
        sys.exit(1)

    docs = list(col.find(query))
    print(f"[INFO] Fetched {len(docs):,} documents ✅")

    all_rows = []
    for doc in docs:
        all_rows.extend(flatten_document(
            doc,
            azimuth_override=azimuth_override,
            tilt_override=tilt_override,
            rainfall_override=rainfall_override,
        ))

    if not all_rows:
        print("[ERROR] No rows extracted after flattening.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"[INFO] Extracted {len(df):,} string-level rows total")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — DATA QUALITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

QC = dict(
    min_irradiance_kw_m2=0.05,   # kW/m². 0.05 = 50 W/m²
    max_irradiance_kw_m2=1.4,    # kW/m². 1.4 = 1400 W/m²
    min_voltage=100,
    max_voltage=1500,
    min_current=0,
    max_current=20,
    min_temp=-10,
    max_temp=85,
    min_capacity_kwp=0.1,
    noon_hour_start=11,
    noon_hour_end=13,
    min_noon_days=5,
)


def check_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs per-string quality checks and returns clean data + quality report."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    irr_col = "irradiance KW/m2"

    df["flag_night"] = df[irr_col] < QC["min_irradiance_kw_m2"]
    df["flag_bad_irr"] = df[irr_col] > QC["max_irradiance_kw_m2"]
    df["flag_bad_voltage"] = (df["voltage_u"] < QC["min_voltage"]) & (df[irr_col] >= QC["min_irradiance_kw_m2"])
    df["flag_bad_current"] = df["current_i"] > QC["max_current"]
    df["flag_bad_temp"] = (df["pv_temperature"] < QC["min_temp"]) | (df["pv_temperature"] > QC["max_temp"])
    df["flag_zero_cap"] = df["pv_Capacity"] < QC["min_capacity_kwp"]
    df["flag_nan"] = df[["current_i", "voltage_u", irr_col]].isna().any(axis=1)
    df["flag_noon"] = (
        (df["hour"] >= QC["noon_hour_start"]) &
        (df["hour"] <= QC["noon_hour_end"]) &
        (df[irr_col] >= 0.1)
    )

    df["any_issue"] = (
        df["flag_bad_irr"] |
        df["flag_bad_voltage"] |
        df["flag_bad_current"] |
        df["flag_bad_temp"] |
        df["flag_zero_cap"] |
        df["flag_nan"]
    )

    report_rows = []
    for label, g in df.groupby("string_label"):
        total = len(g)
        night = int(g["flag_night"].sum())
        daytime = total - night
        bad_rows = int(g["any_issue"].sum())
        noon_days = g[g["flag_noon"]]["timestamp"].dt.date.nunique()
        zero_cap = bool(g["flag_zero_cap"].all())
        pct_bad = round(bad_rows / max(daytime, 1) * 100, 1)

        if zero_cap:
            verdict = "SKIP — capacity unknown (pv_Capacity=0)"
        elif noon_days < QC["min_noon_days"]:
            verdict = f"WARN — only {noon_days} noon days (need ≥{QC['min_noon_days']})"
        elif pct_bad > 30:
            verdict = f"WARN — {pct_bad}% bad rows"
        else:
            verdict = "OK"

        report_rows.append({
            "string_label": label,
            "inverter": label.split("__")[0],
            "mppt": label.split("__")[1] if "__" in label else "",
            "pv": label.split("__")[2] if label.count("__") >= 2 else "",
            "total_rows": total,
            "daytime_rows": daytime,
            "bad_rows": bad_rows,
            "pct_bad": pct_bad,
            "noon_days": noon_days,
            "days_covered": g["timestamp"].dt.date.nunique(),
            "date_from": g["timestamp"].min().date() if total > 0 else None,
            "date_to": g["timestamp"].max().date() if total > 0 else None,
            "has_bad_irr": bool(g["flag_bad_irr"].any()),
            "has_bad_voltage": bool(g["flag_bad_voltage"].any()),
            "has_bad_current": bool(g["flag_bad_current"].any()),
            "has_nan": bool(g["flag_nan"].any()),
            "zero_cap": zero_cap,
            "verdict": verdict,
        })

    report = pd.DataFrame(report_rows).sort_values("verdict") if report_rows else pd.DataFrame()

    df_clean = df[
        (~df["flag_night"]) &
        (~df["any_issue"]) &
        (~df["flag_zero_cap"])
    ].copy()

    return df_clean, report


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — PRINT QUALITY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame, report: pd.DataFrame) -> None:
    total_strings = report["string_label"].nunique() if not report.empty else 0
    ok_strings = int((report["verdict"] == "OK").sum()) if not report.empty else 0
    warn_strings = int(report["verdict"].str.startswith("WARN").sum()) if not report.empty else 0
    skip_strings = int(report["verdict"].str.startswith("SKIP").sum()) if not report.empty else 0

    print("\n" + "=" * 60)
    print("  DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"  Total rows extracted   : {len(df_raw):,}")
    print(f"  Clean rows (usable)    : {len(df_clean):,}")
    print(f"  Rows dropped           : {len(df_raw) - len(df_clean):,}")
    print(f"  Total strings          : {total_strings}")
    print(f"  ✅ OK strings          : {ok_strings}")
    print(f"  ⚠  WARN strings        : {warn_strings}")
    print(f"  ❌ SKIP strings        : {skip_strings}")

    if len(df_raw):
        date_range = df_raw["timestamp"].agg(["min", "max"])
        print(f"\n  Date range : {date_range['min'].date()} → {date_range['max'].date()}")
        print(f"  Days in data           : {df_raw['timestamp'].dt.date.nunique()}")
        print(f"  Min noon days needed   : {QC['min_noon_days']} (quality warning only)")

    if not report.empty and (warn_strings > 0 or skip_strings > 0):
        print("\n  Strings needing attention:")
        problems = report[report["verdict"] != "OK"]
        for _, row in problems.head(30).iterrows():
            print(f"    {row['string_label']}: {row['verdict']}")
        if len(problems) > 30:
            print(f"    ... and {len(problems) - 30} more. Check the quality report file.")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df_clean: pd.DataFrame, report: pd.DataFrame, out_path: str) -> None:
    """Save clean soiling data and a separate quality report."""
    df_out = df_clean.copy()

    # Keep exactly the demo schema columns for the final soiling pipeline file.
    missing_cols = [c for c in SOILING_OUTPUT_COLUMNS if c not in df_out.columns]
    for col in missing_cols:
        df_out[col] = np.nan
    df_out = df_out[SOILING_OUTPUT_COLUMNS].copy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    clean_path = out_path
    if not (clean_path.endswith(".xlsx") or clean_path.endswith(".csv")):
        clean_path += ".xlsx"

    if clean_path.endswith(".csv"):
        df_out.to_csv(clean_path, index=False)
        print(f"[INFO] Clean soiling data saved → {clean_path}")
    else:
        try:
            with pd.ExcelWriter(clean_path, engine="openpyxl") as writer:
                df_out.to_excel(writer, sheet_name="data", index=False)
            print(f"[INFO] Clean soiling data saved → {clean_path}  (sheet: data)")
        except Exception as e:
            csv_path = clean_path.replace(".xlsx", ".csv")
            df_out.to_csv(csv_path, index=False)
            print(f"[WARN] Excel save failed ({e}), saved as CSV → {csv_path}")
            clean_path = csv_path

    print(f"[INFO] Rows: {len(df_out):,} | Strings: {df_out['string_id'].nunique() if len(df_out) else 0}")
    print(f"[INFO] Output columns: {', '.join(df_out.columns)}")

    report_path = clean_path.replace(".xlsx", "_quality_report.xlsx").replace(".csv", "_quality_report.xlsx")
    try:
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            report.to_excel(writer, sheet_name="quality_report", index=False)
        print(f"[INFO] Quality report saved → {report_path}")
    except Exception as e:
        report_csv = report_path.replace(".xlsx", ".csv")
        report.to_csv(report_csv, index=False)
        print(f"[WARN] Quality report Excel save failed ({e}), saved → {report_csv}")

    print("\n[NEXT STEP] Use this clean file in your soiling pipeline:")
    print(f"  {clean_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_quick_test() -> None:
    """Quick test using a mini version of your MongoDB document."""
    sample = {
        "timestamp": {"$date": "2026-02-12T09:15:00.000Z"},
        "Plant": "Coca Cola Faisalabad",
        "radiation_intensity": 0.5662,
        "sns": [
            {
                "snid": "ES23C0014748",
                "inverter_state": 512,
                "temperature": 33.2,
                "Inverter_capacity": 355.68,
                "mppts": [
                    {
                        "mpptId": "MPPT1",
                        "mppt_Capacity": 67.86,
                        "pvs": [
                            {"pvId": "pv1", "pv_P_abd": 8.0898, "i": 6.95, "u": 1164, "pv_Capacity": 16.965, "pv_temperature": 25.7},
                            {"pvId": "pv2", "pv_P_abd": 8.0898, "i": 6.95, "u": 1164, "pv_Capacity": 16.965, "pv_temperature": 25.7},
                            {"pvId": "pv21", "pv_P_abd": 0, "i": 0, "u": 1265.7, "pv_Capacity": 0, "pv_temperature": 25.7},
                        ],
                    }
                ],
            }
        ],
    }

    print("\n[TEST] Flattening sample document...")
    rows = flatten_document(sample)
    df_raw = pd.DataFrame(rows)
    print(df_raw[SOILING_OUTPUT_COLUMNS].to_string(index=False))

    print("\n[TEST] Running quality check...")
    df_clean, report = check_quality(df_raw)
    print(report[["string_label", "total_rows", "noon_days", "zero_cap", "verdict"]].to_string(index=False))

    print("\n[TEST] Clean output preview...")
    print(df_clean[SOILING_OUTPUT_COLUMNS].to_string(index=False))

    assert "irradiance KW/m2" in df_clean.columns
    assert "power kw" in df_clean.columns
    assert set(SOILING_OUTPUT_COLUMNS).issubset(df_clean.columns)
    assert float(df_raw.loc[0, "tilt"]) == 20.0
    assert float(df_raw.loc[0, "azimuth"]) == 167.0
    print("[TEST] PASSED ✅\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten nested MongoDB PV data into demo soiling-pipeline schema.")
    ap.add_argument("--input", "-i", default=None, help="Path to a .json file or folder of .json files")
    ap.add_argument("--mongo", default=None, help="MongoDB URI e.g. mongodb://localhost:27017")
    ap.add_argument("--db", default=None, help="MongoDB database name")
    ap.add_argument("--collection", default=None, help="MongoDB collection name")
    ap.add_argument("--days", type=int, default=90, help="How many days back to fetch. Used with --mongo.")
    ap.add_argument("--date-from", default=None, help="Start date YYYY-MM-DD. Overrides --days. Used with --mongo.")
    ap.add_argument("--date-to", default=None, help="End date YYYY-MM-DD. Default: today. Used with --mongo.")
    ap.add_argument("--output", "-o", default="data/clean_soiling_data.xlsx", help="Output .xlsx or .csv path")
    ap.add_argument("--hour-filter", choices=["daylight", "noon", "all"], default="daylight",
                    help="MongoDB filter mode: daylight, noon, or all. Default: daylight.")
    ap.add_argument("--min-irradiance-kw-m2", type=float, default=0.1,
                    help="Used when --hour-filter daylight. Default: 0.1 kW/m².")
    ap.add_argument("--azimuth", type=float, default=None,
                    help="Override azimuth for all rows. Example for CCI Faisalabad compass azimuth: 167.")
    ap.add_argument("--tilt", type=float, default=None,
                    help="Override tilt for all rows. Example for CCI Faisalabad: 20.")
    ap.add_argument("--rainfall", type=float, default=None,
                    help="Override rainfall for all rows. Default from plant config is 0.")
    ap.add_argument("--test", action="store_true", help="Run quick test with sample document")
    args = ap.parse_args()

    if args.test:
        run_quick_test()
        return

    if args.input:
        df_raw = load_from_json(
            args.input,
            azimuth_override=args.azimuth,
            tilt_override=args.tilt,
            rainfall_override=args.rainfall,
        )
    elif args.mongo:
        if not args.db or not args.collection:
            print("[ERROR] --db and --collection are required with --mongo")
            sys.exit(1)
        df_raw = load_from_mongodb(
            mongo_uri=args.mongo,
            db_name=args.db,
            collection=args.collection,
            days=args.days,
            date_from=args.date_from,
            date_to=args.date_to,
            hour_filter=args.hour_filter,
            min_irradiance_kw_m2=args.min_irradiance_kw_m2,
            azimuth_override=args.azimuth,
            tilt_override=args.tilt,
            rainfall_override=args.rainfall,
        )
    else:
        print("[ERROR] Provide --input (JSON) or --mongo (MongoDB URI)")
        print("        Or run --test to try with a sample document")
        sys.exit(1)

    df_clean, report = check_quality(df_raw)
    print_summary(df_raw, df_clean, report)
    save_outputs(df_clean, report, args.output)


if __name__ == "__main__":
    main()
