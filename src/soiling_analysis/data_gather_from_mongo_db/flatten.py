"""Flatten nested MongoDB documents into four tidy DataFrames."""

from datetime import datetime

import pandas as pd


def _ts(raw) -> datetime:
    if isinstance(raw, dict):
        return pd.to_datetime(raw.get("$date"))
    return pd.to_datetime(raw)


def flatten_all(records: list[dict]) -> dict[str, pd.DataFrame]:
    plant_rows, inverter_rows, mppt_rows, string_rows = [], [], [], []

    for doc in records:
        plant_name = doc.get("Plant", "")
        ts = _ts(doc.get("timestamp"))
        day_hour = doc.get("Day_Hour", "")

        # ── Plant ──────────────────────────────────────────────────
        plant_rows.append({
            "plant_name":          plant_name,
            "timestamp":           ts,
            "day_hour":            day_hour,
            "irradiance_wm2":      doc.get("radiation_intensity"),
            "plant_capacity_kwp":  doc.get("Plant_capacity"),
            "plant_power_kw":      doc.get("Plant_P_abd"),
            "plant_specific_yield": doc.get("specific_yield_PL"),
            "plant_pr":            doc.get("PR_PL"),
            "plant_deviation":     doc.get("Deviation_PL"),
        })

        for inv in doc.get("sns", []):
            inv_id = inv.get("snid", "")

            # ── Inverter ───────────────────────────────────────────
            inverter_rows.append({
                "plant_name":                  plant_name,
                "timestamp":                   ts,
                "day_hour":                    day_hour,
                "inverter_id":                 inv_id,
                "inverter_capacity_kwp":       inv.get("Inverter_capacity"),
                "inverter_power_kw":           inv.get("Inverter_P_abd"),
                "inverter_active_power_kw":    inv.get("active_power"),
                "inverter_reactive_power_kvar": inv.get("reactive_power"),
                "inverter_power_factor":       inv.get("power_factor"),
                "inverter_efficiency_pct":     inv.get("efficiency"),
                "inverter_temp_c":             inv.get("temperature"),
                "grid_frequency_hz":           inv.get("elec_freq"),
                "inverter_state":              inv.get("inverter_state"),
                "inverter_daily_energy_kwh":   inv.get("day_cap"),
                "mppt_total_power_kw":         inv.get("mppt_power"),
                "mppt_total_capacity_kwp":     inv.get("mppt_total_cap"),
                "mppt_1_energy_kwh":           inv.get("mppt_1_cap"),
                "mppt_2_energy_kwh":           inv.get("mppt_2_cap"),
                "mppt_3_energy_kwh":           inv.get("mppt_3_cap"),
                "mppt_4_energy_kwh":           inv.get("mppt_4_cap"),
                "mppt_5_energy_kwh":           inv.get("mppt_5_cap"),
                "mppt_6_energy_kwh":           inv.get("mppt_6_cap"),
                "mppt_7_energy_kwh":           inv.get("mppt_7_cap"),
                "mppt_8_energy_kwh":           inv.get("mppt_8_cap"),
                "mppt_9_energy_kwh":           inv.get("mppt_9_cap"),
                "mppt_10_energy_kwh":          inv.get("mppt_10_cap"),
                "voltage_phase_a_v":           inv.get("a_u"),
                "voltage_phase_b_v":           inv.get("b_u"),
                "voltage_phase_c_v":           inv.get("c_u"),
                "current_phase_a_a":           inv.get("a_i"),
                "current_phase_b_a":           inv.get("b_i"),
                "current_phase_c_a":           inv.get("c_i"),
                "voltage_line_ab_v":           inv.get("ab_u"),
                "voltage_line_bc_v":           inv.get("bc_u"),
                "voltage_line_ca_v":           inv.get("ca_u"),
                "inverter_specific_yield":     inv.get("specific_yield_SN"),
                "inverter_pr":                 inv.get("PR_SN"),
                "inverter_deviation":          inv.get("Deviation_SN"),
                "inverter_sy_deviation":       inv.get("Deviation_SY_SN"),
            })

            for mppt in inv.get("mppts", []):
                mppt_id = mppt.get("mpptId", "")

                # ── MPPT ──────────────────────────────────────────
                mppt_rows.append({
                    "plant_name":        plant_name,
                    "timestamp":         ts,
                    "day_hour":          day_hour,
                    "inverter_id":       inv_id,
                    "mppt_id":           mppt_id,
                    "mppt_capacity_kwp": mppt.get("mppt_Capacity"),
                    "mppt_power_kw":     mppt.get("mppt_P_abd"),
                    "mppt_specific_yield": mppt.get("specific_yield_MPPT"),
                    "mppt_pr":           mppt.get("PR_MPPT"),
                    "mppt_deviation":    mppt.get("Deviation_MPPT"),
                    "mppt_sy_deviation": mppt.get("Deviation_SY_MPPT"),
                })

                for pv in mppt.get("pvs", []):
                    # ── String ────────────────────────────────────
                    string_rows.append({
                        "plant_name":                    plant_name,
                        "timestamp":                     ts,
                        "day_hour":                      day_hour,
                        "inverter_id":                   inv_id,
                        "mppt_id":                       mppt_id,
                        "string_id":                     pv.get("pvId"),
                        "string_capacity_kwp":           pv.get("pv_Capacity"),
                        "string_power_kw":               pv.get("pv_P_abd"),
                        "string_current_a":              pv.get("i"),
                        "string_voltage_v":              pv.get("u"),
                        "string_specific_yield":         pv.get("specific_yield_PV"),
                        "string_pr":                     pv.get("PR_PV"),
                        "string_deviation":              pv.get("Deviation_PV"),
                        "string_sy_deviation_vs_inverter": pv.get("Deviation_SY_PV_inverter"),
                        "pv_temperature":                None,
                    })

    return {
        "plant":    pd.DataFrame(plant_rows),
        "inverter": pd.DataFrame(inverter_rows),
        "mppt":     pd.DataFrame(mppt_rows),
        "string":   pd.DataFrame(string_rows),
    }


COLUMN_COUNTS = {
    "plant":    9,
    "inverter": 31,
    "mppt":     11,
    "string":   15,  # +1 for pv_temperature
}


def build_temperature_lookup(temp_records: list[dict]) -> dict:
    """Build {hour_timestamp_str: pv_temperature} from FM_OD_PRD EMI docs.

    FM_OD_PRD has one EMI (environmental sensor) per plant, so the key is
    timestamp only — no device ID. Sub-hourly (5-min) readings are averaged
    within each hour to match the hourly resolution of the main data.
    Keeps pv_temperature == 0 (valid reading); only skips None / NaN.
    """
    buckets: dict[str, list[float]] = {}
    for doc in temp_records:
        temp = doc.get("pv_temperature")
        ts_raw = doc.get("timestamp")

        if ts_raw is None or temp is None:
            continue
        try:
            if pd.isna(temp):
                continue
        except (TypeError, ValueError):
            pass

        if isinstance(ts_raw, str):
            ts_str = ts_raw[:19]
        else:
            try:
                ts_str = pd.to_datetime(ts_raw).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue

        ts_hour = ts_str[:13] + ":00:00"
        buckets.setdefault(ts_hour, []).append(float(temp))

    return {k: sum(v) / len(v) for k, v in buckets.items()}


def enrich_pv_temperature(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Fill pv_temperature from the EMI timestamp lookup.

    FM_OD_PRD has one EMI sensor per plant, so temperature is the same for
    every string at a given hour — lookup key is timestamp only.
    Tries UTC first, then PKT (UTC+5) if no matches found.
    """
    if not lookup or df.empty:
        return df

    df = df.copy()
    base_ts = pd.to_datetime(df["timestamp"])

    def _try_offset(hours: int):
        ts_keys = (base_ts + pd.Timedelta(hours=hours)).dt.strftime("%Y-%m-%d %H:00:00")
        values = [lookup.get(ts) for ts in ts_keys]
        matched = sum(1 for v in values if v is not None)
        return values, matched

    values, matched = _try_offset(0)

    if matched == 0:
        values_pkt, matched_pkt = _try_offset(5)
        if matched_pkt > 0:
            values = values_pkt

    df["pv_temperature"] = values
    return df
