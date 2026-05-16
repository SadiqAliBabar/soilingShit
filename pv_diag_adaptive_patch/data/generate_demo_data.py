"""Generate realistic demo plant data: 2 inverters x 4 strings, 5-min, 1 month.

Scenarios per string (test items b–f):
  INV01__MPPT1__pv1: clean baseline                 (control)
  INV01__MPPT1__pv2: soiled then washed mid-month   (item e: post-wash clean)
  INV01__MPPT2__pv3: heavy soiling, no recovery     (control hvy)
  INV01__MPPT2__pv4: curtailment-heavy              (item b)
  INV02__MPPT1__pv5: clean, east-facing az=90       (item c: orientation)
  INV02__MPPT1__pv6: aged 2+ yrs, deg signature     (item f: degradation)
  INV02__MPPT2__pv7: partial wash recovery          (item e: partial)
  INV02__MPPT2__pv8: faulty / insufficient data
"""
from __future__ import annotations
import sys, math
from pathlib import Path
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd


# Plant context (test item d)
PLANT_NAME   = "Coca Cola Faisalabad"
LAT, LON     = 31.4504, 73.1350
ALT          = 184.0
TZ_OFFSET_HR = 5.0           # Asia/Karachi
COMMISSIONING = date(2023, 6, 1)
TARIFF_PKR   = 38.0

# Period
START = datetime(2025, 10, 1, 0, 0)
END   = datetime(2025, 10, 31, 23, 55)
FREQ  = "5min"

# Module / string nameplate
PER_MODULE_VMP = 41.7
PER_MODULE_VOC = 49.5
PER_MODULE_IMP = 12.95
PER_MODULE_ISC = 13.85
N_MOD          = 22
STR_VMP        = PER_MODULE_VMP * N_MOD
STR_VOC        = PER_MODULE_VOC * N_MOD
PMP_STR_W      = STR_VMP * PER_MODULE_IMP
PV_CAPACITY_KW = PMP_STR_W / 1000.0


def solar_position(t_local: pd.Timestamp, lat=LAT, lon=LON, tz_off=TZ_OFFSET_HR):
    t_utc = t_local - pd.Timedelta(hours=tz_off)
    doy = t_utc.dayofyear + (t_utc.hour + t_utc.minute/60.0) / 24.0
    gamma = 2*math.pi*(doy-1)/365.0
    eot = 229.18*(0.000075 + 0.001868*math.cos(gamma) - 0.032077*math.sin(gamma)
                  - 0.014615*math.cos(2*gamma) - 0.040849*math.sin(2*gamma))
    decl = (0.006918 - 0.399912*math.cos(gamma) + 0.070257*math.sin(gamma)
            - 0.006758*math.cos(2*gamma) + 0.000907*math.sin(2*gamma)
            - 0.002697*math.cos(3*gamma) + 0.001480*math.sin(3*gamma))
    hr_utc = t_utc.hour + t_utc.minute/60.0
    solar_time = hr_utc + (4*lon + eot) / 60.0
    H = math.pi * (solar_time - 12) / 12.0
    lat_r = math.radians(lat)
    cos_z = math.sin(lat_r)*math.sin(decl) + math.cos(lat_r)*math.cos(decl)*math.cos(H)
    cos_z = max(-1.0, min(1.0, cos_z))
    zen = math.degrees(math.acos(cos_z))
    sin_a = math.cos(decl) * math.sin(H) / max(math.sin(math.radians(zen)), 1e-6)
    cos_a = ((math.sin(lat_r)*math.cos(math.radians(zen)) - math.sin(decl)) /
             max(math.cos(lat_r)*math.sin(math.radians(zen)), 1e-6))
    az = math.degrees(math.atan2(sin_a, cos_a)) + 180.0
    return zen, az


def clearsky_poa(t_local, surf_az=180.0, tilt=25.0, lat=LAT, lon=LON):
    """Tilted POA with Hay-Davies-ish simple decomposition."""
    zen, sun_az = solar_position(t_local, lat, lon)
    if zen >= 90: return 0.0
    cos_z = max(math.cos(math.radians(zen)), 0.05)
    AM = 1.0 / cos_z
    GHI = 1100.0 * math.exp(-0.18 * AM) * cos_z
    tilt_r = math.radians(tilt)
    daz = math.radians(sun_az - surf_az)
    cos_aoi = (math.cos(math.radians(zen))*math.cos(tilt_r) +
               math.sin(math.radians(zen))*math.sin(tilt_r)*math.cos(daz))
    cos_aoi = max(0.0, cos_aoi)
    diffuse_frac = 0.20
    direct  = GHI * (1 - diffuse_frac) * cos_aoi / cos_z
    diffuse = GHI * diffuse_frac * (1 + math.cos(tilt_r)) / 2.0
    return max(direct + diffuse, 0.0)


def build_string_data(string_label, inv_id, mppt_id, pv_id, az, tilt,
                      scenario, ts_index, rng):
    """Return DataFrame for one string."""
    n = len(ts_index)

    # Clear-sky POA per timestamp
    poa = np.array([clearsky_poa(t, az, tilt) for t in ts_index])

    # Cloud overlay: smooth random multiplier 0.7–1.0 plus occasional dips
    cloud = 0.95 + 0.05 * rng.standard_normal(n)
    cloud = np.clip(cloud, 0.55, 1.05)
    # rare overcast hours
    overcast = rng.random(n) < 0.04
    cloud[overcast] *= rng.uniform(0.3, 0.7, overcast.sum())
    poa_meas = poa * cloud + rng.normal(0, 5, n)
    poa_meas = np.clip(poa_meas, 0, 1200)

    # Soiling timeline (NCI) — function of day-of-month
    dom = np.array([t.day + t.hour/24.0 for t in ts_index])
    if scenario == "clean":
        soil = np.ones(n) * 0.98
    elif scenario == "soiled_then_washed":
        # Loss accumulates till day 17, then rain ~12mm, recovers to ~0.97
        soil = np.where(dom < 17, 1.0 - 0.012*(dom - 1), 0.985 - 0.0008*(dom - 17))
        soil = np.clip(soil, 0.4, 1.0)
    elif scenario == "heavy_soil":
        soil = 1.0 - 0.011 * dom
        soil = np.clip(soil, 0.4, 1.0)
    elif scenario == "curtail":
        soil = np.ones(n) * 0.97
    elif scenario == "clean_east":
        soil = np.ones(n) * 0.97
    elif scenario == "degraded":
        soil = np.ones(n) * 0.91          # degradation, not soil
    elif scenario == "partial_wash":
        soil = np.where(dom < 17, 1.0 - 0.012*(dom - 1), 0.92 - 0.003*(dom - 17))
        soil = np.clip(soil, 0.4, 1.0)
    elif scenario == "faulty":
        soil = np.ones(n) * 0.98
    else:
        soil = np.ones(n) * 0.95

    # Cell temperature (NOCT)
    Tc = 25.0 + (poa_meas / 800.0) * 20.0

    # Expected current/power at module level
    Imp_exp = PER_MODULE_IMP * (poa_meas / 1000.0) * (1 + 0.0004 * (Tc - 25))
    I = Imp_exp * soil + rng.normal(0, 0.04, n)
    I = np.clip(I, 0, None)
    Vmp_str = STR_VMP * (1 + (-0.0027) * (Tc - 25))
    V = np.where(poa_meas > 50, Vmp_str + rng.normal(0, 1.2, n), rng.uniform(0, 50, n))
    P_w = V * I
    P_w = np.clip(P_w, 0, None)

    # State code (0 = standby at night, 512 = on-grid otherwise)
    state = np.where(poa_meas < 30, 0, 512)

    # Scenario tweaks
    if scenario == "curtail":
        cap = 6000.0
        mask = P_w > cap
        P_w[mask] = cap + rng.normal(0, 30, mask.sum())
        state[mask] = 513
        I[mask] = P_w[mask] / np.clip(V[mask], 50, None)

    if scenario == "degraded":
        I *= 0.99   # Isc nearly normal
        V *= 0.94   # Voc/Vmp ~6% down
        P_w = V * I

    if scenario == "faulty":
        bad = rng.random(n) < 0.70
        state[bad] = 768
        I[bad] = 0; V[bad] = 0; P_w[bad] = 0

    # Rainfall (zero except a few storms; only show on plant-level pv2 day 17, pv7 day 17)
    rain = np.zeros(n)
    # one big rain event on day 17 affecting everyone — but soiling response differs
    day17_mask = np.array([(t.day == 17 and 13 <= t.hour <= 16) for t in ts_index])
    if scenario in ("soiled_then_washed", "partial_wash"):
        rain[day17_mask] = rng.uniform(0.6, 1.2, day17_mask.sum())  # mm per 5-min
    elif scenario in ("clean","heavy_soil","curtail","clean_east","degraded","faulty"):
        rain[day17_mask] = rng.uniform(0.6, 1.2, day17_mask.sum())  # same rain hits all

    df = pd.DataFrame({
        "timestamp":   ts_index,
        "plant":       PLANT_NAME,
        "inverter_id": inv_id,
        "mppt_id":     mppt_id,
        "string_id":   pv_id,
        "irradiance KW/m2": poa_meas / 1000.0,
        "voltage_u":   V,
        "current_i":   I,
        "power kw":    P_w / 1000.0,
        "pv_temperature": Tc,
        "pv_Capacity": PV_CAPACITY_KW,
        "inverter_state": state,
        "azimuth":     az,
        "tilt":        tilt,
        "rainfall":    rain,
    })
    return df


def main(out_path: str = "/mnt/user-data/outputs/demo_plant_data.xlsx"):
    rng = np.random.default_rng(42)
    ts = pd.date_range(START, END, freq=FREQ)

    strings = [
        ("INV01","MPPT1","pv1", 180, 25, "clean"),
        ("INV01","MPPT1","pv2", 180, 25, "soiled_then_washed"),
        ("INV01","MPPT2","pv3", 180, 25, "heavy_soil"),
        ("INV01","MPPT2","pv4", 180, 25, "curtail"),
        ("INV02","MPPT1","pv5",  90, 25, "clean_east"),
        ("INV02","MPPT1","pv6", 180, 25, "degraded"),
        ("INV02","MPPT2","pv7", 180, 25, "partial_wash"),
        ("INV02","MPPT2","pv8", 180, 25, "faulty"),
    ]

    frames = []
    for inv, mppt, pv, az, tilt, sc in strings:
        lbl = f"{inv}_{mppt}_{pv}"
        print(f"  ... building {lbl} ({sc})")
        frames.append(build_string_data(lbl, inv, mppt, pv, az, tilt, sc, ts, rng))
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["timestamp","inverter_id","mppt_id","string_id"])

    # Metadata sheet
    meta = pd.DataFrame([
        ("plant_name",         PLANT_NAME),
        ("latitude",           LAT),
        ("longitude",          LON),
        ("tariff_pkr_kwh",     TARIFF_PKR),
        ("commissioning_date", COMMISSIONING.isoformat()),
        ("p_ac_max_kw",        100.0),
        ("technology",         "mono-c-Si"),
    ])

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        long.to_excel(w, sheet_name="Data", index=False)
        meta.to_excel(w, sheet_name="Metadata", index=False, header=False)
    print(f"Wrote {out_path}  ({len(long):,} rows, {long['inverter_id'].nunique()} inv, "
          f"{long.groupby(['inverter_id','mppt_id','string_id']).ngroups} strings)")
    return str(out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs/demo_plant_data.xlsx"
    main(out)
