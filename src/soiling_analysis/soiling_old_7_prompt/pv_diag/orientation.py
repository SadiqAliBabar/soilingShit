"""Solar geometry + clear-sky POA + asymmetry expectation + clustering.
Self-contained (no pvlib)."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .config import PipelineConfig


def _solar_position(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """Approximate solar zenith/azimuth (Spencer 1971)."""
    if times.tz is None:
        ts_utc = times - pd.Timedelta(hours=lon / 15.0)
    else:
        ts_utc = times.tz_convert("UTC").tz_localize(None)
    doy = ts_utc.dayofyear.values + (ts_utc.hour.values
                                     + ts_utc.minute.values / 60.0) / 24.0
    gamma = 2 * math.pi * (doy - 1) / 365.0
    eot = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                    - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.001480*np.sin(3*gamma))  # rad

    hr_utc = ts_utc.hour.values + ts_utc.minute.values / 60.0
    solar_time = hr_utc + (4 * lon + eot) / 60.0
    H = math.pi * (solar_time - 12) / 12.0  # hour angle, rad

    lat_r = math.radians(lat)
    cos_z = (np.sin(lat_r) * np.sin(decl)
             + np.cos(lat_r) * np.cos(decl) * np.cos(H))
    cos_z = np.clip(cos_z, -1.0, 1.0)
    zen = np.degrees(np.arccos(cos_z))
    elev = 90.0 - zen
    # azimuth (south=180 convention)
    sin_a = np.cos(decl) * np.sin(H) / np.clip(np.sin(np.radians(zen)), 1e-6, None)
    cos_a = ((np.sin(lat_r)*np.cos(np.radians(zen)) - np.sin(decl))
             / np.clip(np.cos(lat_r)*np.sin(np.radians(zen)), 1e-6, None))
    az = np.degrees(np.arctan2(sin_a, cos_a)) + 180.0
    return pd.DataFrame(dict(zenith=zen, elevation=elev, azimuth=az,
                             declination=np.degrees(decl)), index=times)


def _clearsky_ghi(zen_deg: np.ndarray, altitude_m: float = 217.0) -> np.ndarray:
    """ASHRAE-simple clear-sky GHI."""
    cos_z = np.cos(np.radians(np.clip(zen_deg, 0, 90)))
    AM = 1.0 / np.clip(cos_z, 0.05, None)
    # Linke turbidity ~3.5 plausible; simple Bird-like form
    GHI = 1100.0 * np.exp(-0.18 * AM) * cos_z
    return np.where(zen_deg >= 90, 0.0, np.maximum(GHI, 0.0))


def _ghi_to_poa(ghi, zen_deg, surface_az, surface_tilt, lat):
    """Hay-Davies-ish simple transposition (isotropic diffuse, 20% diffuse fraction)."""
    zen_r = np.radians(np.clip(zen_deg, 0, 90))
    tilt_r = math.radians(surface_tilt)
    # Solar azimuth from _solar_position is south=180; we need angle of incidence
    # Approximate: cos(AOI) = cos(zen)*cos(tilt) + sin(zen)*sin(tilt)*cos(az_diff)
    # We don't have the actual solar azimuth here per-sample, so we use a
    # daily-averaged approximation: assume sun tracks E→W symmetrically.
    cos_aoi = np.cos(zen_r) * math.cos(tilt_r) + np.sin(zen_r) * math.sin(tilt_r)
    cos_aoi = np.clip(cos_aoi, 0, 1)
    diffuse_frac = 0.2
    direct = ghi * (1 - diffuse_frac) * cos_aoi / np.clip(np.cos(zen_r), 0.05, None)
    diffuse = ghi * diffuse_frac * (1 + math.cos(tilt_r)) / 2.0
    return np.maximum(direct + diffuse, 0.0)


def compute_clearsky_poa(df_or_index, lat, lon, azimuth=180.0, tilt=25.0,
                         altitude=217.0):
    if isinstance(df_or_index, pd.DataFrame):
        idx = pd.to_datetime(df_or_index["ts"])
    else:
        idx = pd.to_datetime(df_or_index)
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    sp = _solar_position(idx, lat, lon)
    ghi = _clearsky_ghi(sp["zenith"].values, altitude)
    poa = _ghi_to_poa(ghi, sp["zenith"].values, azimuth, tilt, lat)
    return pd.Series(poa, index=idx, name="POA_clearsky")


def expected_asymmetry(azimuth: float, tilt: float, lat: float) -> float:
    """Geometric AM/PM asymmetry expected from orientation (fraction)."""
    # Pure-south (180°) → 0; East/West shift creates ~|az-180|/180 * tilt-factor
    az_off = abs(azimuth - 180.0) / 180.0  # 0..1
    tilt_f = min(tilt / 45.0, 1.0)         # 0..1
    sign = 1.0 if azimuth < 180 else -1.0  # east leans AM, west leans PM
    # Empirical scaling: 30° azimuth offset ≈ 8% asymmetry at 25° tilt
    return sign * az_off * tilt_f * 0.30


def cluster_by_azimuth_tilt(string_meta: dict, az_tol=15.0, tilt_tol=5.0):
    """Round (az, tilt) to label clusters."""
    out = {}
    for label, m in string_meta.items():
        az = float(m.get("azimuth", 180))
        tl = float(m.get("tilt", 25))
        az_r = int(round(az / az_tol) * az_tol) % 360
        tl_r = int(round(tl / tilt_tol) * tilt_tol)
        out[label] = f"az{az_r:03d}_t{tl_r:02d}"
    return out


def cluster_by_mppt(string_dfs: dict) -> dict:
    """Group by inverter+MPPT id."""
    out = {}
    for label, df in string_dfs.items():
        if len(df) == 0:
            out[label] = "unknown"; continue
        inv = str(df["inverter_id"].iloc[0])
        mppt = str(df["mppt_id"].iloc[0])
        out[label] = f"{inv}__{mppt}"
    return out


def poa_health_check(df: pd.DataFrame, lat: float, lon: float,
                     azimuth: float, tilt: float) -> dict:
    """Compare measured POA to clear-sky POA on bright midday hours."""
    if len(df) == 0 or "POA" not in df.columns:
        return dict(ratio_median=np.nan, note="no data")
    cs = compute_clearsky_poa(df, lat, lon, azimuth, tilt)
    ts = pd.to_datetime(df["ts"])
    if getattr(ts.dt, "tz", None): ts_local = ts.dt.tz_convert(None)
    else: ts_local = ts
    hr = ts_local.dt.hour + ts_local.dt.minute / 60.0
    midday = (hr >= 11) & (hr <= 13)
    meas = pd.to_numeric(df["POA"], errors="coerce")
    valid = midday & (cs.values > 200) & meas.notna()
    if valid.sum() < 5:
        return dict(ratio_median=np.nan, note="too few midday samples")
    ratio = meas[valid].values / cs.values[valid]
    return dict(ratio_median=float(np.nanmedian(ratio)),
                ratio_p25=float(np.nanpercentile(ratio, 25)),
                ratio_p75=float(np.nanpercentile(ratio, 75)),
                note=f"based on {int(valid.sum())} midday samples")
