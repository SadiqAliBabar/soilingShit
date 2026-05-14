"""Static catalogues."""
from __future__ import annotations

HUAWEI_STATES = {
    0:("Standby","STANDBY"), 1:("Standby: detecting","STANDBY"),
    2:("Standby: grid detect","STANDBY"), 3:("Starting","STANDBY"),
    512:("On-grid","RUNNING"), 513:("On-grid P-limited","CURTAILED"),
    514:("On-grid self-derating","CURTAILED"),
    768:("Shutdown: fault","FAULT"), 769:("Shutdown: cmd","FAULT"),
    771:("Shutdown: comms lost","FAULT"), 772:("Shutdown: low DC","FAULT"),
    1280:("Spot-check ready","TRANSIENT"), 1281:("Spot-checking","TRANSIENT"),
    1284:("Grid scheduling P-limit","CURTAILED"),
    1285:("Grid scheduling Q-limit","CURTAILED"),
    1536:("Inspection","TRANSIENT"), 2048:("IV scanning","IV_SCAN"),
    40960:("Standby: no irradiation","STANDBY"),
}
STATE_CATEGORY = {k:v[1] for k,v in HUAWEI_STATES.items()}
STATE_NAME     = {k:v[0] for k,v in HUAWEI_STATES.items()}
CURTAILED_STATES = frozenset(k for k,c in STATE_CATEGORY.items() if c=="CURTAILED")
FAULT_STATES     = frozenset(k for k,c in STATE_CATEGORY.items() if c=="FAULT")
STANDBY_STATES   = frozenset(k for k,c in STATE_CATEGORY.items() if c=="STANDBY")
IV_SCAN_STATES   = frozenset(k for k,c in STATE_CATEGORY.items() if c=="IV_SCAN")
TRANSIENT_STATES = frozenset(k for k,c in STATE_CATEGORY.items() if c=="TRANSIENT")

QUALITY_FLAGS = {
    "OK":0, "NIGHT":1<<0, "COMMS_GAP":1<<1, "V_OUT_OF_RANGE":1<<2,
    "I_OUT_OF_RANGE":1<<3, "P_NEG":1<<4, "G_LOW":1<<5,
    "INVERTER_FAULT":1<<6, "CURT_STATE":1<<7, "CURT_STATISTICAL":1<<8,
    "IV_SCAN":1<<9, "TRANSIENT":1<<10, "STANDBY":1<<11,
}
DISQUALIFYING = (QUALITY_FLAGS["COMMS_GAP"] | QUALITY_FLAGS["V_OUT_OF_RANGE"]
    | QUALITY_FLAGS["I_OUT_OF_RANGE"] | QUALITY_FLAGS["P_NEG"]
    | QUALITY_FLAGS["INVERTER_FAULT"] | QUALITY_FLAGS["CURT_STATE"]
    | QUALITY_FLAGS["CURT_STATISTICAL"] | QUALITY_FLAGS["IV_SCAN"]
    | QUALITY_FLAGS["TRANSIENT"] | QUALITY_FLAGS["STANDBY"])

LAHORE_LAT = 31.5204
LAHORE_LON = 74.3587
LAHORE_TZ  = "Asia/Karachi"
LAHORE_ALT = 217.0
DEFAULT_AZIMUTH_PK = 180.0
DEFAULT_TILT_PK    = 25.0
DEFAULT_TARIFF_PKR_PER_KWH = 38.0
DEFAULT_CURRENCY = "PKR"

TECH_DEFAULTS = {
    "mono-c-Si": dict(vmp_voc=0.842, imp_isc=0.945,
        alpha_isc=0.00046, beta_voc=-0.00260, gamma_pmp=-0.00300,
        annual_degradation=0.0040, lid_loss=0.010),
    "poly-c-Si": dict(vmp_voc=0.830, imp_isc=0.940,
        alpha_isc=0.00050, beta_voc=-0.00310, gamma_pmp=-0.00410,
        annual_degradation=0.0070, lid_loss=0.025),
}
