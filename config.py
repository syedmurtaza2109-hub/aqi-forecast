"""
config.py — shared settings loaded from environment variables.
All other scripts import from here so you only change values once.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

# ── API credentials ──────────────────────────────────────────────────────────
AQICN_TOKEN        = os.getenv("AQICN_TOKEN", "")
OPENWEATHER_KEY    = os.getenv("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_forecast")

# ── City ─────────────────────────────────────────────────────────────────────
CITY_NAME    = os.getenv("CITY_NAME", "Karachi")
CITY_LAT     = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON     = float(os.getenv("CITY_LON", "67.0011"))
AQICN_STATION = os.getenv("AQICN_STATION", "@9530")  # Karachi station ID

# ── Feature Store ────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME            = "aqi_forecaster"

# ── Model settings ───────────────────────────────────────────────────────────
FORECAST_HOURS   = 72   # predict next 72 hours = 3 days
LOOKBACK_HOURS   = 24   # use last 24 hours as input features
TARGET_COLUMN    = "aqi"

# ── AQI level labels (US EPA standard) ───────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",            "#00e400"),
    (51,  100, "Moderate",        "#ffff00"),
    (101, 150, "Unhealthy (Sens)","#ff7e00"),
    (151, 200, "Unhealthy",       "#ff0000"),
    (201, 300, "Very Unhealthy",  "#8f3f97"),
    (301, 500, "Hazardous",       "#7e0023"),
]

def get_aqi_level(aqi_value: float) -> tuple:
    """Return (label, color) for a given AQI value."""
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi_value <= hi:
            return label, color
    return "Hazardous", "#7e0023"
