"""
feature_pipeline/fetch_and_store.py
────────────────────────────────────
Runs every hour via GitHub Actions.

What it does:
  1. Fetches current AQI data from AQICN
  2. Fetches current weather from OpenWeatherMap
  3. Engineers time and derived features
  4. Upserts one row into the Hopsworks Feature Store

Run manually:  python feature_pipeline/fetch_and_store.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import hopsworks
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fetch raw data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_aqicn() -> dict:
    """Fetch current AQI and pollutants from AQICN."""
    url = f"https://api.waqi.info/feed/{config.AQICN_STATION}/?token={config.AQICN_TOKEN}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "ok":
        raise ValueError(f"AQICN returned status: {data['status']}")

    d = data["data"]
    iaqi = d.get("iaqi", {})

    return {
        "aqi":   float(d.get("aqi", 0)),
        "pm25":  float(iaqi.get("pm25", {}).get("v", 0)),
        "pm10":  float(iaqi.get("pm10", {}).get("v", 0)),
        "o3":    float(iaqi.get("o3",   {}).get("v", 0)),
        "no2":   float(iaqi.get("no2",  {}).get("v", 0)),
        "so2":   float(iaqi.get("so2",  {}).get("v", 0)),
        "co":    float(iaqi.get("co",   {}).get("v", 0)),
    }


def fetch_weather() -> dict:
    """Fetch current weather from OpenWeatherMap."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={config.CITY_LAT}&lon={config.CITY_LON}"
        f"&appid={config.OPENWEATHER_KEY}&units=metric"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    d = resp.json()

    return {
        "temperature":  float(d["main"]["temp"]),
        "humidity":     float(d["main"]["humidity"]),
        "pressure":     float(d["main"]["pressure"]),
        "wind_speed":   float(d["wind"]["speed"]),
        "wind_deg":     float(d["wind"].get("deg", 0)),
        "visibility":   float(d.get("visibility", 10000)) / 1000,  # km
        "weather_main": d["weather"][0]["main"],  # e.g. "Clear", "Rain"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Engineer features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(aqi_data: dict, weather_data: dict) -> pd.DataFrame:
    """Combine raw data into a feature row with time and derived features."""
    now = datetime.now(timezone.utc)

    # One-hot encode weather condition
    weather_dummies = {
        "is_clear": 1 if weather_data["weather_main"] == "Clear" else 0,
        "is_rain":  1 if weather_data["weather_main"] in ("Rain", "Drizzle") else 0,
        "is_cloudy":1 if weather_data["weather_main"] in ("Clouds", "Overcast") else 0,
        "is_haze":  1 if weather_data["weather_main"] in ("Haze", "Smoke", "Dust", "Mist") else 0,
    }

    row = {
        # ── Identifiers ──────────────────────────────────────────────────────
        "timestamp":        int(now.timestamp()),
        "city":             config.CITY_NAME,

        # ── Pollutants ───────────────────────────────────────────────────────
        "aqi":              aqi_data["aqi"],
        "pm25":             aqi_data["pm25"],
        "pm10":             aqi_data["pm10"],
        "o3":               aqi_data["o3"],
        "no2":              aqi_data["no2"],
        "so2":              aqi_data["so2"],
        "co":               aqi_data["co"],

        # ── Weather ──────────────────────────────────────────────────────────
        "temperature":      weather_data["temperature"],
        "humidity":         weather_data["humidity"],
        "pressure":         weather_data["pressure"],
        "wind_speed":       weather_data["wind_speed"],
        "wind_deg":         weather_data["wind_deg"],
        "visibility_km":    weather_data["visibility"],

        # ── Weather dummies ──────────────────────────────────────────────────
        **weather_dummies,

        # ── Time features ────────────────────────────────────────────────────
        "hour":             now.hour,
        "day_of_week":      now.weekday(),      # 0=Monday
        "day_of_month":     now.day,
        "month":            now.month,
        "is_weekend":       1 if now.weekday() >= 5 else 0,

        # ── Cyclical time encoding (prevents 23→0 discontinuity) ─────────────
        "hour_sin":         np.sin(2 * np.pi * now.hour / 24),
        "hour_cos":         np.cos(2 * np.pi * now.hour / 24),
        "month_sin":        np.sin(2 * np.pi * now.month / 12),
        "month_cos":        np.cos(2 * np.pi * now.month / 12),

        # ── Wind direction components ─────────────────────────────────────────
        "wind_u":           weather_data["wind_speed"] * np.cos(np.radians(weather_data["wind_deg"])),
        "wind_v":           weather_data["wind_speed"] * np.sin(np.radians(weather_data["wind_deg"])),
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Connect to Feature Store and upsert row
# ─────────────────────────────────────────────────────────────────────────────

def store_features(df: pd.DataFrame):
    """Connect to Hopsworks and insert the feature row."""
    print("Connecting to Hopsworks…")
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    # Get or create feature group
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        description=f"Hourly AQI + weather features for {config.CITY_NAME}",
    )

    print(f"Inserting {len(df)} row(s) for {config.CITY_NAME}…")
    fg.insert(df, write_options={"wait_for_job": False})
    print("✓ Features stored successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now()}] Starting feature pipeline for {config.CITY_NAME}…")

    print("  Fetching AQI data…")
    aqi_data = fetch_aqicn()
    print(f"  AQI: {aqi_data['aqi']}  PM2.5: {aqi_data['pm25']}  PM10: {aqi_data['pm10']}")

    print("  Fetching weather data…")
    weather_data = fetch_weather()
    print(f"  Temp: {weather_data['temperature']}°C  Humidity: {weather_data['humidity']}%  Wind: {weather_data['wind_speed']} m/s")

    print("  Engineering features…")
    df = engineer_features(aqi_data, weather_data)

    print("  Storing in Feature Store…")
    store_features(df)

    print("Done ✓")


if __name__ == "__main__":
    main()
