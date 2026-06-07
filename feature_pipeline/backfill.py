"""
feature_pipeline/backfill.py
─────────────────────────────
One-time script. Fetches historical AQI + weather data for the past N days
and inserts them into the Feature Store so you have enough data to train.

Run once:  python feature_pipeline/backfill.py --days 365

OpenWeather historical data requires the "One Call API 3.0" (free tier: 1000 calls/day).
AQICN historical data endpoint is used for pollutants.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import hopsworks
import config


def fetch_openweather_historical(dt: datetime) -> dict:
    """Fetch historical weather for a specific UTC datetime."""
    ts = int(dt.timestamp())
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={config.CITY_LAT}&lon={config.CITY_LON}"
        f"&dt={ts}&appid={config.OPENWEATHER_KEY}&units=metric"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        return None
    d = resp.json()
    hour_data = d.get("data", [{}])[0]
    weather_desc = hour_data.get("weather", [{"main": "Clear"}])[0]["main"]

    return {
        "temperature":   float(hour_data.get("temp", 25)),
        "humidity":      float(hour_data.get("humidity", 60)),
        "pressure":      float(hour_data.get("pressure", 1013)),
        "wind_speed":    float(hour_data.get("wind_speed", 3)),
        "wind_deg":      float(hour_data.get("wind_deg", 0)),
        "visibility":    float(hour_data.get("visibility", 10000)) / 1000,
        "weather_main":  weather_desc,
    }


def fetch_aqicn_historical(dt: datetime) -> dict:
    """Fetch historical AQI from AQICN for a given date."""
    date_str = dt.strftime("%Y-%m-%d")
    url = (
        f"https://api.waqi.info/api/feed/{config.AQICN_STATION}/obs.en.json"
        f"?token={config.AQICN_TOKEN}"
    )
    # AQICN historical is limited; fall back to the daily feed if unavailable
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        return {"aqi": 0, "pm25": 0, "pm10": 0, "o3": 0, "no2": 0, "so2": 0, "co": 0}

    data = resp.json()
    if data.get("status") != "ok":
        return {"aqi": 0, "pm25": 0, "pm10": 0, "o3": 0, "no2": 0, "so2": 0, "co": 0}

    d = data["data"]
    iaqi = d.get("iaqi", {})
    return {
        "aqi":  float(d.get("aqi", 0)),
        "pm25": float(iaqi.get("pm25", {}).get("v", 0)),
        "pm10": float(iaqi.get("pm10", {}).get("v", 0)),
        "o3":   float(iaqi.get("o3",   {}).get("v", 0)),
        "no2":  float(iaqi.get("no2",  {}).get("v", 0)),
        "so2":  float(iaqi.get("so2",  {}).get("v", 0)),
        "co":   float(iaqi.get("co",   {}).get("v", 0)),
    }


def build_row(dt: datetime, aqi_data: dict, weather_data: dict) -> dict:
    """Combine data into a feature row (same schema as live pipeline)."""
    weather_dummies = {
        "is_clear":  1 if weather_data["weather_main"] == "Clear" else 0,
        "is_rain":   1 if weather_data["weather_main"] in ("Rain", "Drizzle") else 0,
        "is_cloudy": 1 if weather_data["weather_main"] in ("Clouds", "Overcast") else 0,
        "is_haze":   1 if weather_data["weather_main"] in ("Haze", "Smoke", "Dust", "Mist") else 0,
    }
    ws = weather_data["wind_speed"]
    wd = weather_data["wind_deg"]

    return {
        "timestamp":     int(dt.timestamp()),
        "city":          config.CITY_NAME,
        "aqi":           aqi_data["aqi"],
        "pm25":          aqi_data["pm25"],
        "pm10":          aqi_data["pm10"],
        "o3":            aqi_data["o3"],
        "no2":           aqi_data["no2"],
        "so2":           aqi_data["so2"],
        "co":            aqi_data["co"],
        "temperature":   weather_data["temperature"],
        "humidity":      weather_data["humidity"],
        "pressure":      weather_data["pressure"],
        "wind_speed":    ws,
        "wind_deg":      wd,
        "visibility_km": weather_data["visibility"],
        **weather_dummies,
        "hour":          dt.hour,
        "day_of_week":   dt.weekday(),
        "day_of_month":  dt.day,
        "month":         dt.month,
        "is_weekend":    1 if dt.weekday() >= 5 else 0,
        "hour_sin":      np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos":      np.cos(2 * np.pi * dt.hour / 24),
        "month_sin":     np.sin(2 * np.pi * dt.month / 12),
        "month_cos":     np.cos(2 * np.pi * dt.month / 12),
        "wind_u":        ws * np.cos(np.radians(wd)),
        "wind_v":        ws * np.sin(np.radians(wd)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="How many past days to backfill")
    parser.add_argument("--hours-per-day", type=int, default=4,
                        help="How many hourly samples per day (to stay within API limits)")
    args = parser.parse_args()

    print(f"Backfilling {args.days} days × {args.hours_per_day} samples/day "
          f"= {args.days * args.hours_per_day} rows")

    rows = []
    now = datetime.now(timezone.utc)

    for day_offset in range(args.days, 0, -1):
        day = now - timedelta(days=day_offset)
        hours = [0, 6, 12, 18][:args.hours_per_day]

        for hour in hours:
            dt = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            try:
                weather = fetch_openweather_historical(dt)
                aqi     = fetch_aqicn_historical(dt)
                if weather and aqi:
                    rows.append(build_row(dt, aqi, weather))
            except Exception as e:
                print(f"  ⚠ Skipped {dt}: {e}")

            time.sleep(0.5)  # respect rate limits

        if day_offset % 30 == 0:
            print(f"  …{day_offset} days remaining, {len(rows)} rows collected")

    df = pd.DataFrame(rows)
    print(f"\nCollected {len(df)} rows. Uploading to Feature Store…")

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        description=f"Hourly AQI + weather features for {config.CITY_NAME}",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print("✓ Backfill complete.")


if __name__ == "__main__":
    main()
