"""
dashboard/api.py
─────────────────
Flask REST API backend.
The Streamlit frontend calls these endpoints.

Endpoints:
  GET /api/forecast          → 72-hour AQI predictions
  GET /api/current           → latest real AQI reading
  GET /api/shap              → SHAP feature importance
  GET /api/history?days=7    → historical AQI from Feature Store
  GET /health                → liveness check
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, jsonify, request
from datetime import datetime, timezone, timedelta
import pandas as pd

from inference_pipeline.predict import generate_forecast, load_recent_features
from inference_pipeline.explain import get_shap_values
from feature_pipeline.fetch_and_store import fetch_aqicn, fetch_weather
import config

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "city": config.CITY_NAME,
                    "time": datetime.utcnow().isoformat()})


@app.route("/api/current")
def current():
    try:
        aqi_data     = fetch_aqicn()
        weather_data = fetch_weather()
        level, color = config.get_aqi_level(aqi_data["aqi"])
        return jsonify({
            "aqi":         aqi_data["aqi"],
            "level":       level,
            "color":       color,
            "pm25":        aqi_data["pm25"],
            "pm10":        aqi_data["pm10"],
            "o3":          aqi_data["o3"],
            "no2":         aqi_data["no2"],
            "temperature": weather_data["temperature"],
            "humidity":    weather_data["humidity"],
            "wind_speed":  weather_data["wind_speed"],
            "city":        config.CITY_NAME,
            "timestamp":   datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast")
def forecast():
    try:
        hours = int(request.args.get("hours", 72))
        df    = generate_forecast(hours)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def history():
    try:
        days = int(request.args.get("days", 7))
        n    = days * 24
        df   = load_recent_features(n_rows=n)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        cols = ["datetime", "aqi", "pm25", "pm10", "temperature", "humidity"]
        available = [c for c in cols if c in df.columns]
        return jsonify(df[available].to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/shap")
def shap_endpoint():
    try:
        df = get_shap_values(n_samples=50)
        return jsonify(df.head(15).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
