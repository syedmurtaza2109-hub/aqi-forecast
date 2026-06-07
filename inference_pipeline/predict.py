"""
inference_pipeline/predict.py
──────────────────────────────
Called by the dashboard to produce 3-day (72-hour) AQI forecasts.

Logic:
  1. Load the latest model + scaler from Hopsworks (or local cache)
  2. Fetch the most recent feature rows from the Feature Store
  3. Iteratively predict next hour, feed prediction back as a lag feature
  4. Return a DataFrame with timestamp + predicted AQI for 72 hours
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

import hopsworks
import config


# ─────────────────────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────────────────────

_cache = {}   # simple in-process cache so dashboard doesn't re-download every call

def load_artifacts(force_reload=False):
    if _cache and not force_reload:
        return _cache["model"], _cache["scaler"], _cache["feature_cols"], _cache["meta"]

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )
    mr = project.get_model_registry()
    hw_model = mr.get_best_model(
        name=config.MODEL_NAME,
        metric="rmse",
        direction="min",
    )
    model_dir = hw_model.download()

    scaler       = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))

    with open(os.path.join(model_dir, "metrics.json")) as f:
        meta = json.load(f)

    best_name = meta["best_model"].lower().replace(" ", "_")
    if best_name == "tensorflow_lstm":
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(model_dir, "lstm.keras"))
        is_lstm = True
    else:
        key_map = {"ridge_regression": "ridge", "random_forest": "random_forest",
                   "gradient_boosting": "gradient_boosting"}
        fname = key_map.get(best_name, "random_forest") + ".pkl"
        model = joblib.load(os.path.join(model_dir, fname))
        is_lstm = False

    _cache.update({"model": model, "scaler": scaler,
                   "feature_cols": feature_cols, "meta": meta, "is_lstm": is_lstm})
    return model, scaler, feature_cols, meta


# ─────────────────────────────────────────────────────────────────────────────
# Load recent rows from Feature Store
# ─────────────────────────────────────────────────────────────────────────────

def load_recent_features(n_rows=48) -> pd.DataFrame:
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
    )
    df = fg.read()
    df = df.sort_values("timestamp").tail(n_rows).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Iterative forecast
# ─────────────────────────────────────────────────────────────────────────────

def _make_feature_row(history: list[float], last_row: pd.Series, future_dt: datetime) -> np.ndarray:
    """
    Build one feature vector for a future timestep.
    `history` = list of predicted AQI values so far (most recent last).
    `last_row` = last known real row (for weather features, which we hold constant).
    """
    def safe_get(lst, idx, default=0.0):
        try: return lst[idx]
        except IndexError: return default

    lags = {
        "aqi_lag_1h":  safe_get(history, -1),
        "aqi_lag_3h":  safe_get(history, -3),
        "aqi_lag_6h":  safe_get(history, -6),
        "aqi_lag_12h": safe_get(history, -12),
        "aqi_lag_24h": safe_get(history, -24),
    }

    def roll_mean(n):
        window = history[-n:] if len(history) >= n else history
        return float(np.mean(window)) if window else 0.0

    def roll_std(n):
        window = history[-n:] if len(history) >= n else history
        return float(np.std(window)) if len(window) > 1 else 0.0

    changes = {
        "aqi_change_1h":  (history[-1] - history[-2]) if len(history) >= 2 else 0.0,
        "aqi_change_3h":  (history[-1] - history[-4]) if len(history) >= 4 else 0.0,
        "aqi_change_24h": (history[-1] - history[-25]) if len(history) >= 25 else 0.0,
    }

    row = {
        "pm25": last_row.get("pm25", 0),
        "pm10": last_row.get("pm10", 0),
        "o3":   last_row.get("o3", 0),
        "no2":  last_row.get("no2", 0),
        "so2":  last_row.get("so2", 0),
        "co":   last_row.get("co", 0),
        "temperature":   last_row.get("temperature", 25),
        "humidity":      last_row.get("humidity", 60),
        "pressure":      last_row.get("pressure", 1013),
        "wind_speed":    last_row.get("wind_speed", 3),
        "wind_u":        last_row.get("wind_u", 0),
        "wind_v":        last_row.get("wind_v", 0),
        "visibility_km": last_row.get("visibility_km", 10),
        "is_clear":      last_row.get("is_clear", 1),
        "is_rain":       last_row.get("is_rain", 0),
        "is_cloudy":     last_row.get("is_cloudy", 0),
        "is_haze":       last_row.get("is_haze", 0),
        "hour_sin":      np.sin(2 * np.pi * future_dt.hour / 24),
        "hour_cos":      np.cos(2 * np.pi * future_dt.hour / 24),
        "month_sin":     np.sin(2 * np.pi * future_dt.month / 12),
        "month_cos":     np.cos(2 * np.pi * future_dt.month / 12),
        "day_of_week":   future_dt.weekday(),
        "is_weekend":    1 if future_dt.weekday() >= 5 else 0,
        **lags,
        "aqi_roll_mean_3h":  roll_mean(3),
        "aqi_roll_mean_6h":  roll_mean(6),
        "aqi_roll_mean_12h": roll_mean(12),
        "aqi_roll_mean_24h": roll_mean(24),
        "aqi_roll_std_3h":   roll_std(3),
        "aqi_roll_std_6h":   roll_std(6),
        "aqi_roll_std_12h":  roll_std(12),
        "aqi_roll_std_24h":  roll_std(24),
        **changes,
    }
    return row


def generate_forecast(hours: int = 72) -> pd.DataFrame:
    """Return DataFrame with columns: timestamp, datetime, aqi_predicted, level, color."""
    model, scaler, feature_cols, meta = load_artifacts()
    is_lstm = _cache.get("is_lstm", False)

    history_df = load_recent_features(n_rows=48)
    history    = list(history_df["aqi"].values)
    last_row   = history_df.iloc[-1]

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    predictions = []

    for h in range(1, hours + 1):
        future_dt = now + timedelta(hours=h)
        feat_dict = _make_feature_row(history, last_row, future_dt)

        # Align to training feature columns
        vec = np.array([feat_dict.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
        vec_scaled = scaler.transform(vec)

        if is_lstm:
            vec_scaled = vec_scaled.reshape(1, 1, -1)
            pred = float(model.predict(vec_scaled, verbose=0)[0][0])
        else:
            pred = float(model.predict(vec_scaled)[0])

        pred = max(0, round(pred, 1))   # AQI can't be negative
        history.append(pred)

        level, color = config.get_aqi_level(pred)
        predictions.append({
            "timestamp":     int(future_dt.timestamp()),
            "datetime":      future_dt,
            "aqi_predicted": pred,
            "level":         level,
            "color":         color,
            "hour":          future_dt.hour,
            "date":          future_dt.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(predictions)


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 72-hour AQI forecast…")
    df = generate_forecast(72)
    print(df[["datetime", "aqi_predicted", "level"]].to_string(index=False))
