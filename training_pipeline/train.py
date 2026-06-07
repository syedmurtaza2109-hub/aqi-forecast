"""
training_pipeline/train.py
───────────────────────────
Runs daily via GitHub Actions.

Steps:
  1. Pull features from Hopsworks Feature Store
  2. Build lag / rolling-window features
  3. Train Random Forest, Ridge Regression, and TensorFlow LSTM
  4. Evaluate each model (RMSE, MAE, R²)
  5. Save the best model + scaler to Hopsworks Model Registry
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import hopsworks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data from Feature Store
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    print("Connecting to Hopsworks…")
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
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df)} rows from Feature Store.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering: lag & rolling features
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features based on past AQI values."""
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"aqi_roll_mean_{window}h"] = df["aqi"].shift(1).rolling(window).mean()
        df[f"aqi_roll_std_{window}h"]  = df["aqi"].shift(1).rolling(window).std()
    df["aqi_change_1h"]  = df["aqi"].diff(1)
    df["aqi_change_3h"]  = df["aqi"].diff(3)
    df["aqi_change_24h"] = df["aqi"].diff(24)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "wind_u", "wind_v", "visibility_km",
    "is_clear", "is_rain", "is_cloudy", "is_haze",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_3h", "aqi_roll_mean_6h", "aqi_roll_mean_12h", "aqi_roll_mean_24h",
    "aqi_roll_std_3h",  "aqi_roll_std_6h",  "aqi_roll_std_12h",  "aqi_roll_std_24h",
    "aqi_change_1h", "aqi_change_3h", "aqi_change_24h",
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:30s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def train_all(df: pd.DataFrame):
    df = add_lag_features(df)

    # Filter to available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[config.TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False  # time-series: no shuffle
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = []
    trained_models = {}

    # ── Ridge Regression ────────────────────────────────────────────────────
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    results.append(evaluate(y_test, ridge.predict(X_test_s), "Ridge Regression"))
    trained_models["ridge"] = ridge

    # ── Random Forest ───────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    results.append(evaluate(y_test, rf.predict(X_test_s), "Random Forest"))
    trained_models["random_forest"] = rf

    # ── Gradient Boosting ───────────────────────────────────────────────────
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X_train_s, y_train)
    results.append(evaluate(y_test, gb.predict(X_test_s), "Gradient Boosting"))
    trained_models["gradient_boosting"] = gb

    # ── TensorFlow LSTM ─────────────────────────────────────────────────────
    X_train_lstm = X_train_s.reshape(X_train_s.shape[0], 1, X_train_s.shape[1])
    X_test_lstm  = X_test_s.reshape(X_test_s.shape[0],  1, X_test_s.shape[1])
    lstm = build_lstm((1, X_train_s.shape[1]))
    es = EarlyStopping(patience=5, restore_best_weights=True)
    lstm.fit(X_train_lstm, y_train,
             validation_split=0.1, epochs=50, batch_size=32,
             callbacks=[es], verbose=0)
    preds_lstm = lstm.predict(X_test_lstm, verbose=0).flatten()
    results.append(evaluate(y_test, preds_lstm, "TensorFlow LSTM"))
    trained_models["lstm"] = lstm

    return results, trained_models, scaler, available


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save to Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def save_to_registry(results, trained_models, scaler, feature_cols):
    # Pick best model by RMSE
    best = min(results, key=lambda r: r["rmse"])
    best_name = best["model"].lower().replace(" ", "_")
    print(f"\n  Best model: {best['model']} (RMSE={best['rmse']:.2f})")

    # Map display name back to key
    key_map = {
        "ridge_regression":   "ridge",
        "random_forest":      "random_forest",
        "gradient_boosting":  "gradient_boosting",
        "tensorflow_lstm":    "lstm",
    }
    model_key = key_map.get(best_name, "random_forest")
    best_model = trained_models[model_key]

    # Save artefacts locally
    os.makedirs("models", exist_ok=True)
    if model_key == "lstm":
        best_model.save("models/best_model.keras")
    else:
        joblib.dump(best_model, "models/best_model.pkl")

    joblib.dump(scaler,       "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    # Save all model objects for inference ensemble
    for k, m in trained_models.items():
        if k == "lstm":
            m.save(f"models/{k}.keras")
        else:
            joblib.dump(m, f"models/{k}.pkl")

    metrics = {r["model"]: {"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]}
               for r in results}

    with open("models/metrics.json", "w") as f:
        json.dump({
            "best_model": best["model"],
            "trained_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "feature_cols": feature_cols,
        }, f, indent=2)

    # Upload to Hopsworks Model Registry
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )
    mr = project.get_model_registry()

    hw_model = mr.sklearn.create_model(
        name=config.MODEL_NAME,
        metrics={"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]},
        description=f"Best model: {best['model']} trained on {config.CITY_NAME} AQI data",
    )
    hw_model.save("models/")
    print("✓ Model saved to Hopsworks Model Registry.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now()}] Starting training pipeline…\n")
    df = load_features()

    print("Training models…")
    results, trained_models, scaler, feature_cols = train_all(df)

    print("\nAll results:")
    for r in sorted(results, key=lambda x: x["rmse"]):
        print(f"  {r['model']:30s}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}")

    save_to_registry(results, trained_models, scaler, feature_cols)
    print("\nTraining pipeline complete ✓")


if __name__ == "__main__":
    main()
