"""
inference_pipeline/explain.py
──────────────────────────────
Generates SHAP feature importance values for model explainability.
Called by the dashboard to show which features drive AQI predictions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import shap
import joblib

from inference_pipeline.predict import load_artifacts, load_recent_features
import config


def get_shap_values(n_samples: int = 100) -> pd.DataFrame:
    """
    Returns a DataFrame of mean absolute SHAP values per feature,
    sorted descending (most important first).
    """
    model, scaler, feature_cols, meta = load_artifacts()
    is_lstm = False
    best = meta.get("best_model", "").lower()

    # SHAP works best with tree models; fall back to linear explainer for others
    df = load_recent_features(n_rows=n_samples + 30)

    # Add minimal lag features so we have something to explain
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"aqi_roll_mean_{window}h"] = df["aqi"].shift(1).rolling(window).mean()
        df[f"aqi_roll_std_{window}h"]  = df["aqi"].shift(1).rolling(window).std()
    df["aqi_change_1h"]  = df["aqi"].diff(1)
    df["aqi_change_3h"]  = df["aqi"].diff(3)
    df["aqi_change_24h"] = df["aqi"].diff(24)
    df = df.dropna().tail(n_samples)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    X_scaled = scaler.transform(X)

    if "random_forest" in best or "gradient_boosting" in best:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    elif "lstm" in best:
        # Use KernelExplainer for deep models (slower but works)
        background  = shap.kmeans(X_scaled, 10)
        explainer   = shap.KernelExplainer(
            lambda x: model.predict(x.reshape(x.shape[0], 1, x.shape[1]), verbose=0).flatten(),
            background,
        )
        shap_values = explainer.shap_values(X_scaled[:20], nsamples=50)
    else:
        # Ridge — linear explainer
        explainer   = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    result = pd.DataFrame({
        "feature":    available,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return result


if __name__ == "__main__":
    df = get_shap_values()
    print(df.head(15).to_string(index=False))
