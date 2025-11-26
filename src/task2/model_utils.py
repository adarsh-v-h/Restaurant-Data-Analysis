"""
src/task2/model_utils.py
Load the saved RF model & feature list and provide a helper to predict a dataframe.
"""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"

def load_model(model_name="final_rf_tuned.joblib"):
    p = MODELS_DIR / model_name
    alt = MODELS_DIR / "final_rf_turned.joblib"
    if not p.exists():
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Model not found at {MODELS_DIR}.")
    model = joblib.load(p)
    return model

def load_feature_list():
    p = MODELS_DIR / "feature_list.joblib"
    if not p.exists():
        raise FileNotFoundError("feature_list.joblib missing in models/")
    return joblib.load(p)

def predict_rating_for_df(model, df, feature_list):
    """
    Predict rating for df using feature_list order. Returns a copy of df with 'pred_rating'.
    """
    X = df.copy()
    missing = [c for c in feature_list if c not in X.columns]
    if missing:
        # Not fatal: we will try to select intersection and warn
        raise RuntimeError(f"Missing features required for prediction: {missing[:10]}")
    X_sel = X[feature_list]
    preds = model.predict(X_sel)
    out = df.copy()
    out["pred_rating"] = preds
    return out
