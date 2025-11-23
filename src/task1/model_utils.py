"""src/task1/model_utils.py
Load saved model(s) and provide inference helpers.
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ROOT / "models"

def load_model(model_name: str = "final_rf_tuned.joblib", models_dir: str = None):
    mdir = Path(models_dir) if models_dir else DEFAULT_MODELS
    model_path = mdir / model_name
    if not model_path.exists():
        # try common alternate names
        alt = mdir / "final_rf_turned.joblib"
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    return model

def load_feature_list(models_dir: str = None):
    mdir = Path(models_dir) if models_dir else DEFAULT_MODELS
    fl = mdir / "feature_list.joblib"
    if not fl.exists():
        raise FileNotFoundError(f"feature_list.joblib missing at {fl}")
    return joblib.load(fl)

def predict_batch(model, df: pd.DataFrame, feature_list: list):
    # Ensure df has all feature_list columns in order
    X = df.copy()
    missing = [c for c in feature_list if c not in X.columns]
    if missing:
        raise RuntimeError(f"Missing features for prediction: {missing[:10]}")
    X = X[feature_list]
    preds = model.predict(X)
    df_out = df.copy()
    df_out["pred_rating"] = preds
    return df_out
