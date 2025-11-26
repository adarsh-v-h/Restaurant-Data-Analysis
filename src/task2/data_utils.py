"""
src/task2/data_utils.py
Utilities to load the processed table or fall back to X_test splits.
"""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # project root

def load_processed_table(csv_path=None):
    """
    Try to load data/processed/model_data.csv if present. Otherwise return None.
    """
    p = Path(csv_path) if csv_path else ROOT / "data" / "processed" / "model_data.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

def load_splits_from_models(models_dir=None):
    """
    Load X_train/X_test/Y_train/Y_test from models/ if available.
    Returns tuple of (X_train, X_test, y_train, y_test) where any missing is None.
    """
    mdir = Path(models_dir) if models_dir else ROOT / "models"
    def safe_load(fname):
        p = mdir / fname
        return joblib.load(p) if p.exists() else None
    X_train = safe_load("X_train.joblib")
    X_test  = safe_load("X_test.joblib")
    y_train = safe_load("Y_train.joblib")
    y_test  = safe_load("Y_test.joblib")
    return X_train, X_test, y_train, y_test

def get_candidate_table(prefer_csv=True):
    """
    Return a dataframe of candidates we can recommend from.
    If processed CSV exists and prefer_csv=True => use that (contains meta like city, cuisine strings).
    Otherwise, we fall back to X_test (features only).
    """
    if prefer_csv:
        df = load_processed_table()
        if df is not None:
            return df
    # fallback
    _, X_test, _, _ = load_splits_from_models()
    if X_test is not None:
        # X_test is likely a DataFrame containing engineered features
        return X_test.reset_index(drop=True)
    raise FileNotFoundError("No candidate data found. Place processed CSV at data/processed/model_data.csv or X_test.joblib in models/")
