"""src/task1/data_utils.py
Helpers to load processed data and splits.
"""
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
DEFAULT_MODELS = ROOT / "models"
DEFAULT_DATA_PROCESSED = ROOT / "data" / "processed" / "model_data.csv"

def load_processed_df(path: str = None) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_DATA_PROCESSED
    if p.exists():
        return pd.read_csv(p)
    # fallback: try to re-create from X_test if no CSV
    raise FileNotFoundError(f"Processed CSV not found at {p}. Please provide `data/processed/model_data.csv`.")

def load_splits(models_dir: str = None):
    m = Path(models_dir) if models_dir else DEFAULT_MODELS
    X_train = joblib.load(m / "X_train.joblib") if (m / "X_train.joblib").exists() else None
    X_test  = joblib.load(m / "X_test.joblib")  if (m / "X_test.joblib").exists()  else None
    y_train = joblib.load(m / "Y_train.joblib") if (m / "Y_train.joblib").exists() else None
    y_test  = joblib.load(m / "Y_test.joblib")  if (m / "Y_test.joblib").exists()  else None
    return X_train, X_test, y_train, y_test

def ensure_visuals_dir(root: str = None):
    root_dir = Path(root) if root else Path(__file__).resolve().parents[3] / "visuals"
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir
