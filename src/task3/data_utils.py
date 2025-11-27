from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT / "data" / "processed" / "model_data.csv"

def load_processed(path: str = None):
    p = Path(path) if path else PROCESSED_CSV
    df = pd.read_csv(p)
    return df

def build_X_y_all_classes(df):
    cuisine_cols = [c for c in df.columns if c.startswith("Cuisine_Grouped_")]
    if len(cuisine_cols) == 0:
        raise RuntimeError("No cuisine_grouped_ columns found in processed CSV.")

    # Build y: label string without prefix
    y_series = df[cuisine_cols].idxmax(axis=1).str.replace("Cuisine_Grouped_", "", regex=False)

    # Build X: drop cuisine cols and drop aggregate rating (avoid leakage)
    X = df.drop(columns=cuisine_cols + ["Aggregate rating"] if "Aggregate rating" in df.columns else cuisine_cols)

    return X, y_series

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
