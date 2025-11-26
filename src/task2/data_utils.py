from pathlib import Path
import joblib
import pandas as pd

# Go from: task2 -> src -> project root
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

def load_splits():
    """
    ONLY loads joblib files from models/. Nothing else.
    """
    X_train = joblib.load(MODELS_DIR / "X_train.joblib")
    X_test  = joblib.load(MODELS_DIR / "X_test.joblib")
    y_train = joblib.load(MODELS_DIR / "Y_train.joblib")
    y_test  = joblib.load(MODELS_DIR / "Y_test.joblib")

    return X_train, X_test, y_train, y_test


def get_candidate_table():
    """
    Candidates are always X_test. Nothing fancy.
    """
    _, X_test, _, _ = load_splits()

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    return X_test
