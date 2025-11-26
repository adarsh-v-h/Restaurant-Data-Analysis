from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def load_model(name="final_rf_tuned.joblib"):
    return joblib.load(MODELS_DIR / name)


def _prepare_similarity_matrix(X):
    """
    Build a clean feature matrix for similarity:
      - Use meaningful differentiating features
      - Scale numeric fields
    """

    numeric_cols = [
        "Average Cost for two",
        "Price range",
        "Has Online delivery",
        "Has Table booking",
        "Votes",              # added
        "Country Code"        # added
    ]

    cuisine_cols = [c for c in X.columns if c.startswith("Cuisine_Grouped_")]

    use_cols = numeric_cols + cuisine_cols

    X_sub = X[use_cols].copy()

    scaler = StandardScaler()
    X_sub[numeric_cols] = scaler.fit_transform(X_sub[numeric_cols])

    return X_sub, use_cols



def get_similar_items(X, index, top_k=5):
    """
    Returns a sorted list of most similar restaurants for a given index.
    """

    X_processed, feature_list = _prepare_similarity_matrix(X)

    # Compute cosine similarity
    sims = cosine_similarity(X_processed)

    # Row for selected item
    row = sims[index]

    # Sort indices by similarity (descending), skip itself
    similar_indices = np.argsort(row)[::-1]
    similar_indices = [i for i in similar_indices if i != index][:top_k]

    result = pd.DataFrame({
        "item_index": similar_indices,
        "similarity": row[similar_indices]
    })

    return result

def load_feature_list():
    fl_path = MODELS_DIR / "feature_list.joblib"
    return joblib.load(fl_path)

def predict_rating_for_index(model, X_test, index):
    """
    Predict using a DataFrame with proper feature names and order.
    """
    feature_list = load_feature_list()
    # select a single-row DataFrame in the SAME order used for training
    row = X_test.iloc[[index]][feature_list]   # double brackets -> DataFrame
    return model.predict(row)[0]