from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = ROOT / "data" / "raw" / "Dataset.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "model_data.csv"


def load_raw_metadata():
    """
    Load raw dataset with restaurant info.
    """
    return pd.read_csv(RAW_DATA)


def load_processed_data():
    """
    Load the full processed dataset BEFORE splitting.
    """
    return pd.read_csv(PROCESSED_DATA)


def get_metadata_from_original_index(raw_df, original_idx):
    """
    Given the ORIGINAL dataset index (not X_test index!),
    return human-readable metadata.
    """
    row = raw_df.iloc[original_idx]

    return {
        "Restaurant Name": row.get("Restaurant Name", "Unknown"),
        "City": row.get("City", "Unknown"),
        "Cuisines": row.get("Cuisines", "Unknown")
    }
