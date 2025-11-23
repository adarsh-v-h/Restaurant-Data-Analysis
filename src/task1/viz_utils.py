"""src/task1/viz_utils.py
Utility functions for saving model evaluation plots (pred vs actual, residuals, SHAP screenshot saving hints).
"""
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

VIS_DIR = Path(__file__).resolve().parents[3] / "visuals"
VIS_DIR.mkdir(parents=True, exist_ok=True)

def save_pred_vs_actual(y_true, y_pred, out_name="pred_vs_actual_task1.png"):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], "--", color="red")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Predicted vs Actual")
    out = VIS_DIR / out_name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def save_residuals(y_true, y_pred, out_name="residuals_task1.png"):
    res = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8,5))
    plt.hist(res, bins=30, density=False)
    plt.title("Residuals (actual - predicted)")
    plt.xlabel("Residual value")
    plt.ylabel("Count")
    out = VIS_DIR / out_name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
