import argparse
from pathlib import Path
# import pandas as pd

from data_utils import load_splits, ensure_visuals_dir, load_processed_df
from model_utils import load_model, load_feature_list, predict_batch
from viz_utils import save_pred_vs_actual, save_residuals, compute_metrics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT / "models"

def quick_eval(model_name="final_rf_tuned.joblib"):
    model = load_model(model_name, models_dir=DEFAULT_MODELS_DIR)
    feature_list = load_feature_list(models_dir=DEFAULT_MODELS_DIR)
    X_train, X_test, y_train, y_test = load_splits(models_dir=DEFAULT_MODELS_DIR)
    if X_test is None or y_test is None:
        raise RuntimeError("X_test/Y_test not found in models/ — put the splits there or create data/processed/model_data.csv")
    print("Loaded model and splits. Running batch predictions...")
    df_test = X_test.copy()
    df_test["actual_rating"] = y_test
    df_out = predict_batch(model, df_test, feature_list)
    metrics = compute_metrics(df_out["actual_rating"], df_out["pred_rating"])
    print("Metrics:", metrics)
    pv = save_pred_vs_actual(df_out["actual_rating"], df_out["pred_rating"], out_name="pred_vs_actual_task1.png")
    rv = save_residuals(df_out["actual_rating"], df_out["pred_rating"], out_name="residuals_task1.png")
    print("Saved visuals:", pv, rv)
    # Save metrics to models folder
    import json
    Path(DEFAULT_MODELS_DIR).mkdir(exist_ok=True)
    with open(DEFAULT_MODELS_DIR / "task1_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Wrote metrics to models/task1_metrics.json")

def single_predict(idx: int, model_name="final_rf_tuned.joblib"):
    model = load_model(model_name, models_dir=DEFAULT_MODELS_DIR)
    feature_list = load_feature_list(models_dir=DEFAULT_MODELS_DIR)
    X_train, X_test, y_train, y_test = load_splits(models_dir=DEFAULT_MODELS_DIR)
    if X_test is None:
        raise RuntimeError("X_test missing.")
    df_row = X_test.iloc[[idx]].copy()
    pred = model.predict(df_row[feature_list])[0]
    print(f"Index {idx} predicted rating: {pred:.4f}")
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quick_eval", choices=["quick_eval", "single_predict"])
    parser.add_argument("--idx", type=int, default=23)
    parser.add_argument("--model", type=str, default="final_rf_tuned.joblib")
    args = parser.parse_args()
    ensure_visuals_dir()
    if args.mode == "quick_eval":
        quick_eval(args.model)
    elif args.mode == "single_predict":
        single_predict(args.idx, args.model)
