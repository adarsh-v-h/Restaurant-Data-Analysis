"""
src/task2/main.py
Simple CLI to run content-based and hybrid recommendations.
Example:
    python src/task2/main.py --cuisines "North Indian,Italian" --city "New Delhi" --k 10
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from data_utils import get_candidate_table, load_processed_table
from model_utils import load_model, load_feature_list, predict_rating_for_df
from recommend_utils import recommend_content_then_hybrid, recommend_by_similarity

ROOT = Path(__file__).resolve().parents[3]
VISUALS = ROOT / "visuals"
VISUALS.mkdir(exist_ok=True, parents=True)

def simple_predict_wrapper(df):
    model = load_model()
    features = load_feature_list()
    return predict_rating_for_df(model, df, features)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cuisines", type=str, default=None, help="Comma separated cuisine names (Primary Cuisine or grouped names).")
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--use_similarity", action="store_true", help="Do similarity-based narrowing using item index or user pref.")
    p.add_argument("--item_idx", type=int, default=None, help="Index of item (in candidate table) to find similar restaurants.")
    return p.parse_args()

def main():
    args = parse_args()
    df = get_candidate_table(prefer_csv=True)
    if df is None:
        print("No processed CSV; falling back to X_test splits.")
        df = get_candidate_table(prefer_csv=False)

    cuisine_list = None
    if args.cuisines:
        cuisine_list = [c.strip() for c in args.cuisines.split(",")]

    if args.use_similarity and args.item_idx is not None:
        print(f"Running similarity-based recommendations for item index {args.item_idx} ...")
        sim = recommend_by_similarity(df, item_idx=args.item_idx, top_k=args.k)
        out = sim.head(args.k)
    else:
        # run content + hybrid (model boosting)
        out = recommend_content_then_hybrid(
            df,
            cuisine_list=cuisine_list,
            city=args.city,
            item_idx=args.item_idx if args.use_similarity else None,
            model_predict_fn=simple_predict_wrapper,
            top_k=args.k
        )

    # Show minimal columns (try to include name if present)
    show_cols = [c for c in ["Restaurant Name", "City", "Primary Cuisine", "Average Cost for two", "Votes", "pred_rating", "hybrid_score"] if c in out.columns]
    print("Top recommendations:")
    print(out[show_cols].head(args.k).to_string(index=False))

    # save CSV for review
    out_file = VISUALS / "task2_recommendations.csv"
    out.to_csv(out_file, index=False)
    print("Saved recommendations to:", out_file)

if __name__ == "__main__":
    main()
