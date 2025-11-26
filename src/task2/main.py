from data_utils import load_splits
from recommend_utils import load_model, get_similar_items, predict_rating_for_index
from meta_utils import load_raw_metadata, get_metadata_from_original_index


def run():
    print("\n=== Content-Based Restaurant Recommender ===\n")

    X_train, X_test, y_train, y_test = load_splits()
    model = load_model()
    raw_meta = load_raw_metadata()

    print(f"X_test shape: {X_test.shape}")

    idx = int(input(f"Pick an index (0 to {len(X_test)-1}): "))

    # Extract ORIGINAL row index
    original_idx = X_test.index[idx]

    print("\n>>> SELECTED RESTAURANT DETAILS")
    info = get_metadata_from_original_index(raw_meta, original_idx)
    for k, v in info.items():
        print(f"{k}: {v}")

    pred = predict_rating_for_index(model, X_test, idx)
    print(f"Predicted Rating: {pred:.3f}")

    print("\n>>> Finding similar restaurants...\n")

    sims = get_similar_items(X_test, idx, top_k=5)

    for i, row in sims.iterrows():
        test_idx = int(row["item_index"])
        sim = row["similarity"]

        # ORIGINAL index for similar item
        orig_sim_idx = X_test.index[test_idx]

        sim_info = get_metadata_from_original_index(raw_meta, orig_sim_idx)
        sim_info["Similarity"] = round(sim, 4)

        print(f"#{i+1}:")
        for k, v in sim_info.items():
            print(f"  {k}: {v}")
        print()

if __name__ == "__main__":
    run()