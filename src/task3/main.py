from pathlib import Path
from data_utils import load_processed, build_X_y_all_classes, train_test_split_stratified
from model_utils import train_rf_classifier, save_model, save_artifacts
from eval_utils import classification_metrics, print_classification_report, save_confusion_matrix, save_feature_importances

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def run(end_to_end=True):
    print("Loading processed dataset...")
    df = load_processed()
    print("Building X, y for cuisine classification (all classes)...")
    X, y = build_X_y_all_classes(df)
    print("X shape:", X.shape, "y shape:", y.shape)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2, random_state=42)
    print("Train/Test sizes:", X_train.shape, X_test.shape)

    # Train model
    print("Training RandomForest classifier (baseline)...")
    clf = train_rf_classifier(X_train, y_train)
    print("Training complete.")

    # Save model + artifacts
    save_path = save_model(clf, name="cuisine_rf.joblib")
    print("Saved model to:", save_path)

    # Save feature list and label mapping
    feature_list = list(X.columns)
    save_artifacts(feature_list, "cuisine_feature_list.joblib")
    print("Saved feature list (cuisine_feature_list.joblib)")

    # Evaluate on test
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)

    metrics = classification_metrics(y_test, y_pred)
    print("Metrics:", metrics)
    print("\nClassification report:\n")
    print_classification_report(y_test, y_pred)

    # Save confusion matrix (labels ordered)
    labels = np.unique(y)  # all labels
    cm_path = save_confusion_matrix(y_test, y_pred, labels=labels)
    print("Saved confusion matrix to:", cm_path)

    fi_path = save_feature_importances(clf, feature_list, top_k=30)
    print("Saved feature importances to:", fi_path)

    # Save predictions for later inspection
    preds_path = MODELS_DIR / "cuisine_preds.joblib"
    joblib.dump({"y_test": y_test, "y_pred": y_pred}, preds_path)
    print("Saved predictions to:", preds_path)

    print("\nTask 3 complete. Inspect metrics and visuals/ for outputs.")

if __name__ == "__main__":
    run()
