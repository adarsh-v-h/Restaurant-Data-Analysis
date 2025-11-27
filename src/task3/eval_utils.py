from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
VIS_DIR = ROOT / "visuals"
VIS_DIR.mkdir(exist_ok=True)

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1_weighted": float(f1)}

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, zero_division=0))

def save_confusion_matrix(y_true, y_pred, labels, out_name="cuisine_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm_df, annot=False, fmt="d", cmap="viridis")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    out = VIS_DIR / out_name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def save_feature_importances(clf, feature_names, out_name="cuisine_feature_importances.png", top_k=30):
    # sklearn RandomForest feature_importances_
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    top_feats = [feature_names[i] for i in idx]
    vals = importances[idx]

    plt.figure(figsize=(10, max(4, top_k * 0.25)))
    sns.barplot(x=vals, y=top_feats)
    plt.title("Top Feature Importances")
    out = VIS_DIR / out_name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out
