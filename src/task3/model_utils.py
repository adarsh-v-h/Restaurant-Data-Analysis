from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_rf_classifier(X_train, y_train, n_jobs= -1, random_state=42):
    clf = RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def save_model(clf, name="cuisine_rf.joblib"):
    path = MODELS_DIR / name
    joblib.dump(clf, path)
    return path

def save_artifacts(obj, name):
    path = MODELS_DIR / name
    joblib.dump(obj, path)
    return path

def load_model(name="cuisine_rf.joblib"):
    path = MODELS_DIR / name
    return joblib.load(path)
