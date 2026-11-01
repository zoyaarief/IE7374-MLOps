"""Compare two Wine classifiers and save a versioned model and training report."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAB_DIR = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Create the fixed split that training and evaluation will both use."""
    X, y = load_wine(return_X_y=True)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_models():
    """Keep scaling inside the pipeline so each CV fold learns its own scaler."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def train_model():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {len(y_train)} | Reserved test samples: {len(y_test)}")

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}

    # Compare models using only the training partition. Reserve X_test and
    # y_test for the separate evaluation script, after selecting the winner.
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            error_score="raise",
        )
        comparison[name] = {
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
        }
        print(f"{name}: CV macro F1 = {comparison[name]['cv_f1_macro_mean']:.4f}")

    best_name = max(comparison, key=lambda name: comparison[name]["cv_f1_macro_mean"])
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metadata = {
        "version": version,
        "dataset": "sklearn_wine",
        "selected_model": best_name,
        "selection_metric": "cv_f1_macro_mean",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": 5,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn_version,
        "comparison": comparison,
    }

    models_dir = LAB_DIR / "models"
    metrics_dir = LAB_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"wine_model_{version}.joblib"
    report_path = metrics_dir / f"training_{version}.json"
    joblib.dump({"model": best_model, "metadata": metadata}, model_path)
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Selected model: {best_name}")
    print(f"Model saved: {model_path}")
    print(f"Training report saved: {report_path}")
    return model_path, report_path


if __name__ == "__main__":
    train_model()
