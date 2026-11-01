"""Evaluate a saved Wine model on its reserved test split."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[1]


def find_latest_model():
    """UTC timestamps in the filenames sort from oldest to newest."""
    candidates = sorted((LAB_DIR / "models").glob("wine_model_*.joblib"))
    if not candidates:
        raise FileNotFoundError("No saved Wine model found. Run train_model.py first.")
    return candidates[-1]


def evaluate_model(model_path=None):
    model_path = Path(model_path) if model_path is not None else find_latest_model()
    bundle = joblib.load(model_path)
    model = bundle["model"]
    metadata = bundle["metadata"]

    if metadata["dataset"] != "sklearn_wine":
        raise ValueError("This evaluator supports the sklearn Wine dataset only.")

    dataset = load_wine()
    # Recreate this model's original split using its saved settings. No fitting
    # or model selection happens here; the reserved test set is for evaluation.
    _, X_test, _, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=metadata["test_size"],
        random_state=metadata["random_state"],
        stratify=dataset.target,
    )
    if len(y_test) != metadata["test_samples"]:
        raise ValueError("The recreated test split does not match the saved metadata.")

    predictions = model.predict(X_test)
    labels = list(range(len(dataset.target_names)))
    class_names = dataset.target_names.tolist()
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    results = {
        "version": metadata["version"],
        "dataset": metadata["dataset"],
        "selected_model": metadata["selected_model"],
        "model_file": model_path.name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": metadata["random_state"],
        "test_size": metadata["test_size"],
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(
            f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "class_labels": labels,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    metrics_dir = LAB_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / f"evaluation_{metadata['version']}.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Evaluating: {model_path.name}")
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test macro F1: {results['f1_macro']:.4f}")
    print(f"Class order: {', '.join(class_names)}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)
    print(f"Evaluation report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a saved model. Defaults to the latest timestamped Wine model.",
    )
    args = parser.parse_args()
    evaluate_model(args.model)
