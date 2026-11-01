"""Check data separation, saved models, evaluation, and model discovery.

Run from the repository root: python -m pytest lab_03/test -v
"""

import json
from collections import Counter

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from lab_03.src import evaluate_model as evaluation
from lab_03.src import train_model as training


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Train once in temporary storage, independent of the user's saved models."""
    output_dir = tmp_path_factory.mktemp("wine_training")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, "LAB_DIR", output_dir)
        model_path, report_path = training.train_model()
    return model_path, report_path


def test_data_split_is_reproducible_and_disjoint():
    first_split = training.load_data()
    second_split = training.load_data()
    for first, second in zip(first_split, second_split):
        assert first.tolist() == second.tolist()

    X_train, X_test, y_train, y_test = first_split
    assert X_train.shape == (142, 13)
    assert X_test.shape == (36, 13)
    assert set(map(tuple, X_train)).isdisjoint(set(map(tuple, X_test)))
    assert set(y_train) == set(y_test) == {0, 1, 2}


def test_logistic_scaler_learns_only_from_training_data():
    X_train, _, y_train, _ = training.load_data()
    pipeline = training.build_models()["logistic_regression"]
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["standardscaler"]
    assert int(scaler.n_samples_seen_) == len(y_train)
    assert scaler.mean_.tolist() == pytest.approx(X_train.mean(axis=0).tolist())


def test_training_saves_reloadable_model_and_matching_report(trained_run):
    model_path, report_path = trained_run
    bundle = joblib.load(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert bundle["metadata"] == report
    assert model_path.name == f"wine_model_{report['version']}.joblib"
    assert report_path.name == f"training_{report['version']}.json"

    X_train, _, _, _ = training.load_data()
    predictions = bundle["model"].predict(X_train[:5])
    assert predictions.shape == (5,)
    assert set(predictions).issubset({0, 1, 2})
    assert bundle["model"].n_features_in_ == 13


def test_evaluation_matches_predictions_without_refitting(
    trained_run, tmp_path, monkeypatch
):
    model_path, _ = trained_run
    bundle = joblib.load(model_path)
    _, X_test, _, y_test = training.load_data()
    predictions = bundle["model"].predict(X_test)

    def reject_fit(*args, **kwargs):
        raise AssertionError("Evaluation must not fit or retrain a model.")

    monkeypatch.setattr(Pipeline, "fit", reject_fit)
    monkeypatch.setattr(RandomForestClassifier, "fit", reject_fit)
    monkeypatch.setattr(evaluation, "LAB_DIR", tmp_path)
    report_path = evaluation.evaluate_model(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Derive expected counts directly, independently of sklearn's metric calls.
    counts = Counter(zip(y_test.tolist(), predictions.tolist()))
    expected_matrix = [[counts[(actual, predicted)] for predicted in range(3)]
                       for actual in range(3)]
    correct = sum(actual == predicted for actual, predicted in zip(y_test, predictions))
    assert report["confusion_matrix"] == expected_matrix
    assert report["test_samples"] == sum(map(sum, expected_matrix)) == 36
    assert report["accuracy"] == pytest.approx(correct / len(y_test))

    per_class_f1 = []
    for label in range(3):
        true_positive = expected_matrix[label][label]
        actual_count = sum(expected_matrix[label])
        predicted_count = sum(row[label] for row in expected_matrix)
        denominator = actual_count + predicted_count
        per_class_f1.append(2 * true_positive / denominator if denominator else 0.0)
    assert report["f1_macro"] == pytest.approx(sum(per_class_f1) / 3)
    assert report["version"] == bundle["metadata"]["version"]
    assert report["model_file"] == model_path.name
    assert report_path.name == f"evaluation_{report['version']}.json"


def test_latest_model_is_selected_by_version(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    newer = models_dir / "wine_model_20260102T000000000000Z.joblib"
    older = models_dir / "wine_model_20260101T000000000000Z.joblib"
    newer.touch()
    older.touch()  # Creating the older version last must not make it the latest.

    monkeypatch.setattr(evaluation, "LAB_DIR", tmp_path)
    assert evaluation.find_latest_model() == newer


def test_missing_model_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "LAB_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Run train_model.py first"):
        evaluation.evaluate_model()
