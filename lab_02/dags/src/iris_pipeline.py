import base64
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    return Path(os.getenv("AIRFLOW_LAB_DATA_DIR", _project_root() / "data"))


def _artifact_dir() -> Path:
    return Path(os.getenv("AIRFLOW_LAB_ARTIFACT_DIR", _project_root() / "artifacts"))


def _encode(value: Any) -> str:
    """Serialize a small trusted Python value into a JSON-safe string for XCom."""
    return base64.b64encode(pickle.dumps(value)).decode("ascii")


def _decode(value_b64: str) -> Any:
    """Restore a value created by _encode. Never decode untrusted pickle data."""
    return pickle.loads(base64.b64decode(value_b64.encode("ascii")))


def load_data() -> str:
    """Load and validate the Iris CSV, then return it in an XCom-safe form."""
    input_path = _data_dir() / "iris.csv"
    dataframe = pd.read_csv(input_path)

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    print(f"Loaded {len(dataframe)} rows from {input_path}")
    return _encode(dataframe)


def preprocess_data(dataframe_b64: str) -> str:
    """Clean, split, and scale the data without leaking test information."""
    dataframe = _decode(dataframe_b64)
    dataframe = dataframe.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).drop_duplicates()

    features = dataframe[FEATURE_COLUMNS]
    target = dataframe[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    prepared_data = {
        "x_train": x_train_scaled,
        "x_test": x_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
    }

    print(f"Prepared {len(x_train)} training rows and {len(x_test)} test rows")
    return _encode(prepared_data)


def train_and_save_model(prepared_data_b64: str, filename: str) -> dict[str, str]:
    """Train the classifier, save all inference dependencies, and return metadata."""
    prepared_data = _decode(prepared_data_b64)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(prepared_data["x_train"], prepared_data["y_train"])

    artifact_bundle = {
        "model": model,
        "scaler": prepared_data["scaler"],
        "feature_columns": FEATURE_COLUMNS,
        "classes": model.classes_.tolist(),
    }

    artifact_directory = _artifact_dir()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_directory / filename
    with artifact_path.open("wb") as artifact_file:
        pickle.dump(artifact_bundle, artifact_file)

    evaluation_data = {
        "x_test": prepared_data["x_test"],
        "y_test": prepared_data["y_test"],
    }

    print(f"Saved model bundle to {artifact_path}")
    return {
        "artifact_path": str(artifact_path),
        "evaluation_data": _encode(evaluation_data),
    }


def evaluate_and_predict(training_result: dict[str, str]) -> dict[str, Any]:
    """Reload the artifact, evaluate it, and classify one unseen flower."""
    artifact_path = Path(training_result["artifact_path"])
    with artifact_path.open("rb") as artifact_file:
        artifact_bundle = pickle.load(artifact_file)

    evaluation_data = _decode(training_result["evaluation_data"])
    model = artifact_bundle["model"]
    scaler = artifact_bundle["scaler"]

    test_predictions = model.predict(evaluation_data["x_test"])
    accuracy = float(accuracy_score(evaluation_data["y_test"], test_predictions))

    flower = pd.DataFrame(
        [[5.9, 3.0, 5.1, 1.8]],
        columns=artifact_bundle["feature_columns"],
    )
    flower_scaled = scaler.transform(flower)
    predicted_species = str(model.predict(flower_scaled)[0])
    probabilities = model.predict_proba(flower_scaled)[0]
    class_probabilities = {
        str(class_name): round(float(probability), 4)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    result = {
        "test_accuracy": round(accuracy, 4),
        "predicted_species": predicted_species,
        "class_probabilities": class_probabilities,
        "test_rows": int(len(evaluation_data["y_test"])),
    }

    print(f"Test accuracy: {result['test_accuracy']}")
    print(f"Predicted species: {predicted_species}")
    print(f"Class probabilities: {class_probabilities}")
    return result
