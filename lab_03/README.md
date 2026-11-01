# Lab 03: Wine Classification with GitHub Actions

[![Lab 03 - Wine Model Pipeline](https://github.com/zoyaarief/IE7374-MLOps/actions/workflows/lab_03_model_pipeline.yml/badge.svg)](https://github.com/zoyaarief/IE7374-MLOps/actions/workflows/lab_03_model_pipeline.yml)

This lab automates machine learning tests, model training, evaluation, and versioned output storage using GitHub Actions. It compares Logistic Regression and Random Forest classifiers on the Wine dataset, then evaluates the selected model on a reserved test set.

Adapted from [raminmohammadi/MLOps: GitHub Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2). This implementation is submitted as **Lab 03** in this repository.

## Original modifications

| Area | Reference lab | This implementation |
| --- | --- | --- |
| Dataset | Generated synthetic classification data | Built-in Wine dataset with 178 samples, 13 features, and 3 classes |
| Models | Random Forest | StandardScaler + Logistic Regression compared with Random Forest |
| Model selection | One candidate model | Five-fold stratified cross-validation, selecting by mean macro F1 |
| Evaluation | Separately generated synthetic data | A reproducible, stratified 20% held-out test set |
| Reports | F1 score | Accuracy, macro F1, confusion matrix, per-class metrics, and model metadata |
| Validation | Example test folder | Six automated pytest tests |
| Version storage | Workflow commits model and metrics to Git | Timestamped models and reports uploaded as GitHub Actions artifacts; a local reference run is included in Git |

The dataset is provided by [`sklearn.datasets.load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), so the pipeline needs no external dataset download or API credentials.

## Training and evaluation design

1. Split the dataset into **142 training samples** and **36 test samples**, using stratification and `random_state=42`.
2. Compare both models with five-fold stratified cross-validation on the training partition. Macro F1 gives each class equal weight.
3. Keep scaling inside the Logistic Regression pipeline so that each cross-validation fold learns its own scaling parameters.
4. Select the model with the highest mean cross-validation macro F1 and fit it on the full training partition.
5. Save the model with a UTC timestamp and metadata describing its version, split settings, software versions, and comparison results.
6. Load that saved model and recreate its test split using the saved settings. Compute the final test metrics without refitting the model.

## Repository layout

```text
.github/workflows/
    lab_03_model_pipeline.yml
lab_03/
    README.md
    requirements.txt
    src/
        train_model.py
        evaluate_model.py
    test/
        test_pipeline.py
    models/
        wine_model_<version>.joblib
    metrics/
        training_<version>.json
        evaluation_<version>.json
```

The `.venv` environment and Python caches are excluded from Git.

## Run locally

Use **Python 3.13.5**, which is also configured in the GitHub Actions workflow. Run the following commands from the repository root, the folder containing `lab_01`, `lab_02`, and `lab_03`.

```bash
python3 -m venv lab_03/.venv
lab_03/.venv/bin/python -m pip install -r lab_03/requirements.txt
lab_03/.venv/bin/python -m pytest lab_03/test -v
lab_03/.venv/bin/python lab_03/src/train_model.py
lab_03/.venv/bin/python lab_03/src/evaluate_model.py
```

The evaluator defaults to the latest timestamped Wine model. To evaluate a specific saved version:

```bash
lab_03/.venv/bin/python lab_03/src/evaluate_model.py \
  --model lab_03/models/wine_model_20261101T225234644562Z.joblib
```

Dependencies are pinned in `requirements.txt`. The `.joblib` file contains both the fitted model and its metadata. The timestamp connects the model to its training and evaluation JSON reports.

## Local reference results

Reference model version: **`20261101T225234644562Z`**.

| Candidate | Mean cross-validation accuracy | Mean cross-validation macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.9791 | **0.9796** |
| Random Forest | 0.9791 | 0.9789 |

Selected model: **Logistic Regression**.

| Held-out test metric | Result |
| --- | ---: |
| Test samples | 36 |
| Accuracy | 0.9722 |
| Macro F1 | 0.9710 |
| Correct predictions | 35 / 36 |

Confusion matrix, with rows representing actual classes and columns representing predicted classes. Class order: `class_0`, `class_1`, `class_2`.

```text
[[12  0  0]
 [ 0 14  0]
 [ 0  1  9]]
```

One `class_2` sample was classified as `class_1`. These results describe the fixed 36-sample test split.

Included reference files:

- [Saved model](models/wine_model_20261101T225234644562Z.joblib)
- [Training and comparison report](metrics/training_20261101T225234644562Z.json)
- [Evaluation and per-class report](metrics/evaluation_20261101T225234644562Z.json)

## Automated tests

The six tests check:

1. The data split is reproducible and the training and test partitions have no overlapping samples.
2. The Logistic Regression scaler learns its statistics from the training data.
3. The saved model reloads, predicts, and matches its training report.
4. Evaluation does not refit the model, and its metrics agree with independently counted predictions.
5. Model discovery chooses the latest timestamped version.
6. Evaluating without a saved model raises a clear error.

Local validation: **6 tests passed**. Test artifacts are written to temporary folders.

## GitHub Actions

Workflow: [Lab 03 - Wine Model Pipeline](https://github.com/zoyaarief/IE7374-MLOps/actions/workflows/lab_03_model_pipeline.yml).

The workflow runs when Lab 03 files or its workflow file change in a push to `main` or a pull request targeting `main`. It also supports manual runs through **Actions → Lab 03 - Wine Model Pipeline → Run workflow**.

Each successful run:

1. Checks out the repository and sets up Python 3.13.5.
2. Installs the pinned dependencies.
3. Runs all six tests.
4. Trains and saves a new timestamped model.
5. Evaluates that exact newly trained model using its explicit file path.
6. Uploads the model, matching training and evaluation reports, and requirements file as an artifact named `lab-03-wine-<run_id>-<run_attempt>`.

To retrieve a CI-generated model, open a successful workflow run and download its archive from the **Artifacts** section. Artifacts are configured for **90-day retention**. The committed local reference files above provide a persistent example, while workflow runs connect CI artifacts to the source commit that produced them.

References: [manually running workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow), [downloading workflow artifacts](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts).

## Submission

- Repository: [zoyaarief/IE7374-MLOps](https://github.com/zoyaarief/IE7374-MLOps)
- Lab folder: [lab_03](https://github.com/zoyaarief/IE7374-MLOps/tree/main/lab_03)
- Automation evidence: the workflow status badge and the linked GitHub Actions run history.
