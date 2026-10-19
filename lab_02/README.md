# Airflow Lab: Iris Classification Pipeline

## Objective

This lab demonstrates how Apache Airflow can orchestrate a small machine-learning workflow. The pipeline loads a CSV file, preprocesses its values, trains and saves a classification model, and finally reloads the model to evaluate it and make a prediction.

The four tasks intentionally follow the same learning sequence as a typical introductory Airflow ML lab:

```text
load_data -> preprocess_data -> train_and_save_model -> evaluate_and_predict
```

## Learning outcomes

After completing the lab, you should be able to:

- Define an Airflow DAG.
- turn ordinary Python functions into Airflow tasks with `PythonOperator`;
- create dependencies between tasks;
- pass small results between tasks with XCom;
- preprocess data without leaking test-set information;
- train and save an ML model as an artifact;
- reload the saved artifact for evaluation and inference; and
- inspect a DAG, task logs, and XCom results in the Airflow UI.

## Project structure

```text
lab_02/
├── dags/
│   ├── iris_dag.py
│   └── src/
│       ├── __init__.py
│       └── iris_pipeline.py
├── data/
│   └── iris.csv
├── artifacts/               # The trained model is created here
├── logs/                    # Airflow creates task logs here
├── .env.example
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## What each task does

### Task 1: `load_data`

This task reads `data/iris.csv`, validates that the required columns exist, and returns the DataFrame as a Base64-encoded pickle string.

The encoded string is stored by Airflow as an XCom value:

```text
DataFrame -> pickle bytes -> Base64 string -> XCom
```

Encoding is used because a DataFrame is not directly JSON serializable. This is suitable for this small teaching dataset; production pipelines should normally store larger datasets externally and pass only their location through XCom.

### Task 2: `preprocess_data`

This task receives Task 1's XCom output. It:

1. decodes the DataFrame;
2. removes missing and duplicate rows;
3. separates four numerical features from the `species` target;
4. creates stratified training and test sets; and
5. fits `StandardScaler` only on the training data and applies it to both sets.

The scaler and prepared arrays are packaged together and sent to Task 3 through XCom.

### Task 3: `train_and_save_model`

This task trains a multinomial logistic-regression classifier. It saves a bundle containing the model, scaler, feature names, and class names to:

```text
artifacts/iris_classifier.pkl
```

It returns the artifact path and encoded test data to the final task.

### Task 4: `evaluate_and_predict`

This task reloads the saved artifact, calculates test accuracy, and predicts the species of this new flower:

```text
sepal_length = 5.9
sepal_width  = 3.0
petal_length = 5.1
petal_width  = 1.8
```

Its final XCom result contains the accuracy, predicted species, class probabilities, and number of test rows.

## Step-by-step instructions

### 1. Install prerequisites

Install and start Docker Desktop. Allocate at least 4 GB of memory to Docker.

Confirm that Docker and Docker Compose are available:

```bash
docker --version
docker compose version
```

### 2. Open the project directory

```bash
cd airflow_iris_lab
```

### 3. Create the local environment file

On macOS or Linux:

```bash
cp .env.example .env
sed -i.bak "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
rm .env.bak
```

On Windows, copy `.env.example` to `.env` and leave `AIRFLOW_UID=50000` unchanged.

### 4. Create writable directories

```bash
mkdir -p artifacts logs
```

### 5. Build and start Airflow

```bash
docker compose up --build
```

The first build can take several minutes. Airflow standalone prints the administrator username and password in the terminal. Keep the terminal open.

### 6. Open the Airflow UI

Visit:

```text
http://localhost:8080
```

Sign in using the credentials printed by `airflow standalone`.

### 7. Run the DAG

1. Find `iris_classification_pipeline` in the DAG list.
2. Enable the DAG if it is paused.
3. Select **Trigger DAG**.
4. Open the DAG's **Graph** or **Grid** view.
5. Wait until all four tasks are green.

### 8. Inspect the results

Open the log for `evaluate_and_predict`. Near the end, you should see output similar to:

```text
Test accuracy: 0.9...
Predicted species: virginica
```

The precise accuracy can vary if you change the dataset or random seed. Also confirm that this file exists:

```text
artifacts/iris_classifier.pkl
```

### 9. Stop the environment

Press `Ctrl+C`, then run:

```bash
docker compose down
```

To remove the Airflow metadata volume and start completely fresh:

```bash
docker compose down --volumes
```

## DAG concepts demonstrated

The dependency expression in `iris_dag.py` is:

```python
load_task >> preprocess_task >> train_task >> evaluate_task
```

Airflow therefore starts a task only after its upstream task succeeds. Expressions such as `load_task.output` create XCom references; Airflow resolves those references at runtime and supplies the returned value to the next Python function.

The DAG uses `schedule=None`, so it runs only when manually triggered. It also uses `catchup=False`, preventing Airflow from generating historical runs.

## Suggested submission evidence

Include the following in your submission:

1. The source-code folder.
2. A screenshot of the DAG graph showing four successful tasks.
3. A screenshot of the final task log showing accuracy and the prediction.
4. A short explanation of XCom and the dependency chain.
5. A sentence explaining why the scaler is fitted only on training data.

## Short report summary

> This project uses Apache Airflow to orchestrate a four-stage Iris classification pipeline. The first task loads a CSV dataset, the second task cleans, splits, and scales the data, the third task trains and saves a logistic-regression model, and the fourth task reloads the model for evaluation and inference. Airflow controls the execution order through DAG dependencies and passes small serialized results between tasks using XCom. The project demonstrates reproducible workflow orchestration, preprocessing, model persistence, task monitoring, and prediction.

## Notes and limitations

- This is a learning project, not a production deployment.
- Pickle files must never be loaded from untrusted sources.
- XCom is being used because the dataset is deliberately small.
- A production pipeline would store data and artifacts in object storage or an artifact registry.
- The model artifact is written to a mounted local directory, so every task in this single-container setup can access it.
