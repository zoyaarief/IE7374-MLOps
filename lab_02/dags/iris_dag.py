from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.iris_pipeline import (
    evaluate_and_predict,
    load_data,
    preprocess_data,
    train_and_save_model,
)


default_args = {
    "owner": "student",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="iris_classification_pipeline",
    description="Load, preprocess, train, save, and evaluate an Iris classifier",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["learning", "machine-learning", "iris"],
) as dag:
    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        op_args=[load_task.output],
    )

    train_task = PythonOperator(
        task_id="train_and_save_model",
        python_callable=train_and_save_model,
        op_args=[preprocess_task.output, "iris_classifier.pkl"],
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_and_predict",
        python_callable=evaluate_and_predict,
        op_args=[train_task.output],
    )

    load_task >> preprocess_task >> train_task >> evaluate_task


if __name__ == "__main__":
    dag.test()
