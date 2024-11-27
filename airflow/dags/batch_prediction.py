from asyncio import tasks
import json
from textwrap import dedent
import pendulum

import os

from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    "batch_prediction",
    default_args={"retries":2},
    description="gemstone batch prediction on batch",
    schedule="@weekly",
    start_date = pendulum.datetime(2024, 11, 14, tz="UTC"),
    catchup=False,
    tags=["example"],
) as dag:

    def download_files(**kwarg):
        bucket_name = os.getenv("BUCKET_NAME") #we can take this name manually or from env variable like here
        input_dir = "/app/input_files"
        #creating directory
        os.makedirs(input_dir, exist_ok=True)
        #os.system(f"aws s3 sync s3://{bucket_name}/inbox {config.inbox_dir}")


    def batch_prediction(**kwargs):
        from src.pipeline.batch_prediction import BatchPredictionConfig, SensorBatchPrediction
        config = BatchPredictionConfig()
        secsor_batch_prediction = SensorBatchPrediction(batch_config=config)
        secsor_batch_prediction.start_prediction()

    def upload_files(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync {config.archive_dir} s3://{bucket_name}/archive")
        os.system(f"aws s3 sync {config.outbox_dir} s3://{bucket_name}/outbox")

    download_input_files = PythonOperator(
        task_id="download_file",
        python_callable=download_files
    )

    generate_prediction_files = PythonOperator(
        task_id="prediction",
        python_callable=batch_prediction
    )

    upload_prediction_files = PythonOperator(
        task_id="upload_prediction_files",
        python_callable=upload_files
    )
    
    download_input_files >> generate_prediction_files >> upload_prediction_files
