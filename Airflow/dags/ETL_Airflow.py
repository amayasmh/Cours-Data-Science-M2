from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import pandas as pd
import os

CITIES = {
    "Paris": {"latitude": 48.85, "longitude": 2.35},
    "London": {"latitude": 51.51, "longitude": -0.13},
    "Berlin": {"latitude": 52.52, "longitude": 13.41}
}

DATA_PATH = "/opt/airflow/data/weather_data.csv"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1)
}

def extract_data(**context):
    results = []
    for city, coords in CITIES.items():
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['latitude']}&longitude={coords['longitude']}&current_weather=true"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()["current_weather"]
            results.append({
                "city": city,
                "temperature": data["temperature"],
                "windspeed": data["windspeed"],
                "weathercode": data["weathercode"],
                "timestamp": datetime.utcnow().isoformat()
            })
    context['ti'].xcom_push(key="weather_data", value=results)

def transform_data(**context):
    raw_data = context['ti'].xcom_pull(key="weather_data", task_ids="extract")
    df = pd.DataFrame(raw_data)
    context['ti'].xcom_push(key="weather_df", value=df.to_dict(orient="records"))

def load_data(**context):
    records = context['ti'].xcom_pull(key="weather_df", task_ids="transform")
    df_new = pd.DataFrame(records)

    # Load existing data
    if os.path.exists(DATA_PATH):
        df_existing = pd.read_csv(DATA_PATH)
        df_combined = pd.concat([df_existing, df_new])
        df_combined.drop_duplicates(subset=["city", "timestamp"], inplace=True)
    else:
        df_combined = df_new

    df_combined.to_csv(DATA_PATH, index=False)

with DAG(
    dag_id="daily_weather_etl",
    default_args=default_args,
    schedule="0 8 * * *",
    catchup=False,
    description="ETL DAG to fetch daily weather data",
    tags=["weather", "ETL"],
) as dag:

    extract = PythonOperator(
        task_id="extract",
        python_callable=extract_data
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=transform_data
    )

    load = PythonOperator(
        task_id="load",
        python_callable=load_data
    )

    extract >> transform >> load
