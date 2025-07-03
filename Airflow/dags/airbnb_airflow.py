from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder

default_args = {
    'start_date': datetime(2025, 7, 1),
    'retries': 1,
}

dag = DAG(
    'airbnb_price_prediction',
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=['airbnb', 'machine_learning']
)

DATA_PATH = 'opt/ML/Airbnb_Project/'
TRAIN_CSV = '/opt/airflow/data/airbnb_train.csv'
TEST_CSV = '/opt/airflow/data/airbnb_test.csv'
PREDICTION_FILE = DATA_PATH + 'predictions_xgboost.csv'


def load_data(**kwargs):
    kwargs['ti'].xcom_push(key='train_df', value=pd.read_csv(TRAIN_CSV).to_json())
    kwargs['ti'].xcom_push(key='test_df', value=pd.read_csv(TEST_CSV).to_json())


def preprocess_data(**kwargs):
    train_df = pd.read_json(kwargs['ti'].xcom_pull(task_ids='load_data', key='train_df'))
    test_df = pd.read_json(kwargs['ti'].xcom_pull(task_ids='load_data', key='test_df'))

    for col in ['bathrooms', 'bedrooms', 'beds']:
        train_df[col].fillna(train_df[col].median(), inplace=True)
        test_df[col].fillna(test_df[col].median(), inplace=True)

    for col in ['host_has_profile_pic', 'host_identity_verified']:
        train_df[col].fillna(False, inplace=True)
        test_df[col].fillna(False, inplace=True)

    for df in [train_df, test_df]:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['host_age_days'] = (datetime(2023, 1, 1) - df['host_since']).dt.days
        df['host_age_days'].fillna(df['host_age_days'].median(), inplace=True)

    for df in [train_df, test_df]:
        df['review_scores_rating'].fillna(df['review_scores_rating'].mean(), inplace=True)
        df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float)
        df['host_response_rate'].fillna(df['host_response_rate'].median(), inplace=True)
        df['instant_bookable'] = df['instant_bookable'].astype(bool).astype(int)
        df['cleaning_fee'] = df['cleaning_fee'].astype(bool).astype(int)
        df['nb_amenities'] = df['amenities'].apply(lambda x: len(ast.literal_eval(x)) if pd.notnull(x) else 0)
        df['len_description'] = df['description'].fillna("").apply(len)
        df['property_type'] = df['property_type'].apply(lambda x: x if x in df['property_type'].value_counts().nlargest(10).index else 'Other')

    cat_cols = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
    train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    le = LabelEncoder()
    combined_zip = pd.concat([train_df['zipcode'], test_df['zipcode']], axis=0).astype(str)
    le.fit(combined_zip)
    train_df['zipcode_enc'] = le.transform(train_df['zipcode'].astype(str))
    test_df['zipcode_enc'] = le.transform(test_df['zipcode'].astype(str))

    drop_cols = ['id', 'log_price', 'description', 'name', 'amenities', 'first_review',
                 'last_review', 'host_since', 'zipcode', 'neighbourhood',
                 'host_has_profile_pic', 'host_identity_verified']

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['log_price']
    X_test = test_df.drop(columns=drop_cols)

    kwargs['ti'].xcom_push(key='X_train', value=X_train.to_json())
    kwargs['ti'].xcom_push(key='y_train', value=y_train.to_json())
    kwargs['ti'].xcom_push(key='X_test', value=X_test.to_json())


def train_model(**kwargs):
    X_train = pd.read_json(kwargs['ti'].xcom_pull(task_ids='preprocess_data', key='X_train'))
    y_train = pd.read_json(kwargs['ti'].xcom_pull(task_ids='preprocess_data', key='y_train'), typ='series')

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    kwargs['ti'].xcom_push(key='model', value=model)


def predict_and_save(**kwargs):
    import joblib
    X_test = pd.read_json(kwargs['ti'].xcom_pull(task_ids='preprocess_data', key='X_test'))

    X_train = pd.read_json(kwargs['ti'].xcom_pull(task_ids='preprocess_data', key='X_train'))
    y_train = pd.read_json(kwargs['ti'].xcom_pull(task_ids='preprocess_data', key='y_train'), typ='series')

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result_df = pd.DataFrame({'y_pred': y_pred})
    result_df.to_csv(PREDICTION_FILE, index=False)


t1 = PythonOperator(task_id='load_data', python_callable=load_data, dag=dag)
t2 = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
t3 = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
t4 = PythonOperator(task_id='predict_and_save', python_callable=predict_and_save, dag=dag)

t1 >> t2 >> t3 >> t4
