import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import prefect
from prefect import task, flow, utilities, task_runners
from datetime import date, datetime, time
from prefect import get_run_logger

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    name="cron-schedule-deployment",
    flow_location="D:\\Files\\mlops\\mlops-zoomcamp\\mlops-zoomcamp\\03-orchestration\\homework-11-06-2022.py",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Asia/Kolkata"),
)
 
@task
def read_data(path):

    df = pd.read_csv(path)
    return df

@task
def get_paths(pdate):
    logger = get_run_logger()

    date_split = datetime.strptime(pdate, "%Y-%m-%d")

    train_month = '0' + str(int(date_split.month) - 2)
    val_month = '0' + str(int(date_split.month) - 1)

    year = str(date_split.year)

    train_path = 'D:\\Files\\mlops\\datasets\\datasets\\fhv_tripdata_' + year + '-' + train_month + '.csv'
    val_path = 'D:\\Files\\mlops\\datasets\\datasets\\fhv_tripdata_' + year + '-' + val_month + '.csv'

    logger.info(train_path)
    logger.info(val_path)

    return train_path, val_path


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df.dropOff_datetime = pd.to_datetime(df.dropOff_datetime)
    df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime

    logger.info("Initial Duration: ")
    logger.info(df['duration'])

    df['duration'] = df.duration.dt.total_seconds() / 60

    logger.info("Minutes Duration: ")
    logger.info(df['duration'])

    logger.info("Done calculating duration")

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(pdate = None):
    logger = get_run_logger()

    #logger = prefect.context.get("logger")

    if (pdate == None):
        pdate = date.today()

    train_path, val_path = get_paths(pdate).result()

    #train_path, val_path = get_paths(pdate)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)

    logger.info(train_path)

    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    ## train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    #saving the model and dv

    with open('d:\\files\\mlops\\models\\model-'+ pdate + '.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)

    with open('d:\\files\\mlops\\models\\dv-'+ pdate + '.b', 'wb') as f_out:
        pickle.dump(dv, f_out)
    
    
    #now run the model
    run_model(df_val_processed, categorical, dv, lr)


main(pdate = "2021-08-15")
