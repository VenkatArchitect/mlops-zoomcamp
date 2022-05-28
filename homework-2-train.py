import mlflow
mlflow.autolog(disable=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my_experiment")

import argparse
import os
import pickle

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)

        with open("C:\\USERS\\VRAMA\\OUTPUT\\MODELS\\rf.bin", "wb") as f_out:
            pickle.dump(rf, f_out)

        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact(local_path="C:\\USERS\\VRAMA\\OUTPUT\\MODELS\\rf.bin", artifact_path="models_pickle")




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="c:\\users\\vrama\\output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)
