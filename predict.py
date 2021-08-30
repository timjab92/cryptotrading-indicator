import os
from math import sqrt

import joblib
import pandas as pd
from cryptotradingindicator.params import MODEL_NAME, GCP_PATH, PATH_TO_LOCAL_MODEL, BUCKET_NAME
from cryptotradingindicator.data import get_train_data
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error

path= os.path.dirname(os.path.abspath(__file__))
LOCAL_PATH = os.path.join(path, "data/BTCUSD_4hours.csv")


def get_data(local=True, **kwargs):
    """method to get the training data from google cloud bucket"""
    # Add Client() here
    if local:
        df = get_train_data()
    else:
        df = pd.read_csv(GCP_PATH)
    return df


def download_model(model_directory="models", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res



# if __name__ == '__main__':
