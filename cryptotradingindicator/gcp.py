import os

from google.cloud import storage
from termcolor import colored
from cryptotradingindicator.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, PATH_TO_LOCAL_MODEL

import joblib


def storage_upload(model_directory = MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        MODEL_NAME,
        model_directory,
        'model')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model/')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')
# gsutil cp -r model/ gs://BUCKET_NAME/models/crypto/MODEL_VERSION/
    
def get_model_from_gcp():
    client = storage.Client().bucket(BUCKET_NAME)
    model_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{PATH_TO_LOCAL_MODEL}" 
    blob = client.blob(model_location)
    blob.download_to_file(PATH_TO_LOCAL_MODEL)
    return joblib.load(PATH_TO_LOCAL_MODEL)
#  gsutil cp gs://crypto-indicator/models/crypto/v1/model.joblib .

MODEL_VERSION = "v2_01-09-2021"
storage_upload(model_directory = MODEL_VERSION)