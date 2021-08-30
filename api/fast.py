from datetime import datetime
from os import X_OK
import pandas as pd
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from cryptotradingindicator.gcp import get_model_from_gcp
from cryptotradingindicator.data import minmaxscaling, feature_engineer, get_train_data, get_xgecko

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello, crypto trader!"}

@app.get("/predict")
def predict(coin):

    data_train_scaled, scaler = minmaxscaling(feature_engineer(get_train_data())[['log_close']])
    X = get_xgecko()

    # get model from GCP
    model = get_model_from_gcp()
    # model = joblib.load('model.joblib')

    # make prediction
    pred = model.predict(X)

    # convert response from numpy to python type
    pred = np.exp(scaler.inverse_transform(pred))

    return dict(prediction=pred)