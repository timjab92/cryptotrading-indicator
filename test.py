from cryptotradingindicator.params import SELECTED_FEATURES, WINDOW_SIZE
from datetime import datetime
import pandas as pd
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from cryptotradingindicator.data import minmaxscaling, feature_engineer, get_train_data, get_xgecko
from tensorflow.keras.models import load_model


WINDOW_SIZE = 18
SELECTED_FEATURES = ['close','rsi','bollinger_up','bollinger_down','4h Return']


scaler = minmaxscaling(feature_engineer(get_train_data())[['close']])[1]
X = get_xgecko(selected_features = SELECTED_FEATURES, winsize=WINDOW_SIZE)

# get model from GCP
# model = get_model_from_gcp()
model = load_model('model/')

    # make prediction
pred = model.predict(X)
# convert response from numpy to python type
pred = scaler.inverse_transform(pred)

pred = pred[0][0].tolist()

print(dict(prediction=pred))