from datetime import datetime
from os import X_OK
import pandas as pd
import joblib
import numpy as np
from cryptotradingindicator.gcp import get_model_from_gcp
from cryptotradingindicator.data import minmaxscaling, feature_engineer, get_train_data, get_xgecko
from tensorflow.keras.models import load_model

data_train_scaled, scaler = minmaxscaling(feature_engineer(get_train_data())[['log_close']])
X = get_xgecko()

# get model from GCP
# model = get_model_from_gcp()
model = load_model('model.joblib')

# make prediction
prediction = model.predict(X)

# convert response from numpy to python type
prediction = np.exp(scaler.inverse_transform(prediction))

print(np.exp(scaler.inverse_transform(X[-5:][0])),prediction)