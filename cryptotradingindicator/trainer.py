# # In this file we will implement the training of the dataset
# # we need: class Trainer(obj)

# import joblib
# from termcolor import colored
from cryptotradingindicator.utils import minmaxscaling
import numpy as np
from cryptotradingindicator.data import *
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, LSTM
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Masking
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.metrics import MAPE, MeanAbsoluteError
# from keras.callbacks import EarlyStopping
# from tensorflow.keras.layers.experimental.preprocessing import Normalization

from sklearn.pipeline import Pipeline


#get data
#feature engineer
#create xy


## Pipeline for clean dataset
# def testtrainsplit(data,train_percentage=80):
#     train_len = int(len(data) * train_percentage/100)
#     data_train = data[:train_len]
#     data_test = data[train_len:]
#     return data_train, data_test




# class Trainer(object):
#     def __init__(self, X, y):
#         """
#             X: pandas DataFrame
#             y: pandas Series
#         """
#         self.pipeline = None
#         self.X = X
#         self.y = y

#     def set_pipeline(self):
#         """defines the pipeline as a class attribute"""
#         clean_pipe = Pipeline([
#             ('subsequence', subsequence())
#             ('minmax', minmaxscaling(data_train)),
#             ('create_xy', create_xy(data_train_scaled))
#         ])

#         self.pipeline = Pipeline([
#             ('preproc', preproc_pipe),
#-->             ('LSTM', model())
#         ])

#     def run(self):
#         self.set_pipeline()
#         self.pipeline.fit(self.X, self.y)

#     def evaluate(self, X_test, y_test):
#         """evaluates the pipeline on df_test and return the AME"""
#         y_pred = self.pipeline.predict(X_test)
#-->         mae = compute_rmse(y_pred, y_test)
#         return round(rmse, 2)

#     def save_model_locally(self):
#         """Save the model into a .joblib format"""
#         joblib.dump(self.pipeline, 'model.joblib')
#         print(colored("model.joblib saved locally", "green"))

#--> predict
#--> unscale

# if __name__ == "__main__":
#     # Get and clean data
#     df = get raw data
#     df = --> pipline clean_data(df)
#     y = --> define y (Friedas Notebook)
#     X = --> define X (Friedas Notebook)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     # Train and save model, locally and
#     trainer = Trainer(X=X_train, y=y_train)
#     trainer.set_experiment_name('xp2')
#     trainer.run()
#     mae = trainer.evaluate(X_test, y_test)
#     print(f"mae: {mae}")
#     trainer.save_model_locally()
#-->     storage_upload()
