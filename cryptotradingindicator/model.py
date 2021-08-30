from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import metrics

my_mae = metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None)


def get_model(x_train):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=my_mae)
    return model
