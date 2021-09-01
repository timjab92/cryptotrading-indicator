from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import metrics

def get_model(x_train):
    # Build the LSTM model
    from tensorflow.keras import Sequential, layers

# Build the LSTM model
    model = Sequential()
    model.add(layers.LSTM(units=128,
                     return_sequences = True,
                     activation = "tanh"
                    #activation = "relu"
                     #input_shape = X_train[0].shape)
                     ))
    model.add(layers.LSTM(units=64,
                      return_sequences = False,
                      activation = "relu"
                    #activation = "relu"
                     ))
    model.add(layers.Dense(32,
                        activation = "tanh"
                       #activation="relu"
                      ))

    model.add(layers.Dense(8,
                       activation = "relu"
                       #activation="relu"
                      ))
    model.add(layers.Dense(1,
                      activation = "relu"
                      #activation="relu"
                      ))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics='mae')
    return model
