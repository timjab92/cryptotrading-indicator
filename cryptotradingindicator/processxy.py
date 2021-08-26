#df -> scale -> split into subsequence
import numpy as np
from cryptotradingindicator.data import get_train_data, feature_engineer
from tensorflow.keras import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


## Preprocessing ##
def minmaxscaling(data_train):
    """
    applies the minmaxscaler to the training set. Attention! Output needs to be
    defined for data_train_scaled, min1 and range1!!
    """
    minmax_scaler = MinMaxScaler(feature_range = (0,1))
    minmax_scaler.fit(data_train)
    data_train_scaled = minmax_scaler.transform(data_train)
    min1 = minmax_scaler.data_min_ # [5:9] for log_prices
    range1 = minmax_scaler.data_range_ #[5:9]
    return data_train_scaled, min1, range1


my_mae = metrics.MeanAbsoluteError(
    name='mean_absolute_error', dtype=None
)

def get_xy(data_trained_scaled, length=60, horizon=1):
    y_train = []
    x_train = [data_train_scaled[i-length:i, 0] for i in range(length, len(data_train_scaled))]
    y_train = [data_train_scaled[i, 0] for i in range(length, len(data_train_scaled))]

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


## WORKFLOW ##
data_train = get_train_data()
data_train= feature_engineer(data_train)
data_train_scaled, min1, range1 = minmaxscaling(data_train)

# Split the data into x_train and y_train data sets
x_train, y_train = get_xy(data_train_scaled, 60, 1)
