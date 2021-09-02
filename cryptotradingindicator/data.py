from cryptotradingindicator.utils import computeRSI, stoch_rsi, get_bollinger_bands
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import requests
import numpy as np
from cryptotradingindicator.params import *


def hello_world():
    return "Hello, crypto trader!!"


## GET DATA_TRAIN ##

def get_train_data():
    """
    Returns the raw training dataset for the price of bitcoin since 31.12.2011.
    The index is set to the date.
    """
    # path= os.path.dirname(os.path.abspath(__file__))
    # data = pd.read_csv(os.path.join(path, "../data/BTCUSD_4hours.csv"))
    # data['date'] = pd.to_datetime(data.date)
    # data_train = data.drop(columns="Unnamed: 0").set_index("date")

    data = pd.read_csv("data/BTC4h.csv")
    data['date'] = pd.to_datetime(data.date)
    data = data.set_index('date')
    return data

    return data_train


## GET LIVE DATA FROM COINGECKO ##

def get_coingecko():
    #ohlc
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=30"
    response = requests.get(url).json()
    #cleaning
    data_api = pd.DataFrame(response, columns = ['unix_time','open', 'high', 'low', 'close'])
    data_api["Date"] = pd.to_datetime(data_api["unix_time"], unit='ms')
    data_api = data_api.drop(columns='unix_time').set_index('Date')

    #volume
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30"
    response = requests.get(url).json()

    volume = pd.DataFrame(response['total_volumes'], columns=["unix_time","volume"])
    volume['date'] = pd.to_datetime(pd.to_datetime(volume['unix_time'],unit='ms').dt.strftime("%Y/%m/%d, %H:00:00"))
    volume = volume.drop(columns='unix_time').set_index('date')

    #resample hourly into 4h
    volume = volume.resample("4H").mean()

    #concatenate
    volume = volume[-180:]
    data_api = data_api[-180:]
    full = pd.concat([data_api, volume], axis=1)
    full.columns=['open', 'high', 'low', 'close', 'volume']

    for x in ['open', 'high', 'low', 'close']:
        full[f'log_{x}'] = full[x].apply(lambda x: np.log(x))

    data_api = full.copy()

    return data_api

## Feature Engineer ##

def add_ema(data, tspan=[12,26,20,50,34,55]):
    """
    Adds exponential moving averages (EMA) to the dataframe.
    The default timeframes are 12,26,20,50,34 and 55.
    """
    for t in tspan:
        data[f'ema{t}'] = data[CLOSE].ewm(span=t).mean()
        data[f'dist_ema{t}'] = data[CLOSE] - data[f'ema{t}']
    return data


def add_stoch_rsi(data, d_window=3, k_window=3, window=14):
    """
    Adds stochastic RSI to the dataframe.
    """
    data['rsi'] = computeRSI(data[CLOSE], window)
    data['K'], data['D'] = stoch_rsi(data['rsi'], d_window, k_window, window)
    return data


def add_bollinger(data, prices, rate=20):
    """
    Adds the Bollinger Bands to the Dataframe
    """
    data['sma'], data['bollinger_up'], data['bollinger_down'] = get_bollinger_bands(prices)
    for i in ['bollinger_up', 'bollinger_down']:
        data[f'dist_{i}'] = data[CLOSE] - data[i]
    return data


def add_vol_roc(data):
    """
    Computes and adds Volume Rate of Change to the DataFrame
    """
    data['vol_roc'] = data.volume.pct_change()
    return data

#### WORKFLOW ####

def feature_engineer(data):
    """
    Adds the EMAs, StochRSI, BollingerBands and Volume Rate of Change to the dataframe
    """
    add_ema(data)
    add_stoch_rsi(data)
    add_bollinger(data,data[CLOSE])
    add_vol_roc(data)
    data['4h Return'] = data[CLOSE].pct_change()
    data['4h Gradient'] = data[CLOSE].diff()
    data['boll_width'] = data['bollinger_up'] - data['bollinger_down']
    data['stoch_rsi'] = (data['K'] + data['D']) / 2
    return data


## SCALING ##

def minmaxscaling(data_train):
    """
    applies the minmaxscaler to the training set. Attention! Output needs to be
    defined for data_train_scaled, min1 and range1!!
    """
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(data_train)
    data_train_scaled = minmax_scaler.transform(data_train)
    min1 = minmax_scaler.data_min_
    range1 = minmax_scaler.data_range_
    return data_train_scaled, minmax_scaler, min1, range1

    # TODO: get method from notebook
## TURN INTO SEQUENCES ##
## TRAINING DATA ##
def get_xy(data_train_scaled, window_size=WINDOW_SIZE, horizon=HORIZON):
    data_train_subsequences = []
    fake_y_train = []
    for k in range(len(data_train_scaled)-(window_size-1)-horizon):
        data_train_subsequences.append(data_train_scaled[k:k+window_size])
        fake_y_train.append(data_train_scaled[k+window_size:k+window_size+horizon])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(data_train_subsequences), np.array(
        fake_y_train)[:, :, 0]

    return x_train, y_train

## COINGECKO ##
def get_xgecko(selected_features = SELECTED_FEATURES, winsize=WINDOW_SIZE):
    """
    Calls the coingecko API and returns the data used for prediction.
    x_gecko.shape == (no_sequ , winsize, no_features)
    """
    x_gecko = feature_engineer(get_coingecko())[selected_features][-winsize:]
    #get scaler the long way
    data_train = feature_engineer(get_train_data())[selected_features]
    data_train_scaled, scaler,min1,range1 = minmaxscaling(data_train)

    x_gecko_scaled = scaler.transform(x_gecko)
    x_gecko = np.array(x_gecko_scaled[-winsize:])
    x_gecko = np.reshape(x_gecko, (-1, winsize, len(selected_features)))
    return x_gecko


if __name__ == '__main__':
    print("fetching training data...")
    data_train = feature_engineer(get_train_data())
    print("fetching x_gecko...")
    x_gecko = get_xgecko()
    print("x_gecko shape:")
    print(x_gecko.shape)
    print("scaling data...")
    data_train_scaled, scaler,min1,range1 = minmaxscaling(data_train[[CLOSE]])
    print("splitting data...")
    x_train, y_train = get_xy(data_train_scaled, WINDOW_SIZE, HORIZON)
    print("x_train shape")
    print(x_train.shape)
    print("y_train shape")
    print(y_train.shape)
