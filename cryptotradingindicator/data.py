from cryptotradingindicator.utils import computeRSI, stoch_rsi, get_bollinger_bands
import pandas as pd
import os
import requests
import numpy as np

def hello_world():
    return "Hello, crypto trader!!"


## GET DATA_TRAIN ##

def get_train_data():
    """
    Returns the raw training dataset for the price of bitcoin since 31.12.2011.
    The index is set to the date.
    """
    path= os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(path, "../data/BTCUSD_4hours.csv"))
    data['date'] = pd.to_datetime(data.date)
    data_train = data.drop(columns="Unnamed: 0").set_index("date")
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

    #concatinate
    volume = volume[-180:]
    data_api = data_api[-181:-1]
    full = pd.concat([data_api, volume], axis=1)
    full.columns=['open', 'high', 'low', 'close', 'volume']

    for x in ['open', 'high', 'low', 'close']:
        full[f'log_{x}'] = full[x].apply(lambda x: np.log(x))

    data_api = full.copy()

    return data_api


## PROCESS THE DATA ##

def add_ema(data, tspan=[12,26,20,50,34,55]):
    """
    Adds exponential moving averages (EMA) to the dataframe.
    The default timeframes are 12,26,20,50,34 and 55.
    """
    for t in tspan:
        data[f'ema{t}'] = data.log_close.ewm(span=t).mean()
    return data


def add_stoch_rsi(data, d_window=3, k_window=3, window=14):
    """
    Adds stochastic RSI to the dataframe.
    """
    data['rsi'] = computeRSI(data['log_close'], window)
    data['K'], data['D'] = stoch_rsi(data['rsi'], d_window, k_window, window)
    return data


def add_bollinger(data, prices, rate=20):
    """
    Adds the Bollinger Bands to the Dataframe
    """
    data['sma'], data['bollinger_up'], data['bollinger_down'] = get_bollinger_bands(prices)
    return data


def add_vol_roc(data):
    """
    Computes and adds Volume Rate of Change to the DataFrame
    """
    data['vol_roc'] = data.volume.pct_change()
    return data

## FEATURE ENGINEER THE DATA ##

def feature_engineer(data):
    """
    Adds the EMAs, StochRSI, BollingerBands and Volume Rate of Change to the dataframe
    """
    add_ema(data)
    add_stoch_rsi(data)
    add_bollinger(data,data.log_close)
    add_vol_roc(data)
    return data



if __name__ == '__main__':
    train_data= get_train_data()
    add_ema(train_data)
    add_stoch_rsi(train_data)
    add_bollinger(train_data,train_data.log_close)
    add_vol_roc(train_data)
