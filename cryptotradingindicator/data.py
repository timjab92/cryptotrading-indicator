from utils import computeRSI, stoch_rsi, get_bollinger_bands
import pandas as pd

def hello_world():
    return "Hello, crypto trader!!"


## GET DATA ##

def get_csv_data():
    """
    Returns the raw training dataset for the price of bitcoin since 31.12.2011.
    The index is set to the date.
    """
    data = pd.read_csv("../data/BTCUSD_2011-12-31_to_2021-08-23_4hours_Clean.csv")
    data = data.drop(columns="Unnamed: 0").set_index("date")
    return data


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


if __name__ == '__main__':
    data = get_csv_data()
    add_ema(data)
    add_stoch_rsi(data)
    add_bollinger(data,data.log_close)
    add_vol_roc(data)
