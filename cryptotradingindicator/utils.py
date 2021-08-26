import numpy as np
from cryptotradingindicator.data import get_train_data, feature_engineer
from sklearn.preprocessing import MinMaxScaler

## PREPROCESSING ##
def time_selection(df, timeframe):
    """
    This function filter the data points in the choosed dataframe.
    The timeframe has to be choosen in this way. The first element is an
    interger and the second one is a letter,
    which denotes the timeframe. M is for Months and m is for minutes.
    The smallest timeframe is a minites and the highest is months.
    For instance 5m (5 minutes), 5H (5 hours), 5D (5 days), 5M (5 months) and
    5Y (5 years).
    It will return the dataframe with the specified parameters.
    """
    if "m" in timeframe:
        timeframe_list = [char for char in timeframe]
        minutes = int(timeframe_list[0])
        return df[df['date'].dt.minute%minutes==0].dropna()
    else:
        return df.set_index("date").resample(timeframe).mean().dropna().reset_index()




## STOCH RSI ##

def computeRSI (data, window=14):
    """
    Computes the Relative Stregth Index for a given dataset and the window can be defined. Its default value is 14.
    """
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=window-1 so we get decay alpha=1/window
    up_chg_avg   = up_chg.ewm(com=window-1 , min_periods=window).mean()
    down_chg_avg = down_chg.ewm(com=window-1 , min_periods=window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def stoch_rsi(rsi, d_window=3, k_window=3, window=14):
    """
    Computes the stochastic RSI. Default values are d=3, k=3, window=14.
    """
    minrsi = rsi.rolling(window=window, center=False).min()
    maxrsi = rsi.rolling(window=window, center=False).max()
    stoch = ((rsi - minrsi) / (maxrsi - minrsi)) * 100
    K = stoch.rolling(window=k_window, center=False).mean()
    D = K.rolling(window=d_window, center=False).mean()
    return K, D


## BOLLINGER BANDS ##

def get_bollinger_bands(prices, rate=20):
    """
    Computes the Bollinger Bands for a defined price series.
    """
    sma = prices.rolling(rate).mean() # <-- Get SMA for 20 days
    std = prices.rolling(rate).std() # <-- Get rolling standard deviation for 20 days
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2 # Calculate bottom band
    return sma, bollinger_up, bollinger_down
