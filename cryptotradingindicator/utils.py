import numpy as np
import time

def time_selection(df, timeframe):
    """
    This function filter the data points in the choosed dataframe.
    The timeframe has to be choosen in this way. The first element is an interger and the second one is a letter, 
    which denotes the timeframe. M is for Months and m is for minutes.
    The smallest timeframe is a minites and the highest is months.
    For instance 5m (5 minutes), 5H (5 hours), 5D (5 days), 5M (5 months) and 5Y (5 years).
    It will return the dataframe with the specified parameters.
    """
    if "m" in timeframe:
        timeframe_list = [char for char in timeframe]
        minutes = int(timeframe_list[0])
        return df[df['date'].dt.minute%minutes==0].dropna()
    else:
        return df.set_index("date").resample(timeframe).mean().dropna().reset_index()
    
    
    
    
    
    
################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
