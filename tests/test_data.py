## build test for CI

from cryptotradingindicator.data import *


def test_length_of_hello_world():
    assert len(hello_world()) != 0

def test_data_shape():
    data = get_csv_data()
    add_ema(data)
    add_stoch_rsi(data)
    add_bollinger(data,data.log_close)
    add_vol_roc(data)
    assert data.shape == (20840, 22)
