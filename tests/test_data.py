## build test for CI

from cryptotradingindicator.data import *


def test_length_of_hello_world():
    assert len(hello_world()) != 0

def test_data_shape():
    train_data= get_train_data()
    add_ema(train_data)
    add_stoch_rsi(train_data)
    add_bollinger(train_data,train_data.log_close)
    add_vol_roc(train_data)
    assert train_data.shape == (20840, 22)
