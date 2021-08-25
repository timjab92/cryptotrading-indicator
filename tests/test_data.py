## build test for CI

from cryptotradingindicator.data import hello_world
from cryptotradingindicator.data import data

def test_length_of_hello_world():
    assert len(hello_world()) != 0

def test_data_shape():
    assert data.shape == (20840, 22)
