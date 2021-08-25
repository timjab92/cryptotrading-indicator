import pandas as pd

from cryptotradingindicator.data import get_data

def test_instance():
    assert isinstance(get_data(), pd.DataFrame)
    

def test_header():
    data = [get_data().columns[i].lower() for i in range(len(list(get_data().columns)))]
    assert "open" in data and "close" in data