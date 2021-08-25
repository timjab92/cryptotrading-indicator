import pandas as pd
# from cryptotradingindicator.utils import simple_time_tracker


GCP_PATH = "gs://crypto-indicator/data/BTCUSD_2011-12-31_to_2021-08-23_4hours_Clean.csv"
LOCAL_PATH = "/home/ivanfernandes/code/timjab92/cryptotradingindicator/data/BTCUSD_2011-12-31_to_2021-08-23_4hours_Clean.csv"
BUCKET_NAME="crypto-indicator"


# @simple_time_tracker
def get_gcp_data(nrows=10000, local=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    if local:
        path = LOCAL_PATH
    else:
        path = GCP_PATH
    df = pd.read_csv(path, nrows=nrows)
    return df


def hello_world():
    return "Hello, crypto trader!!"





if __name__ == "__main__":
    params = dict(nrows=1000,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  )
    df = get_gcp_data(**params)
