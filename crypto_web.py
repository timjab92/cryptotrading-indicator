import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# import requests
from datetime import datetime, date, timedelta
from cryptotradingindicator.params import MODEL_NAME, GCP_PATH, PATH_TO_LOCAL_MODEL, BUCKET_NAME
# from tensorflow.keras.models import load_model
from cryptotradingindicator.gcp import get_model_from_gcp
from cryptotradingindicator.data import get_xgecko, get_coingecko, get_train_data, feature_engineer, minmaxscaling
import numpy as np
from google.cloud import storage
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Cryp2️Moon", # => Quick reference - Streamlit  2️⃣🌚 🌕
    page_icon="💰",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed


# # # # create credentials file
# # # google_credentials_file = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
# # # if not os.path.isfile(google_credentials_file):

# # #     print(
# # #         "write credentials file 🔥"
# # #         + f"\n- path: {google_credentials_file}")

# # #     # retrieve credentials
# # #     json_credentials = os.environ["GOOGLE_CREDS"]

# # #     # write credentials
# # #     with open(google_credentials_file, "w") as file:

# # #         file.write(json_credentials)

# # # else:

# # #     print("credentials file already exists 🎉")
    
    
    

##### make a checkbox that decides if use the whole data or the coin_gecko data (timeline)
# st.checkbox("")

data_train_scaled, scaler = minmaxscaling(feature_engineer(get_train_data())[['log_close']])
x_gecko = get_xgecko()


# @st.cache    #  put the load model into a function and it will not be reloaded every time the user changes something.
model = get_model_from_gcp()
    # model = joblib.load("model2.joblib")
# # model = load_model("model.joblib")
prediction = model.predict(x_gecko)
prediction = np.exp(scaler.inverse_transform(prediction))

st.write(f'''
The Bitcoin price is expected to close at around US$ {round(prediction[0][0],2)} within the next 4 hours!''')


coins = ["Bitcoin","Ethereum"]
data = get_train_data()
# data = get_coingecko()s
# data = pd.read_csv("raw_data/BTC-USD.csv")
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')



## SIDEBAR
# but1, but2, but3, but4, but5 = st.sidebar.columns(5)
# but2.markdown("# Crypto ")
# but1.image('bitcoin 32x32.png')
# but4.markdown("#     Indicator")

st.sidebar.markdown(f"""
    # Crypto Indicator
    """)

coin = st.sidebar.selectbox(label="Cryptocurrency",
                                options=coins)


d = st.sidebar.date_input(
    "Select the start date for visualization",
    datetime.now()-timedelta(days=180))

d=  d.strftime('%Y-%m-%d %H:%M:%S')

# RESET TO SEE ALL DATA
# but1, but2, but3 = st.sidebar.columns(3)
# if but2.button('Reset graph'):
if st.sidebar.button('    Reset graph    '):
    d = data.Date[0]


'''
# Cryptocurrency price indicator
'''
    
mask = (data.index > d) & (data.index <= datetime.now())
filtered_data = data.loc[mask]

# st.write('Bitcoin price')
# fig1, ax = plt.subplots(1,1, figsize=(15,10))
# ax.plot(data["Date"], data["Adj Close"])
# ax.set_ylabel("BTC price (USD)")
# st.write(fig1)
# st.line_chart(data=data['Adj Close'], width=0, height=0, use_container_width=True)


fig = go.Figure(data=[go.Candlestick(x=filtered_data.index,
                open=filtered_data['open'],
                high=filtered_data['high'],
                low=filtered_data['low'],
                close=filtered_data['close'])])
fig.update_layout(
                # title='Bitcoin price',
                autosize=True,
                #   width=1000, height=400,
                  margin=dict(l=40, r=40, b=40, t=40),
                  xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward",),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date",
        showgrid = False,
        autorange=True
    ),         
                  yaxis = {'showgrid': False, 
                           "separatethousands": True,
                           'autorange': True,
                           "tickprefix":'$',
                           "tickformat" : " ,.2f",
                           "rangemode": "normal"})


st.plotly_chart(fig)






# # download file
# client = storage.Client()
# bucket = client.bucket(BUCKET_NAME)
# blob = bucket.blob(storage_filename)
# blob.download_to_filename(local_filename)

# # df
# df = pd.read_csv(local_filename)
# df

# # upload file
# upload_blob = bucket.blob(upload_storage_filename)
# upload_blob.upload_from_filename(local_filename)




## Add controlers to ask user for input

## Call api
# url = 'https://taxifare-xyj6jtab4q-ew.a.run.app/predict'
# params = {
#     "passenger_count": str(passenger_count)
# }

# # display prediction
# response = requests.get(url, params=params).json()

# st.text('The taxi ride might cost you around {0:.3g}'.format(response["prediction"]))

