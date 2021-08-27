import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# import requests
from datetime import datetime, date, timedelta
from tensorflow.keras.models import load_model
from cryptotradingindicator.data import get_xgecko, get_coingecko, get_train_data, feature_engineer, minmaxscaling
import numpy as np


st.set_page_config(
    page_title="Cryp2️Moon", # => Quick reference - Streamlit  2️⃣🌚
    page_icon="💰",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed


##### make a checkbox that decides if use the whole data or the coin_gecko data (timeline)
# st.checkbox("")
# data = get_train_data()


data_train_scaled, scaler = minmaxscaling(feature_engineer(get_train_data())[['log_close']])
x_gecko = get_xgecko()

model = load_model("model.joblib")
prediction = model.predict(x_gecko)
prediction = np.exp(scaler.inverse_transform(prediction))

st.write(f'''
The Bitcoin price is expected to close at around US$ {round(prediction[0][0],2)} within the next 4 hours!''')
# "${:.2f}".format(prediction)


coins = ["Bitcoin","Ethereum"]
data = get_coingecko()
# data = pd.read_csv("raw_data/BTC-USD.csv")
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')


## SIDEBAR
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
# Cryptocurrency price estimation from a LSTM algorithm
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





## Add controlers to ask user for input

## Call api
# url = 'https://taxifare-xyj6jtab4q-ew.a.run.app/predict'
# params = {
#     "passenger_count": str(passenger_count)
# }

# # display prediction
# response = requests.get(url, params=params).json()

# st.text('The taxi ride might cost you around {0:.3g}'.format(response["prediction"]))

