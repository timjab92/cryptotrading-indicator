import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import requests
# from datetime import datetime, date

st.set_page_config(
    page_title="Crypto Trading", # => Quick reference - Streamlit
    page_icon="💸",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed
    
'''
# Cryptocurrency price estimation from an LSTM-based algorithm
'''



#Yahoo Finance Data
data = pd.read_csv("raw_data/BTC-USD.csv")
data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
# st.write('Bitcoin price')
# plt.plot(data["Date"], data["Close"])
# plt.plot(data["Date"], data["Adj Close"])
# plt.ylabel("BTC price (USD)")
# plt.show()

st.write('Bitcoin price')
fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

fig.show()


st.sidebar.markdown(f"""
    # Crypto Indicator
    """)



## Add controlers to ask user for input


## Call api
# url = 'https://taxifare-xyj6jtab4q-ew.a.run.app/predict'
# params = {
#     "passenger_count": str(passenger_count)
# }

# # display prediction
# response = requests.get(url, params=params).json()

# st.text('The taxi ride might cost you around {0:.3g}'.format(response["prediction"]))