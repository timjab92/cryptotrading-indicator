import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import requests
from datetime import datetime, date, timedelta

st.set_page_config(
    page_title="Crypto Trading", # => Quick reference - Streamlit
    page_icon="💰",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed

data = pd.read_csv("raw_data/BTC-USD.csv")
data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')

## SIDEBAR
st.sidebar.markdown(f"""
    # Crypto Indicator
    """)

coin = st.sidebar.selectbox(label="Cryptocurrency",
                                options=("Bitcoin",
                                         "Ethereum"))

d = st.sidebar.date_input(
    "Select the start date for visualization",
    date.today()-timedelta(days=720))

d=  d.strftime("%Y-%m-%d")

# RESET TO SEE ALL DATA
# but1, but2, but3 = st.sidebar.columns(3)
# if but2.button('Reset graph'):
if st.sidebar.button('    Reset graph    '):
    d = data.Date[0]


'''
# Cryptocurrency price estimation from a LSTM algorithm
'''
    
mask = (data['Date'] > d) & (data['Date'] <= datetime.today())
filtered_data = data.loc[mask]


# st.write('Bitcoin price')
# fig1, ax = plt.subplots(1,1, figsize=(15,10))
# ax.plot(data["Date"], data["Adj Close"])
# ax.set_ylabel("BTC price (USD)")
# st.write(fig1)
# st.line_chart(data=data['Adj Close'], width=0, height=0, use_container_width=True)


fig = go.Figure(data=[go.Candlestick(x=filtered_data['Date'],
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close'])])
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
        showgrid = False
    ),         
                  yaxis = {'showgrid': False, 
                           "separatethousands": True,
                           'autorange': True,
                           "tickprefix":'$',
                           "tickformat" : " ,.2f"})


st.plotly_chart(fig)



fig.update_layout(
    
)




## Add controlers to ask user for input

## Call api
# url = 'https://taxifare-xyj6jtab4q-ew.a.run.app/predict'
# params = {
#     "passenger_count": str(passenger_count)
# }

# # display prediction
# response = requests.get(url, params=params).json()

# st.text('The taxi ride might cost you around {0:.3g}'.format(response["prediction"]))

