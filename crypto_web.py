import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from cryptotradingindicator.params import MODEL_NAME, GCP_PATH, PATH_TO_LOCAL_MODEL, BUCKET_NAME
from cryptotradingindicator.data import get_coingecko, feature_engineer # , minmaxscaling

###SETTING SITE´S OVERHEAD
st.set_page_config(
    page_title="Cryp2️Moon", # => Quick reference - Streamlit  2️⃣🌚 🌕
    page_icon="💰",
    layout="centered", # wide
    initial_sidebar_state="auto"
    ) # collapsed

##### make a checkbox that decides if use the whole data or the coin_gecko data (timeline)
# st.checkbox("")


### RETRIEVING COMPLETE DATASET, from 201X
# train_data = pd.read_csv(GCP_PATH)
train_data = pd.read_csv("https://raw.githubusercontent.com/timjab92/cryptotradingindicator/master/data/BTC4h.csv")
train_data['date'] = pd.to_datetime(train_data.date)
train_data = train_data.set_index("date")
train_data.index = pd.to_datetime(train_data.index, format='%Y-%m-%d %H:%M')  # the 00:00 data is shown as date only, no time. Fix that later.
train_data = train_data.dropna()

### RETRIEVING DATA FROM COINGECKO

coins = ["₿ - Bitcoin", "💰 more coming soon..."]
# data = get_train_data()
@st.cache(allow_output_mutation=True, show_spinner=False)
def coin():
    return get_coingecko()

data = coin()
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M')
# data.index[-1] = [datetime.now().strftime('%Y-%m-%d %H:%M')]

# GET THE CURRENT PRICE
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
current_p = requests.get(url).json()["bitcoin"]["usd"]
# set the last close price to the current price
data.close[-1] = current_p
data.volume[-1] = data.volume[-2]
if current_p > data.high[-1]:
    data.high[-1] = current_p
elif current_p < data.low[-1]:
    data.low[-1] = current_p
else:
    pass
current_price = f'{current_p:9,.2f}'

data = train_data.merge(data, how='outer', left_index=True, right_index=True)
for i in ['close','open','high','low']:
    data[i] = np.where(data[f'{i}_x'].isna(), data[f'{i}_y'], data[f'{i}_x'])
data = data[['close','open','high','low', 'volume_x']]
data.columns = ['close','open','high','low', 'volume']

data = feature_engineer(data)

# # # ## load graph
# # # def load_graph(df):
# # #     fig = go.Figure(data=[
# # #         go.Candlestick(
# # #             x=df.index,  #x=filtered_data.index,
# # #             open=df['open'],  #open=filtered_data['open'],
# # #             high=df['high'],  #high=filtered_data['high'],
# # #             low=df['low'],  #low=filtered_data['low'],
# # #             close=df['close']  #close=filtered_data['close']
# # #         )
# # #     ])
# # #     return fig

## Call api
@st.cache(show_spinner=False)
def prediction():
    url = 'https://cryp2moon-idvayook4a-ew.a.run.app/predict'
    # display prediction
    response = requests.get(url).json()["prediction"]
    return response

### SIDEBAR CONFIGURATION

st.sidebar.markdown(
    "<h1 style='text-align: center; color: #FFC300;'>Cryp2Moon</h1>",
    unsafe_allow_html=True)



coins_select = st.sidebar.selectbox(label="Cryptocurrency",
                                    options=coins)


dd = st.sidebar.date_input("Select the start date for visualization",
                          datetime.now() - timedelta(days=8),
                          min_value=datetime.strptime(data.index[0].strftime('%Y-%m-%d %H:%M'),
                                                      "%Y-%m-%d %H:%M"),
                          max_value=  datetime.now(),
                        )

d=  dd.strftime('%Y-%m-%d %H:%M')
# ### RESET TO SEE ALL DATA
if st.sidebar.button('    Reset graph    '):
    d  = (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d %H:%M')
    ema_curve = False
    bb_curve = False

selected_data = data[data.index >= d]
selected_data_bb = data[data.index >= d]

## visualize indicators
# EMA
ema_curve = st.sidebar.checkbox("Show EMA", value = False)
t = 1
if ema_curve:
    t = st.sidebar.slider(label= " Select the period for EMA", min_value = 1, max_value= 99, value = 12)
ema = selected_data.close.ewm(span=t).mean()
# df.close.rolling(t).mean()  # normal moving average
ema = ema.dropna()

# Bollinger Bands
bb_curve = st.sidebar.checkbox("Show bollinger bands", value = False)
bb = 20
if bb_curve:
    bb = st.sidebar.number_input(label = "Select the period: ", min_value=1, max_value=100, step=1, value=20)
    selected_data_bb = data[data.index >= (dd- timedelta(days = int(bb/5))).strftime('%Y-%m-%d %H:%M')]
sma = selected_data_bb.close.rolling(bb).mean() # <-- Get SMA for 20 days
std = selected_data_bb.close.rolling(bb).std() # <-- Get rolling standard deviation for 20 days
bb_up = sma + std * 2 # Calculate top band
bb_down = sma - std * 2 # Calculate bottom band

# RSI
rsi_curve = st.sidebar.checkbox("Show stochastic RSI", value = False)



###DESIGN MAIN PART OF THE SITE
st.markdown('''

''')

st.markdown(
    "<h1 style='text-align: center; color: #FFC300;'>Cryptocurrency Price Indicator</h1>",
    unsafe_allow_html=True)

# Initialize session state for the button
if 'button_on' not in st.session_state:
    st.session_state.button_on = False

###BUTTON CREATION
col1, col2, col3 = st.columns(3)
if col2.button('    Prediction in 4 Hours    '):
    st.session_state.button_on = True

if st.session_state.button_on:
    # instantiate the prediction function
    pred = prediction()
    price_str = f'{pred:9,.2f}'
    perc_change = round(abs(1 - pred / current_p) * 100, 2)

    pred_df = data.iloc[-1:]
    pred_df.open = data.close[-1]
    pred_df.high = pred
    pred_df.low = data.close[-1]
    pred_df.close = pred
    # pred_df.index = pd.to_datetime([(datetime.now() + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')], format='%Y-%m-%d %H:%M')
    pred_df.index = pd.to_datetime(
        [(selected_data.index[-1] +
          timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')],
        format='%Y-%m-%d %H:%M')
    data = data.append(pred_df)
    selected_data = selected_data.append(pred_df)
    direction = "increase" if pred > current_p else "drop"
    if pred > 1.005 * current_p:
        st.write(
            "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ "
            + price_str +
            "🔼 within the next 4 hours!  <br> The current price of Bitcoin is US$ "
            + current_price + ". An expected " + str(perc_change) + "% " +
            direction + " 🤑. All in! </br></p>",
            unsafe_allow_html=True)
    elif pred < 0.995 * current_p:
        st.write(
            "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ "
            + price_str +
            "🔻 within the next 4 hours!  <br> The current price of Bitcoin is US$ "
            + current_price + ". An expected " + str(perc_change) + "% " +
            direction + ". Go short! </br></p>",
            unsafe_allow_html=True)
    else:
        st.write(
            "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ "
            + price_str +
            " within the next 4 hours!  <br> The current price of Bitcoin is US$ "
            + current_price + ". An expected " + str(perc_change) + "% " +
            direction + "!</br></p>",
            unsafe_allow_html=True)
#### CANDLE PLOT


def load_highlight(df):
    highlight = go.Candlestick(
        x=df.index[[-1]],
        open=df.open[[-1]],
        high=df.high[[-1]],
        low=df.low[[-1]],
        close=df.close[[-1]],
        increasing={'line': {'color': 'forestgreen'}}, # cornflowerblue, springgreen, darkgoldenrod,
        decreasing={'line': {'color': 'darkred'}},
        name=''
        )
    main_data = go.Candlestick(
            x=df.index,  #x=filtered_data.index,
            open=df['open'],  #open=filtered_data['open'],
            high=df['high'],  #high=filtered_data['high'],
            low=df['low'],  #low=filtered_data['low'],
            close=df['close'],  #close=filtered_data['close']
            name=''
        )
    if st.session_state.button_on == True:
        fig = go.Figure(data=[main_data, highlight])
    else:
        fig = go.Figure(data=main_data)
    return fig

fig = load_highlight(selected_data)

# update figure
fig.update_layout(
    autosize=True,
    width=750, height=350,
    margin=dict(l=40, r=40, b=40, t=40),
    showlegend=False,
    xaxis=dict(rangeslider=dict(visible=False),
            type="date",
            showgrid=False,
            # autorange=True,
            range=[d,datetime.now() + timedelta(days = 1)]),
            # xaxis_range=[datetime.datetime(2013, 10, 17),
            #                    datetime.datetime(2013, 11, 20)]),
    yaxis={
        'showgrid': False,
        "separatethousands": True,
        'autorange': True,
        "tickprefix": '$',
        "tickformat": " ,.2f",
        "rangemode": "normal"
    }
    )

# add EMA curve based on user decision
if ema_curve:
    fig.add_trace(
        go.Scatter(x=selected_data.index, y=ema, line=dict(color='orange', width=1), showlegend=True, mode="lines"))

# add bollinger bands based on user decision
if bb_curve:
    fig.add_trace(
        go.Scatter(x=selected_data_bb.index, y=bb_up, line=dict(color='magenta', width=1), showlegend=True, mode="lines"))
    fig.add_trace(
        go.Scatter(x=selected_data_bb.index, y=bb_down, line=dict(color='magenta', width=1), showlegend=True, mode="lines"))


st.plotly_chart(fig)


def stoch_rsi(data):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data.K, mode='lines', name='K'))
    fig2.add_trace(go.Scatter(x=data.index, y=data.D, mode='lines', name='D'))
    fig2.add_shape(type='line',
                   x0=data.index[0],
                   y0=80,
                   x1=data.index[-1],
                   y1=80,
                   line=dict(color='dimgrey', dash="dot", width=1),
                   xref='x',
                   yref='y')
    fig2.add_shape(type='line',
                   x0=data.index[0],
                   y0=20,
                   x1=data.index[-1],
                   y1=20,
                   line=dict(color='dimgrey', dash="dot", width=1),
                   xref='x',
                   yref='y')
    fig2.update_layout(autosize=False,
                       width=750,
                       height=150,
                       margin=dict(l=40, r=40, b=40, t=40),
                       showlegend=False,
                       xaxis=dict(rangeslider=dict(visible=False),
                                  type="date",
                                  showgrid=False,
                                  autorange=True),
                       yaxis={
                           'showgrid': True,
                           'autorange': True,
                           "rangemode": "normal"
                       })
    return fig2

if rsi_curve:
    st.plotly_chart(stoch_rsi(selected_data))




st.markdown("<p> <br> </br><br> </br><br>   </br> </p>", unsafe_allow_html=True)
# st.markdown('**DISCLAIMER**')
st.write("<p style='text-align: justify; font-size: 80%'> <b><b> DISCLAIMER </b></b> <br>Any and all liability for losses resulting from investment transactions carried out by the user is expressly excluded by Cryp2Moon. The information made available here does not serve as a recommendation for investment. </br></p>",
        unsafe_allow_html=True)
# "We recommend that you contact your personal financial advisor before carrying out specific transactions and investments."


hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)


if __name__ == '__main__':
    pred = prediction()
