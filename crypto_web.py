import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from cryptotradingindicator.params import MODEL_NAME, GCP_PATH, PATH_TO_LOCAL_MODEL, BUCKET_NAME
from cryptotradingindicator.data import get_coingecko # , minmaxscaling


GCP_PATH = "gs://crypto-indicator/data/BTCUSD_4hours.csv"


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
# train_data['date'] = pd.to_datetime(train_data.date)
# train_data = train_data.drop(columns="Unnamed: 0").set_index("date")
# data_train_scaled, scaler = minmaxscaling(train_data[['log_close']])
# x_gecko = get_xgecko()

# ###CALLING THE MODEL AND STRING OUTPUT
# model = load_model("model/")
# prediction = model.predict(x_gecko)
# prediction = np.exp(scaler.inverse_transform(prediction))


### RETRIEVING DATA FROM COINGECKO
coins = ["₿ - Bitcoin", "💰 more coming soon..."]
# data = get_train_data()
@st.cache(allow_output_mutation=True)
def coin():
    data = get_coingecko()
    return data

data = coin()
# data = pd.read_csv("raw_data/BTC-USD.csv")
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')

# GET THE CURRENT PRICE
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
current_p = requests.get(url).json()["bitcoin"]["usd"]
# set the last close price to the current price
data.close[-1] = current_p
if current_p > data.high[-1]:
    data.high[-1] = current_p
elif current_p < data.low[-1]:
    data.low[-1] = current_p
else:
    pass        
current_price = f'{current_p:9,.2f}'

## load graph
def load_graph():
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,  #x=filtered_data.index,
            open=data['open'],  #open=filtered_data['open'],
            high=data['high'],  #high=filtered_data['high'],
            low=data['low'],  #low=filtered_data['low'],
            close=data['close']  #close=filtered_data['close']
        )
    ])
    return fig

# load figure
fig = load_graph()

## Call api
@st.cache
def prediction():
    url = 'https://cryp2moon-idvayook4a-ew.a.run.app/predict'
    # display prediction
    response = requests.get(url).json()["prediction"]
    return response
   
### SIDEBAR CONFIGURATION
st.sidebar.markdown(
    "<h1 style='text-align: center; color: gray;'>Crypto Indicator</h1>",
    unsafe_allow_html=True
    )

coins_select = st.sidebar.selectbox(label="Cryptocurrency",
                                options=coins)


# # # # d = st.sidebar.date_input("Select the start date for visualization",
# # # #                           datetime.now() - timedelta(days=180),
# # # #                           min_value=datetime.strptime("2011-12-31 08:00:00",
# # # #                                                       "%Y-%m-%d %H:%M:%S"),
# # # #                           max_value=  datetime.now()
# # # #                         )

# # # # d=  d.strftime('%Y-%m-%d %H:%M:%S')

# # # # ### RESET TO SEE ALL DATA
# # # # # check later this reset
# # # # if st.sidebar.button('    Reset graph    '):
# # # #     d = data.Date[0]

## visualize indicators
# EMA
ema_curve = st.sidebar.checkbox("Show EMA curve", value = False)
t = 1
if ema_curve:
    t = st.sidebar.slider(label= " Select the period for EMA", min_value = 1, max_value= 99, value = 12)
ema = data.close.ewm(span=t).mean()
# df.close.rolling(t).mean()  # normal moving average
ema = ema.dropna()

# Bollinger Bands
bb_curve = st.sidebar.checkbox("Show bollinger bands", value = False)
bb = 20
if bb_curve:
    bb = st.sidebar.number_input(label = "Select the rate: ", min_value=1, max_value=100, step=1, value=20)
sma = data.close.rolling(bb).mean() # <-- Get SMA for 20 days
std = data.close.rolling(bb).std() # <-- Get rolling standard deviation for 20 days
bb_up = sma + std * 2 # Calculate top band
bb_down = sma - std * 2 # Calculate bottom band



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
    perc_change = round(abs(1-pred/current_p)*100,2)
    if data.close[-1] < pred:
        st.write(
        "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ " + str(round(pred,2)) + 
        "🔼 within the next 4 hours!  <br> The current price of bitcoin is US$ " + current_price + ". An expected " + str(perc_change) + "% increase 🤑. All in! </br></p>",
        unsafe_allow_html=True)
    else:
        st.write(
        "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ " + str(round(pred,2)) + 
        "🔻 within the next 4 hours!  <br> The current price of bitcoin is US$ " + current_price + ". An expected " + str(perc_change) + "% drop . Go short! </br></p>",
        unsafe_allow_html=True)


#### CANDEL PLOT

# FILTERING CANDELS
# mask = (data.index > d) & (data.index <= datetime.now())
# filtered_data = data.loc[mask]
# GRAPH

# @st.cache(allow_output_mutation=True)



# update figure
fig.update_layout(
    autosize=True,
    width=750, height=350,
    margin=dict(l=40, r=40, b=40, t=40),
    showlegend=False,
    xaxis=dict(rangeslider=dict(visible=False),
            type="date",
            showgrid=False,
            autorange=True),
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
        go.Scatter(x=data.index, y=ema, line=dict(color='orange', width=1), showlegend=True, mode="lines"))

# add bollinger bands based on user decision
if bb_curve:
    fig.add_trace(
        go.Scatter(x=data.index, y=bb_up, line=dict(color='magenta', width=1), showlegend=True, mode="lines"))
    fig.add_trace(
        go.Scatter(x=data.index, y=bb_down, line=dict(color='magenta', width=1), showlegend=True, mode="lines"))


st.plotly_chart(fig)



# with placeholder.container():
#     pred = prediction()["prediction"]
    # perc_change = round(abs(1-pred/current_p)*100,2)
    # if data.close[-1] < pred:
    #     st.write(
    #     "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ " + str(round(pred,2)) + 
    #     "🔼 within the next 4 hours!  <br> The current price of bitcoin is US$ " + current_price + ". An expected " + str(perc_change) + "% increase 🤑. All in! </br></p>",
    #     unsafe_allow_html=True)
    # else:
    #     st.write(
    #     "<p style='text-align: center'>The Bitcoin price is expected to close at around US$ " + str(round(pred,2)) + 
    #     "🔻 within the next 4 hours!  <br> The current price of bitcoin is US$ " + current_price + ". An expected " + str(perc_change) + "% drop . Go short! </br></p>",
    #     unsafe_allow_html=True)