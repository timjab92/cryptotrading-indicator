import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
from datetime import datetime, date, timedelta
from cryptotradingindicator.params import MODEL_NAME, GCP_PATH, PATH_TO_LOCAL_MODEL, BUCKET_NAME
# from tensorflow.keras.models import load_model
# from cryptotradingindicator.gcp import get_model_from_gcp
from cryptotradingindicator.data import get_xgecko, get_coingecko, get_train_data, feature_engineer, minmaxscaling
import numpy as np

###SETTING SITE´S OVERHEAD
st.set_page_config(
    page_title="Cryp2️Moon", # => Quick reference - Streamlit  2️⃣🌚 🌕
    page_icon="💰",
    layout="centered", # wide
    initial_sidebar_state="auto"
    ) # collapsed


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


### RETRIEVING DATASET
# data_train_scaled, scaler = minmaxscaling(feature_engineer(get_train_data())[['log_close']])
# x_gecko = get_xgecko()


# # @st.cache    #  put the load model into a function and it will not be reloaded every time the user changes something.
# # model = get_model_from_gcp()
# #     # model = joblib.load("model2.joblib")

# ###CALLING THE MODEL AND STRING OUTPUT
# model = load_model("model/")
# prediction = model.predict(x_gecko)
# prediction = np.exp(scaler.inverse_transform(prediction))

# st.write(f'''
# The Bitcoin price is expected to close at around US$ {round(prediction[0][0],2)} within the next 4 hours!'''
# )

### RETRIEVING DATA FROM COINGECKO
#coins = ["Bitcoin","Ethereum"]
coins = ["₿ - Bitcoin", "💰 more coming soon..."]
# data = get_train_data()
data = get_coingecko()
# data = pd.read_csv("raw_data/BTC-USD.csv")
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')



## SIDEBAR
# but1, but2, but3, but4, but5 = st.sidebar.columns(5)
# but2.markdown("# Crypto ")
# but1.image('bitcoin 32x32.png')
# but4.markdown("#     Indicator")


### SIDEBAR CONFIGURATION
st.sidebar.markdown(
    "<h1 style='text-align: center; color: gray;'>Crypto Indicator</h1>",
    unsafe_allow_html=True
    )

coin = st.sidebar.selectbox(label="Cryptocurrency",
                                options=coins)


d = st.sidebar.date_input("Select the start date for visualization",
                          datetime.now() - timedelta(days=180),
                          min_value=datetime.strptime("2011-12-31 08:00:00",
                                                      "%Y-%m-%d %H:%M:%S"),
                          max_value=  datetime.now()
                        )

d=  d.strftime('%Y-%m-%d %H:%M:%S')

### RESET TO SEE ALL DATA
# check later this reset
if st.sidebar.button('    Reset graph    '):
    d = data.Date[0]

# if st.sidebar.button('    Prediction in 4 Hours    '):
#     st.write(f'''
#         The Bitcoin price is expected to close at around US$ {round(prediction[0][0],2)} within the next 4 hours!'''
#              )

###DESIGN MAIN PART OF THE SITE
st.markdown('''

''')



# ## Call api
# url = 'https://cryp2moon-idvayook4a-ew.a.run.app/predict'
# # display prediction
# response = requests.get(url).json()
# st.write(
#     f'''The Bitcoin price is expected to close at around US$ {round(response["prediction"],2)} within the next 4 hours!'''
# )



st.markdown(
    "<h1 style='text-align: center; color: #FFC300;'>Cryptocurrency Price Indicator</h1>",
    unsafe_allow_html=True)

###BUTTON CREATION
col1, col2, col3 = st.columns(3)
if col2.button('    Prediction in 4 Hours    '):
    st.markdown(
        "This is a serious website who cares about you. Are you sure you wanna come to the moon with us?"
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    if col2.button("YES, OF COURSE"):
        st.write(f'''
            We are glad to hear that. Before continue please send a small donation of 5000 Euros to this paypal: \n
            TIMCAREABOUTYOU@THISISNOTASCAM.COM
            '''
        )
        col1, col2, col3 = st.columns(3)
        if col2.button("I´ve sent my small donation "):
            st.write(f'''
                The Bitcoin price is expected to close at around US$ {round(response["prediction"],2)} within the next 4 hours!'''
                     )
    if col4.button("NO, I WANT TO KEEP LIVING MY BORING LIFE"):
        st.write(f'''
            TODO :No test
            ''')

#TO-DO = CREATE CONECTION WITH THE MODEL

# st.write('Bitcoin price')
# fig1, ax = plt.subplots(1,1, figsize=(15,10))
# ax.plot(data["Date"], data["Adj Close"])
# ax.set_ylabel("BTC price (USD)")
# st.write(fig1)
# st.line_chart(data=data['Adj Close'], width=0, height=0, use_container_width=True)



#### CANDEL PLOT

# FILTERING CANDELS
# mask = (data.index > d) & (data.index <= datetime.now())
# filtered_data = data.loc[mask]
# GRAPH
fig = go.Figure(data=[
    go.Candlestick(
        x=data.index,  #x=filtered_data.index,
        open=data['open'],  #open=filtered_data['open'],
        high=data['high'],  #high=filtered_data['high'],
        low=data['low'],  #low=filtered_data['low'],
        close=data['close']  #close=filtered_data['close']
    )
])
fig.update_layout(
    autosize=True,
    width=750, height=350,
    margin=dict(l=40, r=40, b=40, t=40),
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
# url = 'https://cryp2moon-idvayook4a-ew.a.run.app/predict'


# # display prediction
# response = requests.get(url).json()

# st.text('The taxi ride might cost you around {0:.3g}'.format(response["prediction"]))
