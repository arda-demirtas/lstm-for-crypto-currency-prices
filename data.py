import json
import requests
import pandas as pd
import numpy as np
import math
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import mean_squared_error

np.set_printoptions(suppress=True)

api_key = ''
api_secret = ''

client = Client(api_key, api_secret)

def getData(symbol, day):
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, f"{day} day ago UTC")
    df = pd.DataFrame(data=klines, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'], index=range(day))
    cols = list(df.columns.values)
    df = df.drop(columns = cols[6:])
    df['dateTime'] = pd.to_datetime(df['dateTime'], unit='ms')
    df['days'] = range(day)
    return df
