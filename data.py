import numpy as np
from binance.client import Client


from sklearn.metrics import mean_squared_error

np.set_printoptions(suppress=True)

def getData(symbol, day):
    api_key = ''
    api_secret = ''
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, f"{day} day ago UTC")
    return klines
