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
import pickle

from sklearn.metrics import mean_squared_error
from data import getData

class LSTMP:
    def __init__(self, symbol : str, data : pd.DataFrame, n_past : int):
        self.data = data
        self.symbol = symbol
        self.scaler = MinMaxScaler()
        self.model = Sequential()
        self.xtrain, self.ytrain, self.xtest, self.ytest = [], [], [], []
        self.trainTestData(0.8, n_past)

    def scaleData(self, data):
        return self.scaler.fit_transform(data.reshape(data.shape[0], -1))
    
    def trainTestData(self, ratio : float, n_past : int):
        sd = np.array(self.data).astype(float).reshape(-1, 1)
        sd = self.scaleData(sd)
        train_size = int(len(sd) * ratio)
        train_data, test_data = sd[:train_size], sd[train_size:]

        X_train, Y_train, X_test, Y_test = [], [], [], []

        for i in range(n_past, len(train_data)):
            X_train.append(train_data[i - n_past:i, 0])
            Y_train.append(train_data[i, 0])
        
        for i in range(n_past, len(test_data)):
            X_test.append(test_data[i - n_past:i, 0])
            Y_test.append(test_data[i, 0])

        self.xtrain, self.ytrain, self.xtest, self.ytest =  np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
    
    def createModel(self):

        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.xtrain.shape[1], 1)))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1))

        self.model.summary()

        self.model.compile(loss="mean_squared_error", optimizer="adam")

        self.model.fit(self.xtrain, self.ytrain, validation_data=(self.xtest, self.ytest), epochs = 100, batch_size=32, verbose=1)
    
    def inverse(self, data):
        return self.scaler.inverse_transform(data)
    
    def predict(self, data):
        return self.model.predict(data)

    def saveModel(self, name):
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump(self, file)

    def predictAndDraw(self, nextDays):

        x = self.xtest[-1].reshape(1, -1)
        nextDaysList = []
        for _ in range(nextDays):
            nextDay = self.predict(x)
            nextDaysList.append(nextDay[0, 0])
            x = np.roll(x, -1, axis=1)
            x[0, -1] = nextDay

        nextDaysList = self.inverse(np.array(nextDaysList).reshape(-1, 1))
        days = range(nextDays)
        plt.plot(days, nextDaysList, color="red", label='Prediction')
        plt.xlabel("Further days")
        plt.ylabel("Predicted price")
        plt.legend(loc='lower right')
        plt.show()

    def drawGraph(self):
        length = len(self.ytest) + len(self.ytrain) 
        x =  np.concatenate((self.xtrain, self.xtest), axis=0)
        y = np.concatenate((self.ytrain, self.ytest), axis=0).reshape(-1, 1)
        predict = self.predict(x)
        plt.plot(np.array(range(length)).reshape(-1, 1), self.inverse(y), color='green', label='Real data')
        plt.plot(np.array(range(length)).reshape(-1, 1), self.inverse(predict), color='red', label='Prediction')
        plt.ylabel("Price")
        plt.xlabel("Previous Days")
        plt.legend(loc='lower right')
        plt.show()

    
    def meanSquaredError(self):
        test_predict = self.model.predict(self.xtest)
        return math.sqrt(mean_squared_error(test_predict, self.ytest))
    


    


