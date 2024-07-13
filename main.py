import json
import requests
import pandas as pd
import numpy as np
import math
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

from sklearn.metrics import mean_squared_error
from data import getData
from lstm import LSTMP

TF_ENABLE_ONEDNN_OPTS=0

def createModel(symbol, day, n_past, name):
    data = getData(symbol, day)
    data = data.reset_index()["close"]
    lstm = LSTMP(symbol, data, n_past)
    lstm.createModel()
    lstm.saveModel(name)

def loadModel(symbol):
    with open(f"{symbol}.pkl", "rb") as file:
        loaded_model = pickle.load(file) 
    return loaded_model

def predict(model, nextDays):


    x = model.xtest[-1].reshape(1, -1)
    nextDaysList = []
    for _ in range(nextDays):
        nextDay = model.predict(x)
        nextDaysList.append(nextDay[0, 0])
        x = np.roll(x, -1, axis=1)
        x[0, -1] = nextDay
    
    nextDaysList = model.inverse(np.array(nextDaysList).reshape(-1, 1))
    return nextDaysList
def drawGraph(x, y):
    pass

while True:
    print("*****MENU*****")
    print("Type 1 to create a model")
    print("Type 2 to pick a model")
    inp = int(input("Enter the number : "))

    if inp == 1:
        symbol = input("Enter the symbol(ex:BTCUSDT) : ")
        day = int(input("Enter the length of the data in days (365 is recommended) : "))
        npast = int(input("Enter the length of each x value (60 is recommended): "))
        name = input("Enter a name for the model : ")
        createModel(symbol, day, npast, name)

    if inp == 2:
        fileName = input("Enter the model : ")
        model = loadModel(fileName)
        while True:
            print(f"*****SELECTED MODEL : {fileName}*****")
            print("Type 1 to predict")
            print("Type 2 to draw the graph")
            print("Type 3 to show the mean squared error")
            print("Type 4 to go back to main menu")
            ninput = int(input("Enter the number : "))
            if ninput == 1:
                days = int(input("Enter the number days you want to predict : "))
                model.predictAndDraw(days)
                print("Prediction : \n")
                print(predict(model, days))

            elif ninput == 2:
                model.drawGraph()

            elif ninput == 3:
                print("MSE : " + str(model.meanSquaredError()))

            elif ninput == 4:
                break
            else:
                print("Unrecognized number. Try again")

    







