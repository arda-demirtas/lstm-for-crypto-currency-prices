import pandas as pd
import numpy as np
import pickle
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import warnings
warnings.simplefilter('ignore')

from datetime import datetime, timedelta
from data import getData
from lstm import Model


def to_datetime(x):
    return datetime.fromtimestamp(int(x) / 1000)

def load_model(filename):
    with open(f"{filename}.pickle", "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model

while True:

    print("*****MENU*****")
    print("Type 1 to save a model")
    print("Type 2 to pick a model")
    print("Type 3 to exit")


    a = int(input("Enter a number : "))
    if a == 1:
        symbol = input("enter a symbol(EX:ETHUSDT) : ")
        day = int(input("enter the number of learning days (1080 recommended) : "))
        fileName = input("Enter a name for model : ")
        data = np.array(getData(symbol, day=day))
        data = data[:, 0:6]
        df = pd.DataFrame(data=data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.drop([df.shape[0] - 1])
        df.index = df['Date'].apply(to_datetime)
        df.index = df.index.astype('datetime64[ns]')
        df = df.drop(columns=['Date'])
        df = df.astype(float)   
        model = Model(data=df, symbol=symbol, day=90)
        model.build_model()
        model.save_model(filename=fileName)


    if a == 2:
        fileName = input("Enter the model name : ")
        loadedModel = load_model(fileName)

        if loadedModel:
            pass
        else:
            print("Error.")
            exit()

        print(f"***SELECTED MODEL : {fileName}***")
        print("Type 1 to draw price history graph")
        print("Type 2 to draw graph with test prediction")
        print("Type 3 to see future predictions")
        print("Type 4 to see RMSE")
        print("Type 5 to exit to menu")
        while True:
            opt = int(input("Enter a number : "))
            if opt == 1:
                loadedModel.graph()
            if opt == 2:
                loadedModel.model_graph()
            if opt == 3:
                print(str(loadedModel.data.index[-1] + timedelta(1)) + " : " + str(loadedModel.predict_nextday()[0][0]))
            if opt == 4:
                print("RMSE : " + str(loadedModel.rmse()))
            if opt == 5:
                break
            
    if a == 3:
        exit()








