import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import math


class Model:
    def __init__(self, data, symbol, day):
        self.__day =day
        self.__symbol = symbol
        self.__data = data
        self.__dataset = data.values
        self.__scalerX =  MinMaxScaler(feature_range=(0, 1))
        self.__scalerY = MinMaxScaler(feature_range=(0, 1))
        self.__len = self.__dataset.shape[0]
        self.__trainSize = math.ceil(len(self.__dataset) * .8)
        self.__dataTrain = self.__dataset[0:self.__trainSize, :]
        self.__dataTest = self.__dataset[self.__trainSize - day :, :]
        self.__dataTrainX = []
        self.__dataTrainY = []
        self.__dataTestX = []
        self.__dataTestY = []
        self.__model = Sequential()

        for i in range(day, len(self.__dataTrain)):
            self.__dataTrainX.append(self.__dataTrain[i - day : i])
            self.__dataTrainY.append(self.__dataTrain[i, 3])

        self.__dataTrainX = np.array(self.__dataTrainX).astype(float)
        self.__dataTrainY = np.array(self.__dataTrainY).astype(float)

        self.__dataTrainX = np.reshape(self.__dataTrainX, (-1, day, 5))
        self.__dataTrainY = np.reshape(self.__dataTrainY, (-1, 1))

        self.__dataTestY = self.__dataset[self.__trainSize : , 3]

        for i in range(day, len(self.__dataTest)):
            self.__dataTestX.append(self.__dataTest[i - day : i, :])

        self.__dataTestX = np.array(self.__dataTestX).astype(float)
        self.__dataTestY = np.array(self.__dataTestY).astype(float)

        self.__dataTestX = np.reshape(self.__dataTestX, (-1, day, 5))
        self.__dataTestY = np.reshape(self.__dataTestY, (-1, 1))

        #scale
        self.__scdataTrainX = np.reshape(self.__dataTrainX,(self.__dataTrainX.shape[0], -1))
        self.__scdataTrainX = self.__scalerX.fit_transform(self.__scdataTrainX)
        self.__scdataTrainX = np.reshape(self.__scdataTrainX, (self.__dataTrainX.shape[0], day, 5))

        self.__scdataTestX = np.reshape(self.__dataTestX,(self.__dataTestX.shape[0], -1))
        self.__scdataTestX = self.__scalerX.transform(self.__scdataTestX)
        self.__scdataTestX = np.reshape(self.__scdataTestX, (self.__dataTestX.shape[0], day, 5))

        self.__scdataTrainY = self.__scalerY.fit_transform(self.__dataTrainY)

    @property
    def data(self):
        return self.__data
    
    @property
    def dataset(self):
        return self.__dataset
    
    @property
    def datatrain(self):
        return self.__dataTrain
    
    @property
    def datatest(self):
        return self.__dataTest
    
    @property
    def datatrainx(self):
        return self.__dataTrainX
    
    @property
    def datatrainy(self):
        return self.__dataTrainY
    
    @property
    def datatestx(self):
        return self.__dataTestX
    
    @property
    def datatesty(self):
        return self.__dataTestY
    
    @property
    def scalerx(self):
        return self.__scalerX
    
    @property
    def scalery(self):
        return self.__scalerY

    def build_model(self):
        print(self.__dataTestX)
        print(self.__dataTestY)
        self.__model.add(LSTM(64, activation = "relu", return_sequences = True, input_shape=(self.__scdataTrainX.shape[1], self.__scdataTrainX.shape[2])))
        self.__model.add(LSTM(32, activation = "relu", return_sequences = False))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(self.__dataTrainY.shape[1]))
        self.__model.compile(optimizer = "adam", loss="mse")
        self.__model.fit(self.__scdataTrainX, self.__scdataTrainY, epochs = 100, batch_size = 16, verbose = 1)

    def predict(self, data):
        return self.__model.predict(data)
    
    def rmse(self):
        return np.sqrt(np.mean(self.__scalerY.inverse_transform(self.predict(self.__scdataTestX)) - self.__dataTestY) ** 2)
    

    def save_model(self, filename):
        with open(f"{filename}.pickle", 'wb') as file:
            pickle.dump(self, file)

    def model_graph(self):
        train = self.__data[:self.__trainSize]
        valid = self.__data[self.__trainSize:]
        valid['Pred'] = self.__scalerY.inverse_transform(self.predict(self.__scdataTestX))
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16, 8))
        plt.title("Model")
        plt.xlabel("Date", fontsize = 15)
        plt.ylabel("Close Price USD ($)", fontsize = 15)
        plt.plot(train["Close"])
        plt.plot(valid[['Close', 'Pred']])
        plt.legend(['Train', 'Val', 'Predictions'], loc="lower right")
        plt.show()

    def graph(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16, 8))
        plt.title("Model")
        plt.xlabel("Date", fontsize = 15)
        plt.ylabel("Close Price USD ($)", fontsize = 15)
        plt.plot(self.__data["Close"])
        plt.show()

    def predict_nextday(self):
        last_test = self.__scdataTestX[-1]
        last_test = np.reshape(last_test, (1, self.__day, 5))
        prediction = self.__scalerY.inverse_transform(self.predict(last_test))
        return prediction
    


