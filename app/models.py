from darts import TimeSeries
from darts.models import AutoARIMA, NaiveMovingAverage
from darts.utils.missing_values import fill_missing_values
import numpy as np
from pandas import Series
from keras import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense, Flatten


class LstmNetwork:
    def __init__(self) -> None:
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = LSTM_model(input_shape=(X_train.shape[1], 1))
        self.model.compile(loss ='mean_squared_error', 
                  optimizer ='adam', 
                  metrics=["mse"])

        self.model.fit(X_train, y_train, epochs=15)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if self.model:
            return self.model.evaluate(X_test, y_test)
        else:
            raise BrokenPipeError("Model was tested before training")

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        if self.model:
            return self.model.predict(X_pred)
        else:
            raise BrokenPipeError("Model was used for prediction before training")
        
class GruNetwork:
    def __init__(self) -> None:
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = GRU_model(input_shape=(X_train.shape[1], 1))
        self.model.compile(loss ='mean_squared_error', 
                  optimizer ='adam', 
                  metrics=["mse"])

        self.model.fit(X_train, y_train, epochs=15)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if self.model:
            return self.model.evaluate(X_test, y_test)
        else:
            raise BrokenPipeError("Model was tested before training")

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        if self.model:
            return self.model.predict(X_pred)
        else:
            raise BrokenPipeError("Model was used for prediction before training")
        

def ARIMA(data: Series, days: bool=False) -> float:
    if days:
        X = fill_missing_values(TimeSeries.from_series(data, fill_missing_dates=True, freq='B'))
    else:
        X = fill_missing_values(TimeSeries.from_series(data, fill_missing_dates=True))
    model = AutoARIMA().fit(X)
    return model.predict(n=1).values().flatten()[0]

def MA3(data: Series, days: bool=False) -> float:
    if days:
        X = fill_missing_values(TimeSeries.from_series(data, fill_missing_dates=True, freq='B'))
    else:
        X = fill_missing_values(TimeSeries.from_series(data, fill_missing_dates=True))
    model = NaiveMovingAverage(input_chunk_length=3).fit(X)
    return model.predict(n=1).values().flatten()[0]


def LSTM_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, 
                  return_sequences=True, 
                  input_shape=input_shape))
    
    model.add(LSTM(30, 
                  return_sequences=True))
    
    model.add(Dropout(.1))
    model.add(LSTM(30))
    model.add(Dropout(.1))
    
    # model.add(Flatten())
    # model.add(Dense(25, activation='relu'))
    # model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    return model

def GRU_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, 
                  return_sequences=True, 
                  input_shape=input_shape))
    
    model.add(GRU(70, 
                  return_sequences=True))
    model.add(GRU(70))

    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    return model


