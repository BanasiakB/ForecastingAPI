import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras

TEST_SAMPLES = 50
LEARNING_PERIOD = 50


class Stock:
    def __init__(self, time_step:str, predict:str) -> None:
        self.time_step = time_step
        self.data = self.get_data()
        self.predict = predict
        self.scaler = MinMaxScaler(feature_range=(0,1))
        #self.predict_ix = self.data.columns.get_loc(self.predict)
        #self.y_returns = None
        #self.scaler = StandardScaler()

    def read_model(self, method: str):
        try:
            model = keras.models.load_model(f"app/models/BTC-USD_{method}_{self.predict}_{self.time_step}.keras")
        except ValueError:
            raise ValueError('Error occured during reading a model')
        return model

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        prices = self.data[-LEARNING_PERIOD-TEST_SAMPLES:][self.predict].values  
        y_test = prices[-TEST_SAMPLES:]
        
        prices = np.reshape(prices, (-1,1)) 
        scaled_prices = self.scaler.fit_transform(prices)
        
        X = []
        for i in range(LEARNING_PERIOD, len(scaled_prices)):
            X.append(scaled_prices[i-LEARNING_PERIOD:i, 0])

        X = np.array(X) 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y_test

    def forecast(self, methods: list) -> dict[dict]:
        predictions = {}
        pred_score = {}

        X, y_test = self.prepare_data()
        for method in methods:
            model = self.read_model(method)
            y_pred = self.scaler.inverse_transform(model.predict(X))
            
            score = {}
            score["MAE"] = np.round(mean_absolute_error(y_pred[:-1], y_test), 5)
            score["MSE"] = np.round(mean_squared_error(y_pred[:-1], y_test), 5)
            score['result'] = np.round(y_pred[-1], 5)

            predictions[method] = self.combine_with_time(y_pred)
            pred_score[method] = score
        return predictions, pred_score
    
    def historical_data(self) -> dict:
        historical_data = self.data.iloc[-min(100, len(self.data)):][self.predict]
        historical_data.index = historical_data.index
        return historical_data.to_dict()

    def get_data(self) -> pd.DataFrame:
        data = yf.Ticker('BTC-USD').history(interval=self.time_step, period=self._max_period())
        if not len(data):
            raise ValueError('Error occured during downloading data.')
        return data.tail(1000)
    
    def _max_period(self) -> str:
        max_periods = {'1m': '7d', '2m': '60d', '5m': '60d', '15m': '60d', '30m': '60d', '60m': '730d', '90m': '60d', 
                       '1h': '730d', '1d': 'max', '5d': 'max'} 
        return max_periods[self.time_step]

    def combine_with_time(self, data: list) -> dict:
        ix = self.data.iloc[-TEST_SAMPLES:]
        ix_next = ix.index.max() + pd.Timedelta(self.time_step)
        pred_index = list(ix.index) + [ix_next]
        return {key: value for key, value in zip(pred_index, data)}


    def GRU_forecast(self, X_train, y_train, X_test, X_pred):
        model = GruNetwork()
        model.train(X_train=X_train, y_train=y_train)
        y = self.scaler.inverse_transform(model.predict(X_test))
        y_pred = self.scaler.inverse_transform(model.predict(X_pred))
        
        y = list(y.flatten())
        y_true = self.data.iloc[-TEST_SAMPLES:, ][self.predict].values

        score = {}
        score["MAE"] = np.round(mean_absolute_error(y, y_true), 5)
        score["MSE"] = np.round(mean_squared_error(y, y_true), 5)
        y_pred = list(y_pred.flatten())
        score['result'] = np.round(y_pred[0], 5)
        return self.combine_with_time(y + y_pred), score
    
    def split_data(self) -> tuple[np.ndarray]:
        X_train, y_train = self.get_training_data()
        X_test = self.get_test_data()
        X_pred = self.get_x_pred()
        return X_train, y_train, X_test, X_pred

    def get_x_pred(self):
        dataset = self.data.iloc[-LEARNING_PERIOD:, ][self.predict].values
        dataset = np.reshape(dataset, (-1,1)) 
        dataset = self.scaler.transform(dataset) 
        
        X_pred = np.array([dataset[:, 0]])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        return X_pred

    def get_training_data(self):
        dataset = self.data.iloc[:-TEST_SAMPLES, ][self.predict].values
        dataset = np.reshape(dataset, (-1,1)) 
        dataset = self.scaler.fit_transform(dataset) 
        
        X_train = []
        y_train = []
        for i in range(LEARNING_PERIOD, len(dataset)):
            X_train.append(dataset[i-LEARNING_PERIOD:i, 0])
            y_train.append(dataset[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        return X_train, y_train

    def get_test_data(self):
        dataset = self.data.iloc[-TEST_SAMPLES-LEARNING_PERIOD:, ][self.predict].values
        dataset = np.reshape(dataset, (-1,1)) 
        dataset = self.scaler.transform(dataset) 
        
        X_test = []
        for i in range(LEARNING_PERIOD, len(dataset)):
            X_test.append(dataset[i-LEARNING_PERIOD:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_test
