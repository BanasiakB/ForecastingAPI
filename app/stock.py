import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models import LstmNetwork, GruNetwork, ARIMA, MA3


TEST_SAMPLES = 50
LEARNING_PERIOD = 50


class Stock:
    def __init__(self,  ticker:str, time_step:str, predict:str) -> None:
        self.ticker = ticker
        self.time_step = time_step
        self.data = self.get_data()
        self.predict = predict
        self.predict_ix = self.data.columns.get_loc(self.predict)
        self.y_returns = None
        self.scaler = StandardScaler()

    def forecast(self, methods: list) -> dict[dict]:
        results = {}
        score = {}
        if 'MA' in methods:
            results['MA'], score['MA'] = self.MA3_forecast()
        if 'ARIMA' in methods:
            results['ARIMA'], score['ARIMA'] = self.ARIMA_forecast()
        
        if 'LSTM' in methods or 'GRU' in methods:
            X_train, y_train, X_test, X_pred = self.split_data()
        
            if 'LSTM' in methods:
                results['LSTM'], score['LSTM'] = self.LSTM_forecast(X_train, y_train, X_test, X_pred)

            if 'GRU' in methods:
                results['GRU'], score['GRU'] = self.GRU_forecast(X_train, y_train, X_test, X_pred)
        return results, score
    
    def LSTM_forecast(self, X_train, y_train, X_test, X_pred):
        print(X_train.shape, y_train.shape, X_test.shape, X_pred.shape)
        model = LstmNetwork()
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

    def MA3_forecast(self):
        X = [self.data.iloc[i-LEARNING_PERIOD:i, ][self.predict] for i in range(len(self.data)-TEST_SAMPLES, len(self.data))]
        y_true = self.data.iloc[-TEST_SAMPLES:, ][self.predict].values
        X_pred = self.data.iloc[-LEARNING_PERIOD:, ][self.predict]
        
        days = self.time_step[-1] == 'd'
        y = [MA3(pd.Series(x), days) for x in X]
        score = {}
        score["MAE"] = np.round(mean_absolute_error(y, y_true), 5)
        score["MSE"] = np.round(mean_squared_error(y, y_true), 5)

        y_pred = [MA3(pd.Series(X_pred), days)]
        score['result'] = np.round(y_pred[0], 5)
        return self.combine_with_time(y + y_pred), score
        
    def ARIMA_forecast(self):
        X = [self.data.iloc[i-LEARNING_PERIOD:i, ][self.predict] for i in range(len(self.data)-TEST_SAMPLES, len(self.data))]
        y_true = self.data.iloc[-TEST_SAMPLES:, ][self.predict].values
        X_pred = self.data.iloc[-LEARNING_PERIOD:, ][self.predict]
        days = self.time_step[-1] == 'd'
        y = [ARIMA(pd.Series(x, days)) for x in X]
        score = {}
        score["MAE"] = np.round(mean_absolute_error(y, y_true), 5)
        score["MSE"] = np.round(mean_squared_error(y, y_true), 5)
        
        y_pred = [ARIMA(pd.Series(X_pred), days)]
        score['result'] = np.round(y_pred[0], 5)
        return self.combine_with_time(y + y_pred), score
        
    def split_data(self) -> tuple[np.ndarray]:
        X_train, y_train = self.get_training_data()
        X_test = self.get_test_data()
        X_pred = self.get_x_pred()
        return X_train, y_train, X_test, X_pred

    def get_data(self) -> pd.DataFrame:
        data = yf.Ticker(self.ticker).history(interval=self.time_step, period=self._max_period())
        if not len(data):
            raise ValueError('Ticker is incorrect.')
        return data.tail(1000)
    
    def _max_period(self) -> str:
        max_periods = {'1m': '7d', '2m': '60d', '5m': '60d', '15m': '60d', '30m': '60d', '60m': '730d', '90m': '60d', 
                       '1h': '730d', '1d': 'max', '5d': 'max'} 
        return max_periods[self.time_step]

    def historical_data(self) -> dict:
        historical_data = self.data.iloc[-min(100, len(self.data)):][self.predict]
        historical_data.index = historical_data.index
        return historical_data.to_dict()

    def combine_with_time(self, data: list) -> dict:
        ix = self.data.iloc[-TEST_SAMPLES:]
        ix_next = ix.index.max() + pd.Timedelta(self.time_step)
        pred_index = list(ix.index) + [ix_next]
        return {key: value for key, value in zip(pred_index, data)}

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
