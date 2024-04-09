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

    def read_model(self, method: str):
        try:
            model = keras.models.load_model(f"app/models/BTC-USD_{method}_{self.predict}_{self.time_step}.keras")
        except ValueError:
            raise ValueError('Error occured during reading a model')
        return model

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        prices = self.data[-LEARNING_PERIOD-TEST_SAMPLES-1:][self.predict].values  
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
            score['result'] = np.round(float(y_pred[-1]), 5)

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

    def combine_with_time(self, data: np.ndarray) -> dict:
        ix = self.data.iloc[-TEST_SAMPLES:]
        ix_next = ix.index.max() + pd.Timedelta(self.time_step)
        pred_index = list(ix.index) + [ix_next]
        return {key: float(value) for key, value in zip(pred_index, data)}
