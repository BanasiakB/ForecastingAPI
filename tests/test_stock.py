import pytest
from app.stock import Stock
import pandas as pd
import keras

@pytest.fixture
def stock_instance():
    return Stock('1m', 'Open')

def test_stock_atributes(stock_instance):
    assert stock_instance.time_step == '1m'
    assert isinstance(stock_instance.data, pd.DataFrame) 
    assert stock_instance.predict == 'Open'

def test_get_data(stock_instance):
    data = stock_instance._get_data()
    assert isinstance(data, pd.DataFrame) 
    assert not data.empty
    assert 'Open' in data.columns
    assert 'Close' in data.columns
    assert 'High' in data.columns
    assert 'Low' in data.columns

def test_forecast_without_methods(stock_instance):
    assert stock_instance.forecast([]) == ({}, {})

def test_read_model(stock_instance):
    model_path = 'app/models/BTC-USD_GRU_Open_1m.keras'
    model = stock_instance.read_model('GRU')
    assert isinstance(model, keras.models.Model)

def test_read_not_existing_model(stock_instance):
    with pytest.raises(OSError):
        stock_instance.read_model('non_existing_method')

def test_read_model_without_method(stock_instance):
    with pytest.raises(ValueError):
        stock_instance.read_model('')


