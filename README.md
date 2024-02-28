# ForecastingAPI

This is a Python Flask API designed to forecast stock prices using various forecasting models such as ARIMA, MA(3), LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit). The user can provide a stock ticker of company that the user wants to predict and select the desired forecasting model from a dropdown list, and upon submitting the request, the API returns the forecasted stock prices along with a graphical representation for better visualization.


## Setup
1. Clone this repository to your local machine.
```sh
git clone https://github.com/yourusername/stock-forecast-api.git
```

2. Navigate to the project directory.
  ```sh
cd filepath/ForecastingAPI
```

3. Install the required dependencies.
```sh
pip install -r requirements.txt
```
 
4. Run the Flask server.
```sh
python app
```

## Usage
Once the Flask server is running, you can access the API using any HTTP client (e.g., web browser, Postman). 


## Notes
Ensure that the stock symbol provided is valid and matches the format expected by the data source.
The accuracy of the forecasts may vary based on various factors including the chosen model, data quality, and market conditions.

## Disclaimer
This API is for educational and demonstration purposes only. Stock price forecasting is inherently uncertain and should not be solely relied upon for making investment decisions. Always conduct thorough research or consult with a financial advisor before making any investment choices.

## Meta
Distributed under the MIT license. See `LICENSE` for more information.

[https://github.com/BanasiakB/](https://github.com/BanasiakB/)

## Contributing
1. Fork it (<https://github.com/BanasiakB/ForecastingAPI/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
