from flask import Flask, render_template
import plotly.graph_objs as go
from forms import ForecastForm
import datetime
from stock import Stock

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Secret Key'

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ForecastForm()
    if form.validate_on_submit():
        ticker = form.ticker.data
        time_step = form.time_step.data
        predict_option = form.predict_option.data
        methods = form.methods.data

        stock = Stock(ticker, time_step, predict_option)
        forecasts, scores = stock.forecast(methods=methods)
        forecasts['historical data'] = stock.historical_data()
        
        graph_data = []
        for key, values in forecasts.items():
            graph_data.append(go.Scatter(
            x=list(values.keys()),
            y=list(values.values()),
            mode='lines+markers',
            name=key)
            )

        graph_layout = dict(xaxis=dict(title='Date'), yaxis=dict(title=f'{predict_option} Price'))
        fig = go.Figure(data=graph_data, layout=graph_layout)
        graph = fig.to_html(full_html=False) 

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return render_template('result.html', ticker=ticker, time_step=time_step,  current_time=current_time, graph=graph, scores=scores)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
