from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SelectMultipleField, widgets
from wtforms.validators import DataRequired, Length


class ForecastForm(FlaskForm):
    ticker = StringField('Ticker Symbol', validators=[DataRequired(), Length(max=5)])
    time_step = SelectField('Time Step', choices=[('1m', '1 minute'), ('2m', '2 minutes'), ('5m', '5 minutes'), ('15m', '15 minutes'), 
                                                      ('30m', '30 minutes'), ('60m', '60 minutes'), ('90m', '90 minutes'), 
                                                      ('1h', '1 Hour'), ('1d', '1 day'), ('5d', '5 days')], validators=[DataRequired()])
    
    predict_option = SelectField('What do you want to ferecast?', choices=[('Open', 'Open Price'), ('Close', 'Close Price'), 
                                                         ('High', 'Highest Price'),('Low', 'Lowest Price')], validators=[DataRequired()])
    
    methods = SelectMultipleField('Methods', choices=[('ARIMA', 'ARIMA'), 
                                                      ('MA', 'Moving Average (3)'), 
                                                      ('LSTM', 'Long Short-Term Memory'), 
                                                      ('GRU', 'Gated Recurrent Unit')], 
                                                      widget=widgets.ListWidget(prefix_label=False), 
                                                      option_widget=widgets.CheckboxInput(), validators=[Length(min=1)])
