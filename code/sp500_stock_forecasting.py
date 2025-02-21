import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# set ticker to S&P 500 index
ticker = "^GSPC"

# define the time period for the data
start_date = "2000-01-01"
end_date = "2005-01-01"

# download historical data from yahoo finance
stock_data = yf.download(ticker, start=start_date, end=end_date)

# check the first few rows of the dataset
print(stock_data.head())

# visualize the closing price
plt.figure(figsize=(16, 8))
plt.plot(stock_data['Close'], label=ticker)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('S&P 500 Closing Stock Price')
plt.show()

# calculate moving averages for visualization
ma_short = stock_data['Close'].rolling(window=50).mean()
ma_long = stock_data['Close'].rolling(window=200).mean()

# plot the moving averages
plt.figure(figsize=(16, 8))
plt.plot(stock_data['Close'], label='Closing Price')
plt.plot(ma_short, label='Short MA (50)')
plt.plot(ma_long, label='Long MA (200)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('S&P 500 Stock Price with Moving Averages')
plt.legend()
plt.show()

# perform adfuller test to check stationarity
adf = adfuller(stock_data['Close'], autolag='aic')

# print adf results
print('ADF Statistic:', adf[0])
print('p-value:', adf[1])
print('Critical Values:', adf[4])

# interpret adfuller test results
if adf[1] <= 0.05:
    print("reject the null hypothesis: time series is stationary")
else:
    print("fail to reject the null hypothesis: time series is non-stationary")

# decompose the time series if non-stationary
decom = seasonal_decompose(stock_data['Close'], period=20)
decom.plot()
plt.show()

# log transform the data for better stationarity
df_log = np.log(stock_data['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()

# plot log-transformed data with rolling mean and std deviation
plt.title('Log Transformed Closing Price')
plt.plot(moving_avg, label="Rolling Mean")
plt.plot(std_dev, label="Rolling Std Dev", color='black')
plt.legend()
plt.show()

# split data into training and test sets
train, test = df_log[:int(len(df_log)*0.8)], df_log[int(len(df_log)*0.8):]

# plot training and test data
plt.figure(figsize=(10,6))
plt.plot(df_log, label='Log-transformed Price')
plt.plot(train, label='Training Set', color='blue')
plt.plot(test, label='Test Set', color='red')
plt.legend()
plt.show()

# use auto-arima to find the best ARIMA parameters
autoarima_model = auto_arima(train, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1, seasonal=False,
                             trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(autoarima_model.summary())

# fit ARIMA model
model = ARIMA(train, order=autoarima_model.order)
model_fit = model.fit()

# forecast and calculate confidence intervals
forecast_result = model_fit.get_forecast(steps=len(test))
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

# extract confidence intervals
lower_conf = conf_int.iloc[:, 0]
upper_conf = conf_int.iloc[:, 1]

# plot the forecast with confidence intervals
plt.figure(figsize=(10,5))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, color='blue', label='Actual Price')
plt.plot(test.index, forecast, color='orange', label='Predicted Price')
plt.fill_between(test.index, lower_conf, upper_conf, color='k', alpha=0.1)
plt.title('S&P 500 Stock Price Prediction with ARIMA')
plt.xlabel('Date')
plt.ylabel('Log-transformed Price')
plt.legend(loc='best')
plt.show()

# evaluate model performance
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs(forecast - test)/np.abs(test))

# print performance metrics
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')

