#stock price analysis using auto_arima to find ARIMA parameters.
#80% train data.
#ticker and dates can be changed.



import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#name of stock
ticker = "MSFT"

#timeline
start_date = "2000-01-01"
end_date = "2005-01-01" 

#download from yahoo finance
stock_data = yf.download(ticker, start=start_date, end=end_date)

#create csv file from downloaded data
stock_data.to_csv("MSFT_stock_data.csv")

#load datasets
colnames=['Date','Adj Close','Close','High','Low','Open','Volume']

df = pd.read_csv("MSFT_stock_data.csv", skiprows=3, names=colnames, parse_dates=['Date'])

df['Date']

#check for missing entries
print(df.isnull().sum())

#forward fill missing entries, if any
df.fillna(method='ffill', inplace=True)

#visualize the closing price
plt.figure(figsize=(16, 8))
plt.plot(df['Date'],df['Close'], label=ticker)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('MSFT Closing Stock Price')
plt.show()

df['Date']

#calculate and visualize the moving averages
ma_short = df['Close'].rolling(window=50).mean()
ma_long = df['Close'].rolling(window=200).mean()
rolmean = df['Close'].rolling(window=12).mean()
rolstd = df['Close'].rolling(window=12).std()
plt.figure(figsize=(16, 8))
plt.plot(df['Date'],df['Close'], label='Closing Price')
plt.plot(df['Date'],ma_short, label='Short MA (50)')
plt.plot(df['Date'],ma_long, label='Long MA (200)')
plt.plot(df['Date'],rolmean, label='Rolling Mean (12)')
plt.plot(df['Date'],rolstd, label='Rolling St. Dev (12)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Microsoft Stock Price with Moving Averages')
plt.legend()
plt.show()

#perform ADF test to check for stationarity
adf = adfuller(df['Close'], autolag='aic')

#print results
print('ADF Statistic:', adf[0])
print('p-value:', adf[1])
print('Number of lags:', adf[2])
print('Number of Observations:', [3])
print('Critical Values:', adf[4])

#interpret the results
if adf[1] <= 0.05:
    print("Reject the null hypothesis: Time series is stationary")
else:
    print("Fail to reject the null hypothesis: Time series is non-stationary")

#if series non-stationary, decompose to separate trend and seasonality 
decom = seasonal_decompose(df['Close'], period=20)
fig = plt.figure()  
fig = decom.plot()  
fig.set_size_inches(16, 9)

#eliminate trends for non-stationary series
#take log data
df_log = np.log(df['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()

plt.title('Moving Average (12) - Logarithmic')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()

#split train data
train, test = df_log[5:int(len(df_log)*0.8)], df_log[int(len(df_log)*0.8):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.plot(df_log, 'steelblue', label='Train data')
plt.plot(test, 'pink', label='Test data')
plt.legend()


#run autoarima to find arima parameters
autoarima = auto_arima(train, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(autoarima.summary())
autoarima.plot_diagnostics(figsize=(15,8))
print(autoarima.order)
order = autoarima.order
plt.show()



model = ARIMA(train, order=order ) 
model_fit = model.fit()  
print(model_fit.summary())

forecast_result = model_fit.get_forecast(steps=len(test))  # Get forecast for the length of test data
forecast = forecast_result.predicted_mean  # Extract the forecasted values
conf_int = forecast_result.conf_int(alpha=0.05)  # Get the confidence intervals

#extract lower and upper confidence intervals
lower_conf = conf_int.iloc[:, 0]  
upper_conf = conf_int.iloc[:, 1]

#print forecast and confidence intervals
print(forecast)
print(conf_int)

#evaluate the model
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs(forecast - test)/np.abs(test))

print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE: ',(mape))

#make series connected to test data
fc_series = pd.Series(forecast, index=test.index)
lower_series = pd.Series(lower_conf, index=test.index)
upper_series = pd.Series(upper_conf, index=test.index)

#plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(df['Date'][5:int(len(df['Date'])*0.8)],train, label='training data')
plt.plot(df['Date'][int(len(df['Date'])*0.8):],test, color = 'blue', label='Actual Stock Price')
plt.plot(df['Date'][int(len(df['Date'])*0.8):],fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(df['Date'][int(len(df['Date'])*0.8):], lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (log)')
plt.legend(loc='best', fontsize=8)
plt.show()


