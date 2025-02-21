## S&P 500 Stock Price Forecasting using ARIMA

### Overview:  
This project demonstrates how to predict the closing stock price of the S&P 500 index using ARIMA (AutoRegressive Integrated Moving Average). The goal is to forecast future stock prices based on historical data and evaluate the model's performance using common error metrics.

### Objective:  
  - Predict future stock prices for the S&P 500 index.
  - Evaluate model accuracy and interpret ARIMA results.

### Project Structure:
```
├── code/  
│   ├── sp500_stock_forecasting.py  # Main script for data preprocessing, modeling, and forecasting  
├── plots/  
│   ├── sp500_closing_stock_price.png  # Historical closing stock price  
│   ├── sp500_stock_price_ma.png  # Stock price with moving averages  
│   ├── sp500_seasonal_decompose.png  # Seasonal decomposition of stock  
│   ├── log_closing_price.png # Log transformed closing price with rolling mean and standard deviation  
│   ├── sp500_train_test.png # Training and test data  
│   ├── sp500_arima_prediction.png # Forecasting of future stock closing prices with confidence interval  
└── README.md
```

### Technologies Used:  
  - Python (Pandas, NumPy, Matplotlib, Statsmodels, pmdarima, Scikit-Learn)
  - Yahoo Finance API (yfinance)

### Running the Project:  
To run this project:  

  1. Install the required libraries:
```
pip install yfinance pmdarima statsmodels scikit-learn
```
  2. Run the main script:
```
python code/sp500_stock_forecasting.py
```
