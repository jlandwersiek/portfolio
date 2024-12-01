# Jessica Landwersiek - Portfolio

Welcome to my portfolio! This repository showcases a variety of quantitative research projects, including time series forecasting, Monte Carlo simulations, and data analysis with Python.

## Projects

### 1. **Stock Price Forecasting using ARIMA**
   - **Description**: A time series analysis project where ARIMA (AutoRegressive Integrated Moving Average) was used to forecast stock prices based on historical data.
   - **Objective**: Predict future stock prices using ARIMA models and evaluate the model's performance.
   - **Technologies Used**:
     - Python (Pandas, NumPy, Statsmodels)
     - ARIMA for time series forecasting
   - **Key Results**:
     - Successful prediction of stock price trends with a focus on accuracy and model evaluation.
   - **Code**: [ARIMA Forecasting Code](./stock_price_forecasting.py)

### 2. **Monte Carlo Simulation for π Estimation**
   - **Description**: This paper demonstrates the application of Monte Carlo methods, specifically using acceptance/rejection and transformation techniques, to estimate the value of π and analyze Poisson and Gaussian distributions.
   - **Abstract**:  
     "Monte Carlo methods were employed using acceptance/rejection and transformation techniques. The value of π was found to be 3.1404 with an uncertainty of 0.0016. Histograms of Poisson deviates were plotted with μ= 1, μ=10.3, and μ=102.1, with a lower mean producing more accurate outcomes. Box-Muller transformations were used to find Gaussian deviates with reduced χ2 of 1.097 and 1.054 for perpendicular distributions."
   - **Technologies Used**:  
     - Python (NumPy, Matplotlib)
     - Monte Carlo methods (Acceptance/Rejection, Box-Muller Transformations)
   - **Access the Paper**: [Monte Carlo Paper (PDF)](./MonteCarloMethods.pdf)

### 3. **Polarization Data Analysis for Forecasting Unseen Data**
   - **Description**: Analyzing polarization data to forecast data from points unseen by detectors, leveraging statistical models and computational methods to improve prediction accuracy in scientific research.
   - **Objective**: Develop a model to estimate and predict values from regions where direct measurements are unavailable.
   - **Technologies Used**:
     - Python (SciPy, NumPy)
     - Statistical modeling
   - **Key Results**:
     - Successfully created models to extrapolate data and improve predictions from limited measurements.
   - **Code**: [Polarization Data Analysis Code](./polarization_data_analysis.py)

## Skills Highlighted
- **Programming Languages**: Python
- **Data Analysis**: ARIMA, Monte Carlo simulations, statistical modeling
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Time series forecasting
- **Tools**: Git, Jupyter Notebooks

## How to Use This Repository
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/portfolio.git
2. Navigate to the project folder:
   ```bash
   cd portfolio
3. Install dependencies (if necessary):
   ```bash
   pip install -r requirements.txt
4. Run the code (for any specific project):
   ```bash
   python stock_price_forecasting.py
