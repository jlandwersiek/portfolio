# Jessica Landwersiek — Portfolio

**Physicist → Quantitative Finance → Production Systems**

Background in experimental nuclear physics (Jefferson Lab, DOE-funded). Currently building and running automated trading systems on futures and options. I work with noisy data — particle collisions, market prices — and build models that hold up out of sample.

## Featured: The Decay Lab

**[thedecaylab.com](https://thedecaylab.com)** — Quantitative trading research platform. Built from scratch, running live.

- **Backtest engine** covering 7 futures strategies (ORB, EMA, SMA, VWAP, RSI, IBS, EMA+VWAP) and options credit/debit spreads across 14 years of historical data
- **Strategy builder** with free-form rule engine, configurable exits, concurrent position management, and portfolio combination with correlation analysis
- **Options engine** — Black-Scholes with tuned IV model, SABR for longer-dated, 66% bid-ask slippage model (ORATS methodology), matched to 7.5 years of data
- **Statistical validation** on every backtest: Monte Carlo simulation (5,000 bootstraps), walk-forward testing, Deflated Sharpe Ratio, VIX regime breakdown
- **Parameter sweep** engine for grid search optimization across any strategy configuration
- **Paper trading dashboard** with live market data and performance tracking
- **Production infrastructure**: Flask + Gunicorn + Nginx + Let's Encrypt, systemd services, daily data pipelines, Telegram alerting

**Live trading**: Running 3x MNQ (Micro E-mini Nasdaq) on Tradovate with server-side stop protection, non-blocking limit order entry, and automated monitoring. Full trade journal published at [thedecaylab.com/journal](https://thedecaylab.com/journal).

**Stack**: Python, Flask, Gunicorn, Nginx, SQLite, Schwab API, Tradovate API, systemd, Bootstrap

---

## Research Projects

### 1. Polarization Data Analysis
Extrapolation of R_sum from deep inelastic scattering experiments using polynomial regression and chi-squared fitting. Covariance matrix error propagation. Data from Jefferson Lab.

[View Project](./polarization-analysis)

### 2. Monte Carlo Techniques for Statistical Analysis
Monte Carlo methods for estimating pi, computing n-dimensional hypersphere volumes, and simulating Poisson and Gaussian distributions. Acceptance/rejection sampling and Box-Muller transformations.

[View Project](./monte-carlo-statistics)

### 3. Stock Price Forecasting using ARIMA
S&P 500 price forecasting with ARIMA. Time-series decomposition, stationarity testing, model selection.

[View Project](./stock-price-forecasting)

### 4. Power Grid Stability and Forecasting
Power grid simulation with demand fluctuation modeling, K-means clustering for substation stability classification, and ARIMA demand forecasting.

[View Project](./energy-analysis)

---

## Technical Skills

**Languages**: Python, SQL, C++, R

**Statistics & Modeling**: Monte Carlo simulation, walk-forward validation, chi-squared analysis, Bayesian inference, Black-Scholes & SABR options pricing, regression, clustering

**Libraries**: NumPy, Pandas, SciPy, Scikit-learn, Flask, Matplotlib, Reportlab

**Infrastructure**: Schwab API, Tradovate API, Gunicorn, Nginx, systemd, SQLite, Let's Encrypt

**Tools**: Claude Code, Git, Jupyter, Bootstrap

---

## Education

**M.S. in Physics** — Florida International University (2024)
*Experimental nuclear physics, statistical modeling, Monte Carlo methods. Research at Jefferson Lab.*

**B.S. in Physics & B.A. in Natural and Applied Sciences** — Florida International University (2021)

---

## Contact

- **Web**: [thedecaylab.com](https://thedecaylab.com)
- **Telegram**: [@TheDecayLab](https://t.me/TheDecayLab)
- **LinkedIn**: [linkedin.com/in/jessicarland](https://www.linkedin.com/in/jessicarland)
- **Email**: jessicarland@outlook.com
