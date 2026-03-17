# Jessica Landwersiek — Portfolio

**Physicist → Applied ML → Production Systems**

Scientist and applied ML practitioner with a background in experimental nuclear physics (Jefferson Lab, DOE-funded) and a current focus on quantitative finance and production machine learning systems. I build things that extract real signals from noisy data — whether that data is particle collision events or options market prices.

## Featured: The Decay Lab

**[thedecaylab.com](https://thedecaylab.com)** — A production quantitative trading research platform I built from the ground up.

- **Backtest engine** with 7 futures strategies (ORB, EMA, SMA, VWAP, RSI, IBS, EMA+VWAP) and options credit/debit spreads across 14 years of historical data
- **Strategy builder** — free-form rule engine where users define custom entry conditions using any combination of indicators, with configurable exits (TP/SL/trailing/time stop), concurrent position management, and portfolio combination with correlation analysis
- **TIER 1.1 options engine** matched to 7.5 years of research — Black-Scholes with tuned IV model, SABR for longer-dated options, 66% bid-ask slippage model (ORATS methodology)
- **Statistical validation** on every backtest: Monte Carlo simulation (5,000 bootstraps), walk-forward testing, Deflated Sharpe Ratio, VIX regime breakdown
- **Parameter sweep** engine for grid search optimization across any strategy configuration
- **Chain snapshot backtest** using real IQFeed bid/ask/IV/delta data instead of synthetic pricing
- **Paper trading dashboard** with live IQFeed market data, automated trade execution, and performance tracking
- **Production infrastructure**: Flask + Gunicorn + Nginx + Let's Encrypt, 5+ systemd services, automated IQFeed watchdog, daily data pipeline cron jobs

**Live trading record**: Deploying a 1-DTE SPX call credit spread strategy validated by 1,878 days of backtesting (99.1% WR, Sharpe 8.03). Every trade published to [@TheDecayLab](https://t.me/TheDecayLab) on Telegram.

**Stack**: Python, Flask, Gunicorn, Nginx, SQLite, IQFeed, Tradier API, systemd, Bootstrap, Reportlab

---

## Research Projects

### 1. Stock Price Forecasting using ARIMA
Predicts S&P 500 index prices using ARIMA models and historical data analysis. Evaluates model performance against real market data.

[View Project](https://github.com/jlandwersiek/portfolio/tree/stock-price-forecasting)

### 2. Monte Carlo Techniques for Statistical Analysis
Demonstrates Monte Carlo methods for estimating π, computing n-dimensional hypersphere volumes, and analyzing Poisson and Gaussian distributions. Uses acceptance/rejection sampling and Box-Muller transformations.

[View Project](https://github.com/jlandwersiek/portfolio/tree/monte_carlo_statistics)

### 3. Polarization Data Analysis
Analyzes polarization data from nuclear physics experiments to forecast unseen data points using polynomial regression and chi-squared fitting. Extrapolates from limited measurements with quantified uncertainty.

[View Project](https://github.com/jlandwersiek/portfolio/tree/polarization-analysis)

### 4. Power Grid and Energy Efficiency Forecasting
Models energy consumption trends using time-series analysis, K-means clustering for substation stability classification, and ARIMA-based demand forecasting with interactive network visualizations.

[View Project](https://github.com/jlandwersiek/portfolio/tree/energy-analysis)

---

## Other Projects

- [**spy-0dte-gap-trading-dashboard**](https://github.com/jlandwersiek/spy-0dte-gap-trading-dashboard) — SPY 0-DTE gap analysis with sector-level scoring and market internals
- [**credit-spread-dashboard**](https://github.com/jlandwersiek/credit-spread-dashboard) — Options credit spread analytics with premium efficiency modeling
- [**trend-score-app**](https://github.com/jlandwersiek/trend-score-app) — Multi-factor trend scoring system
- [**spy-analyzer**](https://github.com/jlandwersiek/spy-analyzer) — SPY options analysis platform
- [**stock-move-dashboard**](https://github.com/jlandwersiek/stock-move-dashboard) — Stock movement analysis and visualization

---

## Technical Skills

**Languages**: Python (advanced), SQL, C++, R

**ML & Statistics**: Regression, classification, clustering, Monte Carlo simulation, Bayesian inference, walk-forward validation, chi-squared analysis, Black-Scholes & SABR options pricing, Deflated Sharpe Ratio

**Frameworks**: Scikit-learn, NumPy, Pandas, SciPy, Flask, Streamlit, Reportlab

**Infrastructure**: IQFeed, Tradier API, IBKR, Gunicorn, Nginx, systemd, SQLite, Let's Encrypt

**Tooling**: Claude Code, Git, Jupyter, Matplotlib, Bootstrap

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
