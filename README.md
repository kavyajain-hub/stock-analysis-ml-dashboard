# 📈 Stock Analysis ML Dashboard

A comprehensive Python-based stock analysis tool with machine learning predictions, technical indicators, and advanced risk analysis for Indian stock markets (NSE).

## 🚀 Features

### Price Prediction
- **Linear Regression** - Baseline trend prediction
- **ARIMA Model** - Time-series forecasting
- **Ensemble Prediction** - Combined model average

### Technical Indicators
- **RSI (Relative Strength Index)** - Identifies overbought/oversold conditions
- **MACD** - Moving Average Convergence Divergence with signal line
- **Bollinger Bands** - Volatility-based price bands
- **Moving Averages** - 20-day, 50-day, and 200-day MAs

### Risk Analysis
- **Volatility** - Daily and annualized volatility metrics
- **Sharpe Ratio** - Risk-adjusted return measurement
- **Max Drawdown** - Worst historical peak-to-trough decline
- **52-Week High/Low** - Annual price range analysis

### Advanced Visualization
- **Candlestick Charts** - OHLC price visualization with trend lines
- **Volume Analysis** - Trading volume with color-coded bars
- **RSI Chart** - With overbought/oversold zones (70/30 levels)
- **MACD Chart** - With signal line and histogram
- **Daily Returns** - Volatility visualization with color-coded bars
- **Dashboard Summary** - Key metrics display panel

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Usage

Run the script:
```bash
python stock_analyzer.py
```

Enter a stock ticker when prompted (use `.NS` suffix for NSE stocks):
```
Examples:
- IRFC.NS (Indian Railway Finance Corporation)
- ADANIPOWER.NS (Adani Power)
- TATAMOTORS.NS (Tata Motors)
- RELIANCE.NS (Reliance Industries)
```

## 📊 Output

The system provides:
1. **Comprehensive Report** with:
   - Current market status
   - Next-day price predictions (Linear Regression & ARIMA)
   - Technical indicator values
   - Risk metrics and classification

2. **6-Panel Advanced Dashboard**:
   - **Price Analysis** - Candlestick chart with Linear/ARIMA trends, 50/200-day MAs, and Bollinger Bands
   - **Trading Volume** - Color-coded volume bars (green=up, red=down)
   - **RSI Strength** - Relative Strength Index with overbought/oversold zones
   - **MACD Momentum** - MACD line, signal line, and histogram
   - **Daily Volatility** - Daily returns percentage visualization
   - **Dashboard Summary** - Key metrics including price, volatility, Sharpe ratio, and RSI

## 📈 Sample Output

```
============================================================
  COMPREHENSIVE ANALYSIS REPORT: IRFC.NS
============================================================

📊 CURRENT MARKET STATUS:
   Current Price: ₹122.63
   52-Week High: ₹148.74
   52-Week Low: ₹109.51

🔮 PRICE PREDICTIONS (Next Trading Day):
   Linear Regression: ₹118.15
   ARIMA Model: ₹122.23
   ➜ Ensemble Average: ₹120.19

📈 TECHNICAL INDICATORS:
   RSI (14): 51.74 (Neutral)
   MACD: 1.8775 (Bullish 📈)

⚠️ RISK ANALYSIS:
   Daily Volatility: 2.90%
   Annual Volatility: 46.02%
   Sharpe Ratio: 0.336
   Max Drawdown: -48.41%
   ➜ Risk Level: HIGH RISK 🔴
```

## 🛠️ Technologies Used

- **yfinance** - Real-time stock data fetching from Yahoo Finance
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations and array operations
- **scikit-learn** - Linear regression model and metrics
- **statsmodels** - ARIMA time-series modeling
- **matplotlib** - Advanced data visualization and charting
- **mplfinance** - Financial-specific charting tools

## 📸 Dashboard Preview

![Stock Analysis Dashboard](https://github.com/user-attachments/assets/f9ba6194-e036-40ce-80d7-950930154e25)

*6-panel advanced dashboard showing price trends, volume, RSI, MACD, daily volatility, and key metrics summary*

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. Stock predictions are based on historical data and statistical models. **DO NOT** use this as the sole basis for investment decisions. Always:
- Conduct thorough research
- Consult financial advisors
- Consider your risk tolerance
- Diversify your portfolio

Past performance does not guarantee future results.

## 📝 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👤 Author

Kavya Jain

## 🌟 Acknowledgments

- yfinance for stock data API
- statsmodels for ARIMA implementation
- The open-source community
