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
- **Candlestick Charts** - OHLC price visualization
- **Volume Analysis** - Trading volume with color-coded bars
- **RSI Chart** - With overbought/oversold zones
- **MACD Chart** - With signal line and histogram

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

2. **4-Panel Dashboard**:
   - Price chart with trend lines and moving averages
   - Trading volume visualization
   - RSI indicator chart
   - MACD indicator chart

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

- **yfinance** - Stock data fetching
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **scikit-learn** - Linear regression model
- **statsmodels** - ARIMA time-series modeling
- **matplotlib** - Visualization
- **mplfinance** - Financial charting

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


