import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')

# ========================= DATA FETCHING =========================
def fetch_stock_data(ticker_symbol, period="2y"):
    """Fetch stock data with validation"""
    print(f"--- Fetching data for {ticker_symbol} ---")
    
    try:
        stock_data = yf.download(ticker_symbol, period=period, progress=False)
        if stock_data.empty:
            print("Error: No data found. Check ticker symbol.")
            return None
        
        if len(stock_data) < 100:
            print(f"Warning: Only {len(stock_data)} days of data available. Results may be unreliable.")
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ========================= TECHNICAL INDICATORS =========================
def calculate_technical_indicators(df):
    """Calculate RSI, MACD, Bollinger Bands, and other indicators"""
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Moving Averages
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

# ========================= RISK METRICS =========================
def calculate_risk_metrics(df):
    """Calculate Sharpe Ratio, Max Drawdown, and other risk metrics"""
    
    metrics = {}
    
    # Volatility (Annualized)
    metrics['volatility'] = df['Daily_Return'].std()
    metrics['annual_volatility'] = metrics['volatility'] * np.sqrt(252)
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    avg_return = df['Daily_Return'].mean()
    metrics['sharpe_ratio'] = (avg_return * 252) / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + df['Daily_Return']/100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    metrics['max_drawdown'] = drawdown.min()
    
    # Current Price Stats
    metrics['current_price'] = df['Close'].iloc[-1]
    metrics['52w_high'] = df['Close'].tail(252).max()
    metrics['52w_low'] = df['Close'].tail(252).min()
    
    # Current RSI & MACD
    metrics['current_rsi'] = df['RSI'].iloc[-1]
    metrics['current_macd'] = df['MACD'].iloc[-1]
    
    return metrics

# ========================= PREDICTION MODELS =========================
def linear_regression_model(df):
    """Linear Regression baseline model"""
    df_model = df.dropna().copy()
    df_model['Date_Ordinal'] = df_model.index.to_series().apply(lambda date: date.toordinal())
    
    X = df_model[['Date_Ordinal']]
    y = df_model['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Predict next day
    last_date = df_model.index[-1]
    next_date_ordinal = last_date.toordinal() + 1
    next_price_lr = model.predict([[next_date_ordinal]])[0]
    
    # Create trend line aligned with df_model index
    trend_predictions = pd.Series(model.predict(X), index=df_model.index)
    
    return next_price_lr, rmse, r2, trend_predictions

def arima_model(df):
    """ARIMA time-series model for better predictions"""
    try:
        df_model = df['Close'].dropna()
        
        # Fit ARIMA model (5,1,0) - you can tune these parameters
        model = ARIMA(df_model, order=(5, 1, 0))
        fitted_model = model.fit()
        
        # Forecast next day
        forecast = fitted_model.forecast(steps=1)
        next_price_arima = forecast.iloc[0]
        
        # In-sample predictions for plotting
        predictions = fitted_model.fittedvalues
        
        return next_price_arima, predictions
    except Exception as e:
        print(f"ARIMA model failed: {e}")
        return None, None

# ========================= VISUALIZATION =========================
def plot_advanced_dashboard(df, ticker_symbol, trend_line, arima_pred, metrics):
    """Create comprehensive 4-panel dashboard with SMART LAYOUT fixing"""
    
    # Use only rows without NaN for clean plotting
    df_clean = df.dropna()
    
    # --- FIX 1: INCREASE SIZE & USE CONSTRAINED LAYOUT ---
    # layout='constrained' automatically moves things so they don't overlap
    fig = plt.figure(figsize=(20, 18), layout='constrained')
    
    # Grid structure: 4 rows, 2 columns
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1])
    
    # --- PANEL 1: Candlestick Chart (Top, Spans both columns) ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot candlestick lines
    for idx in range(len(df_clean)):
        date = df_clean.index[idx]
        open_p, close_p = df_clean['Open'].iloc[idx], df_clean['Close'].iloc[idx]
        high_p, low_p = df_clean['High'].iloc[idx], df_clean['Low'].iloc[idx]
        
        color = 'green' if close_p >= open_p else 'red'
        ax1.plot([date, date], [low_p, high_p], color=color, linewidth=0.8, alpha=0.7)
        ax1.plot([date, date], [open_p, close_p], color=color, linewidth=3, alpha=0.9)
    
    # Trends
    if trend_line is not None and len(trend_line) > 0:
        ax1.plot(trend_line.index, trend_line.values, color='blue', linewidth=2, label='Linear Trend')
    if arima_pred is not None and len(arima_pred) > 0:
        ax1.plot(arima_pred.index, arima_pred.values, color='purple', linewidth=2, label='ARIMA Trend')
        
    ax1.plot(df_clean.index, df_clean['50_MA'], color='orange', linestyle='--', label='50-Day MA')
    ax1.fill_between(df_clean.index, df_clean['BB_Upper'], df_clean['BB_Lower'], color='gray', alpha=0.1, label='Bollinger Bands')
    
    ax1.set_title(f'{ticker_symbol} - Price Analysis', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    
    # --- PANEL 2: Volume (Row 2, Spans both columns) ---
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    colors_vol = ['green' if df_clean['Close'].iloc[i] >= df_clean['Open'].iloc[i] else 'red' for i in range(len(df_clean))]
    ax2.bar(df_clean.index, df_clean['Volume'], color=colors_vol, alpha=0.5)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    # --- PANEL 3: RSI (Row 3, Left) ---
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(df_clean.index, df_clean['RSI'], color='purple', linewidth=1.5)
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax3.set_ylabel('RSI', fontsize=12)
    ax3.set_title('RSI Strength', fontsize=12)
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(0, 100)
    
    # --- PANEL 4: MACD (Row 3, Right) ---
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax4.plot(df_clean.index, df_clean['MACD'], color='blue', label='MACD')
    ax4.plot(df_clean.index, df_clean['MACD_Signal'], color='red', label='Signal')
    colors_macd = ['green' if x >= 0 else 'red' for x in df_clean['MACD_Hist']]
    ax4.bar(df_clean.index, df_clean['MACD_Hist'], color=colors_macd, alpha=0.3)
    ax4.set_title('MACD Momentum', fontsize=12)
    ax4.grid(True, alpha=0.2)
    
    # --- PANEL 5: Volatility/Returns (Bottom Left) ---
    ax5 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax5.bar(df_clean.index, df_clean['Daily_Return'], color='gray', alpha=0.6)
    ax5.set_ylabel('Return %', fontsize=12)
    ax5.set_title('Daily Volatility', fontsize=12)
    ax5.grid(True, alpha=0.2)
    
    # --- PANEL 6: Summary Metrics (Bottom Right) ---
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off') # Hide axis box
    
    summary_text = (
        f"📊 DASHBOARD SUMMARY\n\n"
        f"Price:      ₹{metrics['current_price']:.2f}\n"
        f"Volatility: {metrics['annual_volatility']:.1f}%\n"
        f"Sharpe:     {metrics['sharpe_ratio']:.2f}\n"
        f"RSI:        {metrics['current_rsi']:.1f}\n\n"
        f"PREDICTIONS:\n"
        f"Linear:     Trend is {'UP' if trend_line.iloc[-1] > trend_line.iloc[0] else 'DOWN'}\n"
        f"ARIMA:      Short-term {'BULLISH' if arima_pred.iloc[-1] > df_clean['Close'].iloc[-1] else 'BEARISH'}"
    )
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace', 
             verticalalignment='center', bbox=dict(facecolor='wheat', alpha=0.3))

    # --- FINAL CLEANUP ---
    # Hide x-labels for all charts except the bottom ones to reduce clutter
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    # Add Main Title
    fig.suptitle(f'Advanced Stock Analysis: {ticker_symbol}', fontsize=20, fontweight='bold')
    
    print("Displaying fixed dashboard...")
    plt.show()

# ========================= MAIN ANALYSIS FUNCTION =========================
def analyze_stock(ticker_symbol):
    """Main analysis pipeline with all improvements"""
    
    # 1. Fetch Data
    stock_data = fetch_stock_data(ticker_symbol, period="2y")
    if stock_data is None:
        return
    
    # 2. Prepare DataFrame
    df = stock_data.copy()
    
    # Flatten multi-level columns if they exist (yfinance sometimes returns multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 3. Calculate Technical Indicators
    df = calculate_technical_indicators(df)
    
    # 4. Calculate Risk Metrics
    metrics = calculate_risk_metrics(df)
    
    # 5. Prediction Models
    print("\n--- Running Prediction Models ---")
    next_price_lr, rmse_lr, r2_lr, trend_line = linear_regression_model(df)
    next_price_arima, arima_pred = arima_model(df)
    
    # 6. Print Comprehensive Report
    print(f"\n{'='*60}")
    print(f"  COMPREHENSIVE ANALYSIS REPORT: {ticker_symbol}")
    print(f"{'='*60}")
    
    print(f"\n📊 CURRENT MARKET STATUS:")
    print(f"   Current Price: ₹{metrics['current_price']:.2f}")
    print(f"   52-Week High: ₹{metrics['52w_high']:.2f}")
    print(f"   52-Week Low: ₹{metrics['52w_low']:.2f}")
    
    print(f"\n🔮 PRICE PREDICTIONS (Next Trading Day):")
    print(f"   Linear Regression: ₹{next_price_lr:.2f} (R²={r2_lr:.3f}, RMSE=₹{rmse_lr:.2f})")
    if next_price_arima:
        print(f"   ARIMA Model: ₹{next_price_arima:.2f}")
        avg_prediction = (next_price_lr + next_price_arima) / 2
        print(f"   ➜ Ensemble Average: ₹{avg_prediction:.2f}")
    
    print(f"\n📈 TECHNICAL INDICATORS:")
    rsi_status = "Overbought ⚠️" if metrics['current_rsi'] > 70 else "Oversold 🟢" if metrics['current_rsi'] < 30 else "Neutral"
    print(f"   RSI (14): {metrics['current_rsi']:.2f} ({rsi_status})")
    macd_signal = "Bullish 📈" if metrics['current_macd'] > 0 else "Bearish 📉"
    print(f"   MACD: {metrics['current_macd']:.4f} ({macd_signal})")
    
    print(f"\n⚠️ RISK ANALYSIS:")
    print(f"   Daily Volatility: {metrics['volatility']:.2f}%")
    print(f"   Annual Volatility: {metrics['annual_volatility']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f} (Higher is better)")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}% (Worst decline)")
    
    risk_level = "HIGH RISK 🔴" if metrics['annual_volatility'] > 40 else "MEDIUM RISK 🟡" if metrics['annual_volatility'] > 25 else "LOW RISK 🟢"
    print(f"   ➜ Risk Level: {risk_level}")
    
    print(f"\n{'='*60}\n")
    
    # 7. Visualization
    print("Generating advanced dashboard...")
    plot_advanced_dashboard(df, ticker_symbol, trend_line, arima_pred, metrics)

# ========================= RUN IT =========================
if __name__ == "__main__":
    print("=" * 60)
    print("  ADVANCED STOCK ANALYSIS SYSTEM")
    print("  Features: ARIMA, RSI, MACD, Sharpe Ratio, Risk Analysis")
    print("=" * 60)
    user_input = input("\nEnter Stock Ticker (e.g., IRFC.NS, ADANIPOWER.NS, TATAMOTORS.NS): ").upper()
    analyze_stock(user_input)