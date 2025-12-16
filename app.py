"""
Options Pricing Calculator using Black-Scholes Model
Simple Streamlit app for calculating option prices and Greeks
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Black-Scholes implementation
class BlackScholes:
    """Black-Scholes model for pricing European options"""
    
    def __init__(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be positive")
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        d1, d2 = self.d1(), self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self):
        d1, d2 = self.d1(), self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    def delta(self, option_type='call'):
        d1 = self.d1()
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type='call'):
        d1, d2 = self.d1(), self.d2()
        common = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            return common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
    
    def vega(self):
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1()) / 100
    
    def rho(self, option_type='call'):
        d2 = self.d2()
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    # Implied Volatility using Newton-Raphson
    def implied_volatility(self, market_price, option_type='call'):
        # Initial guess
        sigma = 0.5
        max_iterations = 100
        tolerance = 0.0001
        
        for i in range(max_iterations):
            # Calculate price with current sigma
            bs_temp = BlackScholes(self.S, self.K, self.T, self.r, sigma)
            if option_type == 'call':
                price = bs_temp.call_price()
            else:
                price = bs_temp.put_price()
            
            # Calculate vega
            vega = bs_temp.vega() * 100
            
            # Check if converged
            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma
            
            # Newton-Raphson update
            if vega != 0:
                sigma = sigma + diff / vega
            else:
                break
            
            # Keep sigma positive
            if sigma <= 0:
                sigma = 0.01
        
        return sigma


# Helper function to get live data
@st.cache_data(ttl=300)
def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

# Calculate historical volatility
@st.cache_data(ttl=300)
def calc_historical_vol(ticker, days=30):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=f'{days+10}d')
        if len(data) > days:
            returns = np.log(data['Close'] / data['Close'].shift(1))
            vol = returns.std() * np.sqrt(252)
            return vol
        return None
    except:
        return None

# Streamlit UI
st.set_page_config(page_title="Options Pricing Calculator", page_icon="📊", layout="wide")

st.title("📊 Options Pricing Calculator")
st.write("Calculate option prices and Greeks using the Black-Scholes model")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Live Data Section
st.sidebar.subheader("📡 Live Data (Optional)")
use_live_data = st.sidebar.checkbox("Use Live Market Data")

if use_live_data:
    # Popular Indian stocks/indices
    ticker_options = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "INFOSYS": "INFY.NS"
    }
    
    selected_ticker = st.sidebar.selectbox("Select Stock/Index", list(ticker_options.keys()))
    ticker_symbol = ticker_options[selected_ticker]
    
    live_price = get_live_price(ticker_symbol)
    if live_price:
        st.sidebar.success(f"Live Price: ₹{live_price:.2f}")
        default_spot = float(live_price)
    else:
        st.sidebar.warning("Could not fetch live data, using default")
        default_spot = 21500.0
    
    # Get historical volatility
    hist_vol_30 = calc_historical_vol(ticker_symbol, 30)
    hist_vol_60 = calc_historical_vol(ticker_symbol, 60)
    
    if hist_vol_30:
        st.sidebar.info(f"30-day HV: {hist_vol_30*100:.2f}%")
    if hist_vol_60:
        st.sidebar.info(f"60-day HV: {hist_vol_60*100:.2f}%")
    
    # Show historical volatility chart
    if st.sidebar.checkbox("Show HV Chart"):
        try:
            stock = yf.Ticker(ticker_symbol)
            data = stock.history(period='6mo', interval='1d')
            if data is None or data.empty or len(data) <= 30:
                st.sidebar.warning("Not enough data for HV (need > 30 daily bars).")
            else:
                # Calculate rolling 30-day volatility
                returns = np.log(data['Close'] / data['Close'].shift(1))
                rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rolling_vol.index, rolling_vol, 'b-', linewidth=1.5)
                
                # draw current input volatility if available
                sigma_line = st.session_state.get("sigma_for_line")
                if sigma_line is not None:
                    ax.axhline(sigma_line*100, color='red', linestyle='--', label=f'Current Input: {sigma_line*100:.1f}%')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Volatility (%)')
                ax.set_title('30-Day Historical Volatility')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.sidebar.pyplot(fig)
        except:
            st.sidebar.error("Could not fetch data for chart")
else:
    default_spot = 21500.0

st.sidebar.markdown("---")

S = st.sidebar.number_input("Spot Price (S)", min_value=1.0, value=default_spot, step=100.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=default_spot, step=100.0)

# Expiry date instead of days slider (simpler for users)
today = datetime.today().date()
default_expiry = today + timedelta(days=30)
expiry_date = st.sidebar.date_input("Expiry Date", value=default_expiry)

# Compute days and T (keep it simple, calendar days)
try:
    days = (expiry_date - today).days
except Exception:
    days = 30
if days < 1:
    days = 1
T = days / 365

r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 15.0, 7.0, 0.5) / 100
sigma = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 18.0, 1.0) / 100
# store for HV chart (so it can read before this block executes again)
st.session_state["sigma_for_line"] = sigma

st.sidebar.markdown("---")
st.sidebar.info(f"Time to expiry: {T:.4f} years ({days} days)")

st.sidebar.markdown("---")
st.sidebar.header("💰 Purchase Prices (Optional)")
st.sidebar.write("Enter your purchase price to see P/L analysis")

call_purchase = st.sidebar.number_input(
    "Call Purchase Price (₹)", 
    min_value=0.0, 
    value=0.0, 
    step=10.0,
    help="Leave as 0 to skip P/L analysis"
)

put_purchase = st.sidebar.number_input(
    "Put Purchase Price (₹)", 
    min_value=0.0, 
    value=0.0, 
    step=10.0,
    help="Leave as 0 to skip P/L analysis"
)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Heatmap Parameters")
st.sidebar.write("Customize the scenario analysis ranges")

min_spot = st.sidebar.number_input(
    "Min Spot Price",
    min_value=1.0,
    value=S * 0.85,
    step=100.0,
    help="Minimum spot price for heatmap"
)

max_spot = st.sidebar.number_input(
    "Max Spot Price",
    min_value=min_spot + 100,
    value=S * 1.15,
    step=100.0,
    help="Maximum spot price for heatmap"
)

min_vol = st.sidebar.slider(
    "Min Volatility (%)",
    min_value=1.0,
    max_value=sigma * 100,
    value=sigma * 50,
    step=1.0,
    help="Minimum volatility for heatmap"
) / 100

max_vol = st.sidebar.slider(
    "Max Volatility (%)",
    min_value=sigma * 100,
    max_value=100.0,
    value=min(sigma * 150, 100.0),
    step=1.0,
    help="Maximum volatility for heatmap"
) / 100

# Calculate prices and greeks
try:
    bs = BlackScholes(S, K, T, r, sigma)
    
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    # Moneyness Indicator
    moneyness = (S - K) / K * 100
    if abs(moneyness) < 2:
        money_status = "ATM (At The Money)"
        money_color = "🟡"
    elif S > K:
        money_status = f"ITM (In The Money) +{moneyness:.1f}%"
        money_color = "🟢"
    else:
        money_status = f"OTM (Out of The Money) {moneyness:.1f}%"
        money_color = "🔴"
    
    st.info(f"{money_color} **Call Option:** {money_status}")
    st.info(f"{money_color} **Put Option:** {'ITM' if S < K else 'OTM'} (opposite of Call)")
    
    st.markdown("---")
    
    # Display prices
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📞 Call Price", f"₹{call_price:.2f}")
        if call_purchase > 0:
            call_pl = call_price - call_purchase
            st.metric("Call P/L", f"₹{call_pl:.2f}", delta=f"{call_pl:.2f}")
    with col2:
        st.metric("📉 Put Price", f"₹{put_price:.2f}")
        if put_purchase > 0:
            put_pl = put_price - put_purchase
            st.metric("Put P/L", f"₹{put_pl:.2f}", delta=f"{put_pl:.2f}")
    with col3:
        # Break-even prices
        call_breakeven = K + call_price
        put_breakeven = K - put_price
        st.metric("Call Break-even", f"₹{call_breakeven:.2f}")
        st.metric("Put Break-even", f"₹{put_breakeven:.2f}")
    
    st.markdown("---")
    
    # Implied Volatility Calculator
    st.subheader("🔍 Implied Volatility Calculator")
    st.write("Enter market option price to calculate implied volatility")
    
    col1, col2 = st.columns(2)
    with col1:
        market_call_price = st.number_input("Market Call Price", min_value=0.0, value=0.0, step=10.0)
        if market_call_price > 0:
            try:
                iv_call = bs.implied_volatility(market_call_price, 'call')
                st.success(f"Implied Vol (Call): **{iv_call*100:.2f}%**")
                st.write(f"Current input vol: {sigma*100:.2f}%")
                diff = (iv_call - sigma) * 100
                if diff > 0:
                    st.info(f"Market is pricing {diff:.2f}% higher volatility")
                else:
                    st.info(f"Market is pricing {abs(diff):.2f}% lower volatility")
            except:
                st.error("Could not calculate IV")
    
    with col2:
        market_put_price = st.number_input("Market Put Price", min_value=0.0, value=0.0, step=10.0)
        if market_put_price > 0:
            try:
                iv_put = bs.implied_volatility(market_put_price, 'put')
                st.success(f"Implied Vol (Put): **{iv_put*100:.2f}%**")
                st.write(f"Current input vol: {sigma*100:.2f}%")
                diff = (iv_put - sigma) * 100
                if diff > 0:
                    st.info(f"Market is pricing {diff:.2f}% higher volatility")
                else:
                    st.info(f"Market is pricing {abs(diff):.2f}% lower volatility")
            except:
                st.error("Could not calculate IV")
    
    st.markdown("---")
    
    # Greeks
    st.subheader("Greeks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Call Greeks**")
        call_delta = bs.delta('call')
        gamma = bs.gamma()
        call_theta = bs.theta('call')
        vega = bs.vega()
        call_rho = bs.rho('call')
        
        greeks_call = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Theta (yearly)', 'Theta (daily)', 'Vega', 'Rho'],
            'Value': [
                f"{call_delta:.4f}",
                f"{gamma:.6f}",
                f"{call_theta:.2f}",
                f"{call_theta/365:.4f}",
                f"{vega:.4f}",
                f"{call_rho:.4f}"
            ]
        })
        st.dataframe(greeks_call, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**Put Greeks**")
        put_delta = bs.delta('put')
        put_theta = bs.theta('put')
        put_rho = bs.rho('put')
        
        greeks_put = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Theta (yearly)', 'Theta (daily)', 'Vega', 'Rho'],
            'Value': [
                f"{put_delta:.4f}",
                f"{gamma:.6f}",
                f"{put_theta:.2f}",
                f"{put_theta/365:.4f}",
                f"{vega:.4f}",
                f"{put_rho:.4f}"
            ]
        })
        st.dataframe(greeks_put, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Payoff diagram
    st.subheader("Payoff Diagram at Expiration")
    
    stock_range = np.linspace(K * 0.7, K * 1.3, 100)
    call_payoff = np.maximum(stock_range - K, 0) - call_price
    put_payoff = np.maximum(K - stock_range, 0) - put_price
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Call payoff
    ax1.plot(stock_range, call_payoff, 'g-', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(K, color='red', linestyle='--', alpha=0.5, label=f'Strike={K}')
    ax1.fill_between(stock_range, call_payoff, 0, where=(call_payoff > 0), alpha=0.3, color='green')
    ax1.fill_between(stock_range, call_payoff, 0, where=(call_payoff < 0), alpha=0.3, color='red')
    ax1.set_xlabel('Stock Price at Expiration')
    ax1.set_ylabel('Profit/Loss')
    ax1.set_title('Call Option Payoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Put payoff
    ax2.plot(stock_range, put_payoff, 'r-', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(K, color='red', linestyle='--', alpha=0.5, label=f'Strike={K}')
    ax2.fill_between(stock_range, put_payoff, 0, where=(put_payoff > 0), alpha=0.3, color='green')
    ax2.fill_between(stock_range, put_payoff, 0, where=(put_payoff < 0), alpha=0.3, color='red')
    ax2.set_xlabel('Stock Price at Expiration')
    ax2.set_ylabel('Profit/Loss')
    ax2.set_title('Put Option Payoff')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # P/L Heatmap Section
    if call_purchase > 0 or put_purchase > 0:
        st.markdown("---")
        st.subheader("🔥 P/L Heatmap - Scenario Analysis")
        st.write("See how your profit/loss changes across different spot prices and volatilities")
        
        # Create heatmap data using custom parameters
        spot_range = np.linspace(min_spot, max_spot, 20)
        vol_range = np.linspace(min_vol, max_vol, 20)
        
        # Calculate P/L for Call and Put
        call_pl_matrix = np.zeros((len(vol_range), len(spot_range)))
        put_pl_matrix = np.zeros((len(vol_range), len(spot_range)))
        
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                try:
                    bs_temp = BlackScholes(spot, K, T, r, vol)
                    call_theoretical = bs_temp.call_price()
                    put_theoretical = bs_temp.put_price()
                    call_pl_matrix[i, j] = call_theoretical - call_purchase if call_purchase > 0 else 0
                    put_pl_matrix[i, j] = put_theoretical - put_purchase if put_purchase > 0 else 0
                except:
                    call_pl_matrix[i, j] = 0
                    put_pl_matrix[i, j] = 0
        
        # Create side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Call Option Heatmap
        if call_purchase > 0:
            im1 = ax1.imshow(call_pl_matrix, cmap='RdYlGn', aspect='auto', 
                          extent=[spot_range[0], spot_range[-1], vol_range[0]*100, vol_range[-1]*100],
                          origin='lower')
            
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('P/L (₹)', rotation=270, labelpad=20, fontsize=11)
            
            # Add values in cells (sample every 4th cell to avoid overcrowding)
            for i in range(0, len(vol_range), 4):
                for j in range(0, len(spot_range), 4):
                    text_color = 'white' if abs(call_pl_matrix[i, j]) > np.max(np.abs(call_pl_matrix)) * 0.5 else 'black'
                    ax1.text(spot_range[j], vol_range[i]*100, f'{call_pl_matrix[i, j]:.1f}',
                            ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
            
            # Add breakeven line
            contour1 = ax1.contour(spot_range, vol_range*100, call_pl_matrix, levels=[0], colors='black', linewidths=2)
            ax1.clabel(contour1, inline=True, fontsize=9, fmt='Breakeven')
            
            ax1.set_xlabel('Spot Price (₹)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
            ax1.set_title(f'CALL Option P/L\nPurchase: ₹{call_purchase:.2f} | Current: ₹{call_price:.2f}', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax1.plot(S, sigma*100, 'b*', markersize=20, label='Current Position', markeredgecolor='white', markeredgewidth=2)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
        else:
            ax1.text(0.5, 0.5, 'No Call Position\nEnter Purchase Price', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14, color='gray')
            ax1.set_title('CALL Option P/L', fontsize=13, fontweight='bold')
        
        # Put Option Heatmap
        if put_purchase > 0:
            im2 = ax2.imshow(put_pl_matrix, cmap='RdYlGn', aspect='auto', 
                          extent=[spot_range[0], spot_range[-1], vol_range[0]*100, vol_range[-1]*100],
                          origin='lower')
            
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('P/L (₹)', rotation=270, labelpad=20, fontsize=11)
            
            # Add values in cells (sample every 4th cell)
            for i in range(0, len(vol_range), 4):
                for j in range(0, len(spot_range), 4):
                    text_color = 'white' if abs(put_pl_matrix[i, j]) > np.max(np.abs(put_pl_matrix)) * 0.5 else 'black'
                    ax2.text(spot_range[j], vol_range[i]*100, f'{put_pl_matrix[i, j]:.1f}',
                            ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
            
            # Add breakeven line
            contour2 = ax2.contour(spot_range, vol_range*100, put_pl_matrix, levels=[0], colors='black', linewidths=2)
            ax2.clabel(contour2, inline=True, fontsize=9, fmt='Breakeven')
            
            ax2.set_xlabel('Spot Price (₹)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
            ax2.set_title(f'PUT Option P/L\nPurchase: ₹{put_purchase:.2f} | Current: ₹{put_price:.2f}', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax2.plot(S, sigma*100, 'b*', markersize=20, label='Current Position', markeredgecolor='white', markeredgewidth=2)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No Put Position\nEnter Purchase Price', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='gray')
            ax2.set_title('PUT Option P/L', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if call_purchase > 0:
                st.metric("Call Max Profit", f"₹{np.max(call_pl_matrix):.2f}")
            else:
                st.metric("Call Max Profit", "N/A")
        with col2:
            if call_purchase > 0:
                st.metric("Call Max Loss", f"₹{np.min(call_pl_matrix):.2f}")
            else:
                st.metric("Call Max Loss", "N/A")
        with col3:
            if put_purchase > 0:
                st.metric("Put Max Profit", f"₹{np.max(put_pl_matrix):.2f}")
            else:
                st.metric("Put Max Profit", "N/A")
        with col4:
            if put_purchase > 0:
                st.metric("Put Max Loss", f"₹{np.min(put_pl_matrix):.2f}")
            else:
                st.metric("Put Max Loss", "N/A")
        
        with st.expander("📊 How to Read This Heatmap"):
            st.write("""
            **This heatmap shows your profit/loss for different market scenarios:**
            
            - **X-axis (Horizontal):** Different spot prices (stock/index levels)
            - **Y-axis (Vertical):** Different volatility levels
            - **Colors:** 
              - 🟢 **Green** = Profit (theoretical price > your purchase price)
              - 🟡 **Yellow** = Near breakeven
              - 🔴 **Red** = Loss (theoretical price < your purchase price)
            - **Blue Star (⭐):** Your current position
            - **Black Line:** Breakeven points (zero P/L)
            
            **Example:** If the spot moves to ₹22,000 and volatility increases to 25%, 
            find that cell to see your expected P/L.
            
            **Use Case:** Before buying an option, check different scenarios to understand your risk.
            """)
    
    # Option Strategies
    st.markdown("---")
    st.subheader("📊 Option Strategies")
    st.write("Popular trading strategies with payoff diagrams")
    
    strategy = st.selectbox("Select Strategy", [
        "Bull Call Spread",
        "Bear Put Spread", 
        "Long Straddle",
        "Iron Condor"
    ])
    
    # Strategy parameters
    col1, col2 = st.columns(2)
    with col1:
        lower_strike = st.number_input("Lower Strike", min_value=1.0, value=K-500, step=100.0)
    with col2:
        upper_strike = st.number_input("Upper Strike", min_value=lower_strike+100, value=K+500, step=100.0)
    
    # Calculate strategy
    spot_prices = np.linspace(K * 0.7, K * 1.3, 100)
    
    if strategy == "Bull Call Spread":
        # Buy lower strike call, sell higher strike call
        bs_lower = BlackScholes(S, lower_strike, T, r, sigma)
        bs_upper = BlackScholes(S, upper_strike, T, r, sigma)
        
        cost = bs_lower.call_price() - bs_upper.call_price()
        
        payoff = []
        for spot in spot_prices:
            long_call = max(spot - lower_strike, 0)
            short_call = max(spot - upper_strike, 0)
            payoff.append(long_call - short_call - cost)
        
        st.info(f"Net Cost: ₹{cost:.2f} | Max Profit: ₹{(upper_strike-lower_strike-cost):.2f} | Max Loss: ₹{cost:.2f}")
    
    elif strategy == "Bear Put Spread":
        # Buy higher strike put, sell lower strike put
        bs_lower = BlackScholes(S, lower_strike, T, r, sigma)
        bs_upper = BlackScholes(S, upper_strike, T, r, sigma)
        
        cost = bs_upper.put_price() - bs_lower.put_price()
        
        payoff = []
        for spot in spot_prices:
            long_put = max(upper_strike - spot, 0)
            short_put = max(lower_strike - spot, 0)
            payoff.append(long_put - short_put - cost)
        
        st.info(f"Net Cost: ₹{cost:.2f} | Max Profit: ₹{(upper_strike-lower_strike-cost):.2f} | Max Loss: ₹{cost:.2f}")
    
    elif strategy == "Long Straddle":
        # Buy call and put at same strike
        bs_k = BlackScholes(S, K, T, r, sigma)
        cost = bs_k.call_price() + bs_k.put_price()
        
        payoff = []
        for spot in spot_prices:
            call_payoff = max(spot - K, 0)
            put_payoff = max(K - spot, 0)
            payoff.append(call_payoff + put_payoff - cost)
        
        st.info(f"Net Cost: ₹{cost:.2f} | Profit if spot moves beyond: ₹{K-cost:.2f} or ₹{K+cost:.2f}")
    
    elif strategy == "Iron Condor":
        # Sell call spread and put spread
        k1 = lower_strike - 500
        k2 = lower_strike
        k3 = upper_strike
        k4 = upper_strike + 500
        
        bs1 = BlackScholes(S, k1, T, r, sigma)
        bs2 = BlackScholes(S, k2, T, r, sigma)
        bs3 = BlackScholes(S, k3, T, r, sigma)
        bs4 = BlackScholes(S, k4, T, r, sigma)
        
        credit = (bs2.put_price() - bs1.put_price()) + (bs3.call_price() - bs4.call_price())
        
        payoff = []
        for spot in spot_prices:
            put_spread = max(k2 - spot, 0) - max(k1 - spot, 0)
            call_spread = max(spot - k3, 0) - max(spot - k4, 0)
            payoff.append(credit - put_spread - call_spread)
        
        st.info(f"Net Credit: ₹{credit:.2f} | Max Loss: ₹{(500-credit):.2f}")
    
    # Plot strategy
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spot_prices, payoff, 'b-', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(S, color='orange', linestyle='--', alpha=0.5, label=f'Current Spot={S}')
    ax.fill_between(spot_prices, payoff, 0, where=(np.array(payoff) > 0), alpha=0.3, color='green')
    ax.fill_between(spot_prices, payoff, 0, where=(np.array(payoff) < 0), alpha=0.3, color='red')
    ax.set_xlabel('Spot Price at Expiration')
    ax.set_ylabel('Profit/Loss')
    ax.set_title(f'{strategy} Payoff Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Info section
    st.markdown("---")
    with st.expander("ℹ️ About the Model"):
        st.write("""
        **Black-Scholes Model** is used for pricing European options. 
        
        **Greeks:**
        - **Delta**: Change in option price for ₹1 change in stock price
        - **Gamma**: Rate of change of Delta
        - **Theta**: Time decay (how much value the option loses per day)
        - **Vega**: Sensitivity to volatility changes
        - **Rho**: Sensitivity to interest rate changes
        
        **Assumptions:**
        - European options (can only be exercised at expiration)
        - No dividends
        - Constant volatility and interest rate
        - Log-normal stock price distribution
        """)

except ValueError as e:
    st.error(f"Error: {e}")
    st.warning("Please check your input values.")

# Footer
st.markdown("---")
st.markdown("Built with Python, NumPy, SciPy, and Streamlit")
