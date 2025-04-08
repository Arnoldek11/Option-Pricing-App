import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Model", layout="wide")
st.title("Black-Scholes Pricing Model")

# ----------------------
# Black-Scholes Function
# ----------------------
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("Black-Scholes Model")
st.sidebar.markdown("Created by:\n\n**Arnold Pienczykowski**")

S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.005)

# ----------------------
# Main Section
# ----------------------
st.markdown("### Black-Scholes Pricing Model")
data = {
    "Current Asset Price": S,
    "Strike Price": K,
    "Time to Maturity (Years)": T,
    "Volatility (σ)": sigma,
    "Risk-Free Interest Rate": r
}
st.dataframe([data])

call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

# ----------------------
# Display Call and Put Prices - Dark Box Style
# ----------------------
st.markdown("""
<style>
.dark-box {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}
.dark-box h1 {
    font-size: 48px;
    margin: 0;
}
.dark-box h3 {
    font-size: 24px;
    margin: 0;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.markdown(f"<div class='dark-box'><h3>Call Option</h3><h1 style='color:#00FF99;'>${call_price:.2f}</h1></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='dark-box'><h3>Put Option</h3><h1 style='color:#FF5555;'>${put_price:.2f}</h1></div>", unsafe_allow_html=True)


# ----------------------
# Heatmap Section
# ----------------------
st.markdown("### Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Min Spot Price", value=80.0, step=1.0)
max_spot = st.sidebar.number_input("Max Spot Price", value=120.0, step=1.0)

min_vol = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1, step=0.01)
max_vol = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.0, 0.5, step=0.01)

spot_range = np.linspace(min_spot, max_spot, 20)
vol_range = np.linspace(min_vol, max_vol, 20)

call_matrix = np.zeros((len(vol_range), len(spot_range)))
put_matrix = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_matrix[i, j] = black_scholes(s, K, T, r, v, 'call')
        put_matrix[i, j] = black_scholes(s, K, T, r, v, 'put')

col1, col2 = st.columns(2)

# Set dark theme for matplotlib
plt.style.use("dark_background")

# Custom dark figure settings
fig1, ax1 = plt.subplots(facecolor="#0e0e0e")
sns.heatmap(call_matrix, 
            xticklabels=np.round(spot_range, 1), 
            yticklabels=np.round(vol_range, 2), 
            ax=ax1, 
            cmap="cividis", 
            cbar_kws={"label": "Call Price"})
ax1.set_title("Call Price Heatmap", color="white")
ax1.set_xlabel("Spot Price", color="white")
ax1.set_ylabel("Volatility", color="white")
ax1.tick_params(colors='lightgray')
col1.pyplot(fig1)

fig2, ax2 = plt.subplots(facecolor="#0e0e0e")
sns.heatmap(put_matrix, 
            xticklabels=np.round(spot_range, 1), 
            yticklabels=np.round(vol_range, 2), 
            ax=ax2, 
            cmap="cividis", 
            cbar_kws={"label": "Put Price"})
ax2.set_title("Put Price Heatmap", color="white")
ax2.set_xlabel("Spot Price", color="white")
ax2.set_ylabel("Volatility", color="white")
ax2.tick_params(colors='lightgray')
col2.pyplot(fig2)

