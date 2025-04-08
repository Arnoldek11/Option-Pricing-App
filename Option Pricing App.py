import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Model", page_icon="📊", layout="wide")
st.title("\U0001F4CA Black-Scholes Pricing Model")

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

col1, col2 = st.columns(2)
col1.markdown("""
<div style='background-color:#b2fab4;padding:20px;border-radius:10px;text-align:center;'>
    <h3>CALL Value</h3>
    <h1 style='color:green;'>${:.2f}</h1>
</div>
""".format(call_price), unsafe_allow_html=True)

col2.markdown("""
<div style='background-color:#ffcdd2;padding:20px;border-radius:10px;text-align:center;'>
    <h3>PUT Value</h3>
    <h1 style='color:red;'>${:.2f}</h1>
</div>
""".format(put_price), unsafe_allow_html=True)

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

fig1, ax1 = plt.subplots()
sns.heatmap(call_matrix, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2), ax=ax1, cmap="YlGnBu")
ax1.set_title("Call Price Heatmap")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility")
col1.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.heatmap(put_matrix, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2), ax=ax2, cmap="YlGnBu")
ax2.set_title("Put Price Heatmap")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility")
col2.pyplot(fig2)