import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go

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
# Greeks Calculation
# ----------------------
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    theta = theta_call if option_type == 'call' else theta_put
    rho = rho_call if option_type == 'call' else rho_put

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }

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
# Display Call and Put Prices - Styled
# ----------------------
st.markdown("""
<style>
.dark-box {
    background-color: #1a1a1a;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: #e0e0e0;
    border: 1px solid #2a2a2a;
}
.dark-box h1 {
    font-size: 42px;
    margin: 0;
    color: #fdea45;
}
.dark-box h1.put {
    color: #93d7c0;
}
.dark-box h3 {
    font-size: 20px;
    margin-bottom: 10px;
    font-weight: 500;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.markdown(f"<div class='dark-box'><h3>Call Option</h3><h1>${call_price:.2f}</h1></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='dark-box'><h3>Put Option</h3><h1 class='put'>${put_price:.2f}</h1></div>", unsafe_allow_html=True)

# ----------------------
# Greeks Styled Display
# ----------------------
call_greeks = black_scholes_greeks(S, K, T, r, sigma, 'call')
put_greeks = black_scholes_greeks(S, K, T, r, sigma, 'put')

st.markdown("""
<style>
.greeks-box {
    background-color: #1a1a1a;
    padding: 20px;
    border-radius: 15px;
    color: #e0e0e0;
    border: 1px solid #2a2a2a;
    margin-bottom: 10px;
}
.greeks-box h4 {
    margin-bottom: 10px;
    color: #fdea45;
}
.greeks-box.put h4 {
    color: #93d7c0;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='greeks-box'><h4>📈 Call Option Greeks</h4>", unsafe_allow_html=True)
    for g, val in call_greeks.items():
        st.markdown(f"**{g}**: {val:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='greeks-box put'><h4>📉 Put Option Greeks</h4>", unsafe_allow_html=True)
    for g, val in put_greeks.items():
        st.markdown(f"**{g}**: {val:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Greeks Bar Chart
# ----------------------
st.markdown("### Visual Comparison of Option Greeks")

greek_names = list(call_greeks.keys())
greek_df = pd.DataFrame({
    "Greek": greek_names,
    "Call Option": [call_greeks[g] for g in greek_names],
    "Put Option": [put_greeks[g] for g in greek_names]
})

greek_melted = greek_df.melt(id_vars="Greek", var_name="Type", value_name="Value")

fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0e0e0e')
sns.barplot(data=greek_melted, x="Greek", y="Value", hue="Type", palette=["#fdea45", "#93d7c0"], ax=ax)
ax.set_title("Option Greeks", color="white")
ax.tick_params(colors='lightgray')
ax.set_xlabel("", color="white")
ax.set_ylabel("Value", color="white")
ax.legend(frameon=False)
st.pyplot(fig)

# ----------------------
# Heatmap Controls
# ----------------------
st.markdown("### Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters.")

st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Min Spot Price", value=80.0, step=1.0)
max_spot = st.sidebar.number_input("Max Spot Price", value=120.0, step=1.0)
min_vol = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1, step=0.01)
max_vol = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.0, 0.5, step=0.01)

# ----------------------
# 3D Surface of a Greek
# ----------------------
greek_choice = st.selectbox("🧭 Choose a Greek to Explore in 3D", ["Delta", "Gamma", "Vega"])

spot_vals = np.linspace(min_spot, max_spot, 30)
vol_vals = np.linspace(min_vol, max_vol, 30)
spot_grid, vol_grid = np.meshgrid(spot_vals, vol_vals)

greek_surface = np.zeros_like(spot_grid)

for i in range(len(vol_vals)):
    for j in range(len(spot_vals)):
        greeks_here = black_scholes_greeks(spot_grid[i, j], K, T, r, vol_grid[i, j])
        greek_surface[i, j] = greeks_here[greek_choice]

fig3d = go.Figure(data=[go.Surface(
    z=greek_surface,
    x=spot_grid,
    y=vol_grid,
    colorscale='Cividis',
    colorbar=dict(title=greek_choice)
)])

fig3d.update_layout(
    title=f'{greek_choice} Surface',
    scene=dict(
        xaxis_title='Spot Price',
        yaxis_title='Volatility',
        zaxis_title=greek_choice,
        xaxis=dict(backgroundcolor="black", color="white"),
        yaxis=dict(backgroundcolor="black", color="white"),
        zaxis=dict(backgroundcolor="black", color="white")
    ),
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    font_color="white",
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig3d, use_container_width=True)

# ----------------------
# Heatmap Section
# ----------------------
spot_range = np.linspace(min_spot, max_spot, 20)
vol_range = np.linspace(min_vol, max_vol, 20)

call_matrix = np.zeros((len(vol_range), len(spot_range)))
put_matrix = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_matrix[i, j] = black_scholes(s, K, T, r, v, 'call')
        put_matrix[i, j] = black_scholes(s, K, T, r, v, 'put')

col1, col2 = st.columns(2)

plt.style.use("dark_background")
fig1, ax1 = plt.subplots(facecolor="#0e0e0e")
sns.heatmap(call_matrix, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2), ax=ax1, cmap="cividis", cbar_kws={"label": "Call Price"})
ax1.set_title("Call Price Heatmap", color="white")
ax1.set_xlabel("Spot Price", color="white")
ax1.set_ylabel("Volatility", color="white")
ax1.tick_params(colors='lightgray')
col1.pyplot(fig1)

fig2, ax2 = plt.subplots(facecolor="#0e0e0e")
sns.heatmap(put_matrix, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2), ax=ax2, cmap="cividis", cbar_kws={"label": "Put Price"})
ax2.set_title("Put Price Heatmap", color="white")
ax2.set_xlabel("Spot Price", color="white")
ax2.set_ylabel("Volatility", color="white")
ax2.tick_params(colors='lightgray')
col2.pyplot(fig2)
