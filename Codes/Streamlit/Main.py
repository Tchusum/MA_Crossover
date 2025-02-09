import streamlit as st

st.title("Backtesting of a Moving Average Crossover Trading Strategy")

st.header("Definition")

st.markdown(r"""
The **Short-Term Exponential Moving Average (EMA)** is defined as:
""")

st.latex(r"""
\mathrm{EMA}_t^S = \alpha_S P_t^{c} + (1 - \alpha_S) \mathrm{EMA}_{t-1}^S
""")

st.markdown(r"""
Similarly, the **Long-Term Exponential Moving Average (EMA)** is given by:
""")

st.latex(r"""
\mathrm{EMA}_t^L = \alpha_L P_t^{c} + (1 - \alpha_L) \mathrm{EMA}_{t-1}^L
""")

st.markdown(r"""
where:
- $\mathrm{EMA}_t^S$ : Short-Term Exponential Moving Average at time $t$.
- $\mathrm{EMA}_t^L$ : Long-Term Exponential Moving Average at time $t$.
- $P_t^{c}$ : Closing price of the asset at time $t$.
- $\alpha_S$ : Smoothing factor for the short EMA, defined as $\alpha_S = \frac{2}{N_S + 1}$, where $N_S$ is the period of the short EMA.
- $\alpha_L$ : Smoothing factor for the long EMA, defined as $\alpha_L = \frac{2}{N_L + 1}$, where $N_L$ is the period of the long EMA.
- $\mathrm{EMA}_{t-1}^S$ : Previous value of the short-term EMA.
- $\mathrm{EMA}_{t-1}^L$ : Previous value of the long-term EMA.
""")

st.markdown(r"""
The **Moving Average Convergence Divergence (MACD)** is computed as:
""")

st.latex(r"""
\mathrm{MACD}_t = \mathrm{EMA}_t^S - \mathrm{EMA}_t^L
""")

st.markdown(r"""
The MACD provides a measure of momentum by indicating the relationship between the short-term and long-term EMAs. A positive MACD suggests an uptrend, while a negative MACD indicates a downtrend.
""")

st.header("Strategy")

st.markdown(r"""
- **Go long** at time $t+1$ if $MACD_t > 0$ and $MACD_{t-1} < 0$  
- **Go short** at time $t+1$ if $MACD_t < 0$ and $MACD_{t-1} > 0$
""")