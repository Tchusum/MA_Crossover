import streamlit as st
from pathlib import Path
import sys

PATH_PYTHON = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PATH_PYTHON))
import Import_Yfinance as yf_data
import Utils as utils
import MA_Backtest as MA
import Optimizer as opt

WINDOW_SIZE_MACD_ST = st.session_state["WINDOW_SIZE_MACD_ST"] 
WINDOW_SIZE_MACD_LT = st.session_state["WINDOW_SIZE_MACD_LT"]
WINDOW_SIZE_MACD_ST_RANGE = range(10, 30, 5)
WINDOW_SIZE_MACD_LT_RANGE = range(30, 100, 5)
YEARS = st.session_state["YEARS"]
CUTOFF_DATE = st.session_state["CUTOFF_DATE"]
RF = st.session_state["RF"]
CAPITAL = st.session_state["CAPITAL"]


st.title('Optimize Short and Long Window')

if st.button("Run Optimization"):

    best_comb = opt.main(CUTOFF_DATE, RF, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, WINDOW_SIZE_MACD_ST_RANGE, WINDOW_SIZE_MACD_LT_RANGE)

    st.subheader("Best Combination")

    st.dataframe(best_comb)