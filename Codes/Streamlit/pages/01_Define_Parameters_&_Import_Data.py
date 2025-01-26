import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Setup the path for imports
PATH_PYTHON = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PATH_PYTHON))
import Import_Yfinance as yf_data

# Initialize session state variables
if "years_imported" not in st.session_state:
    st.session_state["years_imported"] = None
if "WINDOW_SIZE_MACD_ST" not in st.session_state:
    st.session_state["WINDOW_SIZE_MACD_ST"] = 25
if "WINDOW_SIZE_MACD_LT" not in st.session_state:
    st.session_state["WINDOW_SIZE_MACD_LT"] = 95
if "YEARS" not in st.session_state:
    st.session_state["YEARS"] = 15
if "CUTOFF_DATE" not in st.session_state:
    st.session_state["CUTOFF_DATE"] = "2023-01-01"
if "RF" not in st.session_state:
    st.session_state["RF"] = 0.03
if "CAPITAL" not in st.session_state:
    st.session_state["CAPITAL"] = 200

# UI components
st.title("Trading Strategy Parameters")

WINDOW_SIZE_MACD_ST = st.slider("Select Short Window", 1, 50, st.session_state["WINDOW_SIZE_MACD_ST"])
WINDOW_SIZE_MACD_LT = st.slider("Select Long Window", 1, 200, st.session_state["WINDOW_SIZE_MACD_LT"])
YEARS = st.number_input("Number of years backdate", min_value=0, max_value=50, value=st.session_state["YEARS"], step=1)
CUTOFF_DATE = st.text_input("Cutoff Date (YYYY-MM-DD) between training and validation data", value=st.session_state["CUTOFF_DATE"])
RF = st.number_input("Define Risk-Free Rate", min_value=0.00, max_value=1.00, value=st.session_state["RF"], step=0.01)
CAPITAL = st.number_input("Define Capital per Trade", min_value=50, max_value=10000, value=st.session_state["CAPITAL"], step=50)

# Save parameters
col1a, col2a = st.columns([1, 1])

with col1a:
    if st.button("Save Parameters"):
        st.session_state["WINDOW_SIZE_MACD_ST"] = WINDOW_SIZE_MACD_ST
        st.session_state["WINDOW_SIZE_MACD_LT"] = WINDOW_SIZE_MACD_LT
        st.session_state["YEARS"] = YEARS
        try:
            datetime.strptime(CUTOFF_DATE, "%Y-%m-%d")
            st.session_state["CUTOFF_DATE"] = CUTOFF_DATE
        except ValueError:
            st.error("Invalid date format. Please use YYYY-MM-DD.")
        st.session_state["RF"] = RF
        st.session_state["CAPITAL"] = CAPITAL

        if st.session_state["years_imported"] is not None and st.session_state["YEARS"] != st.session_state["years_imported"]:
            st.error("You must import data again to reflect the changes in the number of years.")
        else:
            st.success("Successfully Saved Parameters")

with col2a:
    if st.button("Import Data"):
        try:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            INPUT_DAILY = BASE_DIR / "Input/Daily_Data/"
            INPUT_MAPPING = BASE_DIR / "Input/Mapping/"
            URL_SP = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            URL_TSX = 'https://en.wikipedia.org/wiki/S%26P/TSX_60'  

            yf_data = yf_data.data(URL_SP, URL_TSX, INPUT_DAILY, INPUT_MAPPING, YEARS)
            yf_data.import_data()
            st.session_state["years_imported"] = YEARS
            st.success("Successfully Imported Data")
        except Exception as e:
            st.error(f"Error importing data: {e}")

# Display parameters
st.subheader("Current Parameters")
col1b, col2b = st.columns(2)

with col1b:
    st.write(f"Short Window: {st.session_state['WINDOW_SIZE_MACD_ST']}")
    st.write(f"Long Window: {st.session_state['WINDOW_SIZE_MACD_LT']}")
    st.write(f"Years: {st.session_state['YEARS']}")

with col2b:
    st.write(f"Cutoff Date: {st.session_state['CUTOFF_DATE']}")
    st.write(f"Risk-Free Rate: {st.session_state['RF']}")
    st.write(f"Capital per Trade: {st.session_state['CAPITAL']}")