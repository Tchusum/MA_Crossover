import streamlit as st

WINDOW_SIZE_MACD_ST = st.slider("Select Short Window", 1, 50, 5)

WINDOW_SIZE_MACD_LT = st.slider("Select Long Window", 1, 200, 95)

YEARS = st.number_input("Number of years backdate", min_value=0, max_value=50, value=15, step=1)

CUTOFF_DATE = st.text_input("Cutoff Date (YYYY-MM-DD) between training and validation data", value="2024-01-01")

RF = st.number_input("Define Risk-Free Rate", min_value=0.00, max_value=1.00, value = 0.03, step=0.01)

CAPITAL = st.number_input("Define Capital per Trade", min_value=50, max_value=10000, value=200, step=50)

if st.button("Save Parameters"):
    st.session_state["WINDOW_SIZE_MACD_ST"] = WINDOW_SIZE_MACD_ST 
    st.session_state["WINDOW_SIZE_MACD_LT"] = WINDOW_SIZE_MACD_LT
    st.session_state["YEARS"] = YEARS
    st.session_state["CUTOFF_DATE"] = CUTOFF_DATE
    st.session_state["RF"] = RF
    st.session_state["CAPITAL"] = CAPITAL
    st.success("Successfully saved parameters")


