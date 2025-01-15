import streamlit as st
import Import_Yfinance as yf_data
import Utils as utils
import MA_Backtest as MA

WINDOW_SIZE_MACD_ST = st.session_state["WINDOW_SIZE_MACD_ST"] 
WINDOW_SIZE_MACD_LT = st.session_state["WINDOW_SIZE_MACD_LT"]
YEARS = st.session_state["YEARS"]
CUTOFF_DATE = st.session_state["CUTOFF_DATE"]
RF = st.session_state["RF"]
CAPITAL = st.session_state["CAPITAL"]

INPUT_DAILY = '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/Daily_Data/'
INPUT_MAPPING = '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/Mapping/'
URL_SP = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
URL_TSX = 'https://en.wikipedia.org/wiki/S%26P/TSX_60'

PATHS = {
    "input_daily_path" : '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/Daily_Data/',
    "input_path" : '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/',
    "input_mapping" : '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/Mapping/',
    "output_path" : '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Output/'
}

exchange = st.selectbox("Select Exchange", ["S&P 500", "TSX", "S&P 500 & TSX"])

long_short = st.selectbox("Select Strategy", ["Long", "Short", "Long & Short"])

if st.button("Run Backtest"):

    #Yahoo Finance Data
    data_stocks, tik_log, fx_data, data_etfs = yf_data.main(URL_SP, URL_TSX, INPUT_DAILY, INPUT_MAPPING, YEARS)

    #Dates
    dates_class = utils.Dates(YEARS)
    today = dates_class.today_str()

    #Format date
    data_stocks = yf_data.format_utf8_to_date(data_stocks)
    data_etfs = yf_data.format_utf8_to_date(data_etfs)
    fx_data = dates_class.format_int64_to_date(fx_data)

    #Split data by exchange
    data_stocks_sp500, data_stocks_tsx = yf_data.split_data_exchange(data_stocks)
    data_etfs_sp500, data_etfs_tsx = yf_data.split_data_exchange(data_etfs)

    #Split data training / validation
    training_data_stocks, validation_data_stocks = yf_data.split_data_training_validation(data_stocks, CUTOFF_DATE)
    training_data_stocks_tsx, validation_data_stocks_tsx = yf_data.split_data_training_validation(data_stocks_tsx, CUTOFF_DATE)
    training_data_stocks_sp500, validation_data_stocks_sp500 = yf_data.split_data_training_validation(data_stocks_sp500, CUTOFF_DATE)

    training_data_etfs, validation_data_etfs = yf_data.split_data_training_validation(data_etfs, CUTOFF_DATE)
    training_data_etfs_sp500, validation_data_etfs_sp500 = yf_data.split_data_training_validation(data_etfs_sp500, CUTOFF_DATE)
    training_data_etfs_tsx, validation_data_etfs_tsx = yf_data.split_data_training_validation(data_etfs_tsx, CUTOFF_DATE)

    #Mapping FX
    path = f"{PATHS['input_mapping']}Exchange_Currency.csv"
    mapping_fx = utils.read_csv(path)

    #####Run Backtest#####

    #Define modules
    base_functions = MA.Base_Functions(fx_data, mapping_fx)
    macd_backtest = MA.MA_Backtester(fx_data, mapping_fx, training_data_stocks, validation_data_stocks, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, RF)
    buy_and_hold = MA.BuyAndHold(fx_data, mapping_fx, validation_data_etfs, WINDOW_SIZE_MACD_LT, RF)

    if exchange == "S&P 500":

        macd_backtest.set_parameters(validation_data = validation_data_stocks_sp500)
        buy_and_hold.set_parameters(validation_data = validation_data_etfs_sp500)
        data_sp500, pivot_ticker_sp500, df_date_sp500, df_stats_sp500, trade_df_sp500 = macd_backtest.run_strategy()
        data_long_sp500, pivot_ticker_long_sp500, df_date_long_sp500, df_stats_long_sp500, trade_df_long_sp500 = macd_backtest.run_strategy('long')
        data_short_sp500, pivot_ticker_short_sp500, df_date_short_sp500, df_stats_short_sp500, trade_df_short_sp500 = macd_backtest.run_strategy('short')
        df_buy_hold_sp500, df_date_buy_hold_sp500, df_stats_buy_hold_sp500, pivot_ticker_buy_hold_sp500 = buy_and_hold.run_buy_hold()

    elif exchange == "TSX":

        macd_backtest.set_parameters(validation_data = validation_data_stocks_tsx)
        buy_and_hold.set_parameters(validation_data = validation_data_etfs_tsx)
        data_tsx, pivot_ticker_tsx, df_date_tsx, df_stats_tsx, trade_df_tsx = macd_backtest.run_strategy()
        data_long_tsx, pivot_ticker_long_tsx, df_date_long_tsx, df_stats_long_tsx, trade_df_long_tsx = macd_backtest.run_strategy('long')
        data_short_tsx, pivot_ticker_short_tsx, df_date_short_tsx, df_stats_short_tsx, trade_df_short_tsx = macd_backtest.run_strategy('short')
        df_buy_hold_tsx, df_date_buy_hold_tsx, df_stats_buy_hold_tsx, pivot_ticker_buy_hold_tsx = buy_and_hold.run_buy_hold()

    elif exchange == "S&P 500 & TSX":

        data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy()
        data_long, pivot_ticker_long, df_date_long, df_stats_long, trade_df_long = macd_backtest.run_strategy( 'long')
        data_short, pivot_ticker_short, df_date_short, df_stats_short, trade_df_short = macd_backtest.run_strategy('short')
        df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

    st.success("Backtest completed")