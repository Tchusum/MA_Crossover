import streamlit as st
from pathlib import Path
import sys

PATH_PYTHON = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PATH_PYTHON))
import Import_Yfinance as yf_data
import Utils as utils
import MA_Backtest as MA

WINDOW_SIZE_MACD_ST = st.session_state["WINDOW_SIZE_MACD_ST"] 
WINDOW_SIZE_MACD_LT = st.session_state["WINDOW_SIZE_MACD_LT"]
YEARS = st.session_state["YEARS"]
CUTOFF_DATE = st.session_state["CUTOFF_DATE"]
RF = st.session_state["RF"]
CAPITAL = st.session_state["CAPITAL"]

st.title("Run Backtest")

exchange = st.selectbox("Select Exchange", ["S&P 500", "TSX", "S&P 500 & TSX"])

long_short = st.selectbox("Select Strategy", ["Long", "Short", "Long & Short"])

if st.button("Run Backtest"):

    #Dates
    dates_class = utils.Dates
    today = dates_class.today_str()

    #Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    INPUT_DAILY = BASE_DIR / "Input/Daily_Data/"
    INPUT_MAPPING = BASE_DIR / "Input/Mapping/"

    #####Import data#####
    #stocks
    path = f"{INPUT_DAILY}/Stocks_{today}.csv"
    data_stocks = utils.read_csv(path)

    #ETFs
    path = f"{INPUT_DAILY}/ETFs_{today}.csv"
    data_etfs = utils.read_csv(path)

    #Format date
    data_stocks = utils.Dates.format_col_to_date(data_stocks, "Date")
    data_etfs = utils.Dates.format_col_to_date(data_etfs, "Date")

    #Split data by exchange
    data_stocks_sp500, data_stocks_tsx = yf_data.data.split_data_exchange(data_stocks)
    data_etfs_sp500, data_etfs_tsx = yf_data.data.split_data_exchange(data_etfs)

    #Split data training / validation
    training_data_stocks, validation_data_stocks = yf_data.data.split_data_training_validation(data_stocks, CUTOFF_DATE)
    training_data_stocks_tsx, validation_data_stocks_tsx = yf_data.data.split_data_training_validation(data_stocks_tsx, CUTOFF_DATE)
    training_data_stocks_sp500, validation_data_stocks_sp500 = yf_data.data.split_data_training_validation(data_stocks_sp500, CUTOFF_DATE)

    training_data_etfs, validation_data_etfs = yf_data.data.split_data_training_validation(data_etfs, CUTOFF_DATE)
    training_data_etfs_sp500, validation_data_etfs_sp500 = yf_data.data.split_data_training_validation(data_etfs_sp500, CUTOFF_DATE)
    training_data_etfs_tsx, validation_data_etfs_tsx = yf_data.data.split_data_training_validation(data_etfs_tsx, CUTOFF_DATE)

    #Import FX
    path = f"{INPUT_DAILY}/FX_{today}.csv"
    fx_data = utils.read_csv(path)
    fx_data = utils.Dates.format_col_to_date(fx_data, "Date")

    #Mapping FX
    path = f"{INPUT_MAPPING}/Exchange_Currency.csv"
    mapping_fx = utils.read_csv(path)

    #####Run Backtest#####

    #Define modules
    base_functions = MA.Base_Functions(fx_data, mapping_fx)
    macd_backtest = MA.MA_Backtester(fx_data, mapping_fx, training_data_stocks, validation_data_stocks, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, RF)
    buy_and_hold = MA.BuyAndHold(fx_data, mapping_fx, validation_data_etfs, WINDOW_SIZE_MACD_LT, RF)

    if long_short == "Long":

        if exchange == "S&P 500":

            macd_backtest.set_parameters(validation_data = validation_data_stocks_sp500)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs_sp500)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy('long')
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

        elif exchange == "TSX":

            macd_backtest.set_parameters(validation_data = validation_data_stocks_tsx)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs_tsx)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy('long')
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

        elif exchange == "S&P 500 & TSX":

            macd_backtest.set_parameters(validation_data = validation_data_stocks)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy( 'long')
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()
            df_date_buy_hold = buy_and_hold.return_by_date_all_exchange(df_date_buy_hold)

    elif long_short == "Short":

        if exchange == "S&P 500":

            macd_backtest.set_parameters(validation_data = validation_data_stocks_sp500)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs_sp500)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy('short')
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

        elif exchange == "TSX":

            macd_backtest.set_parameters(validation_data = validation_data_stocks_tsx)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs_tsx)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy('short')
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

        elif exchange == "S&P 500 & TSX":

            macd_backtest.set_parameters(validation_data = validation_data_stocks)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy('short')
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()
            df_date_buy_hold = buy_and_hold.return_by_date_all_exchange(df_date_buy_hold)

    elif long_short == "Long & Short":

        if exchange == "S&P 500":

            macd_backtest.set_parameters(validation_data = validation_data_stocks_sp500)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs_sp500)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy()
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

        elif exchange == "TSX":

            macd_backtest.set_parameters(validation_data = validation_data_stocks_tsx)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs_tsx)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy()
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()

        elif exchange == "S&P 500 & TSX":

            macd_backtest.set_parameters(validation_data = validation_data_stocks)
            buy_and_hold.set_parameters(validation_data = validation_data_etfs)
            data, pivot_ticker, df_date, df_stats, trade_df = macd_backtest.run_strategy()
            df_buy_hold, df_date_buy_hold, df_stats_buy_hold, pivot_ticker_buy_hold = buy_and_hold.run_buy_hold()
            df_date_buy_hold = buy_and_hold.return_by_date_all_exchange(df_date_buy_hold)

    st.success("Backtest completed")

    st.subheader("Statistics")

    #create combine df of stats (Strategy vs Buy & Hold)
    df_stats_combine = {f"{exchange} / {long_short}": df_stats,
      f"{exchange} / Buy & Hold": df_stats_buy_hold}
    
    df_stats = base_functions.stats_df_combine(df_stats_combine)
    
    st.dataframe(df_stats)

    st.subheader("P&L by Ticker")

    st.dataframe(pivot_ticker)

    st.subheader("P&L by Date")

    st.dataframe(df_date)

    st.subheader("Vs Buy & Hold")

    graphs = MA.Graphs(fx_data, mapping_fx, training_data_stocks, validation_data_stocks, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, RF)
    fig = graphs.strategy_vs_bh_graph(df_date, df_date_buy_hold)

    st.plotly_chart(fig)



    