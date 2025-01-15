import yfinance as yf
import polars as pl
import pandas as pd
from datetime import datetime
import Utils as utils
import requests
import certifi
from io import StringIO

def import_tik_list(url: str) -> pl.DataFrame:
    
    response = requests.get(url, verify=certifi.where())
    html_content = StringIO(response.text)
    table = pd.read_html(html_content, keep_default_na=False)
    tickers = table[0]
    tickers = pl.from_pandas(tickers)

    return tickers

def select_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Select specific columns from a DataFrame.
    Parameters:
    df (polars.DataFrame): The input DataFrame from which columns are to be selected.
    Returns:
    polars.DataFrame: A DataFrame containing only the selected columns:
        - "Symbol"
        - "Security"
        - "GICS Sector"
        - "GICS Sub-Industry"
        - "Date added"
    """
      
    df = df.select([
        pl.col("Symbol"), 
        pl.col("Security"), 
        pl.col("GICS Sector"), 
        pl.col("GICS Sub-Industry"), 
        pl.col("Date added")
    ])

    return df

def add_exchange(df: pl.DataFrame, exchange: str) -> pl.DataFrame:
    """
    Adds an 'Exchange' column to the given DataFrame with a specified exchange value.
    Parameters:
    df (polars.DataFrame): The DataFrame to which the exchange column will be added.
    exchange (str): The exchange value to be added to the 'Exchange' column.
    Returns:
    polars.DataFrame: The DataFrame with the added 'Exchange' column.
    """

    df = df.with_columns(
        pl.lit(exchange).alias("Exchange")
    )

    return df

def modify_ticker(tsx60_tickers: pl.DataFrame, sp500_tickers: pl.DataFrame) -> tuple:
    """
    Modify ticker symbols for TSX 60 and S&P 500 tickers.
    This function performs the following modifications:
    1. Replaces periods (".") with hyphens ("-") in the "Symbol" column for both TSX 60 and S&P 500 tickers.
    2. Appends ".TO" to the end of each ticker symbol in the "Symbol" column for TSX 60 tickers.
    Args:
        tsx60_tickers (polars.DataFrame): DataFrame containing TSX 60 ticker symbols.
        sp500_tickers (polars.DataFrame): DataFrame containing S&P 500 ticker symbols.
    Returns:
        tuple: A tuple containing two DataFrames:
            - Modified TSX 60 tickers DataFrame.
            - Modified S&P 500 tickers DataFrame.
    """

    #Replace "."" by "-""
    tsx60_tickers = tsx60_tickers.with_columns(
        pl.col("Symbol").str.replace(r"\.", "-").alias("Symbol")
    )

    sp500_tickers = sp500_tickers.with_columns(
        pl.col("Symbol").str.replace(r"\.", "-").alias("Symbol")
    )

    #Add ".TO" at the end of tickers
    tsx60_tickers = tsx60_tickers.with_columns(
        (pl.col("Symbol") + ".TO").alias("Symbol")
    )
    
    return tsx60_tickers, sp500_tickers

def ticker_lists(tsx60_tickers: pl.DataFrame, sp500_tickers: pl.DataFrame, etfs_tickers: pl.DataFrame) -> tuple:	

    tik_sp500_list = sp500_tickers['Symbol'].to_list()
    tik_tsx_list = tsx60_tickers['Symbol'].to_list()
    tik_etfs_list = etfs_tickers['Symbol'].to_list()

    return tik_tsx_list, tik_sp500_list, tik_etfs_list

def exchange_map_stock(tik_sp500: list, tik_tsx: list) -> dict:	
    """
    Creates a mapping of ticker symbols to their respective exchanges.
    Args:
        tik_sp500 (list): A list of ticker symbols belonging to the S&P 500.
        tik_tsx (list): A list of ticker symbols belonging to the TSX.
    Returns:
        dict: A dictionary where the keys are ticker symbols and the values are the exchange names ("SP500" or "TSX").
    """

    exchange_map = {tik: "SP500" for tik in tik_sp500}
    exchange_map.update({tik: "TSX" for tik in tik_tsx})

    return exchange_map

def exchange_map_etf(etfs_tickers: pl.DataFrame) -> dict:	
 
    exchange_map_etfs = {row[0]: row[1] for row in etfs_tickers.iter_rows()}

    return exchange_map_etfs

def format_date_to_utf8(df: pl.DataFrame) -> pl.DataFrame:
    """
    Formats the 'Date' column of a DataFrame to a UTF-8 string in the format YYYYMMDD.
    Parameters:
    df (polars.DataFrame): The input DataFrame containing a 'Date' column.
    Returns:
    polars.DataFrame: A DataFrame with the 'Date' column formatted as a UTF-8 string.
    """

    df = df.with_columns(
        df["Date"]
        .dt.strftime("%Y%m%d")
        .cast(pl.Utf8)
        .alias("Date")
    )

    return df

def format_utf8_to_date(data: pl.DataFrame) -> pl.DataFrame:
    """
    Converts the 'Date' column in the given DataFrame from UTF-8 string format to Date format.
    Args:
        data (polars.DataFrame): The input DataFrame containing a 'Date' column in UTF-8 string format.
    Returns:
        polars.DataFrame: The DataFrame with the 'Date' column converted to Date format.
    """

    data = data.with_columns(
    pl.col('Date')
    .cast(pl.Utf8)
    .str.strptime(pl.Date, format="%Y%m%d")
    .alias('Date')
    )

    return data

def format_ticker_log(Good_Tik: list, Bad_Tik: list) -> pl.DataFrame:
    """
    Formats two lists of tickers into a DataFrame with equal length columns.
    Parameters:
    Good_Tik (list): A list of good tickers.
    Bad_Tik (list): A list of bad tickers.
    Returns:
    pl.DataFrame: A DataFrame with two columns, 'Good_Tik' and 'Bad_Tik', 
                  where each column is padded with None to match the length 
                  of the longer list.
    """

    max_len = max(len(Good_Tik), len(Bad_Tik))
     
    tik_log = pl.DataFrame({
        "Good_Tik": Good_Tik + [None] * (max_len - len(Good_Tik)),
        "Bad_Tik": Bad_Tik + [None] * (max_len - len(Bad_Tik)),
    })

    return tik_log
     
def download_data_stocks(tik_list: list, start_date: datetime, end_date: datetime, exchange_map: dict) -> tuple:
    """
    Downloads historical stock data for a list of tickers within a specified date range.
    Args:
        tik_list (list): List of stock tickers to download data for.
        start_date (datetime): The start date for the data download.
        end_date (datetime): The end date for the data download.
        exchange_map (dict): A dictionary mapping each ticker to its respective exchange.
    Returns:
        tuple: A tuple containing:
            - all_data (pl.DataFrame): A concatenated DataFrame of all the downloaded stock data.
            - tik_log (dict): A dictionary with two keys:
                - 'good_tik': List of tickers for which data was successfully downloaded.
                - 'bad_tik': List of tickers for which data download failed.
    """
     
    all_data = []
    good_tik = []
    bad_tik = []

    for tik in tik_list:

        data = yf.download(tik, start=start_date, end=end_date)

        if not data.empty:

            data.columns = data.columns.droplevel("Ticker")

            pl_data = pl.DataFrame(
                {
                    "Date": data.index,
                    "Close": data["Close"],
                    "Open": data["Open"],
                    "Ticker": tik,
                    "Exchange": exchange_map[tik],
                }
            )

            all_data.append(pl_data)
            good_tik.append(tik)

        else:
             
             bad_tik.append(tik)
    
    all_data = pl.concat(all_data)

    all_data = format_date_to_utf8(all_data)

    tik_log = format_ticker_log(good_tik, bad_tik)

    return all_data, tik_log

def download_data_fx(currency_tik: str, currency_mapping: str, start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """
    Downloads foreign exchange data for a given currency and date range from Yahoo Finance.
    Args:
        currency_tik (str): The currency ticker symbol to download data for.
        currency_mapping (str): The currency mapping name to download data for.
        start_date (datetime): The start date for the data download.
        end_date (datetime): The end date for the data download.
    Returns:
        pl.DataFrame: A DataFrame containing the downloaded foreign exchange data with columns:
            - "Date": The date of the data point.
            - "Close": The closing price of the currency on that date.
            - "Open": The opening price of the currency on that date.
            - "Ticker": The currency ticker symbol.
    """

    fx_data = yf.download(currency_tik, start=start_date, end=end_date)

    fx_data.columns = fx_data.columns.droplevel("Ticker")

    fx_data = pl.DataFrame({

        "Date": fx_data.index,
        "Close": fx_data["Close"],
        "Open": fx_data["Open"],
        "Ticker": currency_tik,
        "Currency": currency_mapping

    })

    fx_data = format_date_to_utf8(fx_data)

    return fx_data

def download_data_etfs(tik_etfs_list: list, start_date: datetime, end_date: datetime, exchange_map: dict ) -> pl.DataFrame:

    all_etf_data = []

    for etf in tik_etfs_list:

        data = yf.download(etf, start=start_date, end=end_date)

        if not data.empty:

            data.columns = data.columns.droplevel("Ticker")

            pl_data = pl.DataFrame({

                "Date": data.index,
                "Close": data["Close"],
                "Open": data["Open"],
                "Ticker": etf,
                "Exchange": exchange_map[etf]

            })

            all_etf_data.append(pl_data)

    all_etf_data = pl.concat(all_etf_data)

    all_etf_data = format_date_to_utf8(all_etf_data)

    return all_etf_data

def split_data_training_validation(data: pl.DataFrame, cutoff_date: str) -> tuple:
    """
    Splits the given data into training and validation datasets based on the cutoff date.
    Args:
        data (polars.DataFrame): The input data containing a 'Date' column.
        cutoff_date (str): The cutoff date in the format 'YYYY-MM-DD'. Data before this date will be used for training, and data on or after this date will be used for validation.
    Returns:
        tuple: A tuple containing two polars.DataFrame objects:
            - training_data: DataFrame with data before the cutoff date.
            - validation_data: DataFrame with data on or after the cutoff date.
    """

    cutoff_date = datetime.strptime(cutoff_date, '%Y-%m-%d')

    training_data = data.filter(pl.col('Date') < cutoff_date)

    validation_data = data.filter(pl.col('Date') >= cutoff_date)

    return training_data, validation_data

def split_data_exchange(data: pl.DataFrame) -> tuple:
    """
    Splits the given data into separate DataFrames based on the 'Exchange' column.
    Args:
        data (polars.DataFrame): The input data containing an 'Exchange' column.
    Returns:
        tuple: A tuple containing two polars.DataFrame objects:
            - sp500_data: DataFrame containing data for S&P 500 stocks.
            - tsx_data: DataFrame containing data for TSX stocks.
    """

    sp500_data = data.filter(pl.col('Exchange') == 'SP500')

    tsx_data = data.filter(pl.col('Exchange') == 'TSX')

    return sp500_data, tsx_data

def main(url_sp: str, url_tsx: str, Input_Daily: str, Input : str, years: int) -> None:
    """
    Main function to import and process stock data from given URLs.
    Args:
        url_sp (str): URL to import S&P 500 tickers.
        url_tsx (str): URL to import TSX 60 tickers.
        Input (str): Input directory path for saving the output CSV files.
        years (int): Number of years of historical data to download.
    Returns:
        None
    """
    #Fix date
    dates_class = utils.Dates(years)
    today = dates_class.today_str()
    end_date = dates_class.end_date_backtest()
    start_date = dates_class.start_date_backtest(end_date)

    #---SP500---
    sp500_tickers = import_tik_list(url_sp)

    #Select Columns
    sp500_tickers = select_columns(sp500_tickers)

    #Add Exchange
    sp500_tickers = add_exchange(sp500_tickers, "SP500")

    #---TSX60---
    tsx60_tickers = import_tik_list(url_tsx)

    #Add Exchange
    sp500_tickers = add_exchange(sp500_tickers, "TSX")

    #---Modify Ticker (Exchange vs YF)---
    tsx60_tickers, sp500_tickers = modify_ticker(tsx60_tickers, sp500_tickers)

    #---ETFs---
    etfs_tickers = utils.read_csv(f"{Input}ETFs.csv")

    #Ticker list
    tik_tsx_list, tik_sp500_list, tik_etfs_list = ticker_lists(tsx60_tickers, sp500_tickers, etfs_tickers)

    #Exchange map
    exchange_map_stocks = exchange_map_stock(tik_sp500_list, tik_tsx_list)
    exchange_map_etfs = exchange_map_etf(etfs_tickers)

    #Combine tickers list (Stocks)
    tik_list = tik_sp500_list + tik_tsx_list

    #download data
    all_data, tik_log = download_data_stocks(tik_list, start_date, end_date, exchange_map_stocks)
    fx_data = download_data_fx("CAD=X", "USD", start_date, end_date)
    etf_data = download_data_etfs(tik_etfs_list, start_date, end_date, exchange_map_etfs)

    #Write to csv
    #all_data.write_csv(f"{Input_Daily}Stocks_{today}.csv")
    #tik_log.write_csv(f"{Input_Daily}Tik_Log_{today}.csv")
    #fx_data.write_csv(f"{Input_Daily}FX_{today}.csv")
    #etf_data.write_csv(f"{Input_Daily}ETFs_{today}.csv")

    return all_data, tik_log, fx_data, etf_data

if __name__ == "__main__":

    INPUT_DAILY = '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/Daily_Data/'
    INPUT_MAPPING = '/Users/alexandrechisholm/Library/Mobile Documents/com~apple~CloudDocs/Trading/MA_Crossover/Input/Mapping/'
    URL_SP = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    URL_TSX = 'https://en.wikipedia.org/wiki/S%26P/TSX_60'
    YEAR = 15

    all_data, tik_log, fx_data, etf_data = main(URL_SP, URL_TSX, INPUT_DAILY, INPUT_MAPPING, YEAR)