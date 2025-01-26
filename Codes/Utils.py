import polars as pl
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay

def read_csv(path: str) -> pl.DataFrame:
    """
    Reads a CSV file from the given path and returns it as a Polars DataFrame.
    Parameters:
    path (str): The file path to the CSV file.
    Returns:
    pl.DataFrame: The contents of the CSV file as a Polars DataFrame.
    """

    df = pl.read_csv(path)

    return df

def export_dfs_xl(dfs: dict[str, pl.DataFrame], path: str) -> None:
    """
    Export multiple Polars DataFrames to an Excel file with different sheets.
    This function takes a dictionary of Polars DataFrames and exports them to an
    Excel file at the specified path. Each DataFrame is written to a separate sheet.
    
    Args:
        dfs (dict[str, pl.DataFrame]): A dictionary where keys are sheet names and values are Polars DataFrames.
        path (str): The file path where the Excel file will be saved.
    Returns:
        None
    """
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs.items():
            df = df.to_pandas() if hasattr(df, 'to_pandas') else df
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def export_df_csv(df: pl.DataFrame, path: str) -> None:
        """
        Export a Polars DataFrame to a CSV file.
        Parameters:
        df (pl.DataFrame): The DataFrame to be exported.
        path (str): The file path where the CSV will be saved.
        Returns:
        None
        """
        
        df.write_csv(path)
    
class Dates():

    def today_str() -> str:
        """
        Get today's date.
        Returns:
            str: Today's date in 'YYYYMMDD' format.
        """
        return datetime.today().strftime('%Y%m%d')
    
    def start_date_backtest(end_date: datetime, years_lenght: datetime) -> datetime:
        """
        Get the start date for the backtest.
        Returns:
            datetime: The start date for the backtest.
        """
        return end_date.replace(year=end_date.year - years_lenght)
    
    def end_date_backtest() -> datetime:
        
        today = datetime.today()
        end_date = today.date()

        return end_date
    
    def format_col_to_date(data: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """
        Formats a specified column in a Polars DataFrame to a date format.

        Args:
            data (pl.DataFrame): The input DataFrame containing the column to be formatted.
            col_name (str): The name of the column to be formatted to date.

        Returns:
            pl.DataFrame: A new DataFrame with the specified column formatted as a date.
        """

        data = data.with_columns(
        pl.col(col_name)
        .cast(str)
        .str.strptime(pl.Date, format="%Y%m%d")
        .alias(col_name)
        )

        return data
    
    def format_date_utf8(data: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """
        Formats the date column in a Polars DataFrame to a UTF-8 string in the format YYYYMMDD.

        Args:
            data (pl.DataFrame): The input Polars DataFrame containing the date column.
            col_name (str): The name of the column to format.

        Returns:
            pl.DataFrame: A new Polars DataFrame with the formatted date column.
        """

        data = data.with_columns(
            data[col_name]
            .dt.strftime("%Y%m%d")
            .cast(pl.Utf8)
            .alias(col_name)
        )

        return data
