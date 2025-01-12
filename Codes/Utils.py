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

class Dates:

    def __init__(self, years_lenght: int):

        self.years_lenght = years_lenght

    def today_str(self) -> str:
        """
        Get today's date.
        Returns:
            str: Today's date in 'YYYYMMDD' format.
        """
        return datetime.today().strftime('%Y%m%d')
    
    def start_date_backtest(self, end_date: datetime) -> datetime:
        """
        Get the start date for the backtest.
        Returns:
            datetime: The start date for the backtest.
        """
        return end_date.replace(year=end_date.year - self.years_lenght)
    
    def end_date_backtest(self) -> datetime:
        
        today = datetime.today()
        end_date = today.date()

        return end_date
    
    def format_int64_to_date(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Converts the 'Date' column in the given DataFrame from Int64 format to Date format.
        Args:
            data (polars.DataFrame): The input DataFrame containing a 'Date' column in Integer format.
        Returns:
            polars.DataFrame: The DataFrame with the 'Date' column converted to Date format.
        """

        data = data.with_columns(
        pl.col('Date')
        .cast(str)
        .str.strptime(pl.Date, format="%Y%m%d")
        .alias('Date')
        )

        return data