import polars as pl
import math
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Base_Functions():

    def __init__(self, fx_data: pl.DataFrame, mapping_fx: pl.DataFrame):

        self.fx_data = fx_data
        self.mapping_fx = mapping_fx

    def add_fx(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Adds foreign exchange (FX) data to the given DataFrame.

        This method joins the provided DataFrame with FX mapping and FX data based on 
        the 'Exchange', 'Date', and 'Currency' columns. It fills any null values in 
        the 'Close_right' column with 1 and renames it to 'Close_fx'. Additionally, 
        it excludes certain columns from the final DataFrame.

        Parameters:
        data (pl.DataFrame): The input DataFrame containing trading data.

        Returns:
        pl.DataFrame: The DataFrame with added FX data and selected columns.
        """

        data = data.join(self.mapping_fx, on="Exchange", how="left")

        data = data.join(self.fx_data, on=["Date", "Currency"], how="left")

        data = data.with_columns(
            data["Close_right"].fill_null(1).alias("Close_right")
        )

        data = data.rename({"Close_right": "Close_fx"})

        col_unselect = ['Ticker_right', "Open_right"]
        data = data.select(pl.exclude(col_unselect))

        return data
    
    def df_ticker(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Aggregates and sorts the provided DataFrame by the 'Ticker' column.

        This method groups the input DataFrame by the 'Ticker' column, 
        aggregates the last value of the 'P&L_CAD' column for each group, 
        and then sorts the resulting DataFrame in descending order based on 'P&L_CAD'.

        Args:
            data (pl.DataFrame): The input DataFrame containing 'Ticker' and 'P&L_CAD' columns.

        Returns:
            pl.DataFrame: A DataFrame with 'Ticker' and aggregated 'P&L_CAD' columns, sorted by 'P&L_CAD' in descending order.
        """

        pivot_ticker = data.group_by("Ticker").agg([
            pl.last("P&L_CAD").alias("P&L_CAD")
        ])

        pivot_ticker = pivot_ticker.sort("P&L_CAD")

        return pivot_ticker
    
    def stats_df_combine(self, dfs: dict) -> pl.DataFrame:
        
        """
        Combines multiple DataFrames containing strategy statistics into a single DataFrame.
        Args:
            dfs (dict): A dictionary where keys are labels and values are DataFrames containing 
                        strategy statistics.
        Returns:
            pl.DataFrame: A combined DataFrame with an additional "Strategy" column indicating 
                          the label of each strategy.
        The combined DataFrame includes the following columns:
            - "Strategy": The label of the strategy.
            - "Log_Annualized_Return": The log annualized return of the strategy.
            - "Standard_Deviation": The standard deviation of the strategy's returns.
            - "Sharpe_Ratio": The Sharpe ratio of the strategy.
            - "Total P&L_CAD": The total profit and loss in CAD.
            - "Number of Days": The number of days the strategy was active.
        """

        processed_dfs = [
            df.with_columns(pl.lit(label).alias("Strategy"))
            for label, df in dfs.items()
        ]
        
        df_stats_combine = pl.concat(processed_dfs)
        
        df_stats_combine = df_stats_combine.select([
            "Strategy",
            "Log_Annualized_Return",
            "Standard_Deviation",
            "Sharpe_Ratio",
            "Total P&L_CAD",
            "Number of Days"
        ])
        
        return df_stats_combine
    
class MA_Backtester(Base_Functions):

    def __init__(self,
                 fx_data: pl.DataFrame, #Base Functions
                 mapping_fx: pl.DataFrame, #Base Functions
                 training_data: pl.DataFrame, 
                 validation_data: pl.DataFrame,
                 capital: int, 
                 window_size_macd_st: int, 
                 window_size_macd_lt:int,
                 rf: float):

        super().__init__(fx_data, mapping_fx)
        self.capital = capital
        self.training_data = training_data
        self.validation_data = validation_data
        self.window_size_macd_st = window_size_macd_st
        self.window_size_macd_lt = window_size_macd_lt
        self.rf = rf

    def set_parameters(self, window_size_macd_st: int = None, window_size_macd_lt: int = None, training_data: pl.DataFrame = None, validation_data: pl.DataFrame = None):
        """
        Set the parameters for the MACD backtest.
        Parameters:
        window_size_macd_st (int, optional): The short-term window size for the MACD calculation. Defaults to None.
        window_size_macd_lt (int, optional): The long-term window size for the MACD calculation. Defaults to None.
        training_data (pl.DataFrame, optional): The training data to be used for the backtest. Defaults to None.
        validation_data (pl.DataFrame, optional): The validation data to be used for the backtest. Defaults to None.
        """
        
        if window_size_macd_st is not None:
            self.window_size_macd_st = window_size_macd_st

        if window_size_macd_lt is not None:
            self.window_size_macd_lt = window_size_macd_lt

        if training_data is not None:
            self.training_data = training_data

        if validation_data is not None:
            self.validation_data = validation_data

    #Complete DF functions
    def ema(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the Exponential Moving Average (EMA) for the given data.
        This method adds two new columns to the input DataFrame:
        - EMA for the short-term window size (`EMA_{self.window_size_macd_st}`)
        - EMA for the long-term window size (`EMA_{self.window_size_macd_lt}`)
        The EMA is calculated using the 'Close' column of the DataFrame, and the calculation is performed
        separately for each 'Ticker' in the DataFrame.
        Parameters:
        data (pl.DataFrame): The input DataFrame containing at least the 'Close' and 'Ticker' columns.
        Returns:
        pl.DataFrame: The input DataFrame with two additional columns for the short-term and long-term EMAs.
        """

        data = data.with_columns([
            pl.col('Close').ewm_mean(span = self.window_size_macd_st, min_periods = self.window_size_macd_st, ignore_nulls=True).over('Ticker').alias(f'EMA_{self.window_size_macd_st}'),
            pl.col('Close').ewm_mean(span = self.window_size_macd_lt, min_periods = self.window_size_macd_lt, ignore_nulls=True).over('Ticker').alias(f'EMA_{self.window_size_macd_lt}')
        ])

        return data
    
    def macd(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the Moving Average Convergence Divergence (MACD) for the given data.
        Parameters:
        data (pl.DataFrame): A Polars DataFrame containing the necessary EMA columns.
        Returns:
        pl.DataFrame: The input DataFrame with an additional 'MACD' column.
        """

        data = data.with_columns(
            (pl.col(f'EMA_{self.window_size_macd_st}') - pl.col(f'EMA_{self.window_size_macd_lt}')).alias('MACD')
        )

        return data

    def signal(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Generates trading signals based on the Moving Average Convergence Divergence (MACD) indicator.
        This method adds a 'Signal' column to the provided DataFrame, where:
        - A value of 1 indicates a bullish signal (MACD crosses above the signal line).
        - A value of -1 indicates a bearish signal (MACD crosses below the signal line).
        - A value of 0 indicates no signal.
        Parameters:
        data (pl.DataFrame): A Polars DataFrame containing the MACD values.
        Returns:
        pl.DataFrame: The input DataFrame with an additional 'Signal' column.
        """

        data = data.with_columns(
            pl.when((pl.col(f'MACD') > 0) & (pl.col(f'MACD').shift(1) <= 0))
            .then(1)
            .when((pl.col(f'MACD') < 0) & (pl.col(f'MACD').shift(1) >= 0))
            .then(-1)
            .otherwise(0)
            .alias('Signal')
        )

        return data
    
    def trade(self, data: pl.DataFrame , type: str  = None) -> tuple:
        def trade(self, data: pl.DataFrame, type: str = None) -> tuple:
            """
            Executes trades based on the provided signals and returns the updated data and trade details.
            Parameters:
            data (pl.DataFrame): The input data containing trading signals and other relevant columns.
            type (str, optional): The type of trade to execute. Can be 'long', 'short', or None. Defaults to None.
            Returns:
            tuple: A tuple containing the updated data (pl.DataFrame) with trade details and a DataFrame (pl.DataFrame) 
                   containing detailed trade information.
            The function performs the following steps:
            1. Adds a 'Trade_open' column to the data based on the trading signals.
            2. Initializes lists to keep track of trades, costs, and cumulative sums.
            3. Iterates through the data rows to execute trades based on the signals and the specified trade type.
            4. Updates the trade details and cumulative sums accordingly.
            5. Adds the trade details to the data.
            6. Returns the updated data and a DataFrame containing detailed trade information.
            The returned DataFrame contains the following columns:
            - Ticker: The ticker symbol of the traded asset.
            - Buy_Price: The price at which the asset was bought.
            - Sell_Price: The price at which the asset was sold.
            - Quantity: The quantity of the asset traded.
            - Trade_Date_open: The date when the trade was opened.
            - Trade_Date_close: The date when the trade was closed.
            """
        
        data = data.with_columns([
            pl.when((pl.col('Signal').shift(1) == 1) & (pl.col('Ticker').shift(1) == pl.col('Ticker')))
            .then(((self.capital / pl.col('Close_fx')) / pl.col('Open')).floor())    
            .when((pl.col('Signal').shift(1) == -1) & (pl.col('Ticker').shift(1) == pl.col('Ticker'))) 
            .then(((-self.capital / pl.col('Close_fx')) / pl.col('Open')).ceil()) 
            .otherwise(0)                          
            .alias('Trade_open'),
            pl.col('Signal').shift(1).fill_null(0).alias('Signal_shifted'),
        ])

        trade_list = []
        cost_list = []
        cum_sum_list = []
        cum_sum = 0
        prev_ticker = None

        #Create a df containing all trades
        trade_details = {
            "Ticker": [],
            "Buy_Price": [],
            "Sell_Price": [],
            "Quantity": [],
            "Trade_Date_open": [],
            "Trade_Date_close": []
        }

        for row in data.iter_rows(named=True):

            if row["Ticker"] != prev_ticker:
                cum_sum = 0
                cost = 0

            trade = 0

            if type == 'long' and row["Signal_shifted"] == 1:
                
                trade = row["Trade_open"]
                trade_details["Ticker"].append(row["Ticker"])
                trade_details["Buy_Price"].append(row["Open"])
                trade_details["Sell_Price"].append(None)
                trade_details["Quantity"].append(trade)
                trade_details["Trade_Date_open"].append(row["Date"])
                trade_details["Trade_Date_close"].append(None)
                cost = abs(trade) * row["Open"]

            elif type == 'long' and row["Signal_shifted"] == -1 and cum_sum != 0:
                
                trade = -cum_sum
                trade_details["Sell_Price"][-1] = row["Open"]
                trade_details["Trade_Date_close"][-1] = row["Date"]
                cost = 0

            elif type == 'short' and row["Signal_shifted"] == -1:
                
                trade = row["Trade_open"]
                trade_details["Ticker"].append(row["Ticker"])
                trade_details["Buy_Price"].append(row["Open"])
                trade_details["Sell_Price"].append(None)
                trade_details["Quantity"].append(trade)
                trade_details["Trade_Date_open"].append(row["Date"])
                trade_details["Trade_Date_close"].append(None)
                cost = abs(trade) * row["Open"]

            elif type == 'short' and row["Signal_shifted"] == 1 and cum_sum != 0:
                
                trade = -cum_sum
                trade_details["Sell_Price"][-1] = row["Open"]
                trade_details["Trade_Date_close"][-1] = row["Date"]
                cost = 0

            elif type is None and row["Signal_shifted"] != 0:

                if cum_sum == 0:
                    
                    trade = row["Trade_open"]
                    trade_details["Ticker"].append(row["Ticker"])
                    trade_details["Buy_Price"].append(row["Open"])
                    trade_details["Sell_Price"].append(None)
                    trade_details["Quantity"].append(trade)
                    trade_details["Trade_Date_open"].append(row["Date"])
                    trade_details["Trade_Date_close"].append(None)
                    
                else:
                    
                    trade = -cum_sum + row["Trade_open"]
                    trade_details["Sell_Price"][-1] = row["Open"]
                    trade_details["Trade_Date_close"][-1] = row["Date"]
                    trade_details["Ticker"].append(row["Ticker"])
                    trade_details["Buy_Price"].append(row["Open"])
                    trade_details["Sell_Price"].append(None)
                    trade_details["Quantity"].append(row["Trade_open"])
                    trade_details["Trade_Date_open"].append(row["Date"])
                    trade_details["Trade_Date_close"].append(None)
                
                cost = abs(row["Trade_open"]) * row["Open"]

            cum_sum += trade

            trade_list.append(trade)
            cum_sum_list.append(cum_sum)
            cost_list.append(cost)

            prev_ticker = row["Ticker"]

        data = data.with_columns([
            pl.Series("Trade", trade_list, dtype=pl.Float64),
            pl.Series("Position", cum_sum_list, dtype=pl.Float64),
            pl.Series("Cost", cost_list, dtype=pl.Float64)
        ])

        col_unselect = ['Signal_shifted']
        data = data.select(pl.exclude(col_unselect))

        #Convert trade_details to polars
        trade_df = pl.DataFrame(trade_details)

        return data, trade_df

    def cf(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the cumulative cash flow for each ticker in the given DataFrame.
        Args:
            data (pl.DataFrame): A Polars DataFrame containing at least the columns 'Trade', 'Open', and 'Ticker'.
        Returns:
            pl.DataFrame: The input DataFrame with two additional columns:
                - 'CF': The cash flow for each trade, calculated as the negative product of 'Trade' and 'Open'.
                - 'CF_Sum': The cumulative sum of 'CF' for each ticker.
        """

        data = data.with_columns(
            (-pl.col('Trade') * pl.col('Open')).alias('CF')
        )

        data = data.with_columns([
            pl.col('CF').cum_sum().over('Ticker').alias('CF_Sum')
        ])

        return data

    def realized_pnl(self, trade_df: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the realized profit and loss (P&L) for trades and update the main data with cumulative realized P&L.

        Args:
            trade_df (pl.DataFrame): A DataFrame containing trade information with columns:
                - "Sell_Price": The price at which the asset was sold.
                - "Buy_Price": The price at which the asset was bought.
                - "Quantity": The quantity of the asset traded.
                - "Ticker": The ticker symbol of the asset.
                - "Trade_Date_close": The date when the trade was closed.
            data (pl.DataFrame): The main DataFrame containing market data with columns:
                - "Ticker": The ticker symbol of the asset.
                - "Date": The date of the market data.

        Returns:
            pl.DataFrame: The updated main DataFrame with an additional column "Realized_P&L" representing the cumulative realized P&L for each ticker.
        """

        #remove rows with null values in Sell_Price
        #these trades are not closed
        trade_df = trade_df.filter(trade_df["Sell_Price"].is_not_null())


        #Calculate the realized P&L
        trade_df = trade_df.with_columns([

            ((pl.col("Sell_Price") - pl.col("Buy_Price"))* pl.col("Quantity")).alias("Realized_P&L")

        ])

        #group by Ticker and Trade_Date_close
        realized_pnl_df = trade_df.group_by(["Ticker", "Trade_Date_close"]).agg(pl.col("Realized_P&L").sum())

        #join the realized P&L to the main data and cum_sum
        data = data.join(realized_pnl_df, left_on = ["Ticker", "Date"], right_on = ["Ticker", "Trade_Date_close"], how="left")

        data = data.with_columns([
            pl.col('Realized_P&L').fill_null(0).cum_sum().over('Ticker').alias('Realized_P&L')
        ])

        return data

    def position_value(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the value of the position based on the closing price.

        Args:
            data (pl.DataFrame): A Polars DataFrame containing at least the columns "Position" and "Close".

        Returns:
            pl.DataFrame: A Polars DataFrame with an additional column "Position_Value" which is the product of "Position" and "Close".
        """

        data = data.with_columns([
            (pl.col("Position") * pl.col("Close")).alias("Position_Value")
        ])

        return data

    def unrealized_pnl(self, data: pl.DataFrame) -> pl.DataFrame:
            
        """
        Calculate the unrealized profit and loss (P&L) for each position in the given DataFrame.
        Args:
            data (pl.DataFrame): A DataFrame containing trading data with columns "Position", "Position_Value", and "Cost".
        Returns:
            pl.DataFrame: A DataFrame with an additional column "Unrealized_P&L" representing the unrealized profit and loss for each position.
        """
        
        data = data.with_columns([
            pl.when(pl.col("Position") > 0)
            .then(-1)
            .otherwise(1)
            .alias("long/short")

        ])           
        
        data = data.with_columns([
            (pl.col("Position_Value") + pl.col("Cost") * pl.col("long/short")).alias("Unrealized_P&L")
        ])

        return data
    
    def pnl(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the daily and cumulative profit and loss (P&L) for the given data.

        Args:
            data (pl.DataFrame): A Polars DataFrame containing at least the columns "Realized_P&L", 
                                 "Unrealized_P&L", and "Ticker".

        Returns:
            pl.DataFrame: A Polars DataFrame with additional columns:
                          - "P&L": The cumulative profit and loss.
                          - "P&L_Daily": The daily profit and loss.
        """

        data = data.with_columns([
            ((pl.col("Realized_P&L") + pl.col("Unrealized_P&L"))).alias("P&L")
        ])

        #pnl daily
        data = data.with_columns(
            pl.when(pl.col("Ticker") == pl.col("Ticker").shift(1))
            .then(pl.col("P&L") - pl.col("P&L").shift(1))
            .otherwise(0)
            .alias("P&L_Daily")
        )

        data = data.with_columns(
            data["P&L_Daily"].fill_null(0).alias("P&L_Daily"),
            data["P&L"].fill_null(0).alias("P&L")
        )

        return data
    
    def cost_input_return(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Calculate the cost input return for each row in the given DataFrame.

        This method adds a new column "Cost_Input_Return" to the DataFrame, which is calculated as follows:
        - If the "Trade" column value is not 0, the "Cost_Input_Return" is the sum of the current "Cost" and the previous row's "Cost".
        - Otherwise, the "Cost_Input_Return" is the same as the current "Cost".

        Parameters:
        data (pl.DataFrame): The input DataFrame containing at least the columns "Trade" and "Cost".

        Returns:
        pl.DataFrame: The DataFrame with the additional "Cost_Input_Return" column.
        """

        data = data.with_columns([

            pl.when(pl.col("Trade") != 0)
            .then(pl.col("Cost") + pl.col("Cost").shift(1))
            .otherwise(pl.col("Cost"))
            .alias("Cost_Input_Return")

        ])

        return data
    
    def cad_conv(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Converts various financial metrics from their original currency to CAD using the provided exchange rate.

        Parameters:
        data (pl.DataFrame): A Polars DataFrame containing the financial metrics and the exchange rate column 'Close_fx'.

        Returns:
        pl.DataFrame: A new Polars DataFrame with additional columns for each financial metric converted to CAD.
        """

        data = data.with_columns([
            (pl.col("P&L") * pl.col("Close_fx")).alias("P&L_CAD"),
            (pl.col("P&L_Daily") * pl.col("Close_fx")).alias("P&L_Daily_CAD"),
            (pl.col("Cost") * pl.col("Close_fx")).alias("Cost_CAD"),
            (pl.col("CF_Sum") * pl.col("Close_fx")).alias("CF_Sum_CAD"),
            (pl.col('CF') * pl.col('Close_fx')).alias('CF_CAD'),
            (pl.col("Unrealized_P&L") * pl.col("Close_fx")).alias("Unrealized_P&L_CAD"),
            (pl.col("Realized_P&L") * pl.col("Close_fx")).alias("Realized_P&L_CAD"),
            (pl.col("Position_Value") * pl.col("Close_fx")).alias("Position_Value_CAD"),
            (pl.col("Cost_Input_Return") * pl.col("Close_fx")).alias("Cost_Input_Return_CAD")
        ])

        return data

    def select_col_complete(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Select specific columns from the given DataFrame.
        Parameters:
        data (pl.DataFrame): The input DataFrame containing various columns.
        Returns:
        pl.DataFrame: A DataFrame containing only the selected columns
        """

        data = data.select([
            'Date',
            'Ticker',
            'Exchange',
            'Currency',
            'Close_fx',
            'Close',
            'Open',
            f'EMA_{self.window_size_macd_st}',
            f'EMA_{self.window_size_macd_lt}',
            'MACD',
            'Signal',
            'Trade',
            'Position',
            'Cost',
            'Cost_CAD',
            'Cost_Input_Return',
            'Cost_Input_Return_CAD',
            'CF',
            'CF_CAD',
            'CF_Sum',
            'CF_Sum_CAD',
            'Position_Value',
            'Position_Value_CAD',
            'Realized_P&L',
            'Realized_P&L_CAD',
            'Unrealized_P&L',
            'Unrealized_P&L_CAD',
            'P&L_Daily',
            'P&L_Daily_CAD',
            'P&L',
            'P&L_CAD'
        ])

        return data

    #Date DF functions   
    def df_date(self, data: pl.DataFrame) -> pl.DataFrame:

        """
        Processes the input DataFrame to compute various financial metrics and returns a new DataFrame with selected columns.
        Args:
            data (pl.DataFrame): Input DataFrame containing financial data with columns such as 'Date', 'Cost_Input_Return_CAD', 
                                 'P&L_Daily_CAD', 'P&L_Daily', 'Currency', 'Close_fx', 'CF', 'Realized_P&L_CAD', and 'Position_Value_CAD'.
        Returns:
            pl.DataFrame: A DataFrame with the following columns:
                - 'Date': The date of the record.
                - 'P&L_Daily_CAD': The daily profit and loss in CAD.
                - 'P&L_CAD': The cumulative profit and loss in CAD.
                - 'Position_Value_CAD': The position value in CAD.
                - 'HPR_Daily': The daily holding period return.
                - 'Time-Weighted Return': The time-weighted return.
        """

        df_date = data.group_by("Date").agg([
            pl.col("Cost_Input_Return_CAD").sum().alias("Cost_Input_Return_CAD"),
            pl.col("P&L_Daily_CAD").sum().alias("P&L_Daily_CAD"),
            pl.col("P&L_Daily").filter(pl.col("Currency") == "USD").sum().alias("P&L_Daily_USD_only"),
            pl.col("P&L_Daily").filter(pl.col("Currency") == "CAD").sum().alias("P&L_Daily_CAD_only"),
            pl.col("Close_fx").filter(pl.col("Currency") == "USD").last().alias("Close_fx"),
            pl.col("CF").filter(pl.col("Currency") == "USD").sum().alias("CF_USD"),
            pl.col("CF").filter(pl.col("Currency") == "CAD").sum().alias("CF_CAD"),
            pl.col("Realized_P&L_CAD").sum().alias("Realized_P&L_CAD"),
            pl.col("Position_Value_CAD").sum().alias("Position_Value_CAD")
        ])

        df_date = df_date.with_columns(
            (pl.col("Close_fx").fill_null(1)).alias("Close_fx")
        )

        #Sort by Date for next calculations
        df_date = df_date.sort("Date")

        #P&L Daily by currency
        df_date = df_date.with_columns([
            (pl.col("P&L_Daily_CAD_only").cum_sum()).alias("P&L_CAD_only"),
            (pl.col("P&L_Daily_USD_only").cum_sum()).alias("P&L_USD_only")
        ])

        #P&L TOTAl CAD
        df_date = df_date.with_columns([
            (pl.col("P&L_USD_only") * pl.col("Close_fx")).alias("P&L_USD_only_CAD")
        ])

        df_date = df_date.with_columns([
            (pl.col("P&L_USD_only_CAD") + pl.col("P&L_CAD_only")).alias("P&L_CAD")
        ])

        #Returns time-weighted
        df_date = df_date.with_columns([
            (pl.col("P&L_Daily_CAD") / pl.col("Cost_Input_Return_CAD")).alias("HPR_Daily")
        ])

        df_date = df_date.with_columns([
            (pl.col("HPR_Daily") + 1).alias("HPR_Daily1")
        ])

        #éviter d'avoir des 0 dans la colonne lors d'un cum_prod()
        df_date = df_date.with_columns(
            (pl.col("HPR_Daily1").fill_nan(1)).alias("HPR_Daily1"),
        )

        df_date = df_date.with_columns([
            (pl.col("HPR_Daily1").cum_prod()).alias("Total Return_plus1")
        ])

        df_date = df_date.with_columns([
            (pl.col("Total Return_plus1") - 1).alias("Time-Weighted Return")
        ])

        #Enlever les premieres dates qui n'ont pas de trades
        df_date = df_date.slice(self.window_size_macd_lt, df_date.shape[0] - self.window_size_macd_lt)

        #Select Cols
        df_date = df_date.select([
            'Date',
            'P&L_Daily_CAD',
            'P&L_CAD',
            'Position_Value_CAD',
            'HPR_Daily',
            'Time-Weighted Return'
        ])
    
        return df_date
    
    #Stats functions
    def pnl_total_cad(self, df_ticker: pl.DataFrame) -> float:

        """
        Calculate the total profit and loss (P&L) in CAD from the given DataFrame.
        Args:
            df_ticker (pl.DataFrame): A Polars DataFrame containing a column "P&L_CAD" 
                                      which represents the profit and loss values in CAD.
        Returns:
            float: The total P&L in CAD.
        """
        
        pnl = df_ticker.select(pl.col("P&L_CAD").sum()).item()

        return pnl
    
    def nb_days(self, df_date: pl.DataFrame) -> float:
        
        """
        Calculate the number of days in the given DataFrame.

        Args:
            df_date (pl.DataFrame): A Polars DataFrame containing date information.

        Returns:
            float: The number of days (rows) in the DataFrame.
        """

        nb_days = df_date.height

        return nb_days

    def returns_stats(self, df_date: pl.DataFrame, nb_days: float) -> float:

        """
        Calculate the annualized return from a given DataFrame of returns over a specified number of days.

        Args:
            df_date (pl.DataFrame): A Polars DataFrame containing a column "Time-Weighted Return".
            nb_days (float): The number of days over which the returns are calculated.

        Returns:
            float: The annualized return.
        """

        total_return = df_date.select(pl.col("Time-Weighted Return")).tail(1).item()

        annualized_return = (1 + total_return) ** (252 / nb_days) - 1

        return annualized_return
    
    def sd_stats(self, df_date: pl.DataFrame, nb_days) -> float:

        """
        Calculate the standard deviation of the 'HPR_Daily' column over the last `nb_days` days.
        Parameters:
        df_date (pl.DataFrame): A Polars DataFrame containing at least the 'HPR_Daily' column.
        nb_days (int): The number of days over which to calculate the standard deviation.
        Returns:
        float: The standard deviation of the 'HPR_Daily' column over the specified period.
        """

        df_date = df_date.with_columns(
            pl.col('HPR_Daily').fill_nan(0)
        )
        
        sd = df_date.tail(nb_days).select(pl.col("HPR_Daily").std()).item()

        return sd

    def stats(self, df_date: pl.DataFrame, df_ticker: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate and return various statistical metrics for the given data.
        Args:
            df_date (pl.DataFrame): DataFrame containing date-related data.
            df_ticker (pl.DataFrame): DataFrame containing ticker-related data.
            data (pl.DataFrame): DataFrame containing the main data for analysis.
        Returns:
            pl.DataFrame: DataFrame containing the calculated statistics:
                - "Total P&L_CAD": Total profit and loss in CAD.
                - "Annualized_Return": Annualized return.
                - "Log_Annualized_Return": Logarithm of the annualized return.
                - "Standard_Deviation": Annualized standard deviation.
                - "Risk_Free_Rate": Risk-free rate.
                - "Sharpe_Ratio": Sharpe ratio.
                - "Number of Days": Number of days in the period.
        """

        nb_days = self.nb_days(df_date)

        pnl = self.pnl_total_cad(df_ticker)
        
        annualized_return = self.returns_stats(df_date, nb_days)

        log_annualized_return = math.log(1 + annualized_return)

        sd = self.sd_stats(df_date, nb_days)

        sd_annualized = sd * math.sqrt(252)
        
        rf = self.rf

        sharpe_ratio = (log_annualized_return - rf) / sd_annualized

        stats = {
            "Total P&L_CAD": pnl,
            "Annualized_Return": annualized_return,
            "Log_Annualized_Return": log_annualized_return,
            "Standard_Deviation": sd_annualized,
            "Risk_Free_Rate": rf,
            "Sharpe_Ratio": sharpe_ratio,
            "Number of Days": nb_days
        }

        df_stats = pl.DataFrame([stats])

        return df_stats
    
    #functions to run the backtester
    def run_strategy(self, type = None) -> tuple:

        data = self.validation_data

        data = self.ema(data)

        data = self.macd(data)

        data = self.signal(data)

        data = self.add_fx(data)

        if type == 'long': 

            data, trade_df = self.trade(data, 'long')

        elif type == 'short':

            data, trade_df = self.trade(data, 'short')

        else:

            data, trade_df = self.trade(data)

        data = self.cost_input_return(data)

        data = self.realized_pnl(trade_df, data)

        data = self.position_value(data)

        data = self.unrealized_pnl(data)

        data = self.cf(data)

        data = self.pnl(data)

        data = self.cad_conv(data)

        data = self.select_col_complete(data)

        pivot_ticker = self.df_ticker(data)

        df_date = self.df_date(data)

        df_stats = self.stats(df_date, pivot_ticker, data)

        return data, pivot_ticker, df_date, df_stats, trade_df

    def run_backtest_optimizer(self, parameters: list, training_data: pl.DataFrame) -> float:
            
            self.set_parameters(window_size_macd_st = int(parameters[0]), window_size_macd_lt = int(parameters[1]), training_data = training_data)

            data = self.training_data

            data = self.ema(data)

            data = self.macd(data)

            data = self.signal(data)

            data = self.add_fx(data)

            data, trade_df = self.trade(data)

            data = self.cost_input_return(data)

            data = self.realized_pnl(trade_df, data)

            data = self.position_value(data)

            data = self.unrealized_pnl(data)

            data = self.cf(data)

            data = self.pnl(data)

            data = self.cad_conv(data)

            data = self.select_col_complete(data)

            pivot_ticker = self.df_ticker(data)

            df_date = self.df_date(data)

            df_stats = self.stats(df_date, pivot_ticker, data)

            sharpe_ratio = df_stats.select(pl.col("Sharpe_Ratio")).item()

            return sharpe_ratio
    
    def optimize_parameters(self, window_size_st_range: list, window_size_lt_range: list, training_data: pl.DataFrame) -> tuple:

        """
        Optimize the parameters for the moving average crossover strategy.
        This function searches for the best combination of short-term and long-term 
        window sizes for the moving average crossover strategy by evaluating all 
        valid combinations within the provided ranges. A combination is considered 
        valid if the short-term window size is less than the long-term window size.
        Args:
            window_size_st_range (list): A list of possible short-term window sizes.
            window_size_lt_range (list): A list of possible long-term window sizes.
            training_data (pl.DataFrame): The training data to be used for backtesting.
        Returns:
            tuple: A tuple containing the best combination of short-term and long-term 
                   window sizes as a Polars DataFrame with columns "window_size_st" 
                   and "window_size_lt".
        """

        valid_combinations = [(st, lt) for st, lt in itertools.product(window_size_st_range, window_size_lt_range) if st < lt]

        best_combination = None
        best_score = -float('inf')
        
        for st, lt in valid_combinations:
            score = self.run_backtest_optimizer([st, lt], training_data)
            if score > best_score: 
                best_score = score
                best_combination = (st, lt)
        
        #convert to df
        best_combination = pl.DataFrame([best_combination], schema=["window_size_st", "window_size_lt"], orient="row")

        return best_combination

class BuyAndHold(Base_Functions):

    def __init__(self,
                 fx_data: pl.DataFrame,
                 mapping_fx: pl.DataFrame,
                 validation_data: pl.DataFrame,
                 window_size_macd_lt: int,
                 rf: float):
        
        super().__init__(fx_data, mapping_fx)
        self.validation_data = validation_data
        self.window_size_macd_lt = window_size_macd_lt
        self.rf = rf

    def set_parameters(self,window_size_macd_lt: int = None, validation_data: pl.DataFrame = None):
    
        if window_size_macd_lt is not None:
            self.window_size_macd_lt = window_size_macd_lt

        if validation_data is not None:
            self.validation_data = validation_data

    def buy_hold_strategy(self, data: pl.DataFrame) -> pl.DataFrame:
        
        """
        Implements a buy and hold strategy on the provided data.

        Parameters:
        data (pl.DataFrame): A Polars DataFrame containing the trading data with columns "Ticker", "Open", "Close", and "Close_fx".

        Returns:
        pl.DataFrame: A Polars DataFrame with the following columns:
            - "Ticker": The ticker symbol of the stock.
            - "Open": The opening price of the stock.
            - "Close": The closing price of the stock.
            - "Close_fx": The closing price of the stock in CAD.
            - "Cost": The cost price of the stock.
            - "P&L": The profit and loss from the buy and hold strategy.
            - "P&L_Daily": The daily profit and loss.
            - "P&L_Daily_CAD": The daily profit and loss in CAD.
            - "P&L_CAD": The total profit and loss in CAD.
            - "Cost_CAD": The cost price of the stock in CAD.
        """

        df_buy_price = (
            data.group_by("Ticker")
            .agg(pl.col("Open").head(self.window_size_macd_lt).last().alias("Cost"))
        )

        df_buy_hold = data.join(df_buy_price, on="Ticker")

        df_buy_hold = df_buy_hold.with_columns(
            (pl.col("Close") - pl.col("Cost")).alias("P&L")
        )

        df_buy_hold = df_buy_hold.with_columns(
            pl.when(pl.col("Ticker") == pl.col("Ticker").shift(1))
            .then(pl.col("P&L") - pl.col("P&L").shift(1))
            .otherwise(pl.col("P&L"))
            .alias("P&L_Daily")
        )

        df_buy_hold = df_buy_hold.with_columns([
            (pl.col("P&L_Daily") * pl.col("Close_fx")).alias("P&L_Daily_CAD"),
            (pl.col("P&L") * pl.col("Close_fx")).alias("P&L_CAD"),
            (pl.col("Cost") * pl.col("Close_fx")).alias("Cost_CAD")
        ])

        return df_buy_hold
    
    def group_date_buy_hold(self, data: pl.DataFrame) -> pl.DataFrame:

        df_date = data.group_by(["Exchange", "Date"]).agg([
            pl.col("Cost_CAD").sum(),
            pl.col("P&L_Daily_CAD").sum(),
        ])

        df_date = df_date.sort(
            by=["Exchange", "Date"]
        )

        df_date = df_date.with_columns(
            (pl.col("P&L_Daily_CAD") / pl.col("Cost_CAD")).alias("Returns_Daily")
        )

        df_date = df_date.with_columns(
            pl.col("Date").rank("dense").over("Exchange").alias("Rank")
        )

        df_date = df_date.filter(
            pl.col("Rank") > self.window_size_macd_lt
        )

        #éviter d'avoir des 0 dans la colonne lors d'un cum_prod()
        df_date = df_date.with_columns(
            
            (pl.col("Returns_Daily").fill_nan(0)).alias("Returns_Daily")

        )

        df_date = df_date.with_columns(
            ((pl.col("Returns_Daily") + 1).cum_prod().over("Exchange") - 1).alias("Total Return")
        )

        df_date = df_date.select([
            'Exchange',
            'Date',
            'Cost_CAD',
            'P&L_Daily_CAD',
            'Returns_Daily',
            'Total Return',
        ])

        return df_date
    
    def return_by_date_all_exchange(self, df_group_date: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate the mean daily return and cumulative total return by date for all exchanges.

        Parameters:
        df_group_date (pl.DataFrame): A DataFrame containing grouped data by date with a column "Returns_Daily".

        Returns:
        pl.DataFrame: A DataFrame with the mean daily return and cumulative total return by date.
        """

        df_date_return = (
            df_group_date.group_by("Date")
            .agg(pl.col("Returns_Daily").mean().alias("Mean Return")))

        df_date_return = df_date_return.sort("Date")

        df_date_return = df_date_return.with_columns(
            ((pl.col("Mean Return") + 1).cum_prod() - 1).alias("Total Return")
        )

        return df_date_return

    def return_sd_buy_hold(self, df_date: pl.DataFrame) -> float:

        df_date_return = (
            df_date.group_by("Exchange")
            .agg([
                ((pl.col("Returns_Daily") + 1).product() - 1).alias("Total Return"),
                (pl.count("Date")).alias("nb_days")
            ])
        )

        df_date_return = df_date_return.with_columns(
            ((pl.col("Total Return") + 1) ** (252 / pl.col("nb_days")) - 1).alias("Annualized Return")
        )

        df_nb_days = df_date_return.select(["Exchange", "nb_days"])

        df_nb_days = df_nb_days.with_columns(
            pl.col("nb_days").cast(pl.Float64).alias("nb_days")
        )

        df_date_sd = df_date.join(df_nb_days, on="Exchange").group_by("Exchange").agg([
            pl.col("Returns_Daily").alias("Returns_Daily"),
        ])

        df_date_sd = df_date_sd.with_columns(
            pl.col("Returns_Daily").list.eval(pl.element().std()).alias("Std Deviation")
        )

        df_result = df_date_return.join(df_date_sd, on="Exchange")

        df_result = df_result.with_columns(
            pl.col("Std Deviation").list.first().alias("Std Deviation")
        )

        average_return = df_result.select(pl.col("Annualized Return").mean()).item()

        sd = df_result.select(pl.col("Std Deviation").mean()).item()

        return average_return, sd

    def stats_buy_hold(self, df_date: pl.DataFrame) -> pl.DataFrame:

        pnl = float(0)
        
        nb_days = df_date.height

        average_return, sd = self.return_sd_buy_hold(df_date)

        annualized_return = average_return

        log_annualized_return = math.log(1 + annualized_return)

        sd = sd

        sd_annualized = sd * math.sqrt(252)

        rf = self.rf

        sharpe_ratio = (log_annualized_return - rf) / sd_annualized

        stats = {
            "Total P&L_CAD": pnl,
            "Annualized_Return": annualized_return,
            "Log_Annualized_Return": log_annualized_return,
            "Standard_Deviation": sd_annualized,
            "Risk_Free_Rate": rf,
            "Sharpe_Ratio": sharpe_ratio,
            "Number of Days": nb_days
        }

        df_stats = pl.DataFrame([stats])

        return df_stats
    
    def run_buy_hold(self) -> tuple:

        data = self.add_fx(self.validation_data)

        df_buy_hold = self.buy_hold_strategy(data)

        df_buy_hold_date = self.group_date_buy_hold(df_buy_hold)

        pivot_ticker = self.df_ticker(df_buy_hold)

        df_stats_buy_hold = self.stats_buy_hold(df_buy_hold_date)

        return df_buy_hold, df_buy_hold_date, df_stats_buy_hold, pivot_ticker
    
#Graphs functions
class Graphs(MA_Backtester):

    def __init__(self,
                 fx_data,
                 mapping_fx,
                 training_data,
                 validation_data,
                 capital,
                 window_size_macd_st,
                 window_size_macd_lt,
                 rf
                ):
        
        super().__init__(fx_data,
                 mapping_fx,
                 training_data, 
                 validation_data,
                 capital, 
                 window_size_macd_st, 
                 window_size_macd_lt,
                 rf)

    def line_graph_subplot(self, df_date, df_date_long, df_date_short, df_date_buy_hold, 
                        df_date_tsx, df_date_long_tsx, df_date_short_tsx, df_date_buy_hold_tsx,
                        df_date_sp500, df_date_long_sp500, df_date_short_sp500, df_date_buy_hold_sp500):

        colors = {
            "long_short": "blue",
            "short": "red",
            "long": "green",
            "buy_hold": "orange"
        }

        # Create subplots: 1 row for each dataset
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,  
            subplot_titles=('All', 'TSX', 'SP500')
        )

        # --- All Data ---
        # Adjust Date
        nb_days = df_date.height - self.window_size_macd_lt
        df_date = df_date.tail(nb_days)
        df_date_short = df_date_short.tail(nb_days)
        df_date_long = df_date_long.tail(nb_days)
        df_date_buy_hold = df_date_buy_hold.tail(nb_days)

        fig.add_trace(go.Scatter(
            x=df_date['Date'], y=df_date['Time-Weighted Return'], mode='lines',
            name='Long/Short', line=dict(color=colors["long_short"]), legendgroup="group1"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_date['Date'], y=df_date_short['Time-Weighted Return'], mode='lines',
            name='Short', line=dict(color=colors["short"]), legendgroup="group1"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_date['Date'], y=df_date_long['Time-Weighted Return'], mode='lines',
            name='Long', line=dict(color=colors["long"]), legendgroup="group1"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_date['Date'], y=df_date_buy_hold['Total Return'], mode='lines',
            name='Buy & Hold', line=dict(color=colors["buy_hold"]), legendgroup="group1"), row=1, col=1)

        # --- TSX ---
        # Adjust Date
        nb_days = df_date_tsx.height - self.window_size_macd_lt
        df_date_tsx = df_date_tsx.tail(nb_days)
        df_date_short_tsx = df_date_short_tsx.tail(nb_days)
        df_date_long_tsx = df_date_long_tsx.tail(nb_days)
        df_date_buy_hold_tsx = df_date_buy_hold_tsx.tail(nb_days)

        fig.add_trace(go.Scatter(
            x=df_date_tsx['Date'], y=df_date_tsx['Time-Weighted Return'], mode='lines',
            name='Long/Short', line=dict(color=colors["long_short"]), legendgroup="group1", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_date_tsx['Date'], y=df_date_short_tsx['Time-Weighted Return'], mode='lines',
            name='Short', line=dict(color=colors["short"]), legendgroup="group1", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_date_tsx['Date'], y=df_date_long_tsx['Time-Weighted Return'], mode='lines',
            name='Long', line=dict(color=colors["long"]), legendgroup="group1", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_date_tsx['Date'], y=df_date_buy_hold_tsx['Total Return'], mode='lines',
            name='Buy & Hold', line=dict(color=colors["buy_hold"]), legendgroup="group1", showlegend=False), row=2, col=1)

        # --- SP500 ---
        # Adjust Date
        nb_days = df_date_sp500.height - self.window_size_macd_lt
        df_date_sp500 = df_date_sp500.tail(nb_days)
        df_date_short_sp500 = df_date_short_sp500.tail(nb_days)
        df_date_long_sp500 = df_date_long_sp500.tail(nb_days)
        df_date_buy_hold_sp500 = df_date_buy_hold_sp500.tail(nb_days)

        fig.add_trace(go.Scatter(
            x=df_date_sp500['Date'], y=df_date_sp500['Time-Weighted Return'], mode='lines',
            name='Long/Short', line=dict(color=colors["long_short"]), legendgroup="group1", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df_date_sp500['Date'], y=df_date_short_sp500['Time-Weighted Return'], mode='lines',
            name='Short', line=dict(color=colors["short"]), legendgroup="group1", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df_date_sp500['Date'], y=df_date_long_sp500['Time-Weighted Return'], mode='lines',
            name='Long', line=dict(color=colors["long"]), legendgroup="group1", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df_date_sp500['Date'], y=df_date_buy_hold_sp500['Total Return'], mode='lines',
            name='Buy & Hold', line=dict(color=colors["buy_hold"]), legendgroup="group1", showlegend=False), row=3, col=1)

        # Update layout
        fig.update_layout(
            title='MACD Strategy',
            template='simple_white',
            height=900,  
            showlegend=True,
            legend=dict(
                x=1.02,  
                y=1, 
                bordercolor="black",
                borderwidth=1
            )
        )

        # Update x-axes format for all subplots
        fig.update_xaxes(
            tickmode='linear',
            dtick='M1',  
            tickformat='%b %Y'
        )

        return fig
    
    def strategy_vs_bh_graph(self, df_date, df_date_buy_hold):

        fig = go.Figure()

        nb_days = df_date.height
        df_date = df_date.tail(nb_days)
        df_date_buy_hold = df_date_buy_hold.tail(nb_days)

        fig.add_trace(go.Scatter(
            x=df_date['Date'], 
            y=df_date['Time-Weighted Return'], 
            mode='lines',
            name='Long/Short', 
            line=dict(color="blue"), 
            legendgroup="group1"
        ))
        
        fig.add_trace(go.Scatter(
            x=df_date['Date'], 
            y=df_date_buy_hold['Total Return'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color="orange"),
            legendgroup="group1"
        ))

        fig.update_layout(
            title='MA Crossover Strategy',
            template='simple_white',
            height=900,  
            showlegend=True,
            legend=dict(
                x=1.02,  
                y=1, 
                bordercolor="black",
                borderwidth=1
            ),
            yaxis=dict(
                tickformat=".0%",
                title="Cumulative Return (%)",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1 
            )
        )

        return fig