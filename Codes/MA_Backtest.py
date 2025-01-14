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
        Aggregates the given DataFrame by the 'Ticker' column and returns the last 'P&L_CAD' value for each ticker.
        Args:
            data (pl.DataFrame): The input DataFrame containing at least 'Ticker' and 'P&L_CAD' columns.
        Returns:
            pl.DataFrame: A DataFrame with each unique 'Ticker' and its corresponding last 'P&L_CAD' value.
        """

        pivot_ticker = data.group_by("Ticker").agg([
            pl.last("P&L_CAD").alias("P&L_CAD")
        ])

        return pivot_ticker
    
    def stats_df_combine(self, df_stats: pl.DataFrame, df_stats_long: pl.DataFrame, df_stats_short: pl.DataFrame,
                        df_stats_tsx: pl.DataFrame, df_stats_long_tsx: pl.DataFrame, df_stats_short_tsx: pl.DataFrame,
                        df_stats_sp500: pl.DataFrame, df_stats_long_sp500: pl.DataFrame, df_stats_short_sp500: pl.DataFrame,
                        df_stats_buy_hold: pl.DataFrame, df_stats_buy_hold_tsx: pl.DataFrame, df_stats_buy_hold_sp500: pl.DataFrame) -> pl.DataFrame:
        
        """
        Combine the statistics DataFrames for different set-up into a single DataFrame.
        Parameters:
        df_stats (pl.DataFrame): The DataFrame containing the statistics for the main data.
        df_stats_long (pl.DataFrame): The DataFrame containing the statistics for the long trades.
        df_stats_short (pl.DataFrame): The DataFrame containing the statistics for the short trades.
        df_stats_tsx (pl.DataFrame): The DataFrame containing the statistics for the TSX data.
        df_stats_long_tsx (pl.DataFrame): The DataFrame containing the statistics for the long trades on the TSX data.
        df_stats_short_tsx (pl.DataFrame): The DataFrame containing the statistics for the short trades on the TSX data.
        df_stats_sp500 (pl.DataFrame): The DataFrame containing the statistics for the SP500 data.
        df_stats_long_sp500 (pl.DataFrame): The DataFrame containing the statistics for the long trades on the SP500 data.
        df_stats_short_sp500 (pl.DataFrame): The DataFrame containing the statistics for the short trades on the SP500 data.
        df_stats_buy_hold (pl.DataFrame): The DataFrame containing the statistics for the buy and hold strategy on the main data.
        df_stats_buy_hold_tsx (pl.DataFrame): The DataFrame containing the statistics for the buy and hold strategy on the TSX data.
        df_stats_buy_hold_sp500 (pl.DataFrame): The DataFrame containing the statistics for the buy and hold strategy on the SP500 data.
        Returns:
        pl.DataFrame: The combined DataFrame containing the statistics for all the different set-ups.
        """
        df_stats = df_stats.with_columns(pl.lit("All Data - Long/Short").alias("Strategy"))
        df_stats_long = df_stats_long.with_columns(pl.lit("All Data - Long").alias("Strategy"))
        df_stats_short = df_stats_short.with_columns(pl.lit("All Data - Short").alias("Strategy"))
        df_stats_tsx = df_stats_tsx.with_columns(pl.lit("TSX - Long/Short").alias("Strategy"))
        df_stats_long_tsx = df_stats_long_tsx.with_columns(pl.lit("TSX - Long").alias("Strategy"))
        df_stats_short_tsx = df_stats_short_tsx.with_columns(pl.lit("TSX - Short").alias("Strategy"))
        df_stats_sp500 = df_stats_sp500.with_columns(pl.lit("SP500 - Long/Short").alias("Strategy"))
        df_stats_long_sp500 = df_stats_long_sp500.with_columns(pl.lit("SP500 - Long").alias("Strategy"))
        df_stats_short_sp500 = df_stats_short_sp500.with_columns(pl.lit("SP500 - Short").alias("Strategy"))
        df_stats_buy_hold = df_stats_buy_hold.with_columns(pl.lit("All Data - Buy & Hold ").alias("Strategy"))
        df_stats_buy_hold_tsx = df_stats_buy_hold_tsx.with_columns(pl.lit("TSX - Buy & Hold ").alias("Strategy"))
        df_stats_buy_hold_sp500 = df_stats_buy_hold_sp500.with_columns(pl.lit("SP500 - Buy & Hold ").alias("Strategy"))

        df_stats_combine = pl.concat([
            df_stats, df_stats_long, df_stats_short,
            df_stats_tsx, df_stats_long_tsx, df_stats_short_tsx,
            df_stats_sp500, df_stats_long_sp500, df_stats_short_sp500,
            df_stats_buy_hold, df_stats_buy_hold_tsx, df_stats_buy_hold_sp500
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

        data = data.with_columns([
            (pl.col("Position") * pl.col("Close")).alias("Position_Value")
        ])

        return data

    def unrealized_pnl(self, data: pl.DataFrame) -> pl.DataFrame:
            
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

        data = data.with_columns([

            pl.when(pl.col("Trade") != 0)
            .then(pl.col("Cost") + pl.col("Cost").shift(1))
            .otherwise(pl.col("Cost"))
            .alias("Cost_Input_Return")

        ])

        return data
    
    def cad_conv(self, data: pl.DataFrame) -> pl.DataFrame:

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

        #Select Cols
        df_date = df_date.select([
            'Date',
            'P&L_Daily_CAD',
            'P&L_CAD',
            'Position_Value_CAD',
            'Cost_Input_Return_CAD',
            'HPR_Daily',
            'Time-Weighted Return'
        ])
    
        return df_date
    
    #Stats functions
    def pnl_total_cad(self, df_ticker: pl.DataFrame) -> float:
        
        pnl = df_ticker.select(pl.col("P&L_CAD").sum()).item()

        return pnl
    
    def nb_days(self, df_date: pl.DataFrame) -> float:

        nb_days = df_date.height - self.window_size_macd_lt

        return nb_days

    def returns_stats(self, df_date: pl.DataFrame, nb_days: float) -> float:

        total_return = df_date.select(pl.col("Time-Weighted Return")).tail(1).item()

        annualized_return = (1 + total_return) ** (252 / nb_days) - 1

        return annualized_return
    
    def sd_stats(self, df_date: pl.DataFrame, nb_days) -> float:

        df_date = df_date.with_columns(
            pl.col('HPR_Daily').fill_nan(0)
        )
        
        sd = df_date.tail(nb_days).select(pl.col("HPR_Daily").std()).item()

        return sd

    def stats(self, df_date: pl.DataFrame, df_ticker: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:

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