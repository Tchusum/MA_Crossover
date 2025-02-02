import Import_Yfinance as yf_data
import Utils as utils
import MA_Backtest as MA
from pathlib import Path

def main(CUTOFF_DATE, RF, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, WINDOW_SIZE_MACD_ST_RANGE, WINDOW_SIZE_MACD_LT_RANGE):

    #Dates
    dates_class = utils.Dates
    today = dates_class.today_str()

    #Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_DAILY = BASE_DIR / "Input/Daily_Data/"
    INPUT_MAPPING = BASE_DIR / "Input/Mapping/"
    OUTPUT = BASE_DIR / "Output/"

    #Import data
    path = f"{INPUT_DAILY}/Stocks_{today}.csv"
    data = utils.read_csv(path)

    #Format date
    data = utils.Dates.format_col_to_date(data, "Date")

    #Split data training / validation
    training_data, validation_data = yf_data.data.split_data_training_validation(data, CUTOFF_DATE)

    #Import FX
    path = f"{INPUT_DAILY}/FX_{today}.csv"
    fx_data = utils.read_csv(path)
    fx_data = utils.Dates.format_col_to_date(fx_data, "Date")

    #Mapping FX
    path = f"{INPUT_MAPPING}/Exchange_Currency.csv"
    mapping_fx = utils.read_csv(path)

    #base_functions = MA.Base_Functions(fx_data, mapping_fx)
    macd_backtest = MA.MA_Backtester(fx_data, mapping_fx, training_data, validation_data, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, RF)
    macd_backtest.set_parameters(window_size_macd_st = WINDOW_SIZE_MACD_ST, window_size_macd_lt = WINDOW_SIZE_MACD_LT)

    best_comb = macd_backtest.optimize_parameters(WINDOW_SIZE_MACD_ST_RANGE, WINDOW_SIZE_MACD_LT_RANGE, training_data)

    #Export Data
    path = f"{OUTPUT}/best_comb.csv"
    utils.export_df_csv(best_comb, path)

    return best_comb

if __name__ == "__main__":

    #####Define Variables#####
    CUTOFF_DATE = '2023-01-01' #Cut-off date for training data

    RF = 0.01
    CAPITAL = 300
    WINDOW_SIZE_MACD_ST = 25
    WINDOW_SIZE_MACD_LT = 25

    #Optimization parameters
    WINDOW_SIZE_MACD_ST_RANGE = range(10, 30, 5)
    WINDOW_SIZE_MACD_LT_RANGE = range(30, 100, 5)

    main(CUTOFF_DATE, RF, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, WINDOW_SIZE_MACD_ST_RANGE, WINDOW_SIZE_MACD_LT_RANGE)
