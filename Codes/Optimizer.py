import Import_Yfinance as yf_data
import Utils as utils
import MA_Backtest as MA
from pathlib import Path

def main():

    #Dates
    dates_class = utils.Dates(YEARS)
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
    data = yf_data.format_utf8_to_date(data)

    #Split data training / validation
    training_data, validation_data = yf_data.split_data_training_validation(data, CUTOFF_DATE)

    #Import FX
    path = f"{INPUT_DAILY}/FX_{today}.csv"
    fx_data = utils.read_csv(path)
    fx_data = dates_class.format_int64_to_date(fx_data)

    #Mapping FX
    path = f"{INPUT_MAPPING}/Exchange_Currency.csv"
    mapping_fx = utils.read_csv(path)

    tester = MA.MA_Backtester(fx_data, mapping_fx, training_data, validation_data, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, RF)
    tester.set_parameters(window_size_macd_st = WINDOW_SIZE_MACD_ST, window_size_macd_lt = WINDOW_SIZE_MACD_LT)

    best_comb = tester.optimize_parameters(WINDOW_SIZE_MACD_ST_RANGE, WINDOW_SIZE_MACD_LT_RANGE, training_data)

    #Export Data
    path = f"{OUTPUT}/best_comb.csv"
    utils.export_df_csv(best_comb, path)

if __name__ == "__main__":

    #####Define Variables#####
    CUTOFF_DATE = '2023-01-01' #Cut-off date for training data
    YEARS = 15 #Number of years of data to import

    RF = 0.01
    CAPITAL = 300
    WINDOW_SIZE_MACD_ST = 25
    WINDOW_SIZE_MACD_LT = 25

    #Optimization parameters
    WINDOW_SIZE_MACD_ST_RANGE = range(10, 30, 5)
    WINDOW_SIZE_MACD_LT_RANGE = range(30, 100, 5)

    main(CUTOFF_DATE, YEARS, RF, CAPITAL, WINDOW_SIZE_MACD_ST, WINDOW_SIZE_MACD_LT, WINDOW_SIZE_MACD_ST_RANGE, WINDOW_SIZE_MACD_LT_RANGE)
