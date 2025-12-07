from data.btc_iv_fetcher import BtcIVFetcher
from data.preprocessor import DataPreprocessor
from model.model_executor import ModelExecutor

oct_1_2021_epoch_ms = 1633046400000
oct_1_2025_epoch_ms = 1759276800000
btc_file_output = "data/btc_volatility_index.csv"
vix_file_input = "data/VIX_History.csv"
merged_data_file = "data/merged_iv_vix_daily.csv"

def main():
    fetcher = BtcIVFetcher(
        oct_1_2021_epoch_ms,
        oct_1_2025_epoch_ms,
        output_file=btc_file_output
    )
    fetcher.run()

    processor = DataPreprocessor(
        btc=btc_file_output,
        vix=vix_file_input,
        output_file=merged_data_file
    )
    processor.run()

    executor = ModelExecutor(data_file="data/merged_iv_vix_daily.csv")
    executor.run()

if __name__ == "__main__":
    main()