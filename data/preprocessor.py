import pandas as pd

class DataPreprocessor:
    def __init__(self, btc, vix, output_file):
        self.btc = btc
        self.vix = vix
        self.output_file = output_file

    def prep(self):
        btc_df = pd.read_csv(self.btc)
        vix_df = pd.read_csv(self.vix)

        btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
        btc_df['date'] = btc_df['datetime'].dt.date
        btc_df['hour'] = btc_df['datetime'].dt.hour
        btc_df['iv'] = btc_df['close']
        btc_pivot = (
            btc_df
            .pivot_table(index='date', columns='hour', values='iv', aggfunc='mean')
            .reset_index()
        )
        btc_pivot.columns = ['date'] + [f'iv_{h:02d}' for h in range(24)]

        vix_df['DATE'] = pd.to_datetime(vix_df['DATE'], format='%m/%d/%Y')
        vix_df['date'] = vix_df['DATE'].dt.date
        vix_df['vix_close'] = vix_df['CLOSE']

        self.btc_to_daily = btc_pivot
        self.vix_to_daily = vix_df 

        
    def merge(self):
        merged = pd.merge(self.btc_to_daily, self.vix_to_daily[['date', 'vix_close']], on='date', how='inner')

        merged['btc_iv_mean'] = merged[[f'iv_{h:02d}' for h in range(24)]].mean(axis=1)
        merged['btc_iv_std'] = merged[[f'iv_{h:02d}' for h in range(24)]].std(axis=1)
        merged['btc_iv_min'] = merged[[f'iv_{h:02d}' for h in range(24)]].min(axis=1)
        merged['btc_iv_max'] = merged[[f'iv_{h:02d}' for h in range(24)]].max(axis=1)
        merged['btc_iv_range'] = merged['btc_iv_max'] - merged['btc_iv_min']

        merged['vix_t+1'] = merged['vix_close'].shift(-1)
        merged['vix_t+2'] = merged['vix_close'].shift(-2)
        merged['vix_t+3'] = merged['vix_close'].shift(-3)

        merged = merged.dropna(subset=['vix_t+1', 'vix_t+2', 'vix_t+3'])

        merged = merged.drop(columns=['vix_close'])

        merged.to_csv("merged_iv_vix_daily.csv", index=False)

    def run(self):
        self.prep()
        self.merge()