import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class ModelExecutor:
    def __init__(self, data_file):
        self.data_file = data_file

    def split_test_train(self): 
        df = pd.read_csv(self.data_file)
        self.dates = df['date']
        btc_hourly_cols = [f'iv_{i:02d}' for i in range(24)]
        btc_daily_cols = ['btc_iv_mean', 'btc_iv_std', 'btc_iv_min', 'btc_iv_max', 'btc_iv_range']
        vix_cols = ['vix_t+1', 'vix_t+2', 'vix_t+3']
        df = df[btc_hourly_cols + btc_daily_cols + vix_cols]
        df = df.apply(pd.to_numeric, errors='coerce')

        df[btc_hourly_cols] = df[btc_hourly_cols].apply(
            lambda row: row.interpolate(method='linear', limit_direction='both'), axis=1
        )
        df[btc_hourly_cols] = df[btc_hourly_cols].apply(lambda row: row.fillna(row.mean()), axis=1)

        df[vix_cols] = df[vix_cols].ffill()

        feature_cols = btc_hourly_cols + btc_daily_cols
        btc_data = df[feature_cols]
        vix_1day = df['vix_t+1']
        vix_2day = df['vix_t+2']
        vix_3day = df['vix_t+3']

        self.btc_train, self.btc_test, self.vix1_train, self.vix1_test, self.vix2_train, self.vix2_test, self.vix3_train, self.vix3_test = train_test_split(
            btc_data, vix_1day, vix_2day, vix_3day, test_size=0.2, shuffle=False
        )

        self.hourly_scaler = StandardScaler()
        self.daily_scaler = StandardScaler()

        btc_hourly_scaled_train = self.hourly_scaler.fit_transform(self.btc_train[btc_hourly_cols])
        btc_daily_scaled_train = self.daily_scaler.fit_transform(self.btc_train[btc_daily_cols]) * 3

        self.train_scale = np.hstack((btc_hourly_scaled_train, btc_daily_scaled_train))

        btc_hourly_scaled_test = self.hourly_scaler.transform(self.btc_test[btc_hourly_cols])
        btc_daily_scaled_test = self.daily_scaler.transform(self.btc_test[btc_daily_cols]) * 3

        self.test_scale = np.hstack((btc_hourly_scaled_test, btc_daily_scaled_test))

    def evaluate(self, model, btc_test, vix_test, label):
        preds = model.predict(btc_test)
        mse = mean_squared_error(vix_test, preds)
        r2 = r2_score(vix_test, preds)
        print(f"Future Day {label}: MSE: {mse:.4f}, R²: {r2:.4f}")
        return preds
    
    def graph_results(self, actual, results, label):
        dates = self.dates[-len(actual[0]):].reset_index(drop=True)
        step = max(1, len(dates) // 10)
        actual_min = min([act.min() for act in actual]) - 1
        actual_max = max([act.max() for act in actual]) + 1
        results_min = min([res.min() for res in results]) - 1
        results_max = max([res.max() for res in results]) + 1
        line_colors = ['yellow', 'orange', 'red']
        plt.figure(figsize=(10, 10))
        actual_axis = plt.gca()
        actual_axis.plot(dates, actual[0], color='green', linewidth=2, label='Observed VIX')
        actual_axis.set_ylim(actual_min, actual_max)
        actual_axis.set_xlabel('Date')
        actual_axis.set_ylabel('VIX Values')
        actual_axis.set_xticks(dates[::step])
        actual_axis.tick_params(axis='x', rotation=45)

        result_axis = actual_axis.twinx()
        for i, pred in enumerate(results):
            result_axis.plot(
                dates, pred,
                label=f'Predicted {label} {i+1}-Day',
                color=line_colors[i],
                linestyle='--'
            )
        result_axis.set_ylabel('Predicted VIX')
        result_axis.set_ylim(results_min, results_max)
        lines1, labels1 = actual_axis.get_legend_handles_labels()
        lines2, labels2 = result_axis.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.tight_layout()
        plt.show()
    
    def graph_feature_importance(self, day1, day2, day3, model):
        importance = [day1, day2, day3]
        labels = ["1-day", "2-day", "3-day"]
        feature_names = [f'iv_{i:02d}' for i in range(24)] + \
                        ['btc_iv_mean', 'btc_iv_std', 'btc_iv_min', 'btc_iv_max', 'btc_iv_range']
        for importances, label in zip(importance, labels):
            indices = np.argsort(importances)

            plt.figure(figsize=(10, 10))
            plt.title(f"Feature Importances ({label}) - {model}")
            plt.barh([feature_names[i] for i in indices], importances[indices])
            plt.xlabel("Importance")
            plt.show()

    def lasso(self):
        l1 = LassoCV(max_iter=10000)
        l2 = LassoCV(max_iter=10000)
        l3 = LassoCV(max_iter=10000)

        l1.fit(self.train_scale, self.vix1_train)
        l2.fit(self.train_scale, self.vix2_train)
        l3.fit(self.train_scale, self.vix3_train)

        print("Lasso Results:")
        day1 = self.evaluate(l1, self.test_scale, self.vix1_test, "1")
        day2 = self.evaluate(l2, self.test_scale, self.vix2_test, "2")
        day3 = self.evaluate(l3, self.test_scale, self.vix3_test, "3")

        actual = [self.vix1_test.reset_index(drop=True), 
                  self.vix2_test.reset_index(drop=True), 
                  self.vix3_test.reset_index(drop=True)]
        results = [day1, day2, day3]

        self.graph_results(actual, results, "Lasso")

        self.graph_feature_importance(np.abs(l1.coef_), np.abs(l2.coef_), np.abs(l3.coef_), "Lasso")



    def ridge(self):
        alphas = np.logspace(-3, 3, 50)

        r1 = RidgeCV(alphas=alphas)
        r2 = RidgeCV(alphas=alphas)
        r3 = RidgeCV(alphas=alphas)

        r1.fit(self.train_scale, self.vix1_train)
        r2.fit(self.train_scale, self.vix2_train)
        r3.fit(self.train_scale, self.vix3_train)

        print("Ridge Results:")
        day1 = self.evaluate(r1, self.test_scale, self.vix1_test, "1")
        day2 = self.evaluate(r2, self.test_scale, self.vix2_test, "2")
        day3 = self.evaluate(r3, self.test_scale, self.vix3_test, "3")
        actual = [self.vix1_test.reset_index(drop=True), 
                  self.vix2_test.reset_index(drop=True), 
                  self.vix3_test.reset_index(drop=True)]
        results = [day1, day2, day3]

        self.graph_results(actual, results, "Ridge")

        self.graph_feature_importance(np.abs(r1.coef_), np.abs(r2.coef_), np.abs(r3.coef_), "Ridge")


    def random_forest(self):
        rf_params = dict(n_estimators=500)
        rf1 = RandomForestRegressor(**rf_params)
        rf2 = RandomForestRegressor(**rf_params)
        rf3 = RandomForestRegressor(**rf_params)

        rf1.fit(self.train_scale, self.vix1_train)
        rf2.fit(self.train_scale, self.vix2_train)
        rf3.fit(self.train_scale, self.vix3_train)

        print("Random Forest Results:")
        day1 = self.evaluate(rf1, self.test_scale, self.vix1_test, "1")
        day2 = self.evaluate(rf2, self.test_scale, self.vix2_test, "2")
        day3 = self.evaluate(rf3, self.test_scale, self.vix3_test, "3")
        actual = [self.vix1_test.reset_index(drop=True), 
                  self.vix2_test.reset_index(drop=True), 
                  self.vix3_test.reset_index(drop=True)]
        results = [day1, day2, day3]

        self.graph_results(actual, results, "Random Forest")

        self.graph_feature_importance(rf1.feature_importances_, rf2.feature_importances_, rf3.feature_importances_, "Random Forest")

    def random_forest_estimators(self):
        num_trees = [50, 100, 250, 500, 1000]
        oobs = []
        for n in num_trees:
            rf = RandomForestRegressor(
                n_estimators=n,
                oob_score=True,
                bootstrap=True,
                n_jobs=-1,
                random_state=0
            )
            rf.fit(self.train_scale, self.vix1_train)
            oobs.append(1 - rf.oob_score_)

        for n, oob in zip(num_trees, oobs):
            graph_label = f"Trees: {n}"
            plt.plot(n, oob, 'bo')
            plt.text(n, oob, graph_label)
        plt.xlabel("Number of Trees")
        plt.ylabel("OOB Error Rate")
        plt.show()


    def run(self):
        self.split_test_train()
        self.lasso()
        self.ridge()
        # self.random_forest_estimators()
        self.random_forest()
