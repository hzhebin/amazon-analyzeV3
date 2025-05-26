import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor
import optuna

class StrategyOptimizer:
    def __init__(self):
        self.sales_model = None
        self.profit_model = None
        self.df = None
        self.features = ['售价', 'CPC', '广告花费', 'ACOS', 'Sessions', 'CVR']

    def clean_percent(self, col):
        return col.astype(str).str.replace('%', '', regex=False).str.strip().replace('', '0').fillna('0').astype(float)

    def preprocess(self, df):
        df = df.copy()
        df['ACOS'] = self.clean_percent(df['ACOS']) / 100
        df['CVR'] = self.clean_percent(df['CVR']) / 100
        df['结算毛利率'] = self.clean_percent(df['结算毛利率'])
        df['利润'] = df['销售额'] * (df['结算毛利率'] / 100)
        return df.dropna(subset=self.features + ['订单量', '利润'])

    def train(self, df):
        self.df = self.preprocess(df)
        X = self.df[self.features]
        y_sales = self.df['订单量']
        y_profit = self.df['利润']

        self.sales_model = XGBRegressor(n_estimators=100, max_depth=4)
        self.profit_model = XGBRegressor(n_estimators=100, max_depth=4)
        self.sales_model.fit(X, y_sales)
        self.profit_model.fit(X, y_profit)

    def optimize_sales(self):
        def objective(trial):
            input_data = self._sample_input(trial)
            df_input = pd.DataFrame([input_data])
            return self.sales_model.predict(df_input)[0]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        best = study.best_params
        pred_orders = float(self.sales_model.predict(pd.DataFrame([best]))[0])
        pred_profit = float(self.profit_model.predict(pd.DataFrame([best]))[0])
        return {
            "策略类型": "销量优先",
            "建议参数": best,
            "预计14天销量": round(pred_orders * 14, 1),
            "预计14天利润": round(pred_profit * 14, 2),
            "执行建议": "通过略微牺牲利润来换取更高销量，适合清库存、打排名。"
        }

    def optimize_profit(self):
        def objective(trial):
            input_data = self._sample_input(trial)
            df_input = pd.DataFrame([input_data])
            return self.profit_model.predict(df_input)[0]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        best = study.best_params
        pred_orders = float(self.sales_model.predict(pd.DataFrame([best]))[0])
        pred_profit = float(self.profit_model.predict(pd.DataFrame([best]))[0])
        return {
            "策略类型": "利润最大化",
            "建议参数": best,
            "预计14天销量": round(pred_orders * 14, 1),
            "预计14天利润": round(pred_profit * 14, 2),
            "执行建议": "优先考虑利润率，在合理流量前提下提高单位盈利能力。"
        }

    def _sample_input(self, trial):
        return {
            "售价": trial.suggest_float("售价", 5, 25),
            "CPC": trial.suggest_float("CPC", 0.1, 1.5),
            "广告花费": trial.suggest_float("广告花费", 5, 500),
            "ACOS": trial.suggest_float("ACOS", 0.05, 0.4),
            "Sessions": trial.suggest_float("Sessions", 50, 1000),
            "CVR": trial.suggest_float("CVR", 0.01, 0.3)
        }
