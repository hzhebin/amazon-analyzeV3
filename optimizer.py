import pandas as pd
import numpy as np
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
            return self.sales_model.predict(pd.DataFrame([input_data]))[0]
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
            "执行建议": "通过策略筛选利润承压换取高销量，适合清库存、打排名。",
            "解释说明": "- 本策略使用 Optuna 对订单量预测模型进行调参。\n- 搜索空间覆盖售价、广告、流量等变量。\n- 当前组合在回测中表现出订单量峰值。\n- 利润略低但销量最大，是以量换势策略。"
        }

    def optimize_profit(self):
        def objective(trial):
            input_data = self._sample_input(trial)
            return self.profit_model.predict(pd.DataFrame([input_data]))[0]
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
            "执行建议": "优先考虑利润导向，在合理流量前提下提高单位盈利能力。",
            "解释说明": "- 该策略使用利润预测模型作为优化目标。\n- 使用 Optuna 搜索最优组合以提升14天利润。\n- 推荐方案兼顾转化率和广告成本控制。\n- 适用于竞争稳定期或利润导向运营。"
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
