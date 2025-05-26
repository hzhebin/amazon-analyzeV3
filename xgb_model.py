import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import shap

class XGBModel:
    def __init__(self):
        self.model_sales = None
        self.model_profit = None
        self.features = ['售价', 'CPC', '广告花费', 'ACOS', 'Sessions', 'CVR']
        self.explainer = None
        self.shap_values = None

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()
        df['ACOS'] = df['ACOS'].astype(str).str.replace('%', '').astype(float) / 100
        df['CVR'] = df['CVR'].astype(str).str.replace('%', '').astype(float) / 100
        df['结算毛利率'] = df['结算毛利率'].astype(str).str.replace('%', '').astype(float)
        df['利润'] = df['销售额'] * (df['结算毛利率'] / 100)
        return df.dropna(subset=self.features + ['订单量', '利润'])

    def train(self, df: pd.DataFrame):
        df = self.preprocess(df)
        X = df[self.features]
        y_sales = df['订单量']
        y_profit = df['利润']

        self.model_sales = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        self.model_profit = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        self.model_sales.fit(X, y_sales)
        self.model_profit.fit(X, y_profit)

        self.explainer = shap.Explainer(self.model_profit, X)
        self.shap_values = self.explainer(X)

    def predict_sales(self, variables: dict):
        x = self._prepare_input(variables)
        return float(self.model_sales.predict(x))

    def predict_profit(self, variables: dict):
        x = self._prepare_input(variables)
        return float(self.model_profit.predict(x))

    def _prepare_input(self, variables):
        return pd.DataFrame([{k: variables[k] for k in self.features}])

    def shap_summary_plot(self):
        shap.plots.beeswarm(self.shap_values)

    def shap_bar_plot(self):
        shap.plots.bar(self.shap_values)
