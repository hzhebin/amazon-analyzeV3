
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import optuna

class OperatorStrategyEngine:
    def __init__(self, inventory_qty, target_days):
        self.sales_model = None
        self.profit_model = None
        self.df = None
        self.features = ['售价', 'CPC', '广告花费', 'ACOS', 'Sessions', 'CVR']
        self.inventory_qty = inventory_qty
        self.target_days = target_days
        self.daily_target = inventory_qty / target_days

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

    def score_combination(self, variables):
        df_input = pd.DataFrame([variables])
        pred_sales = float(self.sales_model.predict(df_input)[0])
        pred_profit = float(self.profit_model.predict(df_input)[0])
        predicted_days = self.inventory_qty / max(pred_sales, 1e-6)

        # 广告效率指标
        ad_cost = variables["广告花费"]
        sales = pred_sales * variables["售价"]
        ACoAS = ad_cost / (ad_cost + sales) if (ad_cost + sales) else 0
        ASoAS = ad_cost / sales if sales else 0
        CPM = ad_cost / max(variables["Sessions"]/1000, 1e-6)
        CPO = ad_cost / max(pred_sales, 1e-6)

        ad_score = 1 - (ACoAS + CPM/10 + CPO/10) # 广告效率综合分, 权重可调

        # 动销评分 = 趋近目标周转天数的程度（越接近越高）
        deviation = abs(predicted_days - self.target_days)
        turnover_score = max(0, 1 - deviation / self.target_days)

        # 多维综合得分: 利润(60%)、动销(25%)、广告效率(15%)
        total_score = pred_profit * 0.6 + turnover_score * 0.25 * pred_profit + ad_score * 0.15 * pred_profit

        ad_metrics = dict(ACoAS=ACoAS, ASoAS=ASoAS, CPM=CPM, CPO=CPO)
        return total_score, pred_sales, pred_profit, predicted_days, ad_metrics

    def optimize(self):
        def objective(trial):
            x = {
                "售价": trial.suggest_float("售价", 8, 25),
                "CPC": trial.suggest_float("CPC", 0.1, 1.2),
                "广告花费": trial.suggest_float("广告花费", 10, 500),
                "ACOS": trial.suggest_float("ACOS", 0.05, 0.4),
                "Sessions": trial.suggest_float("Sessions", 50, 1000),
                "CVR": trial.suggest_float("CVR", 0.01, 0.3),
            }
            score, *_ = self.score_combination(x)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=80)
        best = study.best_params
        final_score, pred_sales, pred_profit, pred_days, ad_metrics = self.score_combination(best)

        # 运营风格解释和建议
        expl = (f"系统识别目标动销周期为{self.target_days}天, "
                f"本次策略综合考虑售价、广告投放、转化、流量等多维度。"
                f"推荐组合在利润、动销与广告效率之间找到最佳平衡。"
                f"预计{round(pred_days, 1)}天售罄库存, 14天利润${round(pred_profit * 14,2)}。")
        action = ("如库存压力大，可考虑降价+提曝光+优化广告结构，防止积压与资金占用。"
                  "若库存紧张，则建议主攻利润，谨防爆单。")

        return {
            "建议参数": best,
            "预测14天销量": round(pred_sales * 14, 1),
            "预测14天利润": round(pred_profit * 14, 2),
            "预测库存周转天数": round(pred_days, 1),
            "目标周转天数": self.target_days,
            "广告指标": ad_metrics,
            "策略解释": expl,
            "落地建议": action
        }
