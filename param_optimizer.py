import optuna
import numpy as np

def profit_objective(trial, model, X_base, constraint_fn):
    price = trial.suggest_float("price", X_base['price'].min() * 0.8, X_base['price'].max() * 1.2)
    cpc = trial.suggest_float("cpc", X_base['cpc'].min(), X_base['cpc'].max())
    ad_spend = trial.suggest_float("ad_spend", X_base['ad_spend'].min(), X_base['ad_spend'].max())
    X_test = X_base.copy()
    X_test['price'] = price
    X_test['cpc'] = cpc
    X_test['ad_spend'] = ad_spend
    if not constraint_fn(X_test):
        return -99999
    profit_pred = model.predict(X_test)
    return np.mean(profit_pred)

def run_optimization(model, X_base, constraint_fn, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: profit_objective(trial, model, X_base, constraint_fn), n_trials=n_trials)
    return study.best_params, study.best_value
import numpy as np

class StrategyOptimizer:
    def __init__(self, df, stock, target_days):
        self.df = df.copy()
        self.stock = stock
        self.target_days = target_days
        self.target_units_per_day = stock / target_days
        self.clean_data()

    def clean_data(self):
        self.df = self.df.rename(columns=lambda x: x.strip())
        if '售价(总价)' in self.df.columns:
            self.df['售价'] = self.df['售价(总价)']
        if 'Sessions-Total' in self.df.columns:
            self.df['Sessions'] = self.df['Sessions-Total']
        self.df.dropna(subset=['订单量', '售价', 'Sessions'], inplace=True)

    def simulate(self, strategy_type):
        if strategy_type == "销量优先":
            price = self.df['售价'].quantile(0.7)
            cpc = self.df['CPC'].quantile(0.7)
            ad_spend = self.df['广告花费'].quantile(0.7)
        else:
            price = self.df['售价'].quantile(0.3)
            cpc = self.df['CPC'].quantile(0.3)
            ad_spend = self.df['广告花费'].quantile(0.3)

        sessions = np.mean(self.df['Sessions']) * (1.2 if strategy_type == "销量优先" else 0.8)
        cvr = np.mean(self.df['CVR']) * (1.1 if strategy_type == "销量优先" else 1.0)
        units = sessions * cvr
        profit = units * (price * 0.3) - ad_spend

        return {
            "建议售价": f"${price:.2f}",
            "建议 CPC": f"${cpc:.2f}",
            "广告花费": f"${ad_spend:.2f}",
            "Sessions": f"{sessions:.1f}",
            "CVR": f"{cvr*100:.2f}%",
            "14天预计销量": f"{units:.1f} 单",
            "14天预计利润": f"${profit:.2f}",
            "执行建议": "优先动销" if strategy_type == "销量优先" else "优先盈利"
        }

    def optimize_strategies(self):
        return self.simulate("销量优先"), self.simulate("利润最大化")
