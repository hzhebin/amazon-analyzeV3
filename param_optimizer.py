
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
