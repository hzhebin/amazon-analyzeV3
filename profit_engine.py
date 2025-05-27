import numpy as np
from xgb_model import ProfitXGBModel
from param_optimizer import run_optimization

def inventory_constraint(X, min_days=7, max_days=90, target_days=30):
    days_in_stock = X.get('days_in_stock', [target_days])
    if days_in_stock[0] > max_days:
        return False
    if days_in_stock[0] < min_days:
        return False
    return True

def simulate_strategy(df, mode='profit', days_predict=14):
    # 数据字段
    features = ['price', 'cpc', 'ad_spend', 'sessions', 'cvr', 'acos', 'acoas', 'cpm', 'cpo', 'profit_margin', 'days_in_stock']
    target = 'profit'
    xgb = ProfitXGBModel()
    xgb.fit(df, target, features)
    X_base = df[features].iloc[[-1]].copy()
    # 约束函数
    if mode == 'volume':
        def constraint(X): return X['days_in_stock'].iloc[0] > 30  # 需要加速动销
    else:
        def constraint(X): return inventory_constraint(X, target_days=30)
    best_params, best_profit = run_optimization(xgb, X_base, constraint)
    # 预测未来14天利润和销量
    X_pred = X_base.copy()
    for k, v in best_params.items():
        X_pred[k] = v
    pred_profit = xgb.predict(X_pred) * days_predict
    pred_sales = pred_profit / (X_pred['price'].iloc[0] * X_pred['profit_margin'].iloc[0])
    return {
        "params": best_params,
        "pred_profit": pred_profit[0],
        "pred_sales": pred_sales[0],
        "shap": xgb.shap_importance(X_base),
        "explanation": explain_strategy(mode, best_params, pred_profit, pred_sales)
    }

def explain_strategy(mode, params, pred_profit, pred_sales):
    if mode == 'volume':
        return f"【销量优先】当前库存压力较大，建议降价/加投放推动出货，策略建议：售价{params['price']:.2f}，CPC{params['cpc']:.2f}，预计14天可出{int(pred_sales)}单，总利润{pred_profit:.2f}。适用于：清库存/冲排名。"
    else:
        return f"【利润最大化】以利润最优为目标，策略建议：售价{params['price']:.2f}，CPC{params['cpc']:.2f}，预计14天利润{pred_profit:.2f}，单量{int(pred_sales)}。适用于：利润优先、库存压力可控场景。"
