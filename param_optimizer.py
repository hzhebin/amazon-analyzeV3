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
