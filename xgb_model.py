import xgboost as xgb
import shap
import numpy as np
from sklearn.model_selection import train_test_split

class ProfitXGBModel:
    def __init__(self):
        self.model = None
        self.features = None

    def fit(self, df, target_col, feature_cols):
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        self.features = feature_cols
        return self.model.score(X_val, y_val)

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        else:
            raise Exception("Model not trained.")

    def shap_importance(self, X):
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        return shap_values
