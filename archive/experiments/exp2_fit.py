import time

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

def mean_squared_error(y, y_pred):
    return sum((y_pred - y) ** 2) / len(y_pred)

df = pd.read_csv("exp2_results.csv")
X = df[["num_experts", "tot_num_tokens"]]
y = df["time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_poly_pipeline(deg):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=deg)),
        ("linear", LinearRegression())
    ])


models = [
    (create_poly_pipeline(1), "poly-1"),
    (create_poly_pipeline(2), "poly-2"),
    (create_poly_pipeline(3), "poly-3"),
    (create_poly_pipeline(4), "poly-4"),
    (create_poly_pipeline(5), "poly-5"),
    (create_poly_pipeline(6), "poly-6"),
    (RandomForestRegressor(n_estimators=100, random_state=42), "random forest"),
    (Ridge(alpha=1.0), "ridge regression"),
    (Lasso(alpha=1.0), "lasso regression"),
    (ElasticNet(alpha=1.0, l1_ratio=0.5), "elasticnet regression"),
]

for model, name in models:
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    mse = mean_squared_error(y_test, y_pred)
    time_ms = (end-start)*1000
    print(f"{name} ({time_ms:.2f}ms): {mse}")


coefficients = models[1][0].named_steps['linear'].coef_
intercept = models[1][0].named_steps['linear'].intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)