import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.preprocessing import CleanDataReturn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def evaluate_ch4(coal, rh, temp, model, poly):
    input_data = np.array([[coal, rh, temp]])
    input_poly = poly.transform(input_data)
    ch4_pred = model.predict(input_poly)
    return ch4_pred[0]

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=42)

    # Define hyperparameters grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)


    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Mean squared error: {mse}")
    print(f"Mean absolute error: {mae}")
    print(f"R-squared: {r2}")

    return best_model, scaler

