import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
from lib.preprocessing import CleanDataReturn
from lib.ml_pred import train_and_evaluate_model
from lib.ca_use import usedCA
from lib.ml_pred import evaluate_ch4
from scipy.interpolate import make_interp_spline
# Add the path to lib directory


def main():
    # Load and preprocess data
    data = pd.read_csv('C:/Users/supri/OneDrive/Desktop/Metane_Forcasting-main/data/merged_data_1.csv')

    data['Time'] = pd.to_datetime(data['Time'], format='%d-%m-%Y %H:%M')
    data_cleaned = CleanDataReturn(data)
    data_cleaned['Time'] = data['Time'][data_cleaned.index]
    data_cleaned = data_cleaned.sort_values(by='Time')
    # Select features and target
    X = data_cleaned[['Coal', 'RH (%)', 'Temp (deg C)']]
    y = data_cleaned['CH4 (ppm)']


    # Train and evaluate the model (Example with RandomForestRegressor)
    model = RandomForestRegressor(random_state=42)
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 100, 200]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # Example evaluation
    y_pred = best_model.predict(X)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Mean squared error: {mean_squared_error(y, y_pred)}")
    print(f"Mean absolute error: {mean_absolute_error(y, y_pred)}")
    print(f"R-squared: {r2_score(y, y_pred)}")

    # Use Cellular Automata for forecasting
    predicted_ch4_array_alt = best_model.predict(X)  # Example prediction using trained model
    forecast = usedCA(predicted_ch4_array_alt)
    data = data.iloc[:-1]
    # Plotting
    plt.figure(figsize=(120, 60))
    plt.plot(data_cleaned['Time'],data_cleaned['CH4 (ppm)'], color='blue', label='Actual CH4 (ppm)')
    plt.plot(data_cleaned['Time'], forecast, color='green', linestyle='dashed', label='Predicted CH4 (ppm)')
    plt.xlabel('Time')
    plt.ylabel('CH4 (ppm)')
    plt.title('Actual vs Predicted CH4 (ppm) over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()

