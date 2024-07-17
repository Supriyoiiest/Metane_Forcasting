import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from lib.preprocessing import CleanDataReturn
data = pd.read_csv('D:/merged_data_1.csv')
data['Time'] = pd.to_datetime(data['Time'], format='%d-%m-%Y %H:%M')
data_cleaned= CleanDataReturn(data)
X = data_cleaned[['Coal', 'RH (%)', 'Temp (deg C)']]
y = data_cleaned['CH4 (ppm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
degree = 50
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge = Ridge()
ridge.fit(X_train_poly, y_train)
y_pred = ridge.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
intercept = ridge.intercept_
coefficients = ridge.coef_
feature_names = poly.get_feature_names_out(X.columns)

equation_terms = [f"{coeff}*{name}" for coeff, name in zip(coefficients, feature_names)]
equation = " + ".join(equation_terms)

def evaluate_ch4(coal, rh, temp):
    input_data = np.array([[coal, rh, temp]])
    input_poly = poly.transform(input_data)
    ch4 = intercept + np.sum(coefficients * input_poly)
    return ch4
