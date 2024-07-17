import lib.ml_pred as ml_pred
import lib.preprocessing as preprocessing
import lib.ca_use as ca_use #Blank
import matplotlib as plt
from lib.preprocessing import CleanDataReturn
from lib.ml_pred import evaluate_ch4
import pandas as pd

def main():
    data = pd.read_csv('D:/merged_data_1.csv')
    data['Time'] = pd.to_datetime(data['Time'], format='%d-%m-%Y %H:%M')
    data_cleaned = CleanDataReturn(data)
    data_cleaned['Predicted CH4 (ppm)_2'] = data_cleaned.apply(lambda row: evaluate_ch4(row['Coal'], row['RH (%)'], row['Temp (deg C)']), axis=1)
    plt.figure.Figure(figsize=(12, 6))
    data.drop(data.index[-1], inplace=True)
    plt.pyplot.plot(data['Time'], data_cleaned['CH4 (ppm)'], color='blue', label='Actual CH4 (ppm)')
    plt.pyplot.plot(data['Time'], data_cleaned['Predicted CH4 (ppm)_2'], color='green', linestyle='dashed', label='Predicted CH4 (ppm) - Method 2')
    plt.pyplot.xlabel('Time')
    plt.pyplot.ylabel('CH4 (ppm)')
    plt.pyplot.title('Actual vs Predicted CH4 (ppm) over Time')
    plt.pyplot.legend()
    plt.pyplot.grid(True)
    plt.pyplot.show()

if __name__ == "__main__":
    main()