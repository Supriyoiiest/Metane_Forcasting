import pandas as pd
from scipy import stats
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
def CleanDataReturn(data):
    features = data.drop(columns=['Coal'])
    features = features.drop(columns=['Time'])
    target = data['Coal']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    data_scaled = pd.concat([features_scaled_df, target.reset_index(drop=False)], axis=1)
    z_scores = stats.zscore(features_scaled)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores <= 3).all(axis=1)
    data_cleaned = data_scaled[filtered_entries]
    min_values = data_cleaned.min()
    if (min_values < 0).any():
        shift_value = abs(min_values.min()) + 0.01
        data_cleaned.loc[:, ['CH4 (ppm)', 'Temp (deg C)', 'RH (%)', 'Coal']] += shift_value
    return data_cleaned