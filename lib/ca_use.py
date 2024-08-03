import numpy as np
from sklearn.neural_network import MLPRegressor

def quantize(org_value, num_states, min_value, max_value):
    precise_value = (org_value - min_value) / (max_value - min_value) * (num_states - 1)
    return round(precise_value) + 1

def dequantize(state, num_states, min_value, max_value):
    return min_value + (state - 1) * (max_value - min_value) / (num_states - 1)

def fuzzy_membership(value, states, sigma=1.0):
    membership_values = []
    for state in states:
        membership_value = np.exp(-((value - state) ** 2) / (2 * sigma ** 2))
        membership_values.append(membership_value)
    return membership_values

def fuzzy_aggregate(memberships):
    total_weight = sum(memberships)
    if total_weight == 0:
        return 0
    return sum(m * s for m, s in zip(memberships, range(1, len(memberships) + 1))) / total_weight

def train_ml_model(data, num_states, min_value, max_value):
    X, y = [], []
    num_cells = len(data)
    
    for i in range(3, num_cells - 3):
        neighborhood = data[i-1:i+2]  # Immediate neighborhood
        extended_neighborhood = data[i-3:i+4]  # Extended neighborhood (3 cells on each side)
        features = np.concatenate((neighborhood, extended_neighborhood))
        target = data[i]
        X.append(features)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    model = MLPRegressor(hidden_layer_sizes=(500, 500), max_iter=1000)
    model.fit(X, y)
    
    return model

def next_state_ml(model, grid, i):
    neighborhood = grid[i-1:i+2]  # Immediate neighborhood
    extended_neighborhood = grid[i-3:i+4]  # Extended neighborhood (3 cells on each side)
    features = np.concatenate((neighborhood, extended_neighborhood))
    nn_prediction = model.predict([features])[0]
    avg_neighbors = np.mean(neighborhood)
    next_state = (nn_prediction + avg_neighbors) / 2
    return next_state

def usedCA(methane_data, steps=10):
    num_cells = len(methane_data)
    num_states = 399
    grid = np.zeros(num_cells, dtype=float)
    min_value, max_value = np.min(methane_data), np.max(methane_data)
    print(f"num_cells: {num_cells}, num_states: {num_states}, min_value: {min_value}, max_value: {max_value}")
    
    for i in range(num_cells):
        grid[i] = quantize(methane_data[i], num_states, min_value, max_value)
    
    model = train_ml_model(methane_data, num_states, min_value, max_value)

    forecast = np.zeros((steps, num_cells), dtype=float)
    forecast[0] = grid.astype(float)
    
    for t in range(1, steps):
        new_grid = np.zeros(num_cells, dtype=float)
        for i in range(3, num_cells - 3):
            new_grid[i] = next_state_ml(model, forecast[t-1], i)
        forecast[t] = new_grid
    
    final_forecast = np.array([dequantize(state, num_states, min_value, max_value) for state in forecast[-1]])
    return final_forecast
