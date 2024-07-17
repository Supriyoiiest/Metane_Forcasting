import numpy as np

def quantize(org_value, num_states, min_value, max_value):
    precise_value = (org_value - min_value) / (max_value - min_value) * (num_states - 1)
    return round(precise_value) + 1

def dequantize(state, num_states, min_value, max_value):
    return min_value + (state - 1) * (max_value - min_value) / (num_states - 1)

def fuzzy_membership(value, states):
    membership_values = []
    for state in states:
        membership_value = np.exp(-((value - state)**2) / (2 * (state / 10)**2))
        membership_values.append(membership_value)
    return membership_values

def fuzzy_aggregate(memberships):
    total_weight = sum(memberships)
    if total_weight == 0:
        return 0
    return sum(m * s for m, s in zip(memberships, range(1, len(memberships) + 1))) / total_weight

def next_state(current_state, num_states, grid, i):
    neighborhood = 5
    num_cells = len(grid)
    neighborhood_values = []
    for offset in range(-neighborhood, neighborhood + 1):
        index = (i + offset) % num_cells 
        index = (index + num_cells) % num_cells 
        neighborhood_values.append(grid[index])
    
    states = np.linspace(np.min(grid), np.max(grid), num_states)
    fuzzy_memberships = np.zeros(num_states)
    
    for value in neighborhood_values:
        memberships = fuzzy_membership(value, states)
        fuzzy_memberships += np.array(memberships)
    
    next_state_index = int(fuzzy_aggregate(fuzzy_memberships))
    next_state_index = min(max(next_state_index, 1), num_states)
    
    return next_state_index

def usedCA(methane_data, steps=10):
    num_cells = len(methane_data)
    num_states = 399
    grid = np.zeros(num_cells, dtype=float)
    min_value, max_value = np.min(methane_data), np.max(methane_data)
    print(f"num_cells: {num_cells}, num_states: {num_states}, min_value: {min_value}, max_value: {max_value}")
    
    for i in range(num_cells):
        grid[i] = quantize(methane_data[i], num_states, min_value, max_value)

    forecast = np.zeros((steps, num_cells), dtype=int)
    forecast[0] = grid.astype(int)
    
    for t in range(1, steps):
        new_grid = np.zeros(num_cells, dtype=int)
        for i in range(num_cells):
            new_grid[i] = next_state(forecast[t-1], num_states, grid, i)
        forecast[t] = new_grid
    
    final_forecast = np.array([dequantize(state, num_states, min_value, max_value) for state in forecast[-1]])
    return final_forecast
