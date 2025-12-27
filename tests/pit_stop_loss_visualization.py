import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# 1. Enable caching
if not os.path.exists('../data/cache'):
    os.makedirs('../data/cache')
fastf1.Cache.enable_cache('../data/cache')

# 2. Load Bahrain 2023 FP2
session_fp2 = fastf1.get_session(2023, 'Bahrain', 'FP2')
session_fp2.load()

# 3. Extract and Filter Laps for Hamilton (Stint 4, Soft)
ham_laps = session_fp2.laps.pick_driver('HAM')
long_run_fp2 = ham_laps[
    (ham_laps['Stint'] == 4) & 
    (ham_laps['Compound'] == 'SOFT') & 
    (ham_laps['LapNumber'] >= 12) & 
    (ham_laps['LapNumber'] <= 22)
].copy()
long_run_fp2 = long_run_fp2.sort_values(by='LapNumber')

# 4. Fit Model to get 'b' (Slope)
FUEL_BURN_PER_LAP = 1.6
TIME_SENSITIVITY = 0.05
first_lap_number = long_run_fp2['LapNumber'].min()

# Apply Fuel Correction
long_run_fp2['fuel_used_kg'] = (long_run_fp2['LapNumber'] - first_lap_number) * FUEL_BURN_PER_LAP
long_run_fp2['fuel_time_delta'] = long_run_fp2['fuel_used_kg'] * TIME_SENSITIVITY
long_run_fp2['FuelCorrectedLapTime'] = long_run_fp2['LapTime'].dt.total_seconds() + long_run_fp2['fuel_time_delta']
long_run_fp2['stint_lap'] = long_run_fp2['LapNumber'] - first_lap_number

# Fit Linear Regression
X = long_run_fp2[['stint_lap']]
y = long_run_fp2['FuelCorrectedLapTime']
model = LinearRegression()
model.fit(X, y)
b = model.coef_[0]

# 5. Simulation
PIT_LOSS = 22.0
sim_laps = np.arange(0, 41)
incremental_loss = b * sim_laps
cumulative_loss = np.cumsum(incremental_loss)

# 6. Generate Plot
plt.figure(figsize=(10, 6))

# Line 1: Cumulative Degradation Loss
plt.plot(sim_laps, cumulative_loss, label='Cumulative Degradation Loss', color='blue', linewidth=2)

# Horizontal Line: Pit Loss
plt.axhline(y=PIT_LOSS, color='red', linestyle='--', linewidth=2, label=f'Pit Loss ({PIT_LOSS}s)')

# Mark Crossover Point
# Find the first index where cumulative loss exceeds pit loss
crossover_indices = np.where(cumulative_loss >= PIT_LOSS)[0]
if len(crossover_indices) > 0:
    idx = crossover_indices[0]
    x_cross = sim_laps[idx]
    y_cross = cumulative_loss[idx]
    
    plt.scatter(x_cross, y_cross, color='black', zorder=5, s=100)
    plt.annotate(f'Crossover: Lap {x_cross}', 
                 xy=(x_cross, y_cross), 
                 xytext=(x_cross - 5, y_cross + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

# Formatting
plt.title("Pit Window Simulation — Bahrain SOFT — HAM")
plt.xlabel("stint_lap")
plt.ylabel("seconds")
plt.legend()
plt.grid(True)
plt.xlim(0, 40)
plt.ylim(0, PIT_LOSS * 1.5)

# Save plot
plt.savefig('images/pit_window_simulation.png')