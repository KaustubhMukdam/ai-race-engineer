import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
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
# Constants
FUEL_BURN_PER_LAP = 1.6
TIME_SENSITIVITY = 0.05
first_lap_number = long_run_fp2['LapNumber'].min()

# Apply Correction
long_run_fp2['fuel_used_kg'] = (long_run_fp2['LapNumber'] - first_lap_number) * FUEL_BURN_PER_LAP
long_run_fp2['fuel_time_delta'] = long_run_fp2['fuel_used_kg'] * TIME_SENSITIVITY
long_run_fp2['FuelCorrectedLapTime'] = long_run_fp2['LapTime'].dt.total_seconds() + long_run_fp2['fuel_time_delta']
long_run_fp2['stint_lap'] = long_run_fp2['LapNumber'] - first_lap_number

# Fit
X = long_run_fp2[['stint_lap']]
y = long_run_fp2['FuelCorrectedLapTime']
model = LinearRegression()
model.fit(X, y)
b = model.coef_[0]

print(f"Fitted degradation slope (b): {b:.4f} s/lap")

# 5. Simulate Stint
# Create laps 0 to 30
sim_laps = np.arange(0, 31)

# Calculate losses
# Incremental loss at lap i = b * i
incremental_loss = b * sim_laps

# Cumulative loss at lap n = sum(incremental losses from 0 to n)
cumulative_loss = np.cumsum(incremental_loss)

# 6. Store in DataFrame
simulation_df = pd.DataFrame({
    'stint_lap': sim_laps,
    'incremental_loss': incremental_loss,
    'cumulative_loss': cumulative_loss
})

# 7. Print Results
print("\nSimulation Results (First 10 laps):")
print(simulation_df.head(10))
print("\nSimulation Results (Last 5 laps):")
print(simulation_df.tail(5))