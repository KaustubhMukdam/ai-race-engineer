import fastf1
import pandas as pd
import numpy as np
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

# 5. Simulation
PIT_LOSS = 22.0  # seconds

# Create simulation range (0 to 40 laps to be safe)
sim_laps = np.arange(0, 41)

# Calculate losses
incremental_loss = b * sim_laps
cumulative_loss = np.cumsum(incremental_loss)

# Create DataFrame
sim_df = pd.DataFrame({
    'stint_lap': sim_laps,
    'incremental_loss': incremental_loss,
    'cumulative_loss': cumulative_loss
})

# 6. Identify Crossover
# Find first row where cumulative_loss >= PIT_LOSS
crossover = sim_df[sim_df['cumulative_loss'] >= PIT_LOSS].head(1)

# 7. Print Results
if not crossover.empty:
    optimal_lap = int(crossover['stint_lap'].values[0])
    loss_val = crossover['cumulative_loss'].values[0]
    print(f"Optimal Pit Lap (Stint-Based): {optimal_lap}")
    print(f"Corresponding Cumulative Loss: {loss_val:.4f} s")
    print(f"(Pit Loss Constant: {PIT_LOSS} s)")
else:
    print("No crossover found within simulation range.")