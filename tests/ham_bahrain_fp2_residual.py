import fastf1
import pandas as pd
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

# 3. Extract and Filter Laps for Hamilton
ham_laps = session_fp2.laps.pick_driver('HAM')
long_run_fp2 = ham_laps[
    (ham_laps['Stint'] == 4) & 
    (ham_laps['Compound'] == 'SOFT') & 
    (ham_laps['LapNumber'] >= 12) & 
    (ham_laps['LapNumber'] <= 22)
].copy()
long_run_fp2 = long_run_fp2.sort_values(by='LapNumber')

# 4. Apply Fuel Correction
FUEL_BURN_PER_LAP = 1.6
TIME_SENSITIVITY = 0.05
first_lap_number = long_run_fp2['LapNumber'].min()

long_run_fp2['fuel_used_kg'] = (long_run_fp2['LapNumber'] - first_lap_number) * FUEL_BURN_PER_LAP
long_run_fp2['fuel_time_delta'] = long_run_fp2['fuel_used_kg'] * TIME_SENSITIVITY
long_run_fp2['FuelCorrectedLapTime'] = long_run_fp2['LapTime'].dt.total_seconds() + long_run_fp2['fuel_time_delta']

# 5. Create stint_lap
long_run_fp2['stint_lap'] = long_run_fp2['LapNumber'] - first_lap_number

# 6. Fit Linear Regression
X = long_run_fp2[['stint_lap']]
y = long_run_fp2['FuelCorrectedLapTime']

model = LinearRegression()
model.fit(X, y)

# 7. Compute Residuals
long_run_fp2['predicted_laptime'] = model.predict(X)
long_run_fp2['residuals'] = long_run_fp2['FuelCorrectedLapTime'] - long_run_fp2['predicted_laptime']

# 8. Plot Residuals
plt.figure(figsize=(10, 6))
plt.scatter(long_run_fp2['stint_lap'], long_run_fp2['residuals'], color='blue', label='Residuals')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')

# Formatting
plt.title("Residuals of Linear Fit — Bahrain 2023 FP2 — HAM")
plt.xlabel("Stint Lap (Laps since start of stint)")
plt.ylabel("Residuals (Actual - Predicted) [s]")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig('images/ham_residuals_plot.png')