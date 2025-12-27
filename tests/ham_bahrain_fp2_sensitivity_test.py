import fastf1
import pandas as pd
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

# 4. Prepare common variables
TIME_SENSITIVITY = 0.05
first_lap_number = long_run_fp2['LapNumber'].min()
long_run_fp2['stint_lap'] = long_run_fp2['LapNumber'] - first_lap_number
X = long_run_fp2[['stint_lap']]

# 5. Iterate through Fuel Burn Variants
burn_rates = [1.4, 1.6, 1.8]

print(f"{'Fuel Burn (kg/lap)':<20} | {'Degradation Slope (s/lap)':<25}")
print("-" * 50)

for burn_rate in burn_rates:
    # Compute Fuel Correction for this specific burn rate
    fuel_used_kg = (long_run_fp2['LapNumber'] - first_lap_number) * burn_rate
    fuel_time_delta = fuel_used_kg * TIME_SENSITIVITY
    
    # Calculate target variable
    y = long_run_fp2['LapTime'].dt.total_seconds() + fuel_time_delta
    
    # Fit Linear Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Extract slope
    slope = model.coef_[0]
    
    # Print result
    print(f"{burn_rate:<20} | {slope:.5f}")