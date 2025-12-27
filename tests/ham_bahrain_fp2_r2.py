import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

# 5. Create stint_lap (Feature Engineering)
long_run_fp2['stint_lap'] = long_run_fp2['LapNumber'] - first_lap_number

# 6. Fit Linear Regression (OLS)
# X must be 2D for sklearn, y is 1D
X = long_run_fp2[['stint_lap']]
y = long_run_fp2['FuelCorrectedLapTime']

model = LinearRegression()
model.fit(X, y)

# 7. Extract Parameters
intercept = model.intercept_
slope = model.coef_[0]
r2 = r2_score(y, model.predict(X))

# 8. Print Results
print(f"Intercept (a): {intercept}")
print(f"Slope (b): {slope} seconds per lap")
print(f"R² score: {r2}")