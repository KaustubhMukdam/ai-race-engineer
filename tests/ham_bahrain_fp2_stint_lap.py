import fastf1
import pandas as pd
import os

# 1. Enable caching
if not os.path.exists('../data/cache'):
    os.makedirs('../data/cache')
fastf1.Cache.enable_cache('../data/cache')

# 2. Load Bahrain 2023 FP2
session_fp2 = fastf1.get_session(2023, 'Bahrain', 'FP2')
session_fp2.load()

# 3. Extract and Filter Laps for Hamilton (Stint 4, Soft, Laps 12-22)
ham_laps = session_fp2.laps.pick_driver('HAM')
long_run_fp2 = ham_laps[
    (ham_laps['Stint'] == 4) & 
    (ham_laps['Compound'] == 'SOFT') & 
    (ham_laps['LapNumber'] >= 12) & 
    (ham_laps['LapNumber'] <= 22)
].copy()
long_run_fp2 = long_run_fp2.sort_values(by='LapNumber')

# 4. Apply Fuel Correction (Locked Formula)
FUEL_BURN_PER_LAP = 1.6
TIME_SENSITIVITY = 0.05
first_lap_number = long_run_fp2['LapNumber'].min()

long_run_fp2['fuel_used_kg'] = (long_run_fp2['LapNumber'] - first_lap_number) * FUEL_BURN_PER_LAP
long_run_fp2['fuel_time_delta'] = long_run_fp2['fuel_used_kg'] * TIME_SENSITIVITY
long_run_fp2['FuelCorrectedLapTime'] = long_run_fp2['LapTime'].dt.total_seconds() + long_run_fp2['fuel_time_delta']

# 5. Create stint_lap (Feature Engineering)
long_run_fp2['stint_lap'] = long_run_fp2['LapNumber'] - first_lap_number

# 6. Prepare X and y
X = long_run_fp2[['stint_lap']]  # 2D DataFrame for potential sklearn usage
y = long_run_fp2['FuelCorrectedLapTime']

# 7. Verify Data (No Stats)
print(long_run_fp2[['LapNumber', 'stint_lap', 'FuelCorrectedLapTime']].head())