import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Enable caching
if not os.path.exists('../data/cache'):
    os.makedirs('../data/cache')
fastf1.Cache.enable_cache('../data/cache')

# 2. Load Bahrain 2023 FP2
session_fp2 = fastf1.get_session(2023, 'Bahrain', 'FP2')
session_fp2.load()

# 3. Extract Laps for Hamilton
ham_laps = session_fp2.laps.pick_driver('HAM')

# 4. Filter for specific long run
# Stint 4, Soft Compound, Laps 12-22
long_run_fp2 = ham_laps[
    (ham_laps['Stint'] == 4) & 
    (ham_laps['Compound'] == 'SOFT') & 
    (ham_laps['LapNumber'] >= 12) & 
    (ham_laps['LapNumber'] <= 22)
].copy()

# 5. Sort by LapNumber
long_run_fp2 = long_run_fp2.sort_values(by='LapNumber')

# 6. Compute Fuel Correction
# Constants
FUEL_BURN_PER_LAP = 1.6  # kg/lap
TIME_SENSITIVITY = 0.05  # s/kg

# Identify first lap of the run
first_lap_number = long_run_fp2['LapNumber'].min()

# Apply formula: fuel_used_kg = (lap_number - first_lap_number) * 1.6
long_run_fp2['fuel_used_kg'] = (long_run_fp2['LapNumber'] - first_lap_number) * FUEL_BURN_PER_LAP

# Apply formula: fuel_time_delta = fuel_used_kg * 0.05
long_run_fp2['fuel_time_delta'] = long_run_fp2['fuel_used_kg'] * TIME_SENSITIVITY

# Apply formula: fuel_corrected_laptime = raw_laptime + fuel_time_delta
# Note: LapTime must be converted to seconds first
long_run_fp2['FuelCorrectedLapTime'] = long_run_fp2['LapTime'].dt.total_seconds() + long_run_fp2['fuel_time_delta']

# 7. Generate Plot
# X-axis: LapNumber, Y-axis: FuelCorrectedLapTime
plt.plot(long_run_fp2['LapNumber'], long_run_fp2['FuelCorrectedLapTime'], marker='o', linestyle='-')

# Formatting
plt.title("Fuel-Corrected Lap Time — Bahrain 2023 FP2 — HAM — SOFT")
plt.xlabel("LapNumber")
plt.ylabel("FuelCorrectedLapTime (seconds)")
plt.grid(True)

# Save plot
plt.savefig('images/ham_fuel_corrected_plot.png')