import fastf1
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

# 4. Filter for specific long run (Stint 4, Soft, Laps 12-22)
long_run_fp2 = ham_laps[
    (ham_laps['Stint'] == 4) & 
    (ham_laps['Compound'] == 'SOFT') & 
    (ham_laps['LapNumber'] >= 12) & 
    (ham_laps['LapNumber'] <= 22)
].copy()

# 5. Sort by LapNumber
long_run_fp2 = long_run_fp2.sort_values(by='LapNumber')

# 6. Generate Plot
plt.figure(figsize=(10, 6))

# Convert LapTime (Timedelta) to seconds for plotting
x = long_run_fp2['LapNumber']
y = long_run_fp2['LapTime'].dt.total_seconds()

# Plot raw data
plt.plot(x, y, marker='o', linestyle='-', label='Raw Lap Time')

# Formatting
plt.title("Raw Lap Time — Bahrain 2023 FP2 — HAM — SOFT")
plt.xlabel("LapNumber")
plt.ylabel("LapTime (seconds)")
plt.grid(True)
plt.show()