import fastf1
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

# 6. Verify selection (Optional print)
print(f"Selected {len(long_run_fp2)} laps")
print(long_run_fp2[['LapNumber', 'Stint', 'Compound', 'IsAccurate']])