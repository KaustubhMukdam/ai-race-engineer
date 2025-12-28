import fastf1
import pandas as pd
import os

# 1. Enable caching
if not os.path.exists('../../data/cache'):
    os.makedirs('../../data/cache')
fastf1.Cache.enable_cache('../../data/cache')

# 2. Load Bahrain 2023 Race
session_race = fastf1.get_session(2023, 'Bahrain', 'Race')
session_race.load()

# 3. Extract Laps for Hamilton
ham_laps = session_race.laps.pick_driver('HAM')

# 4. Compute and Print Stint Pit Laps
print(f"--- Pit Stop Analysis: Lewis Hamilton (Bahrain 2023) ---")

for stint_id, stint_data in ham_laps.groupby('Stint'):
    # Determine start and end laps of the stint
    stint_start_lap = stint_data['LapNumber'].min()
    race_pit_lap = stint_data['LapNumber'].max()
    
    # Check if the stint ended with a pit stop
    # (PitInTime is not NaT/NaN for the last lap of the stint if they pitted)
    last_lap = stint_data.loc[stint_data['LapNumber'] == race_pit_lap].iloc[0]
    
    if not pd.isnull(last_lap['PitInTime']):
        # Compute stint-based lap number per user formula
        actual_stint_lap = race_pit_lap - stint_start_lap
        
        print(f"Stint {int(stint_id)} Pit Stop:")
        print(f"  Race Lap: {int(race_pit_lap)}")
        print(f"  Stint Lap: {int(actual_stint_lap)}")