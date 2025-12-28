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

# 4. Group by Stint to extract details
print(f"--- Strategy Summary: Lewis Hamilton (Bahrain 2023) ---")

# Iterate through stints sequentially
for stint_id, stint_data in ham_laps.groupby('Stint'):
    
    # Get basic stint info
    compound = stint_data['Compound'].mode().iloc[0] # Most frequent compound in stint
    start_lap = int(stint_data['LapNumber'].min())
    end_lap = int(stint_data['LapNumber'].max())
    
    # Check for Pit In (End of Stint)
    # A lap is a "pit in" lap if PitInTime is recorded (meaning they entered pits at end of lap)
    # or if it's the last lap of a stint before another stint.
    # FastF1 IsAccurate is often False on in/out laps.
    
    # Check specifically if the last lap of this stint has a valid PitInTime
    last_lap_data = stint_data.loc[stint_data['LapNumber'] == end_lap].iloc[0]
    pitted = not pd.isnull(last_lap_data['PitInTime'])
    
    # Output formatting
    stint_summary = f"Stint {int(stint_id)}: Laps {start_lap}–{end_lap}, Compound: {compound}"
    
    if pitted:
        print(f"{stint_summary} | Pit In on Lap {end_lap}")
    else:
        # Likely the final stint (race finish)
        print(f"{stint_summary} | Race Finish")