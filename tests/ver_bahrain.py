import fastf1
import pandas as pd
import os

# 1. Enable caching
if not os.path.exists('../data/cache'):
    os.makedirs('../data/cache')
fastf1.Cache.enable_cache('../data/cache')

# 2. Load Bahrain 2023 FP3
session_fp3 = fastf1.get_session(2023, 'Bahrain', 'FP3')
session_fp3.load()

# 3. Extract and Filter Laps for VER
fp3_laps = session_fp3.laps.pick_driver('VER')
accurate_laps = fp3_laps[fp3_laps['IsAccurate'] == True].copy()

# 4. Find Longest Consecutive Run
best_run = {
    'count': 0,
    'compound': None,
    'stint': None,
    'range': (0, 0)
}

# Group by Stint and Compound to isolate distinct runs
for (stint, compound), group in accurate_laps.groupby(['Stint', 'Compound']):
    
    # Sort by LapNumber to ensure correct consecutive check
    group = group.sort_values('LapNumber')
    
    # Identify consecutive blocks:
    # If the difference between current and previous lap number is not 1, it's a new block
    group['block_id'] = (group['LapNumber'].diff() != 1).cumsum()
    
    # Analyze each continuous block within this stint
    for _, block in group.groupby('block_id'):
        count = len(block)
        
        if count > best_run['count']:
            start_lap = block['LapNumber'].min()
            end_lap = block['LapNumber'].max()
            
            best_run['count'] = count
            best_run['compound'] = compound
            best_run['stint'] = stint
            best_run['range'] = (start_lap, end_lap)

# 5. Print Results
print(f"Longest Consecutive Run found:")
print(f"Compound: {best_run['compound']}")
print(f"Stint Number: {best_run['stint']}")
print(f"Consecutive Laps: {best_run['count']}")
print(f"Lap Range: {best_run['range'][0]} - {best_run['range'][1]}")