"""
Prepare training data for pit window classification

Creates features from historical race data to predict optimal pit timing
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def create_pit_window_features(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for pit window classification
    
    Features:
    - tire_age: Current tire age in laps
    - tire_compound: SOFT/MEDIUM/HARD (encoded)
    - lap_number: Current lap number
    - race_progress: lap_number / total_laps (0-1)
    - position: Current track position
    - track_temp: Current track temperature
    - air_temp: Current air temperature
    - degradation_rate: Avg degradation over last 5 laps (s/lap)
    - lap_time: Current lap time (seconds)
    
    Target:
    - should_pit: Binary (0 = stay out, 1 = pit within next 3 laps)
    """
    
    features = []
    
    # Filter only dry compounds
    laps_df = laps_df[laps_df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
    
    # Get unique races and drivers
    races = laps_df['Race'].unique()
    
    logger.info(f"Processing {len(races)} races")
    
    for race in races:
        race_laps = laps_df[laps_df['Race'] == race]
        total_laps_in_race = race_laps['LapNumber'].max()
        
        for driver in race_laps['Driver'].unique():
            driver_laps = race_laps[race_laps['Driver'] == driver].sort_values('LapNumber')
            
            # Skip if too few laps
            if len(driver_laps) < 10:
                continue
            
            # Process each lap (need 5 laps history, 3 laps lookahead)
            for i in range(5, len(driver_laps) - 3):
                current = driver_laps.iloc[i]
                
                # Skip invalid laps
                if pd.isna(current['LapTime_Seconds']) or current['LapTime_Seconds'] <= 0:
                    continue
                
                # Calculate degradation rate from last 5 laps
                recent_laps = driver_laps.iloc[i-5:i]['LapTime_Seconds'].values
                recent_laps_clean = recent_laps[~np.isnan(recent_laps)]
                
                if len(recent_laps_clean) >= 3:
                    # Linear degradation: (last - first) / num_laps
                    deg_rate = (recent_laps_clean[-1] - recent_laps_clean[0]) / len(recent_laps_clean)
                else:
                    deg_rate = 0.0

                # Check if driver pits in next 5 laps (target)
                next_5_laps = driver_laps.iloc[i+1:i+6]
                
                # A pit is indicated by a change in TyreLife (resets to low number)
                # or by PitInTime being not null
                will_pit = False
                
                # Method 1: Check for stint change (TyreLife reset)
                current_stint = driver_laps.iloc[:i+1]['Stint'].max()
                next_stint = next_5_laps['Stint'].max()
                if next_stint > current_stint:
                    will_pit = True
                
                # Method 2: Check PitInTime (if available)
                if 'PitInTime' in next_5_laps.columns:
                    if any(next_5_laps['PitInTime'].notna()):
                        will_pit = True
                
                # Get position (if available)
                position = None  # Default to midfield

                if position is None:
                    lap_positions = race_laps[race_laps['LapNumber'] == current['LapNumber']].copy()
                    lap_positions = lap_positions.sort_values('LapTime_Seconds')
                    lap_positions['Position'] = range(1, len(lap_positions) + 1)
                    
                    position_row = lap_positions[lap_positions['Driver'] == driver]
                    if not position_row.empty:
                        position = int(position_row.iloc[0]['Position'])
                
                # Create feature row
                feature_row = {
                    'race': race,
                    'driver': driver,
                    'tire_age': int(current['TyreLife']),
                    'tire_compound': current['Compound'],
                    'lap_number': int(current['LapNumber']),
                    'race_progress': float(current['LapNumber']) / float(total_laps_in_race),
                    'position': int(position) if not pd.isna(position) else 10,
                    'track_temp': float(current['TrackTemp']) if not pd.isna(current['TrackTemp']) else 30.0,
                    'air_temp': float(current['AirTemp']) if not pd.isna(current['AirTemp']) else 25.0,
                    'degradation_rate': float(deg_rate),
                    'lap_time': float(current['LapTime_Seconds']),
                    'should_pit': 1 if will_pit else 0
                }
                
                features.append(feature_row)
    
    return pd.DataFrame(features)

def main():
    """Prepare pit window training data"""
    
    logger.info("="*80)
    logger.info("PIT WINDOW DATA PREPARATION")
    logger.info("="*80)
    
    # Load multi-race training data
    train_laps_path = settings.base_dir / 'ml/datasets/train_laps.csv'
    
    if not train_laps_path.exists():
        logger.error(f"Training data not found at {train_laps_path}")
        logger.info("Please run collect_multi_race_data.py first")
        return
    
    logger.info(f"Loading training data from {train_laps_path}")
    train_laps = pd.read_csv(train_laps_path)
    
    logger.info(f"Loaded {len(train_laps)} laps from training data")
    logger.info(f"Races: {train_laps['Race'].nunique()}")
    logger.info(f"Drivers: {train_laps['Driver'].nunique()}")
    
    # Create features
    logger.info("\nCreating pit window features...")
    pit_features = create_pit_window_features(train_laps)
    
    logger.info(f"Created {len(pit_features)} training samples")
    
    # Class distribution
    pit_count = pit_features['should_pit'].sum()
    stay_count = len(pit_features) - pit_count
    
    logger.info("\nClass Distribution:")
    logger.info(f"  Pit (within 3 laps): {pit_count} ({pit_count/len(pit_features)*100:.1f}%)")
    logger.info(f"  Stay out: {stay_count} ({stay_count/len(pit_features)*100:.1f}%)")
    
    # Compound distribution
    logger.info("\nCompound Distribution:")
    for compound in pit_features['tire_compound'].unique():
        count = (pit_features['tire_compound'] == compound).sum()
        logger.info(f"  {compound}: {count} samples")
    
    # Save
    output_path = settings.base_dir / 'ml/datasets/pit_window_features.csv'
    pit_features.to_csv(output_path, index=False)
    
    logger.info(f"\nSaved to {output_path}")
    logger.info("="*80)
    logger.info("Data preparation complete!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
