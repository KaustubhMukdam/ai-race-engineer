"""
Telemetry Preprocessing Module
Cleans and engineers features for tire degradation analysis
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json

from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TelemetryPreprocessor:
    """Preprocesses lap and tire data for degradation analysis"""
    
    def __init__(self):
        self.processed_dir = settings.processed_data_dir
    
    def load_session_data(self, session_path: Path) -> Dict[str, pd.DataFrame]:
        """Load all CSV/JSON data from a session directory"""
        try:
            data = {}
            
            # Load lap times
            lap_file = session_path / "lap_times.csv"
            if lap_file.exists():
                data['laps'] = pd.read_csv(lap_file)
                logger.info(f"Loaded {len(data['laps'])} laps")
            
            # Load tire strategies
            strategy_file = session_path / "tire_strategies.json"
            if strategy_file.exists():
                with open(strategy_file, 'r') as f:
                    data['strategies'] = json.load(f)
                logger.info(f"Loaded strategies for {len(data['strategies'])} drivers")
            
            # Load weather
            weather_file = session_path / "weather.csv"
            if weather_file.exists():
                data['weather'] = pd.read_csv(weather_file)
                logger.info(f"Loaded {len(data['weather'])} weather records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            raise
    
    def calculate_lap_time_seconds(self, lap_time_str: str) -> Optional[float]:
        """Convert lap time string to seconds"""
        try:
            if pd.isna(lap_time_str) or lap_time_str == '':
                return None
            # Handle format: "0 days 00:01:29.504000"
            if 'days' in str(lap_time_str):
                time_part = str(lap_time_str).split('days')[1].strip()
                h, m, s = time_part.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
            return None
        except:
            return None
    
    def process_lap_data(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw lap data with feature engineering"""
        try:
            df = laps_df.copy()
            
            # Convert lap time to seconds
            df['LapTime_Seconds'] = df['LapTime'].apply(self.calculate_lap_time_seconds)
            
            # Remove outliers (pit laps, safety car, etc.)
            if 'LapTime_Seconds' in df.columns:
                # Filter out laps slower than 2 minutes (120s) - likely pit laps
                df = df[df['LapTime_Seconds'] < 120].copy()
                # Filter out laps faster than 70s (impossible on most tracks)
                df = df[df['LapTime_Seconds'] > 70].copy()
            
            # Add rolling average lap time (3-lap window)
            df['RollingAvgLapTime'] = df.groupby('Driver')['LapTime_Seconds'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Calculate lap time delta from baseline (lap 5 - first clean lap)
            df['BaselineLapTime'] = df.groupby(['Driver', 'Stint'])['LapTime_Seconds'].transform(
                lambda x: x.iloc[4] if len(x) > 4 else x.iloc[0]
            )
            df['LapTimeDelta'] = df['LapTime_Seconds'] - df['BaselineLapTime']
            
            # Degradation rate (seconds lost per lap)
            df['DegradationRate'] = df.groupby(['Driver', 'Stint'])['LapTimeDelta'].transform(
                lambda x: x.diff()
            )
            
            logger.info(f"Processed {len(df)} laps with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error processing lap data: {e}")
            raise
    
    def merge_weather_data(
        self, 
        laps_df: pd.DataFrame, 
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge weather data with lap data based on timestamps"""
        try:
            # Convert time columns to datetime
            laps_df['Time'] = pd.to_timedelta(laps_df['Time'])
            weather_df['Time'] = pd.to_timedelta(weather_df['Time'])
            
            # Merge using nearest timestamp (asof merge)
            laps_df = laps_df.sort_values('Time')
            weather_df = weather_df.sort_values('Time')
            
            merged = pd.merge_asof(
                laps_df,
                weather_df[['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']],
                on='Time',
                direction='nearest'
            )
            
            logger.info("Merged weather data with lap data")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging weather data: {e}")
            return laps_df  # Return without weather if merge fails
    
    def calculate_tire_degradation(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive tire degradation metrics
        
        Returns DataFrame with degradation analysis per stint
        """
        try:
            degradation_data = []
            
            for driver in laps_df['Driver'].unique():
                driver_laps = laps_df[laps_df['Driver'] == driver]
                
                for stint in driver_laps['Stint'].unique():
                    stint_laps = driver_laps[driver_laps['Stint'] == stint].copy()
                    
                    if len(stint_laps) < 3:  # Need at least 3 laps for analysis
                        continue
                    
                    # Get clean laps (exclude first 2 laps for tire warmup)
                    clean_laps = stint_laps.iloc[2:] if len(stint_laps) > 2 else stint_laps
                    
                    if len(clean_laps) == 0:
                        continue
                    
                    # Calculate degradation metrics
                    first_clean_lap = clean_laps['LapTime_Seconds'].iloc[0]
                    last_clean_lap = clean_laps['LapTime_Seconds'].iloc[-1]
                    
                    total_degradation = last_clean_lap - first_clean_lap
                    laps_in_stint = len(clean_laps)
                    deg_per_lap = total_degradation / laps_in_stint if laps_in_stint > 0 else 0
                    
                    # Exponential degradation factor (tires degrade faster over time)
                    lap_numbers = clean_laps['TyreLife'].values
                    lap_times = clean_laps['LapTime_Seconds'].values
                    
                    # Fit linear regression for degradation rate
                    if len(lap_numbers) > 1:
                        coef = np.polyfit(lap_numbers, lap_times, 1)
                        linear_deg_rate = coef[0]  # Slope
                    else:
                        linear_deg_rate = 0
                    
                    degradation_data.append({
                        'Driver': driver,
                        'Stint': int(stint),
                        'Compound': stint_laps['Compound'].iloc[0],
                        'TotalLaps': len(stint_laps),
                        'CleanLaps': len(clean_laps),
                        'FirstCleanLapTime': first_clean_lap,
                        'LastCleanLapTime': last_clean_lap,
                        'TotalDegradation_Seconds': total_degradation,
                        'DegradationPerLap_Seconds': deg_per_lap,
                        'LinearDegradationRate': linear_deg_rate,
                        'AvgTrackTemp': stint_laps['TrackTemp'].mean() if 'TrackTemp' in stint_laps.columns else None,
                        'AvgAirTemp': stint_laps['AirTemp'].mean() if 'AirTemp' in stint_laps.columns else None,
                        'StartLap': int(stint_laps['LapNumber'].min()),
                        'EndLap': int(stint_laps['LapNumber'].max())
                    })
            
            degradation_df = pd.DataFrame(degradation_data)
            logger.info(f"Calculated degradation for {len(degradation_df)} stints")
            return degradation_df
            
        except Exception as e:
            logger.error(f"Error calculating tire degradation: {e}")
            raise
    
    def identify_optimal_pit_windows(
        self, 
        degradation_df: pd.DataFrame,
        threshold_seconds: float = 0.15
    ) -> Dict[str, List[int]]:
        """
        Identify optimal pit windows based on degradation rate
        
        Args:
            degradation_df: Degradation analysis DataFrame
            threshold_seconds: Degradation per lap threshold for pit recommendation
        
        Returns:
            Dictionary mapping compound to recommended pit lap ranges
        """
        try:
            pit_windows = {}
            
            # Analyze by compound
            for compound in degradation_df['Compound'].unique():
                compound_stints = degradation_df[degradation_df['Compound'] == compound]
                
                # Find average lap where degradation exceeds threshold
                critical_laps = []
                for _, stint in compound_stints.iterrows():
                    if stint['DegradationPerLap_Seconds'] > threshold_seconds:
                        # Estimate when degradation exceeded threshold
                        critical_lap = stint['StartLap'] + (stint['TotalLaps'] * 0.7)
                        critical_laps.append(int(critical_lap))
                
                if critical_laps:
                    avg_critical_lap = int(np.mean(critical_laps))
                    # Recommend pit window (±3 laps from critical point)
                    pit_windows[compound] = [
                        max(1, avg_critical_lap - 3),
                        avg_critical_lap + 3
                    ]
            
            logger.info(f"Identified pit windows: {pit_windows}")
            return pit_windows
            
        except Exception as e:
            logger.error(f"Error identifying pit windows: {e}")
            return {}
    
    def export_processed_data(
        self,
        session_name: str,
        laps_df: pd.DataFrame,
        degradation_df: pd.DataFrame,
        pit_windows: Dict[str, List[int]]
    ) -> Dict[str, Path]:
        """Export all processed data"""
        try:
            output_dir = self.processed_dir / f"{session_name}_processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported = {}
            
            # Export processed laps
            laps_file = output_dir / "processed_laps.csv"
            laps_df.to_csv(laps_file, index=False)
            exported['laps'] = laps_file
            
            # Export degradation analysis
            deg_file = output_dir / "tire_degradation_analysis.csv"
            degradation_df.to_csv(deg_file, index=False)
            exported['degradation'] = deg_file
            
            # Export pit windows
            pit_file = output_dir / "optimal_pit_windows.json"
            with open(pit_file, 'w') as f:
                json.dump(pit_windows, f, indent=2)
            exported['pit_windows'] = pit_file
            
            logger.info(f"Exported processed data to {output_dir}")
            return exported
            
        except Exception as e:
            logger.error(f"Error exporting processed data: {e}")
            raise


def main():
    """Example usage"""
    preprocessor = TelemetryPreprocessor()
    
    # Load 2024 Abu Dhabi GP data
    session_path = settings.processed_data_dir / "2024_Abu_Dhabi_Grand_Prix_Race"
    
    logger.info(f"Loading data from {session_path}")
    data = preprocessor.load_session_data(session_path)
    
    # Process lap data
    processed_laps = preprocessor.process_lap_data(data['laps'])
    
    # Merge weather
    if 'weather' in data:
        processed_laps = preprocessor.merge_weather_data(processed_laps, data['weather'])
    
    # Calculate tire degradation
    degradation_df = preprocessor.calculate_tire_degradation(processed_laps)
    
    print("\n" + "="*80)
    print("TIRE DEGRADATION ANALYSIS - 2024 ABU DHABI GP")
    print("="*80)
    print(degradation_df.to_string(index=False))
    
    # Identify optimal pit windows
    pit_windows = preprocessor.identify_optimal_pit_windows(degradation_df)
    
    print("\n" + "="*80)
    print("OPTIMAL PIT WINDOWS")
    print("="*80)
    for compound, window in pit_windows.items():
        print(f"{compound:10} → Pit between laps {window[0]}-{window[1]}")
    
    # Export processed data
    exported = preprocessor.export_processed_data(
        session_name="2024_Abu_Dhabi_GP",
        laps_df=processed_laps,
        degradation_df=degradation_df,
        pit_windows=pit_windows
    )
    
    print("\n✅ Data preprocessing complete!")
    print("\nExported files:")
    for data_type, file_path in exported.items():
        print(f"  - {data_type}: {file_path}")


if __name__ == "__main__":
    main()
