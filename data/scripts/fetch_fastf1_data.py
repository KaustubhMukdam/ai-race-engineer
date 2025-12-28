"""
FastF1 Data Fetching Module
Handles downloading and caching F1 telemetry and timing data
"""
import fastf1
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FastF1DataFetcher:
    """Handles fetching and caching F1 data using FastF1 library"""
    
    def __init__(self):
        """Initialize FastF1 with cache directory"""
        fastf1.Cache.enable_cache(str(settings.fastf1_cache_dir))
        logger.info(f"FastF1 cache enabled at: {settings.fastf1_cache_dir}")
    
    def get_event_schedule(self, year: int) -> pd.DataFrame:
        """
        Get complete event schedule for a season
        
        Args:
            year: Season year (e.g., 2024)
        
        Returns:
            DataFrame with event schedule
        """
        try:
            logger.info(f"Fetching event schedule for {year}")
            schedule = fastf1.get_event_schedule(year)
            logger.info(f"Found {len(schedule)} events for {year} season")
            return schedule
        except Exception as e:
            logger.error(f"Error fetching event schedule for {year}: {e}")
            raise
    
    def load_session(
        self,
        year: int,
        event: str,
        session: str = "Race",
        load_telemetry: bool = True,
        load_weather: bool = True,
        load_messages: bool = True
    ) -> fastf1.core.Session:
        """
        Load a complete session with telemetry and timing data
        
        Args:
            year: Season year
            event: Event name or round number
            session: Session identifier (FP1, FP2, FP3, Qualifying, Sprint, Race)
            load_telemetry: Whether to load telemetry data
            load_weather: Whether to load weather data
            load_messages: Whether to load team radio messages
        
        Returns:
            Loaded FastF1 Session object
        """
        try:
            logger.info(f"Loading {year} {event} {session}")
            start_time = time.time()
            
            session_obj = fastf1.get_session(year, event, session)
            session_obj.load(
                telemetry=load_telemetry,
                weather=load_weather,
                messages=load_messages
            )
            
            load_time = time.time() - start_time
            logger.info(f"Session loaded in {load_time:.2f}s")
            logger.info(f"Session date: {session_obj.date}")
            logger.info(f"Circuit: {session_obj.event['EventName']}")
            
            return session_obj
            
        except Exception as e:
            logger.error(f"Error loading session {year} {event} {session}: {e}")
            raise
    
    def get_driver_telemetry(
        self,
        session: fastf1.core.Session,
        driver: str,
        lap_number: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract telemetry for a specific driver
        
        Args:
            session: Loaded FastF1 session
            driver: Driver identifier (3-letter code or name)
            lap_number: Specific lap number (None for all laps)
        
        Returns:
            DataFrame with telemetry data
        """
        try:
            driver_laps = session.laps.pick_driver(driver)
            
            if lap_number:
                logger.info(f"Extracting telemetry for {driver}, Lap {lap_number}")
                lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
                telemetry = lap.get_telemetry()
            else:
                logger.info(f"Extracting all telemetry for {driver}")
                telemetry = driver_laps.get_telemetry()
            
            logger.info(f"Telemetry shape: {telemetry.shape}")
            return telemetry
            
        except Exception as e:
            logger.error(f"Error extracting telemetry for {driver}: {e}")
            raise
    
    def get_lap_times(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract lap times for all drivers
        
        Args:
            session: Loaded FastF1 session
        
        Returns:
            DataFrame with lap times and metadata
        """
        try:
            logger.info("Extracting lap times")
            laps = session.laps
            
            # Key columns for lap analysis
            lap_data = laps[[
                'Time', 'Driver', 'LapTime', 'LapNumber', 
                'Stint', 'Compound', 'TyreLife',
                'TrackStatus', 'IsPersonalBest'
            ]].copy()
            
            logger.info(f"Extracted {len(lap_data)} laps from {laps['Driver'].nunique()} drivers")
            return lap_data
            
        except Exception as e:
            logger.error(f"Error extracting lap times: {e}")
            raise
    
    def get_tire_strategy(self, session: fastf1.core.Session) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract tire strategy for all drivers
        
        Args:
            session: Loaded FastF1 session
        
        Returns:
            Dictionary mapping driver to list of stint information
        """
        try:
            logger.info("Extracting tire strategies")
            laps = session.laps
            strategies = {}
            
            for driver in laps['Driver'].unique():
                driver_laps = laps[laps['Driver'] == driver]
                stints = []
                
                for stint_num in driver_laps['Stint'].unique():
                    stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                    
                    stint_info = {
                        'stint_number': int(stint_num),
                        'compound': stint_laps.iloc[0]['Compound'],
                        'start_lap': int(stint_laps['LapNumber'].min()),
                        'end_lap': int(stint_laps['LapNumber'].max()),
                        'laps_completed': len(stint_laps),
                        'avg_lap_time': stint_laps['LapTime'].mean().total_seconds() if pd.notna(stint_laps['LapTime'].mean()) else None
                    }
                    stints.append(stint_info)
                
                strategies[driver] = stints
            
            logger.info(f"Extracted strategies for {len(strategies)} drivers")
            return strategies
            
        except Exception as e:
            logger.error(f"Error extracting tire strategies: {e}")
            raise
    
    def export_session_data(
        self,
        session: fastf1.core.Session,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Export session data to CSV files
        
        Args:
            session: Loaded FastF1 session
            output_dir: Directory to save files (default: processed_data_dir)
        
        Returns:
            Dictionary mapping data type to file path
        """
        try:
            if output_dir is None:
                output_dir = settings.processed_data_dir
            
            # Create subdirectory for this session
            session_dir = output_dir / f"{session.event.year}_{session.event['EventName']}_{session.name}".replace(" ", "_")
            session_dir.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            
            # Export lap times
            lap_times = self.get_lap_times(session)
            lap_file = session_dir / "lap_times.csv"
            lap_times.to_csv(lap_file, index=False)
            exported_files['lap_times'] = lap_file
            logger.info(f"Exported lap times to {lap_file}")
            
            # Export tire strategies
            strategies = self.get_tire_strategy(session)
            strategy_file = session_dir / "tire_strategies.json"
            import json
            with open(strategy_file, 'w') as f:
                json.dump(strategies, f, indent=2)
            exported_files['tire_strategies'] = strategy_file
            logger.info(f"Exported tire strategies to {strategy_file}")
            
            # Export weather data if available
            if session.weather_data is not None and not session.weather_data.empty:
                weather_file = session_dir / "weather.csv"
                session.weather_data.to_csv(weather_file, index=False)
                exported_files['weather'] = weather_file
                logger.info(f"Exported weather data to {weather_file}")
            
            logger.info(f"All session data exported to {session_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting session data: {e}")
            raise


def main():
    """Example usage and testing"""
    fetcher = FastF1DataFetcher()
    
    # Example: Load 2024 Abu Dhabi GP Race
    try:
        session = fetcher.load_session(
            year=2024,
            event="Abu Dhabi",
            session="Race"
        )
        
        # Export data
        exported_files = fetcher.export_session_data(session)
        print("\n✅ Data fetch complete!")
        print("\nExported files:")
        for data_type, file_path in exported_files.items():
            print(f"  - {data_type}: {file_path}")
        
        # Get Max Verstappen's telemetry from lap 1
        verstappen_telemetry = fetcher.get_driver_telemetry(
            session=session,
            driver="VER",
            lap_number=1
        )
        print(f"\n📊 Verstappen Lap 1 telemetry: {verstappen_telemetry.shape[0]} data points")
        print(verstappen_telemetry.head())
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
