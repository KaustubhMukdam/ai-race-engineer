"""
Session Manager
Handles dynamic loading and caching of multiple F1 race sessions
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import fastf1
import pandas as pd
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime

from config.app_config import settings
from utils.logger import setup_logger
from data.scripts.fetch_fastf1_data import FastF1DataFetcher
from data.scripts.preprocess_telemetry import TelemetryPreprocessor

logger = setup_logger(__name__)


class SessionManager:
    """Manages multiple F1 race sessions and their processed data"""
    
    def __init__(self):
        self.fetcher = FastF1DataFetcher()
        self.preprocessor = TelemetryPreprocessor()
        self.cache_index_file = settings.processed_data_dir / "session_cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """Load cache index tracking all processed sessions"""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
        logger.info("Cache index saved")
    
    def get_available_seasons(self) -> List[int]:
        """Get list of available F1 seasons (2018-current year)"""
        current_year = datetime.now().year
        return list(range(2018, current_year + 1))
    
    def get_season_schedule(self, year: int) -> pd.DataFrame:
        """
        Get event schedule for a specific season
        
        Args:
            year: Season year
        
        Returns:
            DataFrame with event schedule
        """
        try:
            schedule = self.fetcher.get_event_schedule(year)
            return schedule
        except Exception as e:
            logger.error(f"Error fetching schedule for {year}: {e}")
            raise
    
    def get_session_key(self, year: int, event: str, session: str) -> str:
        """Generate unique key for session"""
        return f"{year}_{event.replace(' ', '_')}_{session}"
    
    def is_session_cached(self, year: int, event: str, session: str) -> bool:
        """Check if session data is already processed and cached"""
        key = self.get_session_key(year, event, session)
        return key in self.cache_index
    
    def get_cached_session_path(self, year: int, event: str, session: str) -> Optional[Path]:
        """Get path to cached session data"""
        key = self.get_session_key(year, event, session)
        if key in self.cache_index:
            return Path(self.cache_index[key]['processed_path'])
        return None
    
    def load_and_process_session(
        self,
        year: int,
        event: str,
        session: str = "Race",
        force_reload: bool = False
    ) -> Dict[str, Path]:
        """
        Load and process a complete F1 session
        
        Args:
            year: Season year
            event: Event name or round number
            session: Session type (Race, Qualifying, etc.)
            force_reload: Force reload even if cached
        
        Returns:
            Dictionary with paths to processed data files
        """
        try:
            session_key = self.get_session_key(year, event, session)
            
            # Check cache first
            if not force_reload and self.is_session_cached(year, event, session):
                logger.info(f"Session {session_key} found in cache")
                cached_path = self.get_cached_session_path(year, event, session)
                
                return {
                    'laps': cached_path / 'processed_laps.csv',
                    'degradation': cached_path / 'tire_degradation_analysis.csv',
                    'pit_windows': cached_path / 'optimal_pit_windows.json',
                    'session_key': session_key,
                    'cached': True
                }
            
            logger.info(f"Loading fresh session: {session_key}")
            
            # Step 1: Load session from FastF1
            session_obj = self.fetcher.load_session(
                year=year,
                event=event,
                session=session
            )
            
            # Step 2: Export raw session data
            raw_export = self.fetcher.export_session_data(session_obj)
            raw_session_dir = raw_export['lap_times'].parent
            
            # Step 3: Process telemetry data
            session_data = self.preprocessor.load_session_data(raw_session_dir)
            processed_laps = self.preprocessor.process_lap_data(session_data['laps'])
            
            # Merge weather if available
            if 'weather' in session_data:
                processed_laps = self.preprocessor.merge_weather_data(
                    processed_laps, 
                    session_data['weather']
                )
            
            # Calculate degradation
            degradation_df = self.preprocessor.calculate_tire_degradation(processed_laps)
            pit_windows = self.preprocessor.identify_optimal_pit_windows(degradation_df)
            
            # Step 4: Export processed data
            processed_name = f"{year}_{event.replace(' ', '_')}_{session}_processed"
            exported = self.preprocessor.export_processed_data(
                session_name=session_key,  # Changed from processed_name
                laps_df=processed_laps,
                degradation_df=degradation_df,
                pit_windows=pit_windows
            )
            
            # Step 5: Update cache index
            self.cache_index[session_key] = {
                'year': year,
                'event': event,
                'session': session,
                'processed_path': str(exported['laps'].parent),
                'processed_date': datetime.now().isoformat(),
                'total_laps': len(processed_laps),
                'drivers': int(processed_laps['Driver'].nunique())
            }
            self._save_cache_index()
            
            logger.info(f"Session {session_key} processed and cached successfully")
            
            return {
                'laps': exported['laps'],
                'degradation': exported['degradation'],
                'pit_windows': exported['pit_windows'],
                'session_key': session_key,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Error processing session {year} {event} {session}: {e}")
            raise
    
    def list_cached_sessions(self) -> pd.DataFrame:
        """Get DataFrame of all cached sessions"""
        if not self.cache_index:
            return pd.DataFrame(columns=['session_key', 'year', 'event', 'session', 
                                        'processed_date', 'total_laps', 'drivers'])
        
        data = []
        for key, info in self.cache_index.items():
            data.append({
                'session_key': key,
                **info
            })
        
        return pd.DataFrame(data)
    
    def delete_cached_session(self, year: int, event: str, session: str) -> bool:
        """Delete cached session data"""
        try:
            session_key = self.get_session_key(year, event, session)
            
            if session_key not in self.cache_index:
                logger.warning(f"Session {session_key} not found in cache")
                return False
            
            # Delete files
            cached_path = Path(self.cache_index[session_key]['processed_path'])
            if cached_path.exists():
                import shutil
                shutil.rmtree(cached_path)
                logger.info(f"Deleted cached files at {cached_path}")
            
            # Remove from index
            del self.cache_index[session_key]
            self._save_cache_index()
            
            logger.info(f"Session {session_key} removed from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cached session: {e}")
            return False
    
    def get_session_metadata(self, year: int, event: str, session: str) -> Optional[Dict]:
        """Get metadata for a cached session"""
        session_key = self.get_session_key(year, event, session)
        return self.cache_index.get(session_key)


def main():
    """Test session manager"""
    manager = SessionManager()
    
    print("\n" + "="*80)
    print("SESSION MANAGER TEST")
    print("="*80)
    
    # Test 1: Get available seasons
    seasons = manager.get_available_seasons()
    print(f"\n✅ Available seasons: {seasons}")
    
    # Test 2: Get 2024 schedule
    print("\n📅 Fetching 2024 schedule...")
    schedule_2024 = manager.get_season_schedule(2024)
    print(f"Found {len(schedule_2024)} events in 2024")
    print("\nFirst 5 events:")
    print(schedule_2024[['RoundNumber', 'EventName', 'EventDate']].head())
    
    # Test 3: Load and process a session
    print("\n\n🏎️ Loading 2024 Monaco GP Race...")
    session_data = manager.load_and_process_session(
        year=2024,
        event="Monaco",
        session="Race"
    )
    
    print(f"\n✅ Session processed!")
    print(f"Session Key: {session_data['session_key']}")
    print(f"Cached: {session_data['cached']}")
    print(f"Files:")
    for key, path in session_data.items():
        if key not in ['session_key', 'cached']:
            print(f"  - {key}: {path}")
    
    # Test 4: List all cached sessions
    print("\n\n📊 All Cached Sessions:")
    cached = manager.list_cached_sessions()
    print(cached.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ Session Manager test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
