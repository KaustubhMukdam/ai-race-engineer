"""
Multi-Race Data Collector for LSTM Training
Fetches and processes multiple races to create diverse training dataset
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import fastf1
from typing import List, Dict
from tqdm import tqdm

from config.app_config import settings
from utils.logger import setup_logger
from data.scripts.session_manager import SessionManager

logger = setup_logger(__name__)


class MultiRaceCollector:
    """Collect and process lap data from multiple races"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        settings.ensure_directories()
        
        # Enable FastF1 cache
        fastf1.Cache.enable_cache(str(settings.fastf1_cache_dir))
    
    def get_2024_races(self) -> List[Dict]:
        """
        Define 2024 + 2023 races with variety of track characteristics
        """
        races = [
            # 2024 Season
            {"year": 2024, "gp": "Bahrain", "session": "Race", "characteristics": "High temp, high deg"},
            {"year": 2024, "gp": "Saudi Arabia", "session": "Race", "characteristics": "Street circuit, high speed"},
            {"year": 2024, "gp": "Australia", "session": "Race", "characteristics": "Street circuit, medium deg"},
            {"year": 2024, "gp": "Japan", "session": "Race", "characteristics": "Technical circuit"},
            {"year": 2024, "gp": "China", "session": "Race", "characteristics": "Medium degradation"},
            {"year": 2024, "gp": "Monaco", "session": "Race", "characteristics": "Low speed, low deg"},
            {"year": 2024, "gp": "Spain", "session": "Race", "characteristics": "Balanced circuit"},
            {"year": 2024, "gp": "Austria", "session": "Race", "characteristics": "Short lap, high deg"},
            {"year": 2024, "gp": "Britain", "session": "Race", "characteristics": "Fast corners, medium deg"},
            {"year": 2024, "gp": "Hungary", "session": "Race", "characteristics": "Twisty, high deg"},
            {"year": 2024, "gp": "Abu Dhabi", "session": "Race", "characteristics": "Long straights, high deg"},
            
            # 2023 Season (for diversity)
            {"year": 2023, "gp": "Bahrain", "session": "Race", "characteristics": "High temp"},
            {"year": 2023, "gp": "Saudi Arabia", "session": "Race", "characteristics": "Street circuit"},
            {"year": 2023, "gp": "Spain", "session": "Race", "characteristics": "Balanced"},
            {"year": 2023, "gp": "Monaco", "session": "Race", "characteristics": "Low speed"},
            {"year": 2023, "gp": "Austria", "session": "Race", "characteristics": "Short lap"},
            {"year": 2023, "gp": "Britain", "session": "Race", "characteristics": "Fast corners"},
            {"year": 2023, "gp": "Hungary", "session": "Race", "characteristics": "Twisty"},
            {"year": 2023, "gp": "Belgium", "session": "Race", "characteristics": "High speed"},
            {"year": 2023, "gp": "Abu Dhabi", "session": "Race", "characteristics": "Long straights, high deg"},
        ]
        
        return races

    def process_race(self, year: int, gp: str, session: str) -> pd.DataFrame:
        """
        Process a single race and return lap data
        
        Returns:
            DataFrame with columns: Driver, LapNumber, LapTime_Seconds, 
                                   Compound, TyreLife, Stint, TrackTemp, AirTemp
        """
        try:
            logger.info(f"Processing {year} {gp} {session}...")
            
            # Load session
            session_obj = fastf1.get_session(year, gp, session)
            session_obj.load()
            
            # Get laps
            laps = session_obj.laps
            
            # Filter valid laps
            laps = laps[
                (laps['LapTime'].notna()) &
                (laps['Compound'].notna()) &
                (laps['TyreLife'].notna()) &
                (~laps['IsAccurate'].isna())
            ].copy()
            
            # Convert lap time to seconds
            laps['LapTime_Seconds'] = laps['LapTime'].dt.total_seconds()
            
            # Get weather data
            if hasattr(session_obj, 'weather_data') and session_obj.weather_data is not None:
                weather = session_obj.weather_data
                # Get median weather values
                track_temp = weather['TrackTemp'].median() if 'TrackTemp' in weather.columns else 30.0
                air_temp = weather['AirTemp'].median() if 'AirTemp' in weather.columns else 25.0
            else:
                track_temp = 30.0
                air_temp = 25.0
            
            # Select relevant columns
            processed = laps[[
                'Driver', 'LapNumber', 'LapTime_Seconds', 
                'Compound', 'TyreLife', 'Stint'
            ]].copy()
            
            # Add weather
            processed['TrackTemp'] = track_temp
            processed['AirTemp'] = air_temp
            
            # Add race identifier
            processed['Race'] = f"{year}_{gp.replace(' ', '_')}"
            
            logger.info(f"Processed {len(processed)} laps from {year} {gp}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing {year} {gp}: {e}")
            return pd.DataFrame()
    
    def collect_all_races(self, races: List[Dict], output_file: Path) -> pd.DataFrame:
        """
        Collect data from all races and save combined dataset
        
        Args:
            races: List of race definitions
            output_file: Path to save combined CSV
        
        Returns:
            Combined DataFrame with all race data
        """
        all_laps = []
        
        logger.info(f"Starting collection of {len(races)} races...")
        
        for race_info in tqdm(races, desc="Processing races"):
            year = race_info['year']
            gp = race_info['gp']
            session = race_info['session']
            
            race_laps = self.process_race(year, gp, session)
            
            if not race_laps.empty:
                all_laps.append(race_laps)
        
        if not all_laps:
            logger.error("No race data collected!")
            return pd.DataFrame()
        
        # Combine all races
        combined = pd.concat(all_laps, ignore_index=True)
        
        # Save to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_file, index=False)
        
        logger.info(f"Saved combined dataset: {len(combined)} laps from {len(races)} races")
        logger.info(f"Output file: {output_file}")
        
        # Print summary statistics
        logger.info("\n" + "="*80)
        logger.info("DATASET SUMMARY")
        logger.info("="*80)
        logger.info(f"Total laps: {len(combined)}")
        logger.info(f"Unique drivers: {combined['Driver'].nunique()}")
        logger.info(f"Races: {combined['Race'].nunique()}")
        logger.info(f"\nLaps per compound:")
        logger.info(combined['Compound'].value_counts().to_string())
        logger.info(f"\nLaps per race:")
        logger.info(combined['Race'].value_counts().to_string())
        logger.info("="*80)
        
        return combined
    
    def split_train_test(
        self,
        combined_df: pd.DataFrame,
        train_output: Path,
        test_output: Path,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Split dataset into training and testing by LAPS (not races)
        This ensures test set can use race encodings from training
        """
        from sklearn.model_selection import train_test_split
        
        # Split by laps, stratified by race to get laps from all races in both sets
        train_df, test_df = train_test_split(
            combined_df,
            test_size=test_size,
            random_state=random_state,
            stratify=combined_df['Race']  # Ensure all races in both sets
        )
        
        # Save
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        logger.info(f"✅ Training set: {len(train_df)} laps from {train_df['Race'].nunique()} races")
        logger.info(f"   Races: {train_df['Race'].unique().tolist()}")
        logger.info(f"✅ Test set: {len(test_df)} laps from {test_df['Race'].nunique()} races")
        logger.info(f"   Races: {test_df['Race'].unique().tolist()}")
        logger.info(f"📁 Training data saved to: {train_output}")
        logger.info(f"📁 Test data saved to: {test_output}")

def main():
    """Collect multi-race training dataset"""
    
    collector = MultiRaceCollector()
    
    # Define output paths
    datasets_dir = settings.base_dir / 'ml' / 'datasets'
    combined_file = datasets_dir / 'multi_race_training_data.csv'
    train_file = datasets_dir / 'train_laps.csv'
    test_file = datasets_dir / 'test_laps.csv'
    
    # Get races (current 8 races from 2024)
    races = collector.get_2024_races()
    
    print("\n" + "="*80)
    print("MULTI-RACE LSTM TRAINING DATA COLLECTOR")
    print("="*80)
    print(f"Collecting data from {len(races)} races:")
    for race in races:
        print(f"  - {race['year']} {race['gp']}: {race['characteristics']}")
    print("="*80 + "\n")
    
    # Collect all races
    combined_df = collector.collect_all_races(races, combined_file)
    
    if combined_df.empty:
        print("\n❌ Failed to collect race data")
        return
    
    # Split into train/test BY LAPS (not by entire races)
    collector.split_train_test(
        combined_df=combined_df,
        train_output=train_file,
        test_output=test_file,
        test_size=0.2  # 80% train, 20% test
    )
    
    print("\n" + "="*80)
    print("✅ DATA COLLECTION COMPLETE!")
    print("="*80)
    print(f"Combined dataset: {combined_file}")
    print(f"Training set: {train_file}")
    print(f"Test set: {test_file}")
    print("\n⚠️  IMPORTANT: Train and test both contain ALL races")
    print("   This ensures model learns track baselines and generalizes")
    print("\nNext step: Retrain LSTM model with:")
    print("  python ml/training/train_multi_race_lstm.py")
    print("="*80)


if __name__ == "__main__":
    main()
