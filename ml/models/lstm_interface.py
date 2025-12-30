"""
LSTM Inference Wrapper
Simplified interface for real-time tire degradation predictions
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional

from ml.models.tire_degradation_lstm import TireDegradationPredictor, TireDegradationLSTM
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMInferenceEngine:
    """
    Simplified interface for LSTM tire degradation predictions
    Handles preprocessing and provides easy-to-use prediction methods
    """
    
    def __init__(self):
        self.predictor = TireDegradationPredictor()
        self.model_loaded = False
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load pre-trained model"""
        try:
            if self.predictor.model_path.exists():
                self.predictor.load_model(input_size=8)
                self.model_loaded = True
                logger.info("LSTM model loaded successfully for inference")
            else:
                logger.warning(f"LSTM model not found at {self.predictor.model_path}")
                logger.info("Predictions will use fallback rule-based method")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self.model_loaded = False
    
    def predict_next_lap(
        self,
        driver: str,
        lap_number: int,
        tire_age: int,
        compound: str,
        track_temp: float,
        air_temp: float,
        recent_lap_times: List[float],
        race_name: str = "2024_Monaco",  # 🔥 NEW parameter
        sequence_length: int = 10
    ) -> Optional[float]:
        """Predict next lap time using LSTM model"""
        if not self.model_loaded:
            return None
        
        try:
            if len(recent_lap_times) < sequence_length:
                return None
            
            recent_laps = recent_lap_times[-sequence_length:]
            
            # Encode categorical features
            try:
                compound_encoded = self.predictor.compound_encoder.transform([compound])[0]
            except:
                compound_encoded = 0
            
            try:
                driver_encoded = self.predictor.driver_encoder.transform([driver])[0]
            except:
                driver_encoded = 0
            
            # 🔥 NEW: Encode race/track
            try:
                race_encoded = self.predictor.race_encoder.transform([race_name])[0]
            except:
                logger.warning(f"Unknown race: {race_name}, using default")
                # Use first race in encoder as fallback
                race_encoded = 0
            
            # Build sequence
            sequence = []
            for i, lap_time in enumerate(recent_laps):
                historical_tire_age = tire_age - (sequence_length - i - 1)
                historical_lap_number = lap_number - (sequence_length - i - 1)
                prev_lap_time = recent_laps[i-1] if i > 0 else lap_time
                
                # 🔥 UPDATED: 8 features now
                features = [
                    historical_lap_number,
                    historical_tire_age,
                    track_temp,
                    air_temp,
                    compound_encoded,
                    driver_encoded,
                    race_encoded,      # 🔥 NEW
                    prev_lap_time
                ]
                
                sequence.append(features)
            
            sequence_array = np.array(sequence)
            
            # Use feature_scaler
            sequence_scaled = self.predictor.feature_scaler.transform(pd.DataFrame([sequence[-1]], columns=self.predictor.feature_scaler.feature_names_in_))
            
            # Predict
            predicted_lap_time = self.predictor.predict(sequence_scaled)
            
            return predicted_lap_time
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return None

    
    def predict_stint_degradation(
        self,
        driver: str,
        starting_lap: int,
        stint_length: int,
        compound: str,
        track_temp: float,
        air_temp: float,
        baseline_lap_time: float,
        race_name: str = "2024_Monaco"
    ) -> Dict[str, any]:
        """
        Predict full stint degradation lap-by-lap
        
        Args:
            driver: Driver code
            starting_lap: First lap of stint
            stint_length: Number of laps in stint
            compound: Tire compound
            track_temp: Track temperature
            air_temp: Air temperature
            baseline_lap_time: Expected lap time on fresh tires
        
        Returns:
            Dictionary with predicted lap times and degradation metrics
        """
        if not self.model_loaded:
            return {
                'status': 'error',
                'message': 'LSTM model not loaded'
            }
        
        try:
            predicted_laps = []
            recent_laps = [baseline_lap_time] * 10  # Initialize with baseline
            
            for lap in range(stint_length):
                tire_age = lap + 1
                lap_number = starting_lap + lap
                
                # Predict next lap
                predicted_time = self.predict_next_lap(
                    driver=driver,
                    lap_number=lap_number,
                    tire_age=tire_age,
                    compound=compound,
                    track_temp=track_temp,
                    air_temp=air_temp,
                    recent_lap_times=recent_laps,
                    race_name=race_name
                )
                
                if predicted_time is None:
                    # Fallback to simple degradation model
                    predicted_time = baseline_lap_time + (tire_age * 0.02)
                
                predicted_laps.append({
                    'lap_number': lap_number,
                    'tire_age': tire_age,
                    'predicted_time': round(predicted_time, 3)
                })
                
                # Update recent laps for next prediction
                recent_laps.append(predicted_time)
                if len(recent_laps) > 10:
                    recent_laps.pop(0)
            
            # Calculate degradation metrics
            avg_degradation_per_lap = (predicted_laps[-1]['predicted_time'] - predicted_laps[0]['predicted_time']) / stint_length
            total_time_loss = sum(lap['predicted_time'] for lap in predicted_laps) - (baseline_lap_time * stint_length)
            
            return {
                'status': 'success',
                'driver': driver,
                'compound': compound,
                'race': race_name,
                'stint_length': stint_length,
                'predicted_laps': predicted_laps,
                'avg_degradation_per_lap': round(avg_degradation_per_lap, 4),
                'total_time_loss': round(total_time_loss, 3),
                'baseline_lap_time': baseline_lap_time
            }
            
        except Exception as e:
            logger.error(f"Error predicting stint degradation: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def is_available(self) -> bool:
        """Check if LSTM model is available for predictions"""
        return self.model_loaded


def main():
    """Test LSTM inference"""
    engine = LSTMInferenceEngine()
    
    print("\n" + "="*80)
    print("LSTM INFERENCE ENGINE TEST")
    print("="*80)
    
    if not engine.is_available():
        print("\n❌ LSTM model not available")
        print("Train the model first: python ml/models/tire_degradation_lstm.py")
        return
    
    print("\n✅ LSTM model loaded and ready")
    
    # Test 1: Single lap prediction
    print("\n\n🏎️ TEST 1: Predict Next Lap Time")
    print("-"*80)
    
    # Simulate recent lap times (increasing trend = degradation)
    recent_laps = [90.5, 90.6, 90.7, 90.8, 90.9, 91.0, 91.1, 91.2, 91.3, 91.4]
    
    predicted = engine.predict_next_lap(
        driver="VER",
        lap_number=25,
        tire_age=15,
        compound="MEDIUM",
        track_temp=31.0,
        air_temp=26.5,
        recent_lap_times=recent_laps
    )
    
    if predicted:
        print(f"Recent lap times: {recent_laps[-3:]}")
        print(f"Predicted next lap: {predicted:.3f}s")
        print(f"Expected degradation trend: {predicted - recent_laps[-1]:.3f}s")
    else:
        print("Prediction failed")
    
    # Test 2: Full stint prediction
    print("\n\nTEST 2: Predict Full Stint Degradation")
    print("-"*80)
    
    stint_prediction = engine.predict_stint_degradation(
        driver="VER",
        starting_lap=20,
        stint_length=25,
        compound="MEDIUM",
        track_temp=31.0,
        air_temp=26.5,
        baseline_lap_time=90.5
    )
    
    if stint_prediction['status'] == 'success':
        print(f"Driver: {stint_prediction['driver']}")
        print(f"Compound: {stint_prediction['compound']}")
        print(f"Stint length: {stint_prediction['stint_length']} laps")
        print(f"\nBaseline lap time: {stint_prediction['baseline_lap_time']:.3f}s")
        print(f"Avg degradation: {stint_prediction['avg_degradation_per_lap']:.4f}s/lap")
        print(f"Total time loss: {stint_prediction['total_time_loss']:.3f}s")
        
        print("\nFirst 5 laps:")
        for lap in stint_prediction['predicted_laps'][:5]:
            print(f"  Lap {lap['lap_number']} (age {lap['tire_age']}): {lap['predicted_time']:.3f}s")
        
        print("\nLast 5 laps:")
        for lap in stint_prediction['predicted_laps'][-5:]:
            print(f"  Lap {lap['lap_number']} (age {lap['tire_age']}): {lap['predicted_time']:.3f}s")
    else:
        print(f"Error: {stint_prediction['message']}")
    
    print("\n" + "="*80)
    print("LSTM Inference Engine test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
