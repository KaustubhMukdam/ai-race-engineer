"""
Verstappen Simulator API Routes
Compares aggressive vs baseline strategy using multi-agent system
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

from config.app_config import settings
from utils.logger import setup_logger
from ml.models.lstm_interface import LSTMInferenceEngine

logger = setup_logger(__name__)

# Initialize router
router = APIRouter(prefix="/verstappen", tags=["Verstappen Simulator"])

# Global LSTM engine
lstm_engine: Optional[LSTMInferenceEngine] = None


def initialize_simulator():
    """Initialize LSTM engine for simulator"""
    global lstm_engine
    try:
        lstm_engine = LSTMInferenceEngine()
        if lstm_engine.is_available():
            logger.info("Verstappen simulator initialized with LSTM model")
        else:
            logger.warning("LSTM model not available for simulator")
    except Exception as e:
        logger.error(f"Failed to initialize simulator: {e}")


class SimulationRequest(BaseModel):
    session_key: str
    starting_lap: int
    ending_lap: int
    tire_compound: str
    track_temp: float
    air_temp: float


@router.post("/simulate")
async def simulate_verstappen_strategy(request: SimulationRequest) -> Dict[str, Any]:
    """
    Run Verstappen aggressive vs baseline strategy simulation

    Uses LSTM tire model to predict degradation under two scenarios:
    1. Verstappen aggressive: Higher tire loads, faster lap times, more degradation
    2. Baseline conservative: Normal tire management, slower but consistent
    """
    try:
        if lstm_engine is None or not lstm_engine.is_available():
            raise HTTPException(
                status_code=503,
                detail="LSTM model not available. Train the model first."
            )

        logger.info(f"Running Verstappen simulation: Laps {request.starting_lap}-{request.ending_lap}")

        # Calculate stint length
        stint_length = request.ending_lap - request.starting_lap + 1

        # Baseline lap time (conservative driving)
        baseline_lap_time = 90.0  # Placeholder - would come from session data

        # Verstappen aggressive lap time (0.3s faster per lap)
        verstappen_lap_time = baseline_lap_time - 0.3

        # ==================== VERSTAPPEN AGGRESSIVE STRATEGY ====================
        verstappen_prediction = lstm_engine.predict_stint_degradation(
            driver="VER",
            starting_lap=request.starting_lap,
            stint_length=stint_length,
            compound=request.tire_compound,
            track_temp=request.track_temp,
            air_temp=request.air_temp,
            baseline_lap_time=verstappen_lap_time,  # Faster baseline
            race_name=request.session_key
        )

        # ==================== BASELINE CONSERVATIVE STRATEGY ====================
        baseline_prediction = lstm_engine.predict_stint_degradation(
            driver="VER",
            starting_lap=request.starting_lap,
            stint_length=stint_length,
            compound=request.tire_compound,
            track_temp=request.track_temp,
            air_temp=request.air_temp,
            baseline_lap_time=baseline_lap_time,  # Normal pace
            race_name=request.session_key
        )

        # ==================== COMPARISON ANALYSIS ====================

        # Calculate lap-by-lap comparison
        lap_by_lap = []
        for i in range(stint_length):
            ver_lap = verstappen_prediction['predicted_laps'][i]
            base_lap = baseline_prediction['predicted_laps'][i]

            lap_by_lap.append({
                'lap': ver_lap['lap_number'],
                'verstappen_time': ver_lap['predicted_time'],
                'baseline_time': base_lap['predicted_time'],
                'delta': ver_lap['predicted_time'] - base_lap['predicted_time']
            })

        # Calculate cliff laps (when degradation exceeds threshold)
        def find_cliff_lap(predicted_laps, threshold=0.5):
            for i, lap in enumerate(predicted_laps):
                if i > 0:
                    delta = lap['predicted_time'] - predicted_laps[i-1]['predicted_time']
                    if delta > threshold:
                        return lap['lap_number']
            return None

        verstappen_cliff = find_cliff_lap(verstappen_prediction['predicted_laps'])
        baseline_cliff = find_cliff_lap(baseline_prediction['predicted_laps'])

        # ==================== MULTI-AGENT REASONING ====================
        reasoning = f"""**VERSTAPPEN AGGRESSIVE VS BASELINE ANALYSIS**

**VERSTAPPEN STRATEGY (Aggressive):**
- Starting pace: {verstappen_lap_time:.3f}s per lap (0.3s faster than baseline)
- Average degradation: {verstappen_prediction['avg_degradation_per_lap']:.4f}s/lap
- Total time loss: {verstappen_prediction['total_time_loss']:.2f}s over {stint_length} laps
- Predicted tire cliff: {'Lap ' + str(verstappen_cliff) if verstappen_cliff else 'No cliff detected'}

**BASELINE STRATEGY (Conservative):**
- Starting pace: {baseline_lap_time:.3f}s per lap
- Average degradation: {baseline_prediction['avg_degradation_per_lap']:.4f}s/lap
- Total time loss: {baseline_prediction['total_time_loss']:.2f}s over {stint_length} laps
- Predicted tire cliff: {'Lap ' + str(baseline_cliff) if baseline_cliff else 'No cliff detected'}

**KEY INSIGHTS:**
1. Verstappen's aggressive style gains {verstappen_lap_time - baseline_lap_time:.1f}s per lap initially
2. However, tire degradation is {(verstappen_prediction['avg_degradation_per_lap'] / baseline_prediction['avg_degradation_per_lap'] - 1) * 100:.1f}% higher
3. Over {stint_length} laps, aggressive strategy {'gains' if verstappen_prediction['total_time_loss'] < baseline_prediction['total_time_loss'] else 'loses'} {abs(verstappen_prediction['total_time_loss'] - baseline_prediction['total_time_loss']):.2f}s total

**RECOMMENDATION:**
{'Aggressive strategy is faster overall - push the tires!' if verstappen_prediction['total_time_loss'] < baseline_prediction['total_time_loss'] else 'Conservative strategy is better for tire preservation - manage the pace.'}
"""

        # ==================== RESPONSE ====================
        return {
            "status": "success",
            "verstappen_strategy": {
                "avg_lap_time": verstappen_lap_time,
                "avg_degradation_per_lap": verstappen_prediction['avg_degradation_per_lap'],
                "total_time_loss": verstappen_prediction['total_time_loss'],
                "predicted_cliff_lap": verstappen_cliff,
                "stint_length": stint_length
            },
            "baseline_strategy": {
                "avg_lap_time": baseline_lap_time,
                "avg_degradation_per_lap": baseline_prediction['avg_degradation_per_lap'],
                "total_time_loss": baseline_prediction['total_time_loss'],
                "predicted_cliff_lap": baseline_cliff,
                "stint_length": stint_length
            },
            "comparison": {
                "lap_by_lap": lap_by_lap,
                "total_delta": verstappen_prediction['total_time_loss'] - baseline_prediction['total_time_loss']
            },
            "reasoning": reasoning
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verstappen simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def get_verstappen_comparison(session_key: str) -> Dict[str, Any]:
    """
    Get pre-computed Verstappen vs baseline comparison for a session
    (Future endpoint for cached results)
    """
    return {
        "status": "not_implemented",
        "message": "This endpoint will provide cached comparison results in the future"
    }