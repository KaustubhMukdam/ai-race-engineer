"""
Strategy API routes
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from backend.schemas.strategy_schema import (
    PitStrategyRequest,
    PitStrategyResponse,
    DegradationExplanationRequest,
    DegradationExplanationResponse,
    UndercutAnalysisRequest,
    UndercutAnalysisResponse
)
from agents.strategy_agent import StrategyAgent
from config.app_config import settings
from utils.logger import setup_logger
from ml.models.lstm_interface import LSTMInferenceEngine

logger = setup_logger(__name__)

# Initialize router
router = APIRouter(prefix="/strategy", tags=["Strategy"])

# Global agent instance (loaded once at startup)
strategy_agent: StrategyAgent = None


def get_agent() -> StrategyAgent:
    """Get or initialize strategy agent"""
    global strategy_agent
    if strategy_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Strategy agent not initialized. Call /health to check status."
        )
    return strategy_agent


def initialize_agent():
    """Initialize agent with race data"""
    global strategy_agent
    try:
        strategy_agent = StrategyAgent()
        
        # Load latest processed race data
        degradation_file = settings.processed_data_dir / "2024_Abu_Dhabi_Grand_Prix_Race_processed" / "tire_degradation_analysis.csv"
        pit_windows_file = settings.processed_data_dir / "2024_Abu_Dhabi_Grand_Prix_Race_processed" / "optimal_pit_windows.json"

        strategy_agent.load_race_data(degradation_file, pit_windows_file)
        logger.info("Strategy agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize strategy agent: {e}")
        raise


@router.post("/recommend-pit", response_model=PitStrategyResponse)
async def recommend_pit_strategy(request: PitStrategyRequest) -> PitStrategyResponse:
    """
    Generate pit strategy recommendation
    
    - **driver**: 3-letter driver code (e.g., VER, HAM)
    - **current_lap**: Current lap number
    - **total_laps**: Total race laps
    - **current_compound**: Current tire compound (SOFT, MEDIUM, HARD)
    - **tire_age**: Age of current tires in laps
    - **track_temp**: Track temperature in Celsius
    - **air_temp**: Air temperature in Celsius
    - **race_context**: Optional additional context
    """
    try:
        agent = get_agent()
        
        result = agent.recommend_pit_strategy(
            driver=request.driver,
            current_lap=request.current_lap,
            total_laps=request.total_laps,
            current_compound=request.current_compound,
            tire_age=request.tire_age,
            track_temp=request.track_temp,
            air_temp=request.air_temp,
            race_context=request.race_context or "Normal race conditions"
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result.get('message', 'Unknown error'))
        
        return PitStrategyResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recommend_pit_strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-degradation", response_model=DegradationExplanationResponse)
async def explain_degradation(request: DegradationExplanationRequest) -> DegradationExplanationResponse:
    """
    Explain tire degradation pattern for a driver's stint
    
    - **driver**: 3-letter driver code
    - **stint**: Stint number (1, 2, 3, etc.)
    """
    try:
        agent = get_agent()
        explanation = agent.explain_tire_degradation(request.driver, request.stint)
        return DegradationExplanationResponse(explanation=explanation)
        
    except Exception as e:
        logger.error(f"Error in explain_degradation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-undercut", response_model=UndercutAnalysisResponse)
async def analyze_undercut(request: UndercutAnalysisRequest) -> UndercutAnalysisResponse:
    """
    Analyze undercut opportunity against driver ahead
    
    - **driver**: Your driver code
    - **driver_ahead**: Driver ahead code
    - **gap_seconds**: Gap in seconds
    - **your_tire_age**: Your tire age in laps
    - **their_tire_age**: Their tire age in laps
    """
    try:
        agent = get_agent()
        analysis = agent.analyze_undercut(
            driver=request.driver,
            driver_ahead=request.driver_ahead,
            gap_seconds=request.gap_seconds,
            your_tire_age=request.your_tire_age,
            their_tire_age=request.their_tire_age
        )
        return UndercutAnalysisResponse(analysis=analysis)
        
    except Exception as e:
        logger.error(f"Error in analyze_undercut: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/lstm-status")
async def lstm_status():
    engine = LSTMInferenceEngine()
    return {"available": engine.is_available()}
