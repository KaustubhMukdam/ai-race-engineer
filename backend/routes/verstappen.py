"""
Verstappen Style Simulator API Routes
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from agents.verstappen_simulator import VerstappenStyleSimulator
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/verstappen", tags=["Verstappen Simulator"])

# Global simulator instance
verstappen_simulator: Optional[VerstappenStyleSimulator] = None


def initialize_simulator():
    """Initialize verstappen simulator"""
    global verstappen_simulator
    try:
        verstappen_simulator = VerstappenStyleSimulator()
        logger.info("Verstappen simulator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize verstappen simulator: {e}")
        raise


def get_simulator() -> VerstappenStyleSimulator:
    """Get simulator instance"""
    global verstappen_simulator
    if verstappen_simulator is None:
        raise HTTPException(
            status_code=503,
            detail="Verstappen simulator not initialized"
        )
    return verstappen_simulator


class VerstappenCompareRequest(BaseModel):
    session_key: str
    aggressive_driver: str = "VER"
    baseline_driver: Optional[str] = None


@router.post("/compare")
async def compare_styles(request: VerstappenCompareRequest):
    """
    Compare aggressive vs conservative driving styles
    
    - **session_key**: Session identifier (e.g., "2024_Abu_Dhabi_Grand_Prix_Race")
    - **aggressive_driver**: Driver code for aggressive style (default: VER)
    - **baseline_driver**: Driver code for baseline (None = auto-detect most conservative)
    """
    try:
        simulator = get_simulator()
        
        # Load session data
        processed_path = settings.processed_data_dir / f"{request.session_key}_processed"
        deg_file = processed_path / "tire_degradation_analysis.csv"
        laps_file = processed_path / "processed_laps.csv"
        
        if not deg_file.exists() or not laps_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session data not found: {request.session_key}"
            )
        
        simulator.load_race_data(deg_file, laps_file)
        
        # Run comparison
        comparison = simulator.compare_verstappen_vs_baseline(
            verstappen_driver=request.aggressive_driver,
            baseline_driver=request.baseline_driver
        )
        
        if comparison['status'] == 'error':
            raise HTTPException(status_code=500, detail=comparison['message'])
        
        # Generate AI analysis
        analysis = simulator.generate_llm_analysis(comparison)
        
        return {
            **comparison,
            'ai_analysis': analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verstappen comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{driver}")
async def get_driver_metrics(
    driver: str,
    session_key: str = Query(..., description="Session identifier")
):
    """Get driving style metrics for a specific driver"""
    try:
        simulator = get_simulator()
        
        # Load session data
        processed_path = settings.processed_data_dir / f"{session_key}_processed"
        deg_file = processed_path / "tire_degradation_analysis.csv"
        laps_file = processed_path / "processed_laps.csv"
        
        simulator.load_race_data(deg_file, laps_file)
        
        metrics = simulator.calculate_driving_style_metrics(driver)
        
        if metrics is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for driver {driver}"
            )
        
        style = simulator.classify_driving_style(metrics)
        
        return {
            **metrics,
            'style': style
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
