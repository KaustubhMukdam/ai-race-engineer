"""
Telemetry API Routes
Provides real-time and historical telemetry data
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import pandas as pd

from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize router
router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get("/current/{driver}")
async def get_current_telemetry(
    driver: str,
    session_key: Optional[str] = Query(None, description="Session key (e.g., 2024_Abu_Dhabi_Grand_Prix_Race)")
):
    """
    Get current telemetry snapshot for a driver

    Returns latest lap data including:
    - Position
    - Lap times
    - Tire compound and age
    - Track position gaps
    - Weather conditions
    """
    try:
        # If no session key provided, use default
        if not session_key:
            session_key = "2024_Abu_Dhabi_Grand_Prix_Race"

        # Load processed laps data
        laps_file = settings.processed_data_dir / f"{session_key}_processed" / "processed_laps.csv"

        if not laps_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session data not found for {session_key}"
            )

        laps_df = pd.read_csv(laps_file)

        # Filter for driver
        driver_laps = laps_df[laps_df['Driver'] == driver].copy()

        if driver_laps.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for driver {driver} in session {session_key}"
            )

        # Get latest lap data
        latest_lap = driver_laps.iloc[-1]

        # Calculate gaps (simplified - would need position data for accuracy)
        all_drivers = laps_df[laps_df['LapNumber'] == latest_lap['LapNumber']].sort_values('LapTime_Seconds')
        driver_position = (all_drivers['Driver'] == driver).idxmax()

        # Get driver ahead and behind
        position_idx = all_drivers.index.get_loc(driver_position)
        gap_ahead = None
        gap_behind = None

        if position_idx > 0:
            driver_ahead = all_drivers.iloc[position_idx - 1]
            gap_ahead = latest_lap['LapTime_Seconds'] - driver_ahead['LapTime_Seconds']

        if position_idx < len(all_drivers) - 1:
            driver_behind = all_drivers.iloc[position_idx + 1]
            gap_behind = driver_behind['LapTime_Seconds'] - latest_lap['LapTime_Seconds']

        # Calculate degradation rate from recent laps
        recent_laps = driver_laps.tail(5)
        if len(recent_laps) >= 2:
            degradation_rate = (recent_laps['LapTime_Seconds'].iloc[-1] - recent_laps['LapTime_Seconds'].iloc[0]) / len(recent_laps)
        else:
            degradation_rate = 0.0

        # Calculate total race laps
        total_laps = laps_df['LapNumber'].max()

        # Build response
        telemetry_data = {
            "status": "success",
            "driver": driver,
            "session_key": session_key,
            "current_lap": int(latest_lap['LapNumber']),
            "total_laps": int(total_laps),
            "position": position_idx + 1,
            "lap_time": float(latest_lap['LapTime_Seconds']),
            "tire_compound": str(latest_lap['Compound']),
            "tire_age": int(latest_lap['TyreLife']),
            "track_temp": float(latest_lap['TrackTemp']),
            "air_temp": float(latest_lap['AirTemp']),
            "degradation_rate": float(degradation_rate),
            "gap_ahead": float(gap_ahead) if gap_ahead is not None else None,
            "gap_behind": float(gap_behind) if gap_behind is not None else None,
            "weather": "Clear",  # TODO: Get from session metadata
        }

        # Add warning for wet tires
        if latest_lap['Compound'] in ['INTERMEDIATE', 'WET']:
            telemetry_data['warning'] = f"Wet conditions detected ({latest_lap['Compound']}). AI predictions optimized for dry tires."

        return telemetry_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching telemetry for {driver}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{driver}")
async def get_telemetry_history(
    driver: str,
    session_key: Optional[str] = Query(None),
    start_lap: Optional[int] = Query(None),
    end_lap: Optional[int] = Query(None)
):
    """
    Get historical telemetry data for a driver

    Returns lap-by-lap data for analysis and visualization
    """
    try:
        if not session_key:
            session_key = "2024_Abu_Dhabi_Grand_Prix_Race"

        laps_file = settings.processed_data_dir / f"{session_key}_processed" / "processed_laps.csv"

        if not laps_file.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")

        laps_df = pd.read_csv(laps_file)
        driver_laps = laps_df[laps_df['Driver'] == driver].copy()

        if driver_laps.empty:
            raise HTTPException(status_code=404, detail=f"No data for driver {driver}")

        # Filter by lap range if specified
        if start_lap:
            driver_laps = driver_laps[driver_laps['LapNumber'] >= start_lap]
        if end_lap:
            driver_laps = driver_laps[driver_laps['LapNumber'] <= end_lap]

        # Convert to list of dicts for JSON response
        laps_data = driver_laps[[
            'LapNumber', 'LapTime_Seconds', 'Compound', 'TyreLife',
            'TrackTemp', 'AirTemp', 'Stint'
        ]].to_dict(orient='records')

        return {
            "status": "success",
            "driver": driver,
            "session_key": session_key,
            "total_laps": len(laps_data),
            "laps": laps_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching history for {driver}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pit-probability/{driver}")
async def get_pit_probability(
    driver: str,
    session_key: Optional[str] = Query(None),
    current_lap: Optional[int] = Query(None)
):
    """
    Get XGBoost pit probability for a specific driver and lap

    Returns pit probability, recommended action, and confidence
    """
    try:
        from agents.strategy_agent import StrategyAgent

        if not session_key:
            session_key = "2024_Abu_Dhabi_Grand_Prix_Race"

        # Load telemetry
        laps_file = settings.processed_data_dir / f"{session_key}_processed" / "processed_laps.csv"

        if not laps_file.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")

        laps_df = pd.read_csv(laps_file)
        driver_laps = laps_df[laps_df['Driver'] == driver].copy()

        if driver_laps.empty:
            raise HTTPException(status_code=404, detail=f"No data for driver {driver}")

        # Get lap data
        if current_lap:
            lap_data = driver_laps[driver_laps['LapNumber'] == current_lap]
            if lap_data.empty:
                lap_data = driver_laps.iloc[-1:]
        else:
            lap_data = driver_laps.iloc[-1:]

        lap_data = lap_data.iloc[0]

        # Calculate degradation rate
        recent_laps = driver_laps.tail(5)
        if len(recent_laps) >= 2:
            degradation_rate = (recent_laps['LapTime_Seconds'].iloc[-1] - recent_laps['LapTime_Seconds'].iloc[0]) / len(recent_laps)
        else:
            degradation_rate = 0.0

        # Get position (simplified)
        all_drivers = laps_df[laps_df['LapNumber'] == lap_data['LapNumber']].sort_values('LapTime_Seconds')
        position = (all_drivers['Driver'] == driver).tolist().index(True) + 1

        # Initialize strategy agent to use pit classifier
        agent = StrategyAgent()

        # Prepare features for XGBoost
        pit_features = {
            'tire_age': int(lap_data['TyreLife']),
            'tire_compound': str(lap_data['Compound']),
            'lap_number': int(lap_data['LapNumber']),
            'race_progress': float(lap_data['LapNumber']) / float(laps_df['LapNumber'].max()),
            'position': position,
            'track_temp': float(lap_data['TrackTemp']),
            'air_temp': float(lap_data['AirTemp']),
            'degradation_rate': float(degradation_rate),
            'lap_time': float(lap_data['LapTime_Seconds'])
        }

        # Get XGBoost prediction
        if hasattr(agent.pit_classifier, 'model') and agent.pit_classifier.model is not None:
            pit_prediction = agent.pit_classifier.predict(pit_features)

            return {
                "status": "success",
                "driver": driver,
                "lap": int(lap_data['LapNumber']),
                "should_pit": pit_prediction['should_pit'],
                "pit_probability": pit_prediction['pit_probability'],
                "confidence": pit_prediction['confidence'],
                "threshold_used": pit_prediction.get('threshold_used', 0.5),
                "reason": pit_prediction.get('reason', 'N/A')
            }
        else:
            return {
                "status": "error",
                "message": "XGBoost model not loaded",
                "pit_probability": 0.0
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pit probability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
async def get_driver_comparison(
    driver1: str = Query(..., description="First driver code"),
    driver2: str = Query(..., description="Second driver code"),
    session_key: Optional[str] = Query(None),
    metric: str = Query("lap_time", description="Metric to compare (lap_time, tire_deg, sector_times)")
):
    """
    Compare telemetry between two drivers
    """
    try:
        if not session_key:
            session_key = "2024_Abu_Dhabi_Grand_Prix_Race"

        laps_file = settings.processed_data_dir / f"{session_key}_processed" / "processed_laps.csv"

        if not laps_file.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")

        laps_df = pd.read_csv(laps_file)

        driver1_laps = laps_df[laps_df['Driver'] == driver1].copy()
        driver2_laps = laps_df[laps_df['Driver'] == driver2].copy()

        if driver1_laps.empty or driver2_laps.empty:
            raise HTTPException(status_code=404, detail="One or both drivers not found")

        # Calculate comparison metrics
        comparison = {
            "status": "success",
            "driver1": {
                "code": driver1,
                "avg_lap_time": float(driver1_laps['LapTime_Seconds'].mean()),
                "best_lap_time": float(driver1_laps['LapTime_Seconds'].min()),
                "total_laps": len(driver1_laps)
            },
            "driver2": {
                "code": driver2,
                "avg_lap_time": float(driver2_laps['LapTime_Seconds'].mean()),
                "best_lap_time": float(driver2_laps['LapTime_Seconds'].min()),
                "total_laps": len(driver2_laps)
            },
            "delta": {
                "avg_lap_time_diff": float(driver1_laps['LapTime_Seconds'].mean() - driver2_laps['LapTime_Seconds'].mean()),
                "best_lap_diff": float(driver1_laps['LapTime_Seconds'].min() - driver2_laps['LapTime_Seconds'].min())
            }
        }

        return comparison

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in driver comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))