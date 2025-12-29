"""
Session Management API Routes
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd

from data.scripts.session_manager import SessionManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize router
router = APIRouter(prefix="/sessions", tags=["Sessions"])

# Global session manager
session_manager: SessionManager = None


def initialize_session_manager():
    """Initialize session manager"""
    global session_manager
    try:
        session_manager = SessionManager()
        logger.info("Session manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize session manager: {e}")
        raise


def get_manager() -> SessionManager:
    """Get session manager instance"""
    global session_manager
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized"
        )
    return session_manager


@router.get("/seasons")
async def get_available_seasons() -> List[int]:
    """Get list of available F1 seasons"""
    try:
        manager = get_manager()
        seasons = manager.get_available_seasons()
        return seasons
    except Exception as e:
        logger.error(f"Error getting seasons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule/{year}")
async def get_season_schedule(year: int):
    """
    Get event schedule for a specific season
    
    - **year**: Season year (2018-2025)
    """
    try:
        manager = get_manager()
        schedule = manager.get_season_schedule(year)
        
        # Convert to dict for JSON response
        result = schedule[['RoundNumber', 'EventName', 'EventDate', 'EventFormat', 
                          'Country', 'Location']].to_dict(orient='records')
        
        return {
            "year": year,
            "total_events": len(result),
            "events": result
        }
    except Exception as e:
        logger.error(f"Error getting schedule for {year}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_session(
    year: int = Query(..., description="Season year"),
    event: str = Query(..., description="Event name (e.g., 'Monaco', 'Abu Dhabi')"),
    session: str = Query("Race", description="Session type (Race, Qualifying, etc.)"),
    force_reload: bool = Query(False, description="Force reload even if cached")
):
    """
    Load and process a complete F1 session
    
    - **year**: Season year
    - **event**: Event name
    - **session**: Session type (default: Race)
    - **force_reload**: Force reload even if cached
    """
    try:
        manager = get_manager()
        
        logger.info(f"Loading session: {year} {event} {session}")
        
        session_data = manager.load_and_process_session(
            year=year,
            event=event,
            session=session,
            force_reload=force_reload
        )
        
        metadata = manager.get_session_metadata(year, event, session)
        
        return {
            "status": "success",
            "session_key": session_data['session_key'],
            "cached": session_data['cached'],
            "metadata": metadata,
            "message": "Session loaded and ready for analysis"
        }
        
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cached")
async def list_cached_sessions():
    """Get list of all cached sessions"""
    try:
        manager = get_manager()
        cached = manager.list_cached_sessions()
        
        if cached.empty:
            return {
                "total_cached": 0,
                "sessions": []
            }
        
        return {
            "total_cached": len(cached),
            "sessions": cached.to_dict(orient='records')
        }
    except Exception as e:
        logger.error(f"Error listing cached sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cached")
async def delete_cached_session(
    year: int = Query(..., description="Season year"),
    event: str = Query(..., description="Event name"),
    session: str = Query("Race", description="Session type")
):
    """Delete a cached session"""
    try:
        manager = get_manager()
        success = manager.delete_cached_session(year, event, session)
        
        if success:
            return {
                "status": "success",
                "message": f"Session {year} {event} {session} deleted"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found in cache")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting cached session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
