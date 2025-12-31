"""
Shared state schema for multi-agent system
"""
from typing import List, Dict, Optional, Annotated
from typing_extensions import TypedDict
import operator

class RaceContext(TypedDict):
    """Current race situation"""
    driver: str
    current_lap: int
    total_laps: int
    position: int
    gap_ahead: Optional[float]
    gap_behind: Optional[float]
    track_name: str
    weather: str

class TireState(TypedDict):
    """Current tire status"""
    compound: str
    age: int
    degradation_rate: float
    predicted_cliff_lap: Optional[int]
    temperature: Dict[str, float]

class FuelState(TypedDict):
    """Fuel status"""
    current_fuel_kg: float
    fuel_per_lap_kg: float
    laps_remaining_on_fuel: int

class CompetitorState(TypedDict):
    """Individual competitor info"""
    driver: str
    position: int
    tire_compound: str
    tire_age: int
    gap_to_us: float
    last_pit_lap: Optional[int]
    predicted_pit_lap: Optional[int]

class RaceControlState(TypedDict):
    """Race control information"""
    safety_car: bool
    virtual_safety_car: bool
    red_flag: bool
    yellow_flags: List[str]
    drs_enabled: bool
    incidents: List[str]

class AgentState(TypedDict):
    """Main state that flows through all agents"""
    # Input context
    race_context: RaceContext
    tire_state: TireState
    fuel_state: FuelState
    competitors: List[CompetitorState]
    race_control: RaceControlState
    
    # Agent outputs
    messages: List[str]  # 🔥 REMOVED operator.add to prevent duplicates
    strategy_recommendation: Optional[Dict]
    telemetry_analysis: Optional[Dict]
    competitor_analysis: Optional[Dict]
    race_control_analysis: Optional[Dict]
    
    # Final decision
    final_decision: Optional[Dict]
    confidence: Optional[float]
    reasoning: Optional[str]
