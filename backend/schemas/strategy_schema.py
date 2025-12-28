"""
Pydantic schemas for strategy API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal


class PitStrategyRequest(BaseModel):
    """Request model for pit strategy recommendation"""
    driver: str = Field(..., description="3-letter driver code", example="VER")
    current_lap: int = Field(..., gt=0, description="Current lap number")
    total_laps: int = Field(..., gt=0, description="Total race laps")
    current_compound: Literal["SOFT", "MEDIUM", "HARD"] = Field(..., description="Current tire compound")
    tire_age: int = Field(..., ge=0, description="Age of current tires in laps")
    track_temp: float = Field(..., description="Track temperature in Celsius")
    air_temp: float = Field(..., description="Air temperature in Celsius")
    race_context: Optional[str] = Field(None, description="Additional race context")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "driver": "VER",
                "current_lap": 20,
                "total_laps": 58,
                "current_compound": "MEDIUM",
                "tire_age": 20,
                "track_temp": 31.5,
                "air_temp": 26.7,
                "race_context": "P1, gap to P2 is 3.2 seconds"
            }
        }
    )


class PitStrategyResponse(BaseModel):
    """Response model for pit strategy recommendation"""
    status: str
    driver: str
    current_lap: int
    recommendation: str
    llm_model: str  # Changed from model_used to avoid Pydantic conflict
    
    model_config = ConfigDict(protected_namespaces=())


class DegradationExplanationRequest(BaseModel):
    """Request model for tire degradation explanation"""
    driver: str = Field(..., description="3-letter driver code")
    stint: int = Field(..., gt=0, description="Stint number")


class DegradationExplanationResponse(BaseModel):
    """Response model for degradation explanation"""
    explanation: str


class UndercutAnalysisRequest(BaseModel):
    """Request model for undercut analysis"""
    driver: str = Field(..., description="Your driver code")
    driver_ahead: str = Field(..., description="Driver ahead code")
    gap_seconds: float = Field(..., gt=0, description="Gap to driver ahead in seconds")
    your_tire_age: int = Field(..., ge=0, description="Your tire age in laps")
    their_tire_age: int = Field(..., ge=0, description="Their tire age in laps")


class UndercutAnalysisResponse(BaseModel):
    """Response model for undercut analysis"""
    analysis: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    agent_loaded: bool
    llm_model: str  # Changed from model
