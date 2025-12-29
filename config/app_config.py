"""
Application Configuration Management
Handles all environment variables and settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    groq_api_key: str
    
    # Data Configuration
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    fastf1_cache_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw" / "fastf1_cache")
    processed_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    datasets_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "datasets")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Groq Model Configuration (Updated Dec 2025)
    # PRIMARY MODELS (Production-ready)
    groq_primary_model: str = "llama-3.3-70b-versatile"        # Main reasoning & strategy
    groq_fast_model: str = "llama-3.3-70b-specdec"             # Fast real-time responses
    groq_vision_model: str = "llama-3.2-90b-vision-preview"    # If we add telemetry charts
    
    # ALTERNATIVE MODELS
    groq_reasoning_model: str = "llama-3.3-70b-versatile"      # Complex multi-step reasoning
    groq_commentary_model: str = "llama-3.1-8b-instant"        # Fast commentary/chat
    
    # MODEL PARAMETERS
    groq_temperature: float = 0.7
    groq_max_tokens: int = 8192  # Updated to match model limits
    
    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.fastf1_cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create ML directories
        ml_dir = self.base_dir / "ml" / "saved_models"
        ml_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
