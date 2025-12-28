"""
Application Configuration Management
Handles all environment variables and settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    groq_api_key: str
    
    # Data Configuration
    fastf1_cache_dir: Path = Path("./data/raw/fastf1_cache")
    processed_data_dir: Path = Path("./data/processed")
    datasets_dir: Path = Path("./data/datasets")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Groq Model Configuration
    groq_primary_model: str = "llama-3.3-70b-versatile"
    groq_reasoning_model: str = "deepseek-r1-distill-llama-70b"
    groq_temperature: float = 0.7
    groq_max_tokens: int = 4096
    
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


# Global settings instance
settings = Settings()
settings.ensure_directories()
