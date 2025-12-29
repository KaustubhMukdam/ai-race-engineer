"""
FastAPI Application
Main entry point for AI Race Engineer API
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routes import strategy
from backend.schemas.strategy_schema import HealthCheckResponse
from config.app_config import settings
from utils.logger import setup_logger
from backend.routes import sessions

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events - startup and shutdown"""
    # Startup
    logger.info("Starting AI Race Engineer API...")
    try:
        strategy.initialize_agent()
        logger.info("Strategy agent loaded successfully")

        sessions.initialize_session_manager()
        logger.info("Session manager loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load strategy agent: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Race Engineer API...")


# Initialize FastAPI app
app = FastAPI(
    title="AI Race Engineer API",
    description="LLM-powered F1 race strategy and tire management system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(strategy.router)
app.include_router(sessions.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Race Engineer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    agent_loaded = strategy.strategy_agent is not None
    return HealthCheckResponse(
        status="healthy" if agent_loaded else "degraded",
        agent_loaded=agent_loaded,
        llm_model=settings.groq_primary_model  # Changed from model
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
