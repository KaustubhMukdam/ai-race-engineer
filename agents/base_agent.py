"""
Base Agent Class
Foundation for all AI agents in the system
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from groq import Groq

from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all AI agents"""
    
    def __init__(
        self,
        name: str,
        model: str = settings.groq_primary_model,
        temperature: float = settings.groq_temperature
    ):
        """
        Initialize agent with Groq client
        
        Args:
            name: Agent identifier
            model: Groq model to use
            temperature: Sampling temperature
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=settings.groq_api_key)
        logger.info(f"Initialized {name} with model {model}")
    
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = settings.groq_max_tokens,
        temperature: Optional[float] = None
    ) -> str:
        """
        Call Groq LLM with error handling and retry logic
        
        Args:
            system_prompt: System role definition
            user_prompt: User query/context
            max_tokens: Maximum response tokens
            temperature: Override default temperature
        
        Returns:
            LLM response text
        """
        try:
            temp = temperature if temperature is not None else self.temperature
            
            logger.info(f"{self.name} calling Groq API with {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(
                    f"Tokens used - Prompt: {response.usage.prompt_tokens}, "
                    f"Completion: {response.usage.completion_tokens}, "
                    f"Total: {response.usage.total_tokens}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and generate agent-specific output
        Must be implemented by subclasses
        """
        pass
