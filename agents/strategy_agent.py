"""
Strategy Agent
Analyzes tire degradation and recommends pit strategies
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, Any, Optional
import json

from agents.base_agent import BaseAgent
from agents.prompts.strategy_prompts import (
    SYSTEM_PROMPT,
    STRATEGY_ANALYSIS_PROMPT,
    TIRE_DEGRADATION_EXPLANATION_PROMPT,
    UNDERCUT_ANALYSIS_PROMPT
)
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StrategyAgent(BaseAgent):
    """AI agent for race strategy and tire management"""
    
    def __init__(self):
        super().__init__(
            name="StrategyAgent",
            model=settings.groq_primary_model,
            temperature=0.7
        )
        self.degradation_df = None
        self.pit_windows = None
    
    def load_race_data(self, degradation_csv: Path, pit_windows_json: Path):
        """Load preprocessed race data"""
        try:
            self.degradation_df = pd.read_csv(degradation_csv)
            with open(pit_windows_json, 'r') as f:
                self.pit_windows = json.load(f)
            logger.info(f"Loaded race data: {len(self.degradation_df)} stints")
        except Exception as e:
            logger.error(f"Error loading race data: {e}")
            raise
    
    def get_driver_degradation(self, driver: str) -> pd.DataFrame:
        """Get degradation data for specific driver"""
        if self.degradation_df is None:
            raise ValueError("Race data not loaded. Call load_race_data() first.")
        return self.degradation_df[self.degradation_df['Driver'] == driver]
    
    def recommend_pit_strategy(
        self,
        driver: str,
        current_lap: int,
        total_laps: int,
        current_compound: str,
        tire_age: int,
        track_temp: float,
        air_temp: float,
        race_context: str = "Normal race conditions"
    ) -> Dict[str, Any]:
        """
        Generate pit strategy recommendation
        
        Args:
            driver: 3-letter driver code
            current_lap: Current lap number
            total_laps: Total race laps
            current_compound: Current tire compound
            tire_age: Age of current tires in laps
            track_temp: Current track temperature
            air_temp: Current air temperature
            race_context: Additional context (safety car, traffic, etc.)
        
        Returns:
            Dictionary with recommendation and reasoning
        """
        try:
            logger.info(f"Generating strategy for {driver} on lap {current_lap}")
            
            # Get driver's degradation data
            driver_deg = self.get_driver_degradation(driver)
            
            if driver_deg.empty:
                return {
                    "status": "error",
                    "message": f"No degradation data found for {driver}"
                }
            
            # Format degradation data for prompt
            deg_summary = driver_deg.to_string(index=False)
            
            # Format pit windows
            pit_windows_str = "\n".join([
                f"- {compound}: Laps {window[0]}-{window[1]}"
                for compound, window in self.pit_windows.items()
            ])
            
            # Build prompt
            user_prompt = STRATEGY_ANALYSIS_PROMPT.format(
                driver=driver,
                current_lap=current_lap,
                total_laps=total_laps,
                current_compound=current_compound,
                tire_age=tire_age,
                degradation_data=deg_summary,
                pit_windows=pit_windows_str,
                track_temp=track_temp,
                air_temp=air_temp,
                race_context=race_context
            )
            
            # Get LLM response
            response = self.call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            return {
                "status": "success",
                "driver": driver,
                "current_lap": current_lap,
                "recommendation": response,
                "llm_model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy recommendation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def explain_tire_degradation(self, driver: str, stint: int) -> str:
        """Generate natural language explanation of tire degradation"""
        try:
            driver_deg = self.get_driver_degradation(driver)
            stint_data = driver_deg[driver_deg['Stint'] == stint]
            
            if stint_data.empty:
                return f"No data found for {driver} stint {stint}"
            
            user_prompt = TIRE_DEGRADATION_EXPLANATION_PROMPT.format(
                driver=driver,
                stint_data=stint_data.to_string(index=False)
            )
            
            response = self.call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.5
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error explaining degradation: {e}")
            return f"Error: {str(e)}"
    
    def analyze_undercut(
        self,
        driver: str,
        driver_ahead: str,
        gap_seconds: float,
        your_tire_age: int,
        their_tire_age: int
    ) -> str:
        """Analyze undercut opportunity"""
        try:
            your_deg = self.get_driver_degradation(driver)
            their_deg = self.get_driver_degradation(driver_ahead)
            
            # Get latest stint degradation rate
            your_rate = your_deg.iloc[-1]['DegradationPerLap_Seconds'] if not your_deg.empty else 0
            their_rate = their_deg.iloc[-1]['DegradationPerLap_Seconds'] if not their_deg.empty else 0
            
            user_prompt = UNDERCUT_ANALYSIS_PROMPT.format(
                driver=driver,
                driver_ahead=driver_ahead,
                gap_seconds=gap_seconds,
                your_tire_age=your_tire_age,
                their_tire_age=their_tire_age,
                your_deg=your_rate,
                their_deg=their_rate
            )
            
            response = self.call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing undercut: {e}")
            return f"Error: {str(e)}"
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and generate strategy output"""
        action = input_data.get('action', 'recommend_pit_strategy')
        
        if action == 'recommend_pit_strategy':
            return self.recommend_pit_strategy(**input_data.get('params', {}))
        elif action == 'explain_degradation':
            return {"explanation": self.explain_tire_degradation(**input_data.get('params', {}))}
        elif action == 'analyze_undercut':
            return {"analysis": self.analyze_undercut(**input_data.get('params', {}))}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


def main():
    """Test the Strategy Agent"""
    agent = StrategyAgent()
    
    # Load Abu Dhabi GP data
    degradation_file = settings.processed_data_dir / "2024_Abu_Dhabi_GP_processed" / "tire_degradation_analysis.csv"
    pit_windows_file = settings.processed_data_dir / "2024_Abu_Dhabi_GP_processed" / "optimal_pit_windows.json"
    
    agent.load_race_data(degradation_file, pit_windows_file)
    
    print("\n" + "="*80)
    print("AI RACE ENGINEER - STRATEGY AGENT TEST")
    print("="*80)
    
    # Test 1: Pit strategy for Verstappen on lap 20
    print("\nTEST 1: Verstappen Strategy - Lap 20")
    print("-"*80)
    result = agent.recommend_pit_strategy(
        driver="VER",
        current_lap=20,
        total_laps=58,
        current_compound="MEDIUM",
        tire_age=20,
        track_temp=31.5,
        air_temp=26.7,
        race_context="P1, gap to P2 is 3.2 seconds"
    )
    
    if result['status'] == 'success':
        print(result['recommendation'])
        print(f"\nModel used: {result['llm_model']}")
    else:
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return
    
    # Test 2: Explain Verstappen's MEDIUM tire degradation
    print("\n\nTEST 2: Explain Verstappen's MEDIUM Tire Degradation")
    print("-"*80)
    explanation = agent.explain_tire_degradation(driver="VER", stint=1)
    print(explanation)
    
    # Test 3: Undercut analysis
    print("\n\nTEST 3: Undercut Analysis - Verstappen vs Norris")
    print("-"*80)
    undercut = agent.analyze_undercut(
        driver="VER",
        driver_ahead="NOR",
        gap_seconds=2.8,
        your_tire_age=18,
        their_tire_age=20
    )
    print(undercut)
    
    print("\n" + "="*80)
    print("Strategy Agent test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
