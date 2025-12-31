"""
Telemetry Agent - Monitors tire degradation, fuel, and car performance
"""
import logging
from typing import Dict
import json
from ml.orchestration.agent_state import AgentState

logger = logging.getLogger(__name__)

class TelemetryAgent:
    """Analyzes real-time telemetry data"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        
    def analyze(self, state: AgentState) -> Dict:
        """Analyze telemetry and provide insights"""
        
        tire = state['tire_state']
        fuel = state['fuel_state']
        race = state['race_context']
        
        # Simple rule-based analysis (no LLM needed for MVP)
        tire_condition_score = max(1, 10 - (tire['age'] / 3))  # Degrades with age
        
        # Determine cliff risk
        if tire['compound'] == 'SOFT' and tire['age'] > 15:
            cliff_risk = "High"
        elif tire['compound'] == 'MEDIUM' and tire['age'] > 20:
            cliff_risk = "High"
        elif tire['compound'] == 'HARD' and tire['age'] > 30:
            cliff_risk = "Medium"
        else:
            cliff_risk = "Low"
        
        # Fuel check
        fuel_critical = fuel['laps_remaining_on_fuel'] < 5
        
        # Temperature issues
        max_temp = max(tire['temperature'].values())
        temp_issues = "Overheating" if max_temp > 110 else "None"
        
        # Recommendation
        if cliff_risk == "High" or tire_condition_score < 4:
            recommendation = "Pit now"
        elif cliff_risk == "Medium" or tire_condition_score < 6:
            recommendation = "Monitor closely"
        else:
            recommendation = "Can extend"
        
        key_insight = f"Tire condition {tire_condition_score:.1f}/10, cliff risk {cliff_risk}"
        
        analysis = {
            'tire_condition_score': round(tire_condition_score, 1),
            'tire_cliff_risk': cliff_risk,
            'fuel_critical': fuel_critical,
            'temp_issues': temp_issues,
            'key_insight': key_insight,
            'recommendation': recommendation
        }
        
        logger.info(f"Telemetry analysis: {key_insight}")
        
        return {
            'status': 'success',
            'analysis': analysis,
            'agent': 'telemetry'
        }

    def __call__(self, state: AgentState) -> AgentState:
        """LangGraph node function"""
        analysis = self.analyze(state)
        
        state['telemetry_analysis'] = analysis
        
        # Only add message if not already present
        msg = f"[TELEMETRY] {analysis['analysis']['key_insight']}"
        if msg not in state['messages']:
            state['messages'].append(msg)
        
        return state

