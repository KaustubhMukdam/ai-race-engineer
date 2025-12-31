"""
Race Control Agent - Monitors safety cars, flags, and race incidents
"""
import logging
from typing import Dict
from ml.orchestration.agent_state import AgentState

logger = logging.getLogger(__name__)

class RaceControlAgent:
    """Analyzes race control situations"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        
    def analyze(self, state: AgentState) -> Dict:
        """Analyze race control situation"""
        
        race_control = state['race_control']
        race = state['race_context']
        tire = state['tire_state']
        
        # Determine situation severity
        if race_control['red_flag']:
            severity = "Critical"
            pit_window_open = True
            opportunity = "Free tire change during red flag"
            risks = "None - mandatory stop"
            recommendation = "Pit now"
        elif race_control['safety_car']:
            severity = "Significant"
            pit_window_open = True
            opportunity = "Cheap pit stop under SC"
            risks = "Pit lane may be crowded"
            recommendation = "Pit now"
        elif race_control['virtual_safety_car']:
            severity = "Significant"
            pit_window_open = True
            opportunity = "Reduced pit stop time loss"
            risks = "VSC may end quickly"
            recommendation = "Pit now"
        elif race_control['yellow_flags']:
            severity = "Minor"
            pit_window_open = False
            opportunity = "None"
            risks = "Potential SC deployment"
            recommendation = "Wait for SC"
        else:
            severity = "Normal"
            pit_window_open = False
            opportunity = "None"
            risks = "None"
            recommendation = "Pit on schedule"
        
        key_insight = f"{severity} race control situation"
        if race_control['safety_car'] or race_control['virtual_safety_car']:
            key_insight += " - PIT WINDOW OPEN"
        
        analysis = {
            'situation_severity': severity,
            'pit_window_open': pit_window_open,
            'strategic_opportunity': opportunity,
            'risks': risks,
            'key_insight': key_insight,
            'recommendation': recommendation
        }
        
        logger.info(f"Race control analysis: {key_insight}")
        
        return {
            'status': 'success',
            'analysis': analysis,
            'agent': 'race_control'
        }

    def __call__(self, state: AgentState) -> AgentState:
        """LangGraph node function"""
        analysis = self.analyze(state)
        
        state['race_control_analysis'] = analysis
        state['messages'].append(f"[RACE CONTROL] {analysis['analysis']['key_insight']}")
        
        return state
