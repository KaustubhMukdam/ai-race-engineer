"""
LangGraph workflow for multi-agent race strategy system
"""
import logging
from typing import Dict
from langgraph.graph import StateGraph, END
from ml.orchestration.agent_state import AgentState
from ml.orchestration.agents.telemetry_agent import TelemetryAgent
from ml.orchestration.agents.competitor_agent import CompetitorAgent
from ml.orchestration.agents.race_control_agent import RaceControlAgent
from ml.orchestration.agents.orchestrator_agent import OrchestratorAgent
from agents.strategy_agent import StrategyAgent

logger = logging.getLogger(__name__)

class RaceStrategyGraph:
    """Multi-agent race strategy system using LangGraph"""
    
    def __init__(self):
        self.strategy_agent = StrategyAgent()
        self.telemetry_agent = TelemetryAgent()
        self.competitor_agent = CompetitorAgent()
        self.race_control_agent = RaceControlAgent()
        self.orchestrator_agent = OrchestratorAgent()
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes with unique names (can't match any state keys)
        workflow.add_node("analyze_strategy", self._run_strategy_agent)
        workflow.add_node("analyze_telemetry", self.telemetry_agent)
        workflow.add_node("analyze_competitors", self.competitor_agent)
        workflow.add_node("analyze_race_control", self.race_control_agent)
        workflow.add_node("make_decision", self.orchestrator_agent)
        
        # Define workflow sequence
        workflow.set_entry_point("analyze_strategy")
        workflow.add_edge("analyze_strategy", "analyze_telemetry")
        workflow.add_edge("analyze_telemetry", "analyze_competitors")
        workflow.add_edge("analyze_competitors", "analyze_race_control")
        workflow.add_edge("analyze_race_control", "make_decision")
        workflow.add_edge("make_decision", END)
        
        # Compile graph
        return workflow.compile()
    
    def _run_strategy_agent(self, state: AgentState) -> AgentState:
        """Adapter for Strategy Agent to work with LangGraph"""
        race = state['race_context']
        tire = state['tire_state']
        
        # Build race context string for Strategy Agent
        race_context_str = f"""Position: P{race['position']}
    Gap ahead: {race.get('gap_ahead', 'N/A')}s
    Gap behind: {race.get('gap_behind', 'N/A')}s
    Weather: {race['weather']}"""
        
        try:
            # Build minimal race data for Strategy Agent
            race_data = {
                'laps': [
                    {
                        'lap_number': race['current_lap'],
                        'driver': race['driver'],
                        'tire_compound': tire['compound'],
                        'tire_age': tire['age'],
                        'lap_time': 90.0,  # placeholder
                        'position': race['position']
                    }
                ],
                'stints': []
            }
            
            # Call Strategy Agent's actual method
            # Check if strategy_agent has 'recommend_pit_stop' or similar method
            # For now, use a simple recommendation based on XGBoost
            if hasattr(self.strategy_agent, 'pit_classifier'):
                pit_features = {
                    'tire_age': tire['age'],
                    'tire_compound': tire['compound'],
                    'lap_number': race['current_lap'],
                    'race_progress': race['current_lap'] / race['total_laps'],
                    'position': race['position'],
                    'track_temp': 30.0,  # placeholder
                    'air_temp': 25.0,    # placeholder
                    'degradation_rate': tire['degradation_rate'],
                    'lap_time': 90.0
                }
                
                pit_result = self.strategy_agent.pit_classifier.predict(pit_features)
                
                if pit_result['should_pit']:
                    recommendation = f"Pit now (probability: {pit_result['pit_probability']:.1%})"
                else:
                    recommendation = f"Stay out (probability: {pit_result['stay_out_probability']:.1%})"
            else:
                recommendation = "Strategy analysis complete"
            
            state['strategy_recommendation'] = {
                'recommendation': recommendation,
                'confidence': 0.85,
                'agent': 'strategy'
            }
            state['messages'].append(f"[STRATEGY] {recommendation}")
            logger.info(f"Strategy recommendation: {recommendation}")
            
        except Exception as e:
            logger.error(f"Strategy agent error: {e}")
            recommendation = "Unable to generate strategy recommendation"
            state['strategy_recommendation'] = {
                'recommendation': recommendation,
                'error': str(e),
                'agent': 'strategy'
            }
            state['messages'].append(f"[STRATEGY] Error: {str(e)}")
        
        return state

    
    def run(self, state: Dict) -> Dict:
        """
        Execute the multi-agent workflow
        
        Args:
            state: Initial race state
            
        Returns:
            Final state with decision
        """
        logger.info("=" * 80)
        logger.info("🏎️  MULTI-AGENT RACE STRATEGY SYSTEM")
        logger.info("=" * 80)
        
        # Initialize messages list if not present
        if 'messages' not in state:
            state['messages'] = []
        
        # Run the graph
        final_state = self.graph.invoke(state)
        
        logger.info("=" * 80)
        logger.info("✅ ANALYSIS COMPLETE")
        logger.info("=" * 80)
        
        return final_state
