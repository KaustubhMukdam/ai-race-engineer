"""
Competitor Agent - Tracks rival strategies and predicts their moves
"""
import logging
from typing import Dict, List
from ml.orchestration.agent_state import AgentState, CompetitorState

logger = logging.getLogger(__name__)

class CompetitorAgent:
    """Analyzes competitor strategies"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        
    def analyze(self, state: AgentState) -> Dict:
        """Analyze competitors and predict their strategies"""
        
        race = state['race_context']
        competitors = state['competitors']
        our_tire = state['tire_state']
        
        # Find main threat (closest competitor)
        relevant_competitors = [
            c for c in competitors 
            if abs(c['gap_to_us']) < 20.0 and c['position'] != race['position']
        ]
        
        if not relevant_competitors:
            return {
                'status': 'success',
                'analysis': {
                    'main_threat': 'None',
                    'undercut_opportunity': {'feasibility': 'Low'},
                    'undercut_risk': {'risk_level': 'Low'},
                    'key_insight': 'No nearby competitors',
                    'recommendation': 'Pit on schedule'
                },
                'agent': 'competitor'
            }
        
        # Find closest ahead and behind
        ahead = [c for c in relevant_competitors if c['gap_to_us'] > 0]
        behind = [c for c in relevant_competitors if c['gap_to_us'] < 0]
        
        main_threat = min(ahead, key=lambda x: x['gap_to_us'])['driver'] if ahead else \
                      max(behind, key=lambda x: x['gap_to_us'])['driver']
        
        # Undercut opportunity (on car ahead with old tires)
        undercut_target = None
        for comp in ahead:
            if comp['tire_age'] > our_tire['age'] + 5:
                undercut_target = comp
                break
        
        if undercut_target:
            undercut_opp = {
                'driver': undercut_target['driver'],
                'window': f"Lap {race['current_lap']}-{race['current_lap']+3}",
                'feasibility': 'High' if undercut_target['gap_to_us'] < 15 else 'Medium'
            }
            recommendation = "Pit early"
        else:
            undercut_opp = {'feasibility': 'Low'}
            recommendation = "Pit on schedule"
        
        # Undercut risk (from car behind with fresh tires)
        undercut_risk_driver = None
        for comp in behind:
            if comp['tire_age'] < our_tire['age'] - 5:
                undercut_risk_driver = comp
                break
        
        if undercut_risk_driver:
            undercut_risk = {
                'driver': undercut_risk_driver['driver'],
                'risk_level': 'High' if abs(undercut_risk_driver['gap_to_us']) < 10 else 'Medium',
                'expected_pit_lap': race['current_lap'] + 2
            }
        else:
            undercut_risk = {'risk_level': 'Low'}
        
        key_insight = f"Main threat: {main_threat}, undercut feasibility: {undercut_opp['feasibility']}"
        
        analysis = {
            'main_threat': main_threat,
            'undercut_opportunity': undercut_opp,
            'undercut_risk': undercut_risk,
            'key_insight': key_insight,
            'recommendation': recommendation
        }
        
        logger.info(f"Competitor analysis: {key_insight}")
        
        return {
            'status': 'success',
            'analysis': analysis,
            'agent': 'competitor'
        }

    def __call__(self, state: AgentState) -> AgentState:
        """LangGraph node function"""
        analysis = self.analyze(state)
        
        state['competitor_analysis'] = analysis
        state['messages'].append(f"[COMPETITOR] {analysis['analysis']['key_insight']}")
        
        return state
