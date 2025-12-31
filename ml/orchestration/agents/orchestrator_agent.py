"""
Orchestrator Agent - Makes final strategic decisions using LLM reasoning
"""
import logging
from typing import Dict
import json
from ml.orchestration.agent_state import AgentState
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Coordinates all agents and makes final decision using LLM"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile", use_llm: bool = True):
        # Pass required 'name' parameter to BaseAgent
        super().__init__(name="OrchestratorAgent", model=model)
        self.use_llm = use_llm  # Toggle between LLM and rule-based
    
    def process(self, race_data: Dict) -> Dict:
        """
        Required abstract method from BaseAgent
        This is called by the base agent when needed
        """
        # For orchestrator, this is a pass-through
        # Actual decision-making happens in __call__
        return {"status": "orchestrator uses __call__ instead"}
    
    def make_decision_with_llm(self, state: AgentState) -> Dict:
        """Use LLM to synthesize all agent inputs into nuanced decision"""
        
        # Gather all agent recommendations
        strategy = state.get('strategy_recommendation', {})
        telemetry = state.get('telemetry_analysis', {}).get('analysis', {})
        competitor = state.get('competitor_analysis', {}).get('analysis', {})
        race_control = state.get('race_control_analysis', {}).get('analysis', {})
        
        race = state['race_context']
        tire = state['tire_state']
        fuel = state['fuel_state']
        
        # Build comprehensive prompt for LLM
        prompt = f"""You are a Formula 1 chief strategist making a critical pit stop decision.

RACE SITUATION:
- Driver: {race['driver']}
- Position: P{race['position']}
- Lap: {race['current_lap']}/{race['total_laps']} ({race['current_lap']/race['total_laps']*100:.1f}% complete)
- Gap ahead: {race.get('gap_ahead', 'Leading')}s
- Gap behind: {race.get('gap_behind', 'N/A')}s
- Weather: {race['weather']}

TIRE STATUS:
- Compound: {tire['compound']}
- Age: {tire['age']} laps
- Degradation rate: {tire['degradation_rate']:.4f} s/lap
- Predicted cliff: Lap {tire.get('predicted_cliff_lap', 'Unknown')}
- Max temperature: {max(tire['temperature'].values()):.1f}°C

FUEL STATUS:
- Current: {fuel['current_fuel_kg']:.1f} kg
- Laps remaining on fuel: {fuel['laps_remaining_on_fuel']}

AGENT RECOMMENDATIONS:

1. STRATEGY AGENT (ML-Powered):
   {strategy.get('recommendation', 'No recommendation')}

2. TELEMETRY AGENT:
   - Tire condition: {telemetry.get('tire_condition_score', 'N/A')}/10
   - Cliff risk: {telemetry.get('tire_cliff_risk', 'N/A')}
   - Temperature issues: {telemetry.get('temp_issues', 'N/A')}
   - Recommendation: {telemetry.get('recommendation', 'N/A')}

3. COMPETITOR AGENT:
   - Main threat: {competitor.get('main_threat', 'None')}
   - Undercut opportunity: {competitor.get('undercut_opportunity', {}).get('feasibility', 'N/A')}
   - Undercut risk: {competitor.get('undercut_risk', {}).get('risk_level', 'N/A')}
   - Recommendation: {competitor.get('recommendation', 'N/A')}

4. RACE CONTROL AGENT:
   - Situation: {race_control.get('situation_severity', 'N/A')}
   - Pit window open: {race_control.get('pit_window_open', False)}
   - Strategic opportunity: {race_control.get('strategic_opportunity', 'None')}
   - Recommendation: {race_control.get('recommendation', 'N/A')}

Based on ALL the above information, make your FINAL DECISION. Consider:
- Tire degradation vs remaining race distance
- Competitor strategies (undercut opportunities/risks)
- Race control situations (safety cars, incidents)
- Track position importance
- Fuel strategy

Respond ONLY with valid JSON in this exact format:
{{
    "decision": "PIT_NOW|STAY_OUT|PIT_IN_X_LAPS",
    "laps_to_wait": <number or null>,
    "target_tire": "SOFT|MEDIUM|HARD",
    "confidence": <0.0-1.0>,
    "reasoning": "<2-3 sentences explaining your decision, weighing all factors>",
    "key_factors": ["<most important factor 1>", "<most important factor 2>", "<most important factor 3>"],
    "risks": "<main risk of this decision>",
    "alternative_strategy": "<what we'd do if situation changes>"
}}"""
        
        try:
            response = self.call_llm(
                system_prompt="You are an expert F1 strategist. Respond ONLY with valid JSON.",
                user_prompt=prompt,
                temperature=0.3  # Lower temperature for consistent decisions
            )
            
            # Parse JSON response
            decision = json.loads(response)
            
            logger.info(f"FINAL DECISION (LLM): {decision['decision']} (Confidence: {decision['confidence']:.2%})")
            logger.info(f"Reasoning: {decision['reasoning']}")
            
            return {
                'status': 'success',
                'decision': decision,
                'method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"LLM orchestrator error: {e}")
            # Fallback to rule-based
            logger.warning("Falling back to rule-based decision")
            return self.make_decision_rule_based(state)
    
    def make_decision_rule_based(self, state: AgentState) -> Dict:
        """Fallback: Rule-based voting system"""
        
        strategy = state.get('strategy_recommendation', {})
        telemetry = state.get('telemetry_analysis', {}).get('analysis', {})
        competitor = state.get('competitor_analysis', {}).get('analysis', {})
        race_control = state.get('race_control_analysis', {}).get('analysis', {})
        
        race = state['race_context']
        tire = state['tire_state']
        
        # Simple voting system
        votes_pit_now = 0
        votes_stay_out = 0
        
        # Strategy agent vote (weighted 2x)
        if 'pit' in strategy.get('recommendation', '').lower():
            votes_pit_now += 2
        else:
            votes_stay_out += 2
        
        # Telemetry vote
        if telemetry.get('recommendation') == 'Pit now':
            votes_pit_now += 1
        elif telemetry.get('recommendation') == 'Can extend':
            votes_stay_out += 1

        undercut_feasibility = competitor.get('undercut_opportunity', {}).get('feasibility', 'N/A')
        
        # Competitor vote
        if competitor.get('recommendation') == 'Pit early':
            votes_pit_now += 1

        if undercut_feasibility == 'High':  # 🔥 NEW
            votes_pit_now += 1 
        
        # Race control vote (weighted 3x)
        if race_control.get('pit_window_open'):
            votes_pit_now += 3
        
        # Make decision
        if votes_pit_now > votes_stay_out:
            decision = "PIT_NOW"
            target_tire = "HARD" if tire['compound'] == 'MEDIUM' else "MEDIUM"
            laps_to_wait = None
        else:
            decision = "STAY_OUT"
            target_tire = tire['compound']
            laps_to_wait = 5
        
        confidence = max(votes_pit_now, votes_stay_out) / (votes_pit_now + votes_stay_out) if (votes_pit_now + votes_stay_out) > 0 else 0.5
        
        reasoning = f"Decision based on {votes_pit_now} votes for pit, {votes_stay_out} for stay out. "
        reasoning += f"Key factors: Telemetry={telemetry.get('tire_cliff_risk', 'N/A')}, "
        reasoning += f"Competitor={competitor.get('main_threat', 'N/A')}, "
        reasoning += f"RaceControl={race_control.get('situation_severity', 'N/A')}"
        
        result = {
            'decision': decision,
            'laps_to_wait': laps_to_wait,
            'target_tire': target_tire,
            'confidence': confidence,
            'reasoning': reasoning,
            'key_factors': [
                f"Tire condition: {telemetry.get('tire_condition_score', 'N/A')}/10",
                f"Cliff risk: {telemetry.get('tire_cliff_risk', 'N/A')}",
                f"Competitor threat: {competitor.get('main_threat', 'None')}"
            ],
            'risks': "May lose track position" if decision == "PIT_NOW" else "Tire degradation may worsen",
            'alternative_strategy': "Stay out if SC deploys" if decision == "PIT_NOW" else "Pit if VSC/SC"
        }
        
        logger.info(f"FINAL DECISION (Rule-based): {decision} (Confidence: {confidence:.2%})")
        
        return {
            'status': 'success',
            'decision': result,
            'method': 'rule_based'
        }

    def __call__(self, state: AgentState) -> AgentState:
        """LangGraph node function"""
        if self.use_llm:
            result = self.make_decision_with_llm(state)
        else:
            result = self.make_decision_rule_based(state)
        
        state['final_decision'] = result['decision']
        state['confidence'] = result['decision'].get('confidence', 0.0)
        state['reasoning'] = result['decision'].get('reasoning', '')
        
        msg = f"[ORCHESTRATOR] Final decision: {result['decision']['decision']}"
        if msg not in state['messages']:
            state['messages'].append(msg)
        
        return state
