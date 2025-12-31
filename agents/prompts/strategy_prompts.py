"""
Prompts for Strategy Agent
Race engineer-style reasoning and communication
"""

SYSTEM_PROMPT = """You are an expert Formula 1 Race Engineer with 15+ years of experience in strategy and tire management.

Your role is to:
1. Analyze tire degradation data and race context
2. Recommend optimal pit stop strategies with clear reasoning
3. Explain tradeoffs between undercut/overcut opportunities
4. Communicate like a real F1 engineer - concise, data-driven, decisive

Communication Style:
- Use F1 terminology naturally (deg, delta, overcut, graining, etc.)
- Give lap-specific recommendations
- Explain WHY, not just WHAT
- Consider multiple factors: tire deg, track position, traffic, weather
- Be direct and confident in recommendations

You think step-by-step but communicate the final conclusion clearly."""


STRATEGY_ANALYSIS_PROMPT = """Analyze the following race data and provide a pit strategy recommendation:

**DRIVER**: {driver}
**CURRENT LAP**: {current_lap}
**TOTAL RACE LAPS**: {total_laps}
**CURRENT TIRE**: {current_compound}
**TIRE AGE**: {tire_age} laps

**TIRE DEGRADATION DATA**:
{degradation_data}

**LSTM MODEL INSIGHT**:
{lstm_summary}

XGBoost Pit Window Analysis:  
{xgb_summary}

**OPTIMAL PIT WINDOWS**:
{pit_windows}

**TRACK CONDITIONS**:
- Track Temp: {track_temp}°C
- Air Temp: {air_temp}°C

**RACE CONTEXT**:
{race_context}

Provide your strategy recommendation in this format:

**RECOMMENDATION**: [Pit now / Stay out X laps / Optimal window lap Y-Z]

**REASONING**:
1. Current tire condition: [analysis]
2. Degradation projection: [analysis] (LSTM)
3. Pit Window timing: [analysis] (XGBoost)
3. Strategic considerations: [undercut/overcut/track position]

**RISKS**:
- [Key risk 1]
- [Key risk 2]

Be specific with lap numbers and degradation figures."""


TIRE_DEGRADATION_EXPLANATION_PROMPT = """Explain the tire degradation pattern for this driver in plain English:

**DRIVER**: {driver}
**STINT DATA**:
{stint_data}

Explain:
1. What's happening to the tires lap-by-lap
2. Whether this is normal or unusual degradation
3. What causes this pattern (temperature, driving style, track characteristics)
4. When the tires will become uncompetitive

Keep it under 4 sentences. Use racing terminology but make it understandable."""


UNDERCUT_ANALYSIS_PROMPT = """Analyze undercut opportunity:

**YOUR DRIVER**: {driver}
**DRIVER AHEAD**: {driver_ahead}
**GAP**: {gap_seconds}s
**YOUR TIRE AGE**: {your_tire_age} laps
**THEIR TIRE AGE**: {their_tire_age} laps

**DEGRADATION DATA**:
Your degradation: {your_deg} s/lap
Their degradation: {their_deg} s/lap

**PIT STOP LOSS**: ~22 seconds (typical)

Can we undercut? Explain the math and strategic timing."""


VERSTAPPEN_STYLE_ANALYSIS_PROMPT = """Analyze how Max Verstappen's aggressive driving style affects tire strategy compared to a conservative baseline driver:

**VERSTAPPEN'S DATA**:
{verstappen_data}

**BASELINE (CONSERVATIVE) DRIVER DATA**:
{baseline_data}

Provide a comprehensive analysis covering:

1. **Driving Style Comparison**:
   - Quantify the aggressiveness difference
   - What does this mean in practical terms?

2. **Degradation Impact**:
   - Why does Verstappen's style cause more/less tire wear?
   - Which compound benefits most from each style?

3. **Strategic Implications**:
   - Optimal pit timing for each style
   - Race scenarios where each style wins
   - Advantages and disadvantages of aggressive approach

4. **Tactical Recommendations**:
   - When should a driver adopt aggressive style?
   - When to be conservative?
   - How does this affect race strategy decisions?

Use F1 engineering terminology. Be specific with numbers. Explain the physics and strategy tradeoffs."""

