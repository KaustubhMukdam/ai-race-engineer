"""
Enhanced Streamlit Dashboard for AI Race Engineer
Multi-race support with session selection
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import json

# Configure page
st.set_page_config(
    page_title="AI Race Engineer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'available_drivers' not in st.session_state:
    st.session_state.available_drivers = []

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e10600;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .recommendation-box {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e10600;
        font-family: 'Monaco', 'Courier New', monospace;
        line-height: 1.6;
    }
    .recommendation-box h1, 
    .recommendation-box h2, 
    .recommendation-box h3 {
        color: #e10600 !important;
    }
    .recommendation-box strong {
        color: #ff6b6b;
    }
    .session-info {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏎️ AI RACE ENGINEER</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LLM-Powered F1 Strategy & Tire Management System</p>', unsafe_allow_html=True)

# Check API health
def check_api_health():
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data['agent_loaded']:
                st.success(f"✅ API Online | Strategy Agent Ready | Model: {health_data['llm_model']}")
                return True
            else:
                st.error("❌ Strategy Agent Not Loaded")
                return False
        else:
            st.error("❌ API Not Responding")
            return False
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
        st.info("💡 Make sure backend is running: `python backend/app.py`")
        return False

api_healthy = check_api_health()

if not api_healthy:
    st.stop()

# Sidebar - Session Selection
st.sidebar.header("🏁 Session Selection")

# Get available seasons
try:
    seasons_response = requests.get(f"{API_BASE_URL}/sessions/seasons")
    available_seasons = seasons_response.json() if seasons_response.status_code == 200 else [2024]
except:
    available_seasons = [2024]

selected_year = st.sidebar.selectbox("Season", available_seasons, index=len(available_seasons)-1)

# Get schedule for selected year
try:
    schedule_response = requests.get(f"{API_BASE_URL}/sessions/schedule/{selected_year}")
    if schedule_response.status_code == 200:
        schedule_data = schedule_response.json()
        events = schedule_data['events']
        event_names = [event['EventName'] for event in events]
    else:
        event_names = ["Abu Dhabi Grand Prix"]
except:
    event_names = ["Abu Dhabi Grand Prix"]

selected_event = st.sidebar.selectbox("Event", event_names)

session_type = st.sidebar.selectbox("Session", ["Race", "Qualifying", "Sprint"])

# Load Session Button
col1, col2 = st.sidebar.columns(2)
with col1:
    load_button = st.button("🔄 Load Session", type="primary", use_container_width=True)
with col2:
    force_reload = st.checkbox("Force", help="Force reload even if cached")

if load_button:
    with st.spinner(f"Loading {selected_year} {selected_event} {session_type}..."):
        try:
            load_response = requests.post(
                f"{API_BASE_URL}/sessions/load",
                params={
                    "year": selected_year,
                    "event": selected_event,
                    "session": session_type,
                    "force_reload": force_reload
                },
                timeout=180  # 3 minutes timeout for first-time loads
            )
            
            if load_response.status_code == 200:
                result = load_response.json()
                st.session_state.current_session = result
                st.sidebar.success(f"✅ Session Loaded {'(Cached)' if result['cached'] else '(Fresh)'}")
                
                # Load session data for driver selection
                from config.app_config import settings
                session_key = result['session_key']
                processed_path = settings.processed_data_dir / f"{session_key}_processed"
                
                laps_file = processed_path / "processed_laps.csv"
                if laps_file.exists():
                    laps_df = pd.read_csv(laps_file)
                    st.session_state.available_drivers = sorted(laps_df['Driver'].unique().tolist())
                    st.session_state.session_data = laps_df
            else:
                st.sidebar.error(f"Failed to load: {load_response.json()}")
                
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Display current session info
if st.session_state.current_session:
    session_info = st.session_state.current_session
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Session")
    st.sidebar.info(f"""
    **{session_info['session_key'].replace('_', ' ')}**
    
    Status: {'Cached' if session_info['cached'] else 'Fresh'}
    
    Drivers: {session_info['metadata']['drivers']}
    
    Total Laps: {session_info['metadata']['total_laps']}
    """)

st.sidebar.markdown("---")

# Race Configuration
st.sidebar.header("📊 Strategy Parameters")

# Driver selection
if st.session_state.available_drivers:
    selected_driver = st.sidebar.selectbox("Select Driver", st.session_state.available_drivers)
else:
    selected_driver = st.sidebar.selectbox("Select Driver", ["VER", "NOR", "HAM"])

# Race parameters
current_lap = st.sidebar.slider("Current Lap", 1, 70, 25)
total_laps = st.sidebar.number_input("Total Race Laps", 40, 80, 58)
tire_age = st.sidebar.slider("Tire Age (laps)", 0, 50, 25)

# Tire compound
compound = st.sidebar.selectbox("Current Tire Compound", ["SOFT", "MEDIUM", "HARD"])

# Track conditions
st.sidebar.subheader("🌡️ Track Conditions")
track_temp = st.sidebar.slider("Track Temperature (°C)", 20.0, 50.0, 31.0, 0.5)
air_temp = st.sidebar.slider("Air Temperature (°C)", 15.0, 40.0, 26.5, 0.5)

# Race context
race_context = st.sidebar.text_area(
    "Race Context",
    "P1, gap to P2 is 3.2 seconds",
    help="Describe current race situation"
)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Pit Strategy",
    "📈 Tire Degradation",
    "⚡ Undercut Analysis",
    "📊 Multi-Driver Comparison",
    "🗂️ Session Manager"
])

# Tab 1: Pit Strategy Recommendation
with tab1:
    st.header("Pit Strategy Recommendation")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚦 Get Strategy Recommendation", type="primary", use_container_width=True):
            with st.spinner("AI Race Engineer analyzing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/strategy/recommend-pit",
                        json={
                            "driver": selected_driver,
                            "current_lap": current_lap,
                            "total_laps": total_laps,
                            "current_compound": compound,
                            "tire_age": tire_age,
                            "track_temp": track_temp,
                            "air_temp": air_temp,
                            "race_context": race_context
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.markdown("---")
                        st.markdown("### 📡 Race Engineer Communication")
                        st.markdown(f'<div class="recommendation-box">{data["recommendation"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.info(f"**Analysis generated using**: {data['llm_model']}")
                    else:
                        st.error(f"Error: {response.json()}")
                        
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Driver", selected_driver)
        st.metric("Lap", f"{current_lap}/{total_laps}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tire Age", f"{tire_age} laps")
        st.metric("Compound", compound)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Tire Degradation Explanation
with tab2:
    st.header("Tire Degradation Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stint_number = st.number_input("Stint Number", 1, 5, 1)
        
        if st.button("🔍 Explain Tire Degradation", use_container_width=True):
            with st.spinner("Analyzing tire degradation pattern..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/strategy/explain-degradation",
                        json={
                            "driver": selected_driver,
                            "stint": stint_number
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown("---")
                        st.markdown("### 🔬 Degradation Pattern Analysis")
                        st.info(data["explanation"])
                    else:
                        st.error(f"Error: {response.json()}")
                        
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    
    with col2:
        st.metric("Driver", selected_driver)
        st.metric("Stint", stint_number)
    
    # Visualize degradation if data available
    if st.session_state.session_data is not None:
        st.markdown("---")
        st.subheader("📈 Lap Time Evolution")
        
        driver_data = st.session_state.session_data[
            st.session_state.session_data['Driver'] == selected_driver
        ].copy()
        
        if not driver_data.empty:
            fig = px.line(
                driver_data,
                x='LapNumber',
                y='LapTime_Seconds',
                color='Compound',
                title=f"{selected_driver} - Lap Times by Compound",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Lap Number",
                yaxis_title="Lap Time (seconds)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Undercut Analysis
with tab3:
    st.header("Undercut Opportunity Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.available_drivers:
            driver_ahead_options = [d for d in st.session_state.available_drivers if d != selected_driver]
        else:
            driver_ahead_options = ["NOR", "HAM", "LEC"]
        
        driver_ahead = st.selectbox("Driver Ahead", driver_ahead_options)
        gap_seconds = st.number_input("Gap (seconds)", 0.0, 30.0, 2.8, 0.1)
        your_tire_age = st.slider("Your Tire Age", 0, 50, 18, key="undercut_your_tire")
        their_tire_age = st.slider("Their Tire Age", 0, 50, 20, key="undercut_their_tire")
        
        if st.button("⚡ Analyze Undercut", type="primary", use_container_width=True):
            with st.spinner("Calculating undercut opportunity..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/strategy/analyze-undercut",
                        json={
                            "driver": selected_driver,
                            "driver_ahead": driver_ahead,
                            "gap_seconds": gap_seconds,
                            "your_tire_age": your_tire_age,
                            "their_tire_age": their_tire_age
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown("---")
                        st.markdown("### ⚡ Undercut Strategy Analysis")
                        st.success(data["analysis"])
                    else:
                        st.error(f"Error: {response.json()}")
                        
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    
    with col2:
        st.metric("Your Position", selected_driver)
        st.metric("Target", driver_ahead if 'driver_ahead' in locals() else "N/A")
        st.metric("Gap", f"{gap_seconds}s")

# Tab 4: Multi-Driver Comparison
with tab4:
    st.header("Multi-Driver Comparison")
    
    # Check if session data is loaded
    if st.session_state.session_data is None:
        st.info("📥 Load a session first to compare drivers")
        
        if st.session_state.current_session:
            st.warning(f"✅ Session detected: **{st.session_state.current_session['session_key']}**")
            st.info("But driver data not loaded. Click below to load it:")
            
            if st.button("🔄 Load Driver Data", type="primary", key="load_driver_data"):
                with st.spinner("Loading driver data..."):
                    try:
                        from config.app_config import settings
                        session_key = st.session_state.current_session['session_key']
                        processed_path = settings.processed_data_dir / f"{session_key}_processed"
                        
                        laps_file = processed_path / "processed_laps.csv"
                        deg_file = processed_path / "tire_degradation_analysis.csv"
                        
                        if laps_file.exists():
                            laps_df = pd.read_csv(laps_file)
                            st.session_state.available_drivers = sorted(laps_df['Driver'].unique().tolist())
                            st.session_state.session_data = laps_df
                            
                            if deg_file.exists():
                                st.session_state.degradation_data = pd.read_csv(deg_file)
                            
                            st.success(f"✅ Loaded data for {len(st.session_state.available_drivers)} drivers!")
                            st.rerun()
                        else:
                            st.error(f"❌ Data file not found: {laps_file}")
                            st.info("Try reloading the session with 'Force' checkbox enabled")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.code(str(e))
        
        st.stop()
    
    # If we get here, session_data exists
    # Driver selection for comparison
    comparison_drivers = st.multiselect(
        "Select Drivers to Compare",
        st.session_state.available_drivers,
        default=st.session_state.available_drivers[:3] if len(st.session_state.available_drivers) >= 3 else st.session_state.available_drivers
    )
    
    if not comparison_drivers:
        st.warning("👆 Select at least one driver to compare")
        st.stop()
    
    comparison_data = st.session_state.session_data[
        st.session_state.session_data['Driver'].isin(comparison_drivers)
    ]
    
    # Lap time comparison
    st.subheader("📊 Lap Time Comparison")
    fig = px.line(
        comparison_data,
        x='LapNumber',
        y='LapTime_Seconds',
        color='Driver',
        title="Lap Times - All Selected Drivers",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Degradation comparison
    st.subheader("📉 Degradation Rate Comparison")
    
    if st.session_state.current_session:
        from config.app_config import settings
        session_key = st.session_state.current_session['session_key']
        deg_file = settings.processed_data_dir / f"{session_key}_processed" / "tire_degradation_analysis.csv"
        
        if deg_file.exists():
            deg_df = pd.read_csv(deg_file)
            filtered_deg = deg_df[deg_df['Driver'].isin(comparison_drivers)]
            
            if not filtered_deg.empty:
                fig2 = px.bar(
                    filtered_deg,
                    x='Driver',
                    y='DegradationPerLap_Seconds',
                    color='Compound',
                    title="Degradation Rate by Driver & Compound",
                    barmode='group'
                )
                fig2.update_layout(
                    xaxis_title="Driver",
                    yaxis_title="Degradation Per Lap (seconds)",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show table
                st.dataframe(
                    filtered_deg[['Driver', 'Stint', 'Compound', 'TotalLaps', 
                                 'DegradationPerLap_Seconds', 'LinearDegradationRate']],
                    use_container_width=True
                )
            else:
                st.warning("No degradation data found for selected drivers")
        else:
            st.warning("Degradation analysis file not found")

# Tab 5: Session Manager
with tab5:
    st.header("Session Manager")
    
    # Get cached sessions
    try:
        cached_response = requests.get(f"{API_BASE_URL}/sessions/cached")
        if cached_response.status_code == 200:
            cached_data = cached_response.json()
            
            st.subheader(f"📦 Cached Sessions ({cached_data['total_cached']})")
            
            if cached_data['total_cached'] > 0:
                cached_df = pd.DataFrame(cached_data['sessions'])
                cached_df['processed_date'] = pd.to_datetime(cached_df['processed_date']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    cached_df[['session_key', 'year', 'event', 'session', 
                              'drivers', 'total_laps', 'processed_date']],
                    use_container_width=True,
                    height=400
                )
                
                # Delete session
                st.markdown("---")
                st.subheader("🗑️ Delete Cached Session")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    del_year = st.number_input("Year", 2018, 2025, 2024, key="del_year")
                with col2:
                    del_event = st.text_input("Event", "Abu Dhabi Grand Prix", key="del_event")
                with col3:
                    del_session = st.selectbox("Session", ["Race", "Qualifying", "Sprint"], key="del_session")
                with col4:
                    if st.button("🗑️ Delete", type="secondary"):
                        try:
                            del_response = requests.delete(
                                f"{API_BASE_URL}/sessions/cached",
                                params={
                                    "year": del_year,
                                    "event": del_event,
                                    "session": del_session
                                }
                            )
                            if del_response.status_code == 200:
                                st.success("Session deleted!")
                                st.rerun()
                            else:
                                st.error(f"Error: {del_response.json()}")
                        except Exception as e:
                            st.error(f"Delete failed: {str(e)}")
            else:
                st.info("No cached sessions yet. Load a session to get started!")
    except Exception as e:
        st.error(f"Error loading cached sessions: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        <p>AI Race Engineer v1.0 | Powered by Groq (Llama 3.3 70B) + FastF1</p>
        <p>{'Current Session: ' + st.session_state.current_session['session_key'] if st.session_state.current_session else 'No session loaded'}</p>
    </div>
    """,
    unsafe_allow_html=True
)
