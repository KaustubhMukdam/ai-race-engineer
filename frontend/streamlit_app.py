"""
Streamlit Dashboard for AI Race Engineer
Quick interactive interface for testing
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="AI Race Engineer",
    page_icon="🏎️",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e10600;
        text-align: center;
        font-weight: bold;
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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏎️ AI RACE ENGINEER</p>', unsafe_allow_html=True)
st.markdown("**LLM-Powered F1 Strategy & Tire Management**")

# Check API health
try:
    health_response = requests.get(f"{API_BASE_URL}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data['agent_loaded']:
            st.success(f"✅ Strategy Agent Online | Model: {health_data['llm_model']}")
        else:
            st.error("❌ Strategy Agent Not Loaded")
    else:
        st.error("❌ API Not Responding")
except:
    st.error("❌ Cannot connect to API. Make sure backend is running on port 8000")
    st.stop()

# Sidebar for input
st.sidebar.header("📊 Race Configuration")

# Driver selection
drivers = ["VER", "NOR", "HAM", "LEC", "SAI", "RUS", "PIA", "ALO", "GAS", "HUL"]
selected_driver = st.sidebar.selectbox("Select Driver", drivers)

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
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Pit Strategy",
    "📈 Tire Degradation",
    "⚡ Undercut Analysis",
    "📊 Data Explorer"
])

# Tab 1: Pit Strategy Recommendation
with tab1:
    st.header("Pit Strategy Recommendation")
    
    col1, col2 = st.columns([2, 1])
    
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
        st.metric("Driver", selected_driver)
        st.metric("Lap", f"{current_lap}/{total_laps}")
        st.metric("Tire Age", f"{tire_age} laps")
        st.metric("Compound", compound)

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

# Tab 3: Undercut Analysis
with tab3:
    st.header("Undercut Opportunity Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        driver_ahead = st.selectbox("Driver Ahead", [d for d in drivers if d != selected_driver])
        gap_seconds = st.number_input("Gap (seconds)", 0.0, 30.0, 2.8, 0.1)
        your_tire_age = st.slider("Your Tire Age", 0, 50, 18)
        their_tire_age = st.slider("Their Tire Age", 0, 50, 20)
        
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
        st.metric("Target", driver_ahead)
        st.metric("Gap", f"{gap_seconds}s")

# Tab 4: Data Explorer
with tab4:
    st.header("Race Data Explorer")
    
    # Load processed data
    try:
        from config.app_config import settings
        
        # Load lap times
        lap_times_file = settings.processed_data_dir / "2024_Abu_Dhabi_GP_processed" / "processed_laps.csv"
        if lap_times_file.exists():
            lap_data = pd.read_csv(lap_times_file)
            
            st.subheader("📊 Lap Times Data")
            driver_filter = st.multiselect("Filter Drivers", lap_data['Driver'].unique(), default=[selected_driver])
            
            if driver_filter:
                filtered_data = lap_data[lap_data['Driver'].isin(driver_filter)]
                st.dataframe(filtered_data[['LapNumber', 'Driver', 'LapTime_Seconds', 'Compound', 
                                           'TyreLife', 'TrackTemp', 'DegradationRate']], 
                           use_container_width=True, height=400)
                
                st.download_button(
                    "⬇️ Download Filtered Data",
                    filtered_data.to_csv(index=False),
                    "filtered_lap_data.csv",
                    "text/csv"
                )
        
        # Load degradation analysis
        deg_file = settings.processed_data_dir / "2024_Abu_Dhabi_GP_processed" / "tire_degradation_analysis.csv"
        if deg_file.exists():
            st.markdown("---")
            st.subheader("🔧 Tire Degradation Summary")
            deg_data = pd.read_csv(deg_file)
            
            if driver_filter:
                filtered_deg = deg_data[deg_data['Driver'].isin(driver_filter)]
                st.dataframe(filtered_deg, use_container_width=True)
            else:
                st.dataframe(deg_data, use_container_width=True)
                
    except Exception as e:
        st.error(f"Could not load race data: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>AI Race Engineer | Powered by Groq (Llama 3.3 70B) + FastF1 | 2024 Abu Dhabi GP Data</p>
    </div>
    """,
    unsafe_allow_html=True
)
