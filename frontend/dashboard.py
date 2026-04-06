"""
frontend/dashboard.py
---------------------
Real-time Streamlit dashboard that natively polls the lightweight backend API 
and constructs a live updating GUI for monitoring the Pygame player.

Run with:
    streamlit run frontend/dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import time

# Pointed at the standalone backend port
HISTORY_URL = "http://127.0.0.1:8001/history"
LATEST_URL = "http://127.0.0.1:8001/latest"

st.set_page_config(page_title="Live Game AI", page_icon="🧠", layout="wide")

st.title("🧠 Live Cognitive Game Dashboard")
st.caption("Auto-refreshing neural telemetry from the Pygame Engine.")
st.divider()

# Core container we will dynamically overwrite every refresh tick
placeholder = st.empty()

while True:
    try:
        resp_history = requests.get(HISTORY_URL, timeout=2.0)
        resp_latest = requests.get(LATEST_URL, timeout=2.0)
        
        if resp_history.status_code == 200 and resp_latest.status_code == 200:
            history = resp_history.json().get("history", [])
            latest = resp_latest.json()
            
            with placeholder.container():
                if not history or not latest:
                    st.info("Waiting for Pygame telemetry... Ensure the game and server are running.")
                else:
                    
                    # ── Row 1: KPI Blocks ──
                    st.subheader("Latest Neural Readout")
                    c1, c2, c3 = st.columns(3)
                    
                    c1.metric("Cognitive State", latest["state"])
                    c2.metric("CII Score", f"{latest['cii']:+.3f}")
                    
                    action_color = "🟢" if latest["action"]=="increase" else ("🔴" if latest["action"]=="decrease" else "⚪")
                    c3.metric("Difficulty Output", f"{action_color} {latest['action'].upper()}")
                    
                    st.divider()
                    
                    # ── Row 2: Live Chart ──
                    st.subheader("📈 Live CII Trend")
                    
                    # Flatten history into a DataFrame for Streamlit charting
                    df = pd.DataFrame([{ "Step": i, "CII": x["cii"] } for i, x in enumerate(history)])
                    st.line_chart(df.set_index("Step"), y="CII", color="#ab42f5")
                    
                    with st.expander("Latest Telemetry Payload"):
                        st.json(latest["telemetry"])
                        st.json(latest["nlp"])
                        
        else:
            with placeholder.container():
                st.error("Server API returned an error.")
                
        # Sleep exactly 1.0 second before redrawing the whole screen
        time.sleep(1.0)
        
    except requests.exceptions.RequestException:
        with placeholder.container():
            st.error(f"Cannot connect to the backend at {LATEST_URL}. Is 'uvicorn backend.main:app --port 8001' running?")
        time.sleep(2.0)
