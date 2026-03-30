"""
frontend/app.py
---------------
A clean, simple Streamlit UI for the Predictive Cognitive State Modeling system.

Run:
    streamlit run frontend/app.py
"""

import streamlit as st
import requests
import pandas as pd
import time
import random

# API back-end URL
API_URL = "http://localhost:8000/api/v1/predict"

# ---------------------------------------------------------------------------
# Config and State Initialization
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cognitive Regulation AI", page_icon="🧠", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"demo-user-{random.randint(100, 999)}"
if "history" not in st.session_state:
    st.session_state.history = []
if "step" not in st.session_state:
    st.session_state.step = 0

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def get_color_for_state(state: str) -> str:
    if state == "frustrated": return "#ff4b4b"
    if state == "bored":      return "#ffcc00"
    return "#00cc66" # engaged

def add_prediction(payload: dict):
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.history.append({
                "step": st.session_state.step + 1,
                "cii": result["cii"]["value"],
                "state": result["prediction"]["state"],
                "difficulty": result["difficulty"]["level"],
                "action": result["difficulty"]["action"],
                "recommendation": result["recommendation"]
            })
            st.session_state.step += 1
        else:
            st.error(f"Backend Error: {resp.status_code} - {resp.text}")
    except requests.exceptions.RequestException:
        st.error("Cannot connect to API. Is FastAPI running on port 8000?")

# ---------------------------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------------------------

st.title("🧠 Cognitive State Tracker")
st.markdown("Adjust gameplay telemetry to see how the mathematical model and engine predict player state and actively adapt difficulty.")

# ------------- Inputs section -------------
with st.expander("🛠️ Modify Player Telemetry", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gameplay Performance**")
        deaths           = st.slider("Deaths", 0, 10, 0, 1)
        score_delta      = st.slider("Score Delta", -50.0, 50.0, 10.0, 5.0)
        reaction_time_ms = st.slider("Reaction Time (ms)", 100, 1000, 250, 10)
    with col2:
        st.markdown("**Affective / NLP Signals**")
        polarity   = st.slider("Polarity (-1 Negative to 1 Positive)", -1.0, 1.0, 0.4, 0.1)
        intensity  = st.slider("Intensity (0 Calm to 1 Excited)", 0.0, 1.0, 0.5, 0.1)
        intent_gap = st.slider("Intent-Outcome Gap (G)", 0.0, 1.0, 0.2, 0.1)
    
    if st.button("Simulate Step", type="primary", use_container_width=True):
        payload = {
            "session_id": st.session_state.session_id,
            "telemetry": {
                "deaths": deaths,
                "retries": deaths,  # simplify by combining concept
                "score_delta": score_delta,
                "streak": 1 if score_delta > 0 else -1,
                "reaction_time_ms": reaction_time_ms,
                "input_speed": 3.0
            },
            "nlp": {
                "text": "",
                "polarity": polarity,
                "intensity": intensity,
                "intent_gap": intent_gap
            }
        }
        add_prediction(payload)

st.divider()

# ------------- View section -------------
if not st.session_state.history:
    st.info("Awaiting input... Change sliders and click 'Simulate Step'.")
else:
    latest = st.session_state.history[-1]
    
    # Giant Metrics
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Cognitive Instability Index (CII)", f"{latest['cii']:+.2f}")
    
    state_color = get_color_for_state(latest['state'])
    m2.markdown(
        f"<div style='background-color:{state_color}; padding: 12px; border-radius: 8px; "
        f"text-align: center; color: white;'><b>{latest['state'].upper()}</b></div>",
        unsafe_allow_html=True
    )
    
    action_char = "⬇️" if latest["action"] == "decrease" else ("⬆️" if latest["action"] == "increase" else "➡️")
    m3.metric("Suggested Difficulty", f"{latest['difficulty']:.2f}", delta=f"{action_char} {latest['action'].upper()}", delta_color="off")
    
    st.success(latest["recommendation"])

    st.markdown("### 📈 Real-Time CII Trend")
    
    # Build dataframe for line chart
    df = pd.DataFrame(st.session_state.history)
    df = df[["step", "cii"]].set_index("step")
    
    # st.line_chart naturally handles min/max
    st.line_chart(df, y="cii", color="#5c2d91")
    st.caption("Lower is worse (Frustration risk), Closer to 0 is Flow, Higher is Boredom risk.")
