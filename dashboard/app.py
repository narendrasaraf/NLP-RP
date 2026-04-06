import streamlit as st
import requests
import time

st.set_page_config(page_title="CogReg Live Dashboard", layout="centered")

API_LATEST = "http://localhost:8001/latest"

st.title("🧠 Live Cognitive AI Monitor")
st.divider()

try:
    # Fetch data directly from the FastAPI endpoint
    r = requests.get(API_LATEST, timeout=2)
    
    if r.status_code == 200:
        data = r.json()
        if data:
            # 1. Core Models
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CII (Cognitive Instability Index)", f"{data.get('cii', 0):+.3f}")
            with col2:
                st.metric("Predicted Player State", data.get('state', 'Unknown'))
            
            st.divider()
            
            # 2. NLP Analysis Block
            st.subheader("NLP Chat Analysis")
            nlp = data.get("nlp", {})
            chat_text = data.get("chat", "No chat recorded yet...")
                
            st.info(f"**Latest Chat:** {chat_text}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Polarity", f"{nlp.get('polarity', 0.0):.2f}")
            with c2:
                st.metric("Emotional Intensity", f"{nlp.get('emotional_intensity', 0.0):.2f}")
            with c3:
                # Determine prominent emotion dynamically
                scores = {
                    "Frustration": nlp.get("frustration_score", 0.0),
                    "Anger": nlp.get("anger_score", 0.0),
                    "Confidence": nlp.get("confidence_score", 0.0)
                }
                
                # Prevent baseline 0.5 confidence from swallowing true neutral messages.
                if scores["Confidence"] <= 0.5:
                    scores["Confidence"] = 0.0
                    
                detected = max(scores, key=scores.get) if max(scores.values()) > 0.0 else "Neutral"
                st.metric("Detected Emotion", detected)
        else:
            st.info("Waiting for live game data... (No payloads received yet from the player)")
            
except requests.exceptions.RequestException:
    st.error("Could not connect to backend. Ensure FastAPI server is running on port 8001!")

# Auto-refresh loop native to Streamlit (safest way to avoid freezing)
time.sleep(1.0)
st.rerun()
