"""
dashboard/app.py
-----------------
Streamlit real-time dashboard for the Cognitive Regulation System.

Features:
  - Live CII trend chart (line chart updating per step)
  - Player state badge (Frustrated / Bored / Engaged)
  - Frustration / Boredom / Engagement probability bars
  - Difficulty level gauge + action history
  - Quick input form for manual demo testing
  - Session summary analytics at end of session

Run with:
    streamlit run dashboard/app.py
"""

import time
import random
import requests
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "CogReg Dashboard",
    page_icon  = "🧠",
    layout     = "wide",
)

API_BASE = "http://localhost:8000/api/v1"

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history     = []
if "session_id" not in st.session_state:
    st.session_state.session_id  = f"demo-{random.randint(1000,9999)}"
if "step" not in st.session_state:
    st.session_state.step        = 0
if "difficulty" not in st.session_state:
    st.session_state.difficulty  = 5.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_COLORS = {
    "frustrated": "#FF4B4B",
    "bored":      "#FFD700",
    "engaged":    "#00CC66",
}

STATE_LABELS = {
    "frustrated": "🔴 FRUSTRATED",
    "bored":      "🟡 BORED",
    "engaged":    "🟢 ENGAGED",
}

def post_step(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/process", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def get_summary(session_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/session/{session_id}/summary", timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def reset_session(session_id: str):
    try:
        requests.post(f"{API_BASE}/session/{session_id}/reset", timeout=5)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧠 Predictive Cognitive State Modeling")
st.caption("Adaptive Game Difficulty Regulation via CII  |  Real-time Dashboard")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session Control")
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id

    if st.button("Reset Session", type="secondary"):
        reset_session(session_id)
        st.session_state.history    = []
        st.session_state.step       = 0
        st.session_state.difficulty = 5.0
        st.success("Session reset.")

    st.divider()
    st.header("Manual Input")

    polarity      = st.slider("Polarity P(t)",    -1.0, 1.0,  0.0, 0.05)
    intensity     = st.slider("Intensity I(t)",    0.0, 1.0,  0.3, 0.05)
    deaths        = st.number_input("Deaths",      0, 20, 0, step=1)
    retries       = st.number_input("Retries",     0, 20, 0, step=1)
    score_delta   = st.slider("Score Delta",     -50.0, 50.0, 5.0, 1.0)
    reaction_ms   = st.slider("Reaction Time ms", 100, 800, 280, 10)
    intent_gap    = st.slider("Intent Gap G(t)",   0.0, 1.0, 0.2, 0.05)
    player_text   = st.text_input("Player Text", "")

    send = st.button("Submit Step", type="primary", use_container_width=True)

    st.divider()
    st.header("Auto-Simulate")
    scenario = st.selectbox(
        "Scenario",
        ["Escalating Frustration", "Boredom Onset", "Stable Flow"]
    )
    auto_steps = st.slider("Steps", 3, 15, 5)
    run_auto   = st.button("Run Scenario", use_container_width=True)

# ---------------------------------------------------------------------------
# Auto-simulation data
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Escalating Frustration": [
        dict(polarity=-0.1, intensity=0.3, deaths=1, retries=1, score_delta=-5,  reaction_time_ms=320, intent_gap=0.2, text="almost had it"),
        dict(polarity=-0.3, intensity=0.5, deaths=3, retries=2, score_delta=-12, reaction_time_ms=380, intent_gap=0.4, text="why does this keep happening"),
        dict(polarity=-0.6, intensity=0.7, deaths=5, retries=4, score_delta=-20, reaction_time_ms=450, intent_gap=0.6, text="this is impossible"),
        dict(polarity=-0.8, intensity=0.9, deaths=8, retries=7, score_delta=-30, reaction_time_ms=510, intent_gap=0.8, text="I QUIT this is trash"),
        dict(polarity=-0.9, intensity=0.95,deaths=10,retries=9, score_delta=-35, reaction_time_ms=600, intent_gap=0.9, text="completely broken game"),
    ],
    "Boredom Onset": [
        dict(polarity=0.8, intensity=0.6,  deaths=0, retries=0, score_delta=25, reaction_time_ms=180, intent_gap=0.05, text="easy win"),
        dict(polarity=0.7, intensity=0.4,  deaths=0, retries=0, score_delta=30, reaction_time_ms=170, intent_gap=0.05, text="too easy"),
        dict(polarity=0.5, intensity=0.2,  deaths=0, retries=0, score_delta=28, reaction_time_ms=160, intent_gap=0.02, text="..."),
        dict(polarity=0.3, intensity=0.1,  deaths=0, retries=0, score_delta=22, reaction_time_ms=155, intent_gap=0.01, text="this is boring ngl"),
        dict(polarity=0.2, intensity=0.05, deaths=0, retries=0, score_delta=18, reaction_time_ms=150, intent_gap=0.01, text="give me something harder"),
    ],
    "Stable Flow": [
        dict(polarity=0.4, intensity=0.5,  deaths=1, retries=1, score_delta=10, reaction_time_ms=240, intent_gap=0.2,  text="nice challenge"),
        dict(polarity=0.5, intensity=0.55, deaths=0, retries=1, score_delta=12, reaction_time_ms=230, intent_gap=0.15, text="getting better"),
        dict(polarity=0.35,intensity=0.5,  deaths=1, retries=2, score_delta=8,  reaction_time_ms=250, intent_gap=0.25, text="tough but fair"),
        dict(polarity=0.45,intensity=0.6,  deaths=1, retries=1, score_delta=11, reaction_time_ms=235, intent_gap=0.2,  text="close one"),
        dict(polarity=0.5, intensity=0.5,  deaths=0, retries=1, score_delta=14, reaction_time_ms=220, intent_gap=0.18, text="love this game"),
    ],
}

def build_payload(d: dict) -> dict:
    return {
        "session_id": st.session_state.session_id,
        "timestamp":  time.time(),
        "telemetry": {
            "deaths":           d.get("deaths", 0),
            "retries":          d.get("retries", 0),
            "score_delta":      d.get("score_delta", 0),
            "streak":           0,
            "reaction_time_ms": d.get("reaction_time_ms", 280),
            "input_speed":      d.get("input_speed", 3.0),
        },
        "nlp": {
            "text":       d.get("text", ""),
            "polarity":   d.get("polarity", 0.0),
            "intensity":  d.get("intensity", 0.3),
            "intent_gap": d.get("intent_gap", 0.2),
        },
    }

# ---------------------------------------------------------------------------
# Process inputs
# ---------------------------------------------------------------------------

results_to_process = []

if send:
    results_to_process.append(build_payload(dict(
        polarity=polarity, intensity=intensity,
        deaths=int(deaths), retries=int(retries),
        score_delta=score_delta, reaction_time_ms=reaction_ms,
        intent_gap=intent_gap, text=player_text,
    )))

if run_auto:
    steps_data = SCENARIOS[scenario][:auto_steps]
    for d in steps_data:
        results_to_process.append(build_payload(d))

for payload in results_to_process:
    result = post_step(payload)
    if result:
        st.session_state.history.append(result)
        st.session_state.step += 1
        st.session_state.difficulty = result["difficulty_level"]

# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

history = st.session_state.history

if not history:
    st.info("No data yet. Use the sidebar to submit a step or run an auto-scenario.")
    st.stop()

latest = history[-1]

# ── Row 1: KPI blocks ──────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Step", latest["timestep"])
with col2:
    cii_val = latest["cii"]
    st.metric("CII(t)", f"{cii_val:+.4f}")
with col3:
    st.metric("Difficulty", f"{latest['difficulty_level']:.2f}",
              delta=f"{latest['difficulty_delta']:+.3f}")
with col4:
    state = latest["player_state"]
    st.markdown(
        f"<div style='background-color:{STATE_COLORS[state]};padding:10px;"
        f"border-radius:8px;text-align:center;font-weight:bold;color:white'>"
        f"{STATE_LABELS[state]}</div>",
        unsafe_allow_html=True,
    )
with col5:
    action = latest["difficulty_action"]
    action_map = {"decrease": "⬇️ DECREASE", "increase": "⬆️ INCREASE", "maintain": "➡️ MAINTAIN"}
    st.info(f"Action: **{action_map[action]}**")

st.divider()

# ── Row 2: Charts ──────────────────────────────────────────────────────────
col_cii, col_probs = st.columns([2, 1])

with col_cii:
    st.subheader("CII Trajectory")
    df = pd.DataFrame([
        {
            "Step":       h["timestep"],
            "CII":        h["cii"],
            "Momentum":   h["emotional_state"]["momentum"],
            "Difficulty": h["difficulty_level"],
        }
        for h in history
    ])
    st.line_chart(df.set_index("Step")[["CII", "Momentum"]])

    # CII reference lines annotation
    st.caption("CII < -0.40 = Frustration zone | CII ∈ [-0.40, +0.30] = Flow | CII > +0.30 = Boredom zone")

with col_probs:
    st.subheader("State Probabilities")
    p_frus = latest["frustration_prob"]
    p_bore = latest["boredom_prob"]
    p_eng  = latest["engagement_prob"]

    st.markdown("**Frustrated**")
    st.progress(p_frus, text=f"{p_frus:.1%}")
    st.markdown("**Bored**")
    st.progress(p_bore, text=f"{p_bore:.1%}")
    st.markdown("**Engaged**")
    st.progress(p_eng,  text=f"{p_eng:.1%}")

st.divider()

# ── Row 3: Difficulty timeline + Emotional state breakdown ─────────────────
col_diff, col_state = st.columns(2)

with col_diff:
    st.subheader("Difficulty Timeline")
    diff_df = pd.DataFrame([
        {"Step": h["timestep"], "Difficulty": h["difficulty_level"]}
        for h in history
    ])
    st.line_chart(diff_df.set_index("Step"))

with col_state:
    st.subheader("Emotional State Components")
    es = latest["emotional_state"]
    comp_df = pd.DataFrame({
        "Component": ["P(t) Polarity", "I(t) Intensity", "D(t) Perf Dev", "G(t) Intent Gap",
                      "M(t) Momentum", "A(t) Acceleration"],
        "Value":     [es["polarity"], es["intensity"], es["performance_dev"], es["intent_gap"],
                      es["momentum"], es["acceleration"]],
    })
    st.bar_chart(comp_df.set_index("Component"))

st.divider()

# ── Row 4: Step history table ──────────────────────────────────────────────
st.subheader("Step History")
table_data = [
    {
        "Step":       h["timestep"],
        "CII":        f"{h['cii']:+.4f}",
        "State":      h["player_state"].upper(),
        "P_Frus":     f"{h['frustration_prob']:.2%}",
        "P_Bore":     f"{h['boredom_prob']:.2%}",
        "Action":     h["difficulty_action"].upper(),
        "Difficulty": f"{h['difficulty_level']:.2f}",
    }
    for h in history
]
st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ── Session summary ────────────────────────────────────────────────────────
summary = get_summary(session_id)
if summary:
    st.divider()
    st.subheader("Session Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Avg CII",           f"{summary['avg_cii']:+.4f}")
    s2.metric("Flow Rate",         f"{summary['flow_rate']:.1%}")
    s3.metric("Frustration Events",summary["frustration_events"])
    s4.metric("Boredom Events",    summary["boredom_events"])
    s5.metric("Difficulty Changes",summary["difficulty_changes"])
