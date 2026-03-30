"""
run_demo.py
-----------
End-to-end demo of the Predictive Cognitive State Modeling pipeline.

This script runs a simulated player session (escalating frustration followed by recovery)
through the entire standalone 4-step pipeline:

    [ Input ]
       ↓
    1. Preprocessing (Data extraction & normalization)
       ↓
    2. CII Modeling (Cognitive Instability Index computation)
       ↓
    3. Prediction (Frustration vs Boredom prediction)
       ↓
    4. Adaptation (Difficulty action & step sizing)

Usage:
    python run_demo.py
"""

import time
from typing import List, Dict

# Import the 4 core standalone modules
from utils.preprocessing     import DataPipeline
from utils.cii               import CIIComputer
from utils.predictor         import StatePredictor
from utils.difficulty_engine import DifficultyEngine

# ---------------------------------------------------------------------------
# Simulated Player Scenario
# ---------------------------------------------------------------------------

SCENARIO = [
    # ── Phase 1: Normal Gameplay (Engaged) ──
    {
        "desc": "Player starts, playing normally",
        "telemetry": {"deaths": 0, "retries": 0, "score_delta": 10.0, "reaction_time_ms": 250},
        "nlp": {"text": "nice game so far"}
    },
    {
        "desc": "Player encounters first challenge",
        "telemetry": {"deaths": 1, "retries": 1, "score_delta": -5.0, "reaction_time_ms": 300},
        "nlp": {"text": "oh that was close"}
    },
    
    # ── Phase 2: Escalating Frustration ──
    {
        "desc": "Player fails repeatedly on same section",
        "telemetry": {"deaths": 3, "retries": 2, "score_delta": -15.0, "reaction_time_ms": 380},
        "nlp": {"text": "why didn't my jump register?"}
    },
    {
        "desc": "Player is tilting (High intent gap)",
        "telemetry": {"deaths": 5, "retries": 4, "score_delta": -25.0, "reaction_time_ms": 450},
        "nlp": {"text": "this is literally impossible, the controls are broken"}
    },
    {
        "desc": "Player reaches acute frustration",
        "telemetry": {"deaths": 8, "retries": 8, "score_delta": -40.0, "reaction_time_ms": 550},
        "nlp": {"text": "I absolute HATE this garbage game Im quitting"}
    },

    # ── Phase 3: Recovery (After difficulty was lowered) ──
    {
        "desc": "Player tries again, difficulty is now lower",
        "telemetry": {"deaths": 1, "retries": 1, "score_delta": +15.0, "reaction_time_ms": 280},
        "nlp": {"text": "finally, okay I got past it"}
    },
    {
        "desc": "Player returns to flow state",
        "telemetry": {"deaths": 0, "retries": 0, "score_delta": +20.0, "reaction_time_ms": 240},
        "nlp": {"text": "that next part was actually fun"}
    },
]

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def run_pipeline_demo():
    print("=" * 70)
    print(" =* COGNITIVE REGULATION SYSTEM - END-TO-END DEMO *= ")
    print("=" * 70)
    
    # Initialize the 4-stage pipeline
    pipeline = DataPipeline()
    computer = CIIComputer()
    predictor = StatePredictor()
    engine = DifficultyEngine()
    
    # formatting helper
    def color_val(val, threshold=0): return f"{val:+.3f}"
    
    print(f"\n[ INIT ] Starting difficulty: {engine.difficulty:.1f}/10.0\n")

    for step, event in enumerate(SCENARIO, 1):
        print("-" * 70)
        print(f"STEP {step} | Context: {event['desc']}")
        print(f"Input   : {event['nlp']['text']}")
        print(f"          (Deaths: {event['telemetry']['deaths']} | Score Delta: {event['telemetry']['score_delta']})")
        
        # ── 1. Data Pipeline ───────────────────────────────────────────
        # Validates and normalizes raw inputs into structured features
        features = pipeline.process(event["telemetry"], event["nlp"])
        
        # ── 2. CII Computation ─────────────────────────────────────────
        # Computes mathematical cognitive instability, momentum, and acceleration
        cii_result = computer.compute(
            polarity        = features.polarity,
            intensity       = features.intensity,
            performance_dev = features.perf_dev,
            intent_gap      = features.intent_gap,
        )
        
        # ── 3. State Prediction ────────────────────────────────────────
        # Uses Gaussian scoring / acceleration rules bounds to classify state
        pred_result = predictor.predict(cii_result)
        
        # ── 4. Difficulty Adaptation ───────────────────────────────────
        # Translates predictions into concrete difficulty numbers with cooldowns
        diff_result = engine.adapt(pred_result, cii_result.value)
        
        # ── Printing the Result ────────────────────────────────────────
        
        # 1. Math Output
        print(f"\n[MATH]  CII: {cii_result.value:+.3f}  |  "
              f"Momentum: {cii_result.components.momentum:+.2f}  |  "
              f"Accel: {cii_result.components.acceleration:+.2f}")
              
        # 2. ML Prediction Output
        conf = f"{pred_result.confidence:.1%}"
        state_str = pred_result.state.value.upper()
        if state_str == "FRUSTRATED": state_str = f"[!] {state_str}"
        elif state_str == "BORED":    state_str = f"[-] {state_str}"
        else:                         state_str = f"[+] {state_str}"
        
        print(f"[PRED]  State: {state_str} (Conf: {conf})")
        
        # 3. Engine Action Output
        if diff_result.action.value == "decrease":
            act = f"[-] REDUCE by {abs(diff_result.difficulty_delta):.2f}"
        elif diff_result.action.value == "increase":
            act = f"[+] INCREASE by {diff_result.difficulty_delta:.2f}"
        else:
            act = "[=] MAINTAIN"

        print(f"[CTRL]  Action: {act:<20} -> New Diff: {diff_result.difficulty_level:.2f}")
        
        time.sleep(1.0) # small delay for demo readability

    print("=" * 70)
    print("SESSION COMPLETE")
    print("Summary Analytics:")
    for k, v in engine.session_summary().items():
        print(f"  - {k:<20}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_demo()
