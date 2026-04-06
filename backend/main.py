"""
backend/main.py
---------------
FastAPI entry point for the Cognitive Regulation game AI.

Pipeline: chat → NLP → temporal → telemetry → CII → prediction
All processing is delegated to backend/pipeline.py.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Optional

# Single import — replaces backend.model + backend.nlp_module call chains
from backend.pipeline import pipeline as cognitive_pipeline

app = FastAPI(
    title="Simple CogReg API", 
    description="Standalone inference server for the 2D Shooter Game"
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class Telemetry(BaseModel):
    kill_count: int
    death_count: int
    miss_count: int
    reaction_time: float
    score: int

class NLP(BaseModel):
    polarity: float = 0.0
    emotional_intensity: float = 0.0

class PredictRequest(BaseModel):
    telemetry: Telemetry
    chat: Optional[str] = None
    nlp: Optional[NLP] = NLP()

class PredictResponse(BaseModel):
    cii:            float
    interpretation: Literal["low", "medium", "high"]
    state:          Literal["Frustrated", "Engaged", "Bored"]
    action:         Literal["increase", "decrease", "maintain"]
    confidence:     float = 0.0

# ---------------------------------------------------------------------------
# In-Memory State for Live Dashboard
# ---------------------------------------------------------------------------
latest_data = {}
history_log = []

@app.get("/latest")
def get_latest():
    """Returns the most recent payload and ML classification."""
    return latest_data

@app.get("/history")
def get_history():
    """Returns the entire rolling sequence of CII values and predicted states."""
    return {"history": history_log}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict_cognitive_state(payload: PredictRequest):
    """
    Full cognitive pipeline: chat → NLP → temporal → telemetry → CII → prediction.
    All processing is delegated to CognitivePipeline in backend/pipeline.py.
    """
    global latest_data

    print(f"[DEBUG] Started pipeline run for tick...")
    # ── Run the full pipeline (one call replaces ~50 lines of scattered logic) ──
    result = cognitive_pipeline.run_full({
        "chat":      payload.chat,
        "telemetry": payload.telemetry.model_dump(),
    })

    print(f"[DEBUG] Pipeline run finished. CII: {result['cii']}")

    # ── Convenience aliases ────────────────────────────────────────────────────
    nlp_feats  = result["features"]["nlp"]
    telem_feat = result["features"]["telemetry"]

    # ── Update live dashboard store ────────────────────────────────────────────
    latest_data = {
        "chat":       nlp_feats["chat"],
        "cii":        result["cii"],
        "state":      result["state"],
        "action":     result["action"],
        "confidence": result["confidence"],
        "level":      result["level"],
        # NLP signals (kept for Streamlit dashboard compatibility)
        "nlp": {
            "polarity":            nlp_feats["polarity"],
            "emotional_intensity": nlp_feats["intensity"],
            "frustration_score":   nlp_feats["frustration"],
            "anger_score":         nlp_feats["anger"],
            "confidence_score":    nlp_feats["confidence"],
        },
        # Telemetry snapshot (kept for Streamlit dashboard compatibility)
        "telemetry": {
            "kill_count":   telem_feat["kill_count"],
            "death_count":  telem_feat["death_count"],
            "miss_count":   telem_feat["miss_count"],
            "reaction_time": telem_feat["reaction_time_ms"],
        },
    }

    # ── Append to history log ──────────────────────────────────────────────────
    history_log.append({
        "cii":        result["cii"],
        "state":      result["state"],
        "action":     result["action"],
        "confidence": result["confidence"],
        "telemetry":  latest_data["telemetry"],
        "nlp":        latest_data["nlp"],
    })
    if len(history_log) > 500:
        history_log.pop(0)

    # ── Console logging ────────────────────────────────────────────────────────
    if nlp_feats["chat"]:
        emotion_scores = {
            "frustration": nlp_feats["frustration"],
            "anger":       nlp_feats["anger"],
            "confidence":  nlp_feats["confidence"],
        }
        dominant = (
            max(emotion_scores, key=emotion_scores.get)
            if max(emotion_scores.values()) > 0.0
            else "neutral"
        )
        print("\n" + "="*44)
        print(f"  Chat     : {nlp_feats['chat']}")
        print(f"  Polarity : {nlp_feats['polarity']:+.3f}  "
              f"Intensity: {nlp_feats['intensity']:.3f}")
        print(f"  Emotion  : {dominant}")
        print(f"  CII      : {result['cii']:+.4f}  [{result['level'].upper()}]")
        print(f"  State    : {result['state']}  (conf={result['confidence']:.3f})")
        print("="*44 + "\n")

    # ── Return API response (PredictResponse shape) ────────────────────────────
    return {
        "cii":            result["cii"],
        "interpretation": result["level"],
        "state":          result["state"],
        "action":         result["action"],
        "confidence":     result["confidence"],
    }

# Run with: uvicorn backend.main:app --reload --port 8001
