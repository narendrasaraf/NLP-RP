"""
backend/main.py
---------------
FastAPI lightweight entry point for the Cognitive Regulation game AI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from backend.model import predict_state_and_action

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
    polarity: float
    emotional_intensity: float

class PredictRequest(BaseModel):
    telemetry: Telemetry
    nlp: NLP

class PredictResponse(BaseModel):
    cii: float
    interpretation: Literal["low", "medium", "high"]
    state: Literal["Frustrated", "Engaged", "Bored"]
    action: Literal["increase", "decrease", "maintain"]

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
    Main pipeline exposed via REST API.
    Awaits the game's telemetry dictionary every X seconds.
    """
    # Convert Pydantic models to dicts for the math engine
    telemetry_dict = payload.telemetry.model_dump()
    nlp_dict = payload.nlp.model_dump()
    
    global latest_data
    # Run the classification and adaptation layer
    result = predict_state_and_action(telemetry_dict, nlp_dict)
    
    # 1. Store latest request + response
    latest_data = {
        "telemetry": telemetry_dict,
        "nlp": nlp_dict,
        "cii": result["cii"],
        "state": result["state"],
        "action": result["action"],
        "confidence": result.get("confidence", 0.0)
    }
    
    # 2. Maintain history list of CII and State
    history_log.append({
        "cii": result["cii"],
        "state": result["state"],
        
        # Kept these to ensure the Streamlit app can still render properly
        "action": result["action"],
        "telemetry": telemetry_dict,
        "nlp": nlp_dict
    })
    
    # Cap memory passively to prevent crashes if left running
    if len(history_log) > 500:
        history_log.pop(0)
    
    return result

# You can run this directly with `uvicorn backend.main:app --reload --port 8001`
