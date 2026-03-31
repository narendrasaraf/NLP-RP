"""
backend/main.py
---------------
FastAPI lightweight entry point for the Cognitive Regulation game AI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Optional

from backend.model import predict_state_and_action
from backend.nlp_module import analyzer

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
    
    # Check if a raw chat message was provided
    if payload.chat and payload.chat.strip():
        # Call nlp_module to extract features dynamically
        features = analyzer.process_chat(payload.chat)
        nlp_dict = {
            "polarity": features["polarity"],
            "emotional_intensity": features["intensity"],
            "anger_score": features["anger"],
            "frustration_score": features["frustration"],
            "confidence_score": features["confidence"]
        }
    else:
        # Fallback to existing nlp payload
        nlp_dict = payload.nlp.model_dump()
        
    global latest_data
    # Run the classification and adaptation layer
    # Final input to model: { telemetry, nlp_features }
    result = predict_state_and_action(telemetry_dict, nlp_dict)
    
    # 1. Store latest request + response
    latest_data = {
        "telemetry": telemetry_dict,
        "chat": payload.chat if payload.chat else "",
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
        
    # --- Backend Logging --- 
    if payload.chat and payload.chat.strip():
        scores = {
            "frustration": nlp_dict.get("frustration_score", 0.0),
            "anger": nlp_dict.get("anger_score", 0.0),
            "confidence": nlp_dict.get("confidence_score", 0.0)
        }
        # Find dominant emotion
        detected_emotion = max(scores, key=scores.get) if max(scores.values()) > 0.0 else "neutral"
        
        print("\n" + "═"*40)
        print(f"Chat: {payload.chat}")
        print(f"Polarity: {nlp_dict['polarity']} | Intensity: {nlp_dict['emotional_intensity']}")
        print(f"Detected Emotion: {detected_emotion}")
        print(f"CII: {result['cii']}")
        print("═"*40 + "\n")
    
    return result

# You can run this directly with `uvicorn backend.main:app --reload --port 8001`
