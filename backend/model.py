"""
backend/model.py
----------------
Math engine that computes CII and predicts the resulting player state.
"""

from backend.utils import normalize_value

# Global variable to track state securely between API requests for this simple demo
_previous_momentum = 0.0

def compute_cii(telemetry: dict, nlp: dict):
    global _previous_momentum
    
    # 1. Emotional Momentum: M = polarity * emotional_intensity
    polarity  = nlp.get("polarity", 0.0)
    intensity = nlp.get("emotional_intensity", 0.0)
    M = polarity * intensity
    
    # 2. Emotional Acceleration: A = current M - previous M
    A = M - _previous_momentum
    
    # 3. Performance Deviation: D = death_count + miss_count - kill_count
    deaths = telemetry.get("death_count", 0)
    misses = telemetry.get("miss_count", 0)
    kills  = telemetry.get("kill_count", 0)
    D = deaths + misses - kills
    
    # 4. Intent Gap (G): Mismatch between NLP confidence and actual performance.
    # If confidence is high but performance is poor (high D), the cognitive friction (G) is high.
    confidence = nlp.get("confidence_score", 0.5)
    G = confidence * max(0.0, D)
    
    # 5. Compute: CII = weighted sum
    # High M/A (positive) drive engagement/boredom, but high Intent Gap drives instability (negative).
    cii = (0.4 * M) + (0.3 * A) + (0.2 * D) - (0.2 * G)
    
    # Ensure previous state is stored correctly for the next API call
    _previous_momentum = M
    
    # Interpretation (low, medium, high) based on magnitude of instability
    abs_cii = abs(cii)
    if abs_cii < 1.0:
        interpretation = "low"
    elif abs_cii < 3.0:
        interpretation = "medium"
    else:
        interpretation = "high"
        
    return round(cii, 3), interpretation

from backend.predictor import predictor_engine

def predict_state_and_action(telemetry: dict, nlp: dict) -> dict:
    """
    End-to-end pipeline: Input -> CII -> Classification -> Action.
    Returns the exact dictionary shape required by the game.
    """
    # 1. Compute Math Layer
    cii, interpretation = compute_cii(telemetry, nlp)
    
    # 2. Predict State (Classification) via ML-ready module
    pred_result = predictor_engine.predict(cii)
    state = pred_result["state"]
    confidence = pred_result["confidence"]
        
    # 3. Formulate Action Policy
    if state == "Frustrated":
        action = "decrease"
    elif state == "Bored":
        action = "increase"
    else:
        action = "maintain"
        
    return {
        "cii": cii,
        "interpretation": interpretation,
        "state": state,
        "action": action,
        "confidence": confidence
    }
