"""
backend/predictor.py
--------------------
Prediction module mapping CII to discreet Cognitive States.
Currently utilizes a rule-based engine, but cleanly abstracting it 
into a class allows an ML model (e.g. LSTM/Random Forest) to 
eventually be dropped right into the `predict()` method.
"""

class StatePredictor:
    def __init__(self, frustration_threshold: float = 1.0, boredom_threshold: float = -1.0):
        # NOTE: Using the exact rule logic requested:
        # CII > positive threshold = Frustrated
        # CII < negative threshold = Bored
        self.frustration_threshold = frustration_threshold
        self.boredom_threshold = boredom_threshold

    def predict(self, cii: float) -> dict:
        """
        Classifies cognitive state and generates a pseudo-confidence score.
        """
        if cii > self.frustration_threshold:
            state = "Frustrated"
            # Confidence scales up as CII climbs higher past the threshold
            confidence = 0.60 + ((cii - self.frustration_threshold) * 0.15)
            
        elif cii < self.boredom_threshold:
            state = "Bored"
            # Confidence scales up as CII drops deeper past the threshold
            confidence = 0.60 + ((abs(cii) - abs(self.boredom_threshold)) * 0.15)
            
        else:
            state = "Engaged"
            # Max confidence at exactly CII = 0.0, decreases as it approaches thresholds
            max_bound = max(abs(self.frustration_threshold), abs(self.boredom_threshold))
            confidence = 0.95 - ((abs(cii) / max_bound) * 0.45)

        # Clamp confidence strictly between 0.0 and 1.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "state": state,
            "confidence": round(confidence, 3)
        }

# Instantiate a single global instance for the API to use
predictor_engine = StatePredictor(frustration_threshold=1.0, boredom_threshold=-1.0)
