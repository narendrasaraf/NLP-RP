"""
app/core/predictor.py
----------------------
Cognitive state predictor — maps CII(t) and emotional state features
to player state labels and probability estimates.

Two prediction modes:
  1. Rule-Based  : Threshold policy over CII and acceleration (default, no deps)
  2. LSTM-Augmented: Exponentially-weighted CII sequence proxy for LSTM
                     (Replace SimulatedLSTM with trained CIILSTMPredictor
                      from app/models/lstm.py for production)

The predictor outputs:
  player_state    : "frustrated" | "bored" | "engaged"
  frustration_prob: float ∈ [0, 1]
  boredom_prob    : float ∈ [0, 1]
  engagement_prob : float ∈ [0, 1]
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

from app.core.emotional_dynamics import EmotionalState
from app.core.config import PredictorConfig, LSTMConfig, settings


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    player_state:     str    # "frustrated" | "bored" | "engaged"
    frustration_prob: float
    boredom_prob:     float
    engagement_prob:  float


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


# ---------------------------------------------------------------------------
# Simulated LSTM predictor (sequence-aware, no PyTorch dependency)
# Replaces with trained model in production.
# ---------------------------------------------------------------------------

class SimulatedLSTMPredictor:
    """
    Simulates LSTM temporal prediction using exponentially-weighted
    moving average (EWA) + linear trend over CII history.

    Captures the key LSTM insight:
      - Recent steps are weighted more heavily (memory gate analog)
      - Linear slope = temporal gradient across the sequence

    To use the real PyTorch LSTM:
        from app.models.lstm import CIILSTMPredictor
        model = CIILSTMPredictor.load("checkpoints/lstm.pt")
        p_frus, p_bore = model.predict(feature_sequence)
    """

    def __init__(self, cfg: LSTMConfig = None, decay: float = 0.75):
        cfg = cfg or settings.lstm
        self.window  = cfg.seq_len
        self.decay   = decay
        self._history: deque = deque(maxlen=self.window)

    def update(self, state: EmotionalState):
        self._history.append(state.cii)

    def predict(self) -> Tuple[float, float]:
        """Returns (p_frus, p_bore) from EWA + trend over CII history."""
        h = list(self._history)
        n = len(h)
        if n < 2:
            return 0.0, 0.0

        # Exponentially-weighted average
        weights = [self.decay ** (n - 1 - i) for i in range(n)]
        w_sum   = sum(weights)
        ewa_cii = sum(w * c for w, c in zip(weights, h)) / w_sum

        # Linear trend (slope over sequence)
        x_mean = (n - 1) / 2.0
        cov = sum((i - x_mean) * h[i] for i in range(n))
        var = sum((i - x_mean) ** 2 for i in range(n))
        trend = cov / var if var > 0 else 0.0

        p_frus = _sigmoid(-ewa_cii * 3.0 - trend * 2.0)
        p_bore = _sigmoid( ewa_cii * 3.0 + trend * 2.0)

        # Prevent joint probability > 1
        total = p_frus + p_bore + 1e-9
        if total > 1.0:
            p_frus /= total
            p_bore /= total

        return round(p_frus, 4), round(p_bore, 4)

    def reset(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# Rule-based predictor
# ---------------------------------------------------------------------------

class RuleBasedPredictor:
    """
    Threshold policy predictor over CII and Emotional Acceleration.

    Fast, interpretable, and requires no training data.
    Serves as the primary prediction engine in prototype mode and
    as a fallback in production when the LSTM has insufficient history.
    """

    def __init__(self, cfg: PredictorConfig = None):
        self.cfg = cfg or settings.predictor

    def predict(self, state: EmotionalState) -> Tuple[str, float, float, float]:
        """
        Returns (state_label, p_frus, p_bore, p_eng).

        Uses composite score -> softmax for probabilities,
        with an early-warning override on acceleration.
        """
        cii   = state.cii
        accel = state.acceleration

        # Composite raw scores (higher = more likely)
        frus_score = max(0.0, -cii + max(0.0, -accel))
        bore_score = max(0.0,  cii - abs(accel) * 0.5)
        eng_score  = max(0.0, 1.0 - abs(cii) - 0.3 * abs(accel))

        total     = frus_score + bore_score + eng_score + 1e-9
        p_frus    = frus_score / total
        p_bore    = bore_score / total
        p_eng     = eng_score  / total

        # Hard decision (acceleration-aware early warning)
        if cii < self.cfg.frustration_threshold or (
            cii < -0.20 and accel < self.cfg.accel_warning
        ):
            label = "frustrated"
        elif cii > self.cfg.boredom_threshold:
            label = "bored"
        else:
            label = "engaged"

        return label, round(p_frus, 4), round(p_bore, 4), round(p_eng, 4)


# ---------------------------------------------------------------------------
# Blended predictor (rule + LSTM)
# ---------------------------------------------------------------------------

class CognitiveStatePredictor:
    """
    Production predictor that blends:
      60% weight -> RuleBasedPredictor (fast, stable)
      40% weight -> SimulatedLSTMPredictor (temporal awareness)

    The blend is per-session stateful (LSTM maintains its own history).
    """

    RULE_WEIGHT = 0.60
    LSTM_WEIGHT = 0.40

    def __init__(self, cfg: PredictorConfig = None):
        self.rule_predictor = RuleBasedPredictor(cfg)
        self.lstm_predictor = SimulatedLSTMPredictor()

    def predict(self, state: EmotionalState) -> PredictionResult:
        # Rule-based prediction
        rule_label, rule_p_frus, rule_p_bore, rule_p_eng = (
            self.rule_predictor.predict(state)
        )

        # LSTM update + prediction
        self.lstm_predictor.update(state)
        lstm_p_frus, lstm_p_bore = self.lstm_predictor.predict()

        # Blended probabilities
        p_frus = self.RULE_WEIGHT * rule_p_frus + self.LSTM_WEIGHT * lstm_p_frus
        p_bore = self.RULE_WEIGHT * rule_p_bore + self.LSTM_WEIGHT * lstm_p_bore
        p_eng  = max(0.0, 1.0 - p_frus - p_bore)

        # Re-evaluate label from blended probs
        probs = {"frustrated": p_frus, "bored": p_bore, "engaged": p_eng}
        label = max(probs, key=probs.get)

        return PredictionResult(
            player_state     = label,
            frustration_prob = round(p_frus, 4),
            boredom_prob     = round(p_bore, 4),
            engagement_prob  = round(p_eng,  4),
        )

    def reset(self):
        self.lstm_predictor.reset()


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

class PredictorRegistry:
    def __init__(self):
        self._predictors: Dict[str, CognitiveStatePredictor] = {}

    def get_or_create(self, session_id: str) -> CognitiveStatePredictor:
        if session_id not in self._predictors:
            self._predictors[session_id] = CognitiveStatePredictor()
        return self._predictors[session_id]

    def drop_session(self, session_id: str):
        self._predictors.pop(session_id, None)


predictor_registry = PredictorRegistry()
