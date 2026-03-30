"""
services/cognitive_service.py
-------------------------------
CognitiveService — single-call orchestration layer.

Combines utils/preprocessing, utils/cii, utils/predictor,
and utils/difficulty_engine into one stateful per-session pipeline.

This is the ONLY place the API layer needs to import ML logic from.
Routes call service methods; they never touch utils/ directly.

Design:
  - One CognitiveService instance per server lifetime
  - Per-session state managed internally via session_id keys
  - All utils modules instantiated and held here
  - Exposes three public methods:
      .predict(session_id, payload) -> PredictResult
      .get_summary(session_id)      -> SessionStats
      .reset_session(session_id)
      .drop_session(session_id)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from utils.preprocessing   import DataPipeline
from utils.cii             import CIIComputer, CIIHistory, InstabilityLevel
from utils.predictor       import StatePredictor, PredictionHistory
from utils.difficulty_engine import DifficultyEngine


# ---------------------------------------------------------------------------
# Service-level data types (decoupled from Pydantic/FastAPI)
# ---------------------------------------------------------------------------

@dataclass
class CIIOutput:
    value:        float
    level:        str
    zone:         str
    momentum:     float
    acceleration: float
    perf_dev:     float
    intent_gap:   float
    polarity:     float
    intensity:    float
    contribution: Dict[str, float]


@dataclass
class PredictionOutput:
    state:           str
    confidence:      float
    scores:          Dict[str, float]
    recommended_action: str
    reasoning:       List[str]


@dataclass
class DifficultyOutput:
    action:          str
    level:           float
    delta:           float
    prev_level:      float
    reward:          float
    on_cooldown:     bool


@dataclass
class PredictResult:
    """
    Full pipeline output for one /predict call.
    All fields are plain Python — Pydantic serialises at the API boundary.
    """
    session_id:   str
    timestep:     int
    timestamp:    float
    cii:          CIIOutput
    prediction:   PredictionOutput
    difficulty:   DifficultyOutput
    recommendation: str
    warnings:     List[str]


@dataclass
class SessionStats:
    session_id:         str
    total_steps:        int
    avg_cii:            float
    min_cii:            float
    max_cii:            float
    std_cii:            float
    flow_rate:          float
    frustration_events: int
    boredom_events:     int
    acute_events:       int
    engagement_rate:    float
    avg_confidence:     float
    avg_reward:         float
    total_reward:       float
    final_difficulty:   float
    avg_difficulty:     float
    interventions:      int
    difficulty_changes: int
    dominant_cii_driver: str
    trend:              str


# ---------------------------------------------------------------------------
# Per-session state bundle
# ---------------------------------------------------------------------------

@dataclass
class _SessionState:
    pipeline:   DataPipeline
    cii_comp:   CIIComputer
    predictor:  StatePredictor
    engine:     DifficultyEngine
    cii_hist:   CIIHistory
    pred_hist:  PredictionHistory
    timestep:   int = 0


# ---------------------------------------------------------------------------
# CognitiveService
# ---------------------------------------------------------------------------

class CognitiveService:
    """
    Stateful per-session cognitive regulation service.

    Usage (in FastAPI route):
        service = CognitiveService()           # created once at app startup

        @router.post("/predict")
        def predict(payload: PredictRequest):
            result = service.predict(payload.session_id, payload.dict())
            return result_to_response(result)
    """

    def __init__(self):
        self._sessions: Dict[str, _SessionState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, session_id: str, payload: dict) -> PredictResult:
        """
        Run the full 4-stage pipeline for one player timestep.

        Pipeline:
            1. preprocessing  -> validated + normalised feature vector
            2. cii.py         -> CII(t), M(t), A(t), components
            3. predictor.py   -> state label + confidence + per-class scores
            4. difficulty_engine.py -> action + new difficulty + reward

        Args:
            session_id : Unique player/session string
            payload    : dict with keys 'telemetry' and 'nlp'
                         (same structure as PredictRequest Pydantic model)

        Returns:
            PredictResult dataclass — converted to JSON at API boundary
        """
        sess = self._get_or_create(session_id)
        sess.timestep += 1

        telemetry = payload.get("telemetry", {})
        nlp       = payload.get("nlp", {})

        # ── Stage 1: Data pipeline ────────────────────────────────────────
        feature_vec = sess.pipeline.process(telemetry, nlp)

        # ── Stage 2: CII computation ──────────────────────────────────────
        cii_result = sess.cii_comp.compute(
            polarity        = feature_vec.polarity,
            intensity       = feature_vec.intensity,
            performance_dev = feature_vec.perf_dev,
            intent_gap      = feature_vec.intent_gap,
        )
        sess.cii_hist.record(cii_result)

        # ── Stage 3: State prediction ─────────────────────────────────────
        pred_result = sess.predictor.predict(cii_result)
        sess.pred_hist.record(pred_result)

        # ── Stage 4: Difficulty adaptation ───────────────────────────────
        diff_result = sess.engine.adapt(pred_result, cii_result.value)

        # ── Build recommendation string ───────────────────────────────────
        recommendation = self._make_recommendation(
            cii_result.level.value,
            pred_result.state.value,
            pred_result.confidence,
            diff_result.action.value,
            diff_result.difficulty_level,
        )

        return PredictResult(
            session_id = session_id,
            timestep   = sess.timestep,
            timestamp  = time.time(),
            cii        = CIIOutput(
                value        = cii_result.value,
                level        = cii_result.level.value,
                zone         = cii_result.zone.value,
                momentum     = cii_result.components.momentum,
                acceleration = cii_result.components.acceleration,
                perf_dev     = cii_result.components.performance_dev,
                intent_gap   = cii_result.components.intent_gap,
                polarity     = cii_result.components.polarity,
                intensity    = cii_result.components.intensity,
                contribution = cii_result.contribution,
            ),
            prediction = PredictionOutput(
                state              = pred_result.state.value,
                confidence         = pred_result.confidence,
                scores             = pred_result.scores,
                recommended_action = pred_result.recommended_action,
                reasoning          = pred_result.reasoning,
            ),
            difficulty = DifficultyOutput(
                action      = diff_result.action.value,
                level       = diff_result.difficulty_level,
                delta       = diff_result.difficulty_delta,
                prev_level  = diff_result.prev_difficulty,
                reward      = diff_result.reward,
                on_cooldown = diff_result.on_cooldown,
            ),
            recommendation = recommendation,
            warnings       = feature_vec.warnings,
        )

    def get_summary(self, session_id: str) -> Optional[SessionStats]:
        """Return aggregate session analytics. None if session not found."""
        sess = self._sessions.get(session_id)
        if not sess or sess.timestep == 0:
            return None

        cii_sum  = sess.cii_hist.summary()
        pred_sum = sess.pred_hist.summary()
        eng_sum  = sess.engine.session_summary()

        return SessionStats(
            session_id          = session_id,
            total_steps         = sess.timestep,
            avg_cii             = cii_sum.get("mean_cii",            0.0),
            min_cii             = cii_sum.get("min_cii",             0.0),
            max_cii             = cii_sum.get("max_cii",             0.0),
            std_cii             = cii_sum.get("std_cii",             0.0),
            flow_rate           = cii_sum.get("flow_rate",           0.0),
            frustration_events  = cii_sum.get("frustration_events",  0),
            boredom_events      = cii_sum.get("boredom_events",      0),
            acute_events        = cii_sum.get("acute_events",        0),
            engagement_rate     = pred_sum.get("engagement_rate",    0.0),
            avg_confidence      = pred_sum.get("avg_confidence",     0.0),
            avg_reward          = eng_sum.get("avg_reward",          0.0),
            total_reward        = eng_sum.get("total_reward",        0.0),
            final_difficulty    = eng_sum.get("final_difficulty",    5.0),
            avg_difficulty      = eng_sum.get("avg_difficulty",      5.0),
            interventions       = eng_sum.get("interventions",       0),
            difficulty_changes  = (eng_sum.get("decreases", 0) +
                                   eng_sum.get("increases", 0)),
            dominant_cii_driver = cii_sum.get("dominant_driver",    "n/a"),
            trend               = sess.cii_hist.trend(),
        )

    def reset_session(self, session_id: str):
        """Reset all temporal state for a session (keep session alive)."""
        sess = self._sessions.get(session_id)
        if sess:
            sess.pipeline.reset()
            sess.cii_comp.reset()
            sess.engine.reset()
            sess.pred_hist.clear()
            sess.cii_hist.clear()
            sess.timestep = 0

    def drop_session(self, session_id: str):
        """Free all memory for a session."""
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, session_id: str) -> _SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = _SessionState(
                pipeline  = DataPipeline(),
                cii_comp  = CIIComputer(),
                predictor = StatePredictor(),
                engine    = DifficultyEngine(),
                cii_hist  = CIIHistory(),
                pred_hist = PredictionHistory(),
            )
        return self._sessions[session_id]

    @staticmethod
    def _make_recommendation(
        cii_level:   str,
        state:       str,
        confidence:  float,
        action:      str,
        difficulty:  float,
    ) -> str:
        """Generate a human-readable action recommendation string."""
        conf_label = (
            "high confidence" if confidence > 0.75
            else "moderate confidence" if confidence > 0.50
            else "low confidence"
        )
        action_desc = {
            "decrease": f"Reduce difficulty to {difficulty:.1f}",
            "maintain": f"Maintain difficulty at {difficulty:.1f}",
            "increase": f"Increase difficulty to {difficulty:.1f}",
        }.get(action, f"Adjust to {difficulty:.1f}")

        urgency = {
            "acute_frustration": "URGENT: ",
            "high_instability":  "WARNING: ",
            "mild_instability":  "MONITOR: ",
            "stable_flow":       "",
            "boredom_risk":      "NOTICE: ",
        }.get(cii_level, "")

        return (
            f"{urgency}{action_desc} — "
            f"{state.capitalize()} detected ({conf_label})"
        )


# ---------------------------------------------------------------------------
# Singleton — shared across all FastAPI requests
# ---------------------------------------------------------------------------

cognitive_service = CognitiveService()
