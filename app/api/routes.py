"""
app/api/routes.py
------------------
FastAPI route definitions for the Cognitive Regulation System API.

Endpoints (legacy pipeline — uses app/core modules):
  GET  /health                      -- Health check
  POST /process                     -- Full pipeline, low-level response
  GET  /session/{id}/summary        -- Session aggregate (legacy)
  POST /session/{id}/reset          -- Reset session (legacy)
  DELETE /session/{id}              -- Drop session (legacy)
  GET  /sessions                    -- List active sessions

Endpoints (service layer — uses services/cognitive_service.py):
  POST /predict                     -- Clean, unified predict endpoint
  POST /predict/batch               -- Process multiple steps at once
  GET  /session/{id}/stats          -- Richer session analytics
  POST /session/{id}/service-reset  -- Reset service-layer session
"""

from fastapi import APIRouter, HTTPException
from typing import List

from app.api.schemas import (
    # Legacy pipeline schemas
    SessionInput,
    PredictionResponse,
    SessionSummaryResponse,
    HealthResponse,
    EmotionalStateResponse,
    # Service-layer schemas
    PredictRequest,
    PredictResponse,
    CIIDetail,
    PredictionDetail,
    DifficultyDetail,
    SessionStatsResponse,
)
from app.core.telemetry          import telemetry_registry
from app.core.nlp                import nlp_registry
from app.core.emotional_dynamics import dynamics_registry
from app.core.predictor          import predictor_registry
from app.core.controller         import controller_registry
from services.cognitive_service  import cognitive_service

router = APIRouter()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse()


# ---------------------------------------------------------------------------
# Core processing endpoint
# ---------------------------------------------------------------------------

@router.post("/process", response_model=PredictionResponse, tags=["Prediction"])
def process_timestep(payload: SessionInput):
    """
    Process one game timestep for the given session.

    Pipeline:
      1. Telemetry -> D(t)
      2. NLP       -> P(t), I(t), G(t)
      3. Dynamics  -> E(t), M(t), A(t), CII(t)
      4. Predictor -> player_state, probabilities
      5. Controller-> difficulty action + reward

    Returns full PredictionResponse with all computed values.
    """
    sid = payload.session_id

    # ── 1. Telemetry Processing ────────────────────────────────────────────
    tracker  = telemetry_registry.get_or_create(sid)
    perf_dev = tracker.update(payload.telemetry)

    # ── 2. NLP Feature Extraction ──────────────────────────────────────────
    extractor = nlp_registry.get_or_create(sid)
    nlp_inp   = payload.nlp or __import__(
        "app.api.schemas", fromlist=["NLPInput"]
    ).NLPInput()
    nlp_feats = extractor.extract(nlp_inp)

    # ── 3. Emotional Dynamics Engine ───────────────────────────────────────
    engine = dynamics_registry.get_or_create(sid)
    state  = engine.process(
        polarity        = nlp_feats["polarity"],
        intensity       = nlp_feats["intensity"],
        performance_dev = perf_dev,
        intent_gap      = nlp_feats["intent_gap"],
    )

    # ── 4. State Prediction ────────────────────────────────────────────────
    predictor  = predictor_registry.get_or_create(sid)
    prediction = predictor.predict(state)

    # ── 5. Adaptive Difficulty Control ────────────────────────────────────
    controller = controller_registry.get_or_create(sid)
    ctrl_out   = controller.step(prediction, state)

    # ── Build response ─────────────────────────────────────────────────────
    return PredictionResponse(
        session_id        = sid,
        timestep          = engine.timestep,
        emotional_state   = EmotionalStateResponse(
            polarity        = state.polarity,
            intensity       = state.intensity,
            performance_dev = state.performance_dev,
            intent_gap      = state.intent_gap,
            momentum        = state.momentum,
            acceleration    = state.acceleration,
            cii             = state.cii,
        ),
        player_state      = prediction.player_state,
        frustration_prob  = prediction.frustration_prob,
        boredom_prob      = prediction.boredom_prob,
        engagement_prob   = prediction.engagement_prob,
        cii               = state.cii,
        difficulty_action = ctrl_out.action,
        difficulty_level  = ctrl_out.difficulty_level,
        difficulty_delta  = ctrl_out.difficulty_delta,
    )


# ---------------------------------------------------------------------------
# Session analytics
# ---------------------------------------------------------------------------

@router.get(
    "/session/{session_id}/summary",
    response_model=SessionSummaryResponse,
    tags=["Analytics"],
)
def session_summary(session_id: str):
    """Return aggregate statistics for a session."""
    summary = controller_registry.get_summary(session_id)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or has no recorded steps."
        )
    return SessionSummaryResponse(session_id=session_id, **summary)


@router.get("/sessions", response_model=List[str], tags=["Analytics"])
def list_sessions():
    """List all active session IDs."""
    return telemetry_registry.active_sessions()


# ---------------------------------------------------------------------------
# Session lifecycle management
# ---------------------------------------------------------------------------

@router.post("/session/{session_id}/reset", tags=["Session"])
def reset_session(session_id: str):
    """Reset all state for a session (restart tracking)."""
    telemetry_registry.get_or_create(session_id).reset()
    nlp_registry.get_or_create(session_id).reset()
    dynamics_registry.get_or_create(session_id).reset()
    predictor_registry.get_or_create(session_id).reset()
    controller_registry.get_or_create(session_id).reset()
    return {"status": "reset", "session_id": session_id}


@router.delete("/session/{session_id}", tags=["Session"])
def drop_session(session_id: str):
    """Drop all state for a session (free memory)."""
    telemetry_registry.drop_session(session_id)
    nlp_registry.drop_session(session_id)
    dynamics_registry.drop_session(session_id)
    predictor_registry.drop_session(session_id)
    controller_registry.drop_session(session_id)
    return {"status": "dropped", "session_id": session_id}


# ===========================================================================
# SERVICE-LAYER ENDPOINTS  (use services/cognitive_service.py)
# ===========================================================================

def _build_predict_response(result) -> PredictResponse:
    """Convert PredictResult dataclass -> PredictResponse Pydantic model."""
    return PredictResponse(
        session_id  = result.session_id,
        timestep    = result.timestep,
        timestamp   = result.timestamp,
        cii         = CIIDetail(
            value        = result.cii.value,
            level        = result.cii.level,
            zone         = result.cii.zone,
            momentum     = result.cii.momentum,
            acceleration = result.cii.acceleration,
            perf_dev     = result.cii.perf_dev,
            intent_gap   = result.cii.intent_gap,
            polarity     = result.cii.polarity,
            intensity    = result.cii.intensity,
            contribution = result.cii.contribution,
        ),
        prediction  = PredictionDetail(
            state              = result.prediction.state,
            confidence         = result.prediction.confidence,
            frustration_score  = result.prediction.scores.get("frustrated", 0.0),
            engagement_score   = result.prediction.scores.get("engaged",    0.0),
            boredom_score      = result.prediction.scores.get("bored",      0.0),
            recommended_action = result.prediction.recommended_action,
            reasoning          = result.prediction.reasoning,
        ),
        difficulty  = DifficultyDetail(
            action      = result.difficulty.action,
            level       = result.difficulty.level,
            delta       = result.difficulty.delta,
            prev_level  = result.difficulty.prev_level,
            reward      = result.difficulty.reward,
            on_cooldown = result.difficulty.on_cooldown,
        ),
        recommendation = result.recommendation,
        warnings       = result.warnings,
    )


@router.post(
    "/predict",
    response_model = PredictResponse,
    tags           = ["Predict"],
    summary        = "Predict cognitive state and adjust difficulty",
    description    = """
Main prediction endpoint. Accepts player telemetry + NLP input,
runs the full 4-stage pipeline, and returns:
  - **CII** — Cognitive Instability Index with full decomposition
  - **prediction** — player state (frustrated/engaged/bored) + confidence
  - **difficulty** — recommended action + new level + RL reward
  - **recommendation** — plain-English summary string

All telemetry fields are optional and default to neutral values.
Provide either `nlp.text` (auto-processed) or `nlp.polarity`/`intensity`
(pre-computed from your own model).
""",
)
def predict(payload: PredictRequest) -> PredictResponse:
    """
    Example request::

        POST /api/v1/predict
        {
            "session_id": "player-042-run-1",
            "telemetry": {
                "deaths": 5, "retries": 4, "score_delta": -20.0,
                "streak": -3, "reaction_time_ms": 450, "input_speed": 2.1
            },
            "nlp": {
                "text": "this is impossible", "intent_gap": 0.6
            }
        }

    Example response::

        {
            "session_id": "player-042-run-1",
            "timestep": 1,
            "cii": { "value": -0.47, "level": "high_instability", "zone": "frustration", ... },
            "prediction": { "state": "frustrated", "confidence": 0.87, ... },
            "difficulty": { "action": "decrease", "level": 4.31, "delta": -0.69, ... },
            "recommendation": "WARNING: Reduce difficulty to 4.3 — Frustrated detected (high confidence)"
        }
    """
    result = cognitive_service.predict(
        session_id = payload.session_id,
        payload    = payload.model_dump(),
    )
    return _build_predict_response(result)


@router.post(
    "/predict/batch",
    response_model = List[PredictResponse],
    tags           = ["Predict"],
    summary        = "Process multiple timesteps in sequence",
    description    = "Accepts a list of timestep payloads and returns predictions in order. "
                     "Temporal state (acceleration, difficulty) is maintained across steps.",
)
def predict_batch(payloads: List[PredictRequest]) -> List[PredictResponse]:
    responses = []
    for payload in payloads:
        result = cognitive_service.predict(
            session_id = payload.session_id,
            payload    = payload.model_dump(),
        )
        responses.append(_build_predict_response(result))
    return responses


@router.get(
    "/session/{session_id}/stats",
    response_model = SessionStatsResponse,
    tags           = ["Analytics"],
    summary        = "Rich session analytics (service layer)",
)
def session_stats(session_id: str) -> SessionStatsResponse:
    """Returns aggregate analytics including CII trend, flow rate, reward history."""
    stats = cognitive_service.get_summary(session_id)
    if not stats:
        raise HTTPException(
            status_code = 404,
            detail      = f"Session '{session_id}' not found or has no steps.",
        )
    return SessionStatsResponse(**vars(stats))


@router.post(
    "/session/{session_id}/service-reset",
    tags    = ["Session"],
    summary = "Reset service-layer session state",
)
def service_reset(session_id: str):
    """Resets all state for the service-layer pipeline (CII history, predictor, engine)."""
    cognitive_service.reset_session(session_id)
    return {"status": "reset", "session_id": session_id, "layer": "service"}
