"""
app/api/schemas.py
------------------
Pydantic models for FastAPI request validation and response serialization.
All API data contracts are defined here.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


# ---------------------------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------------------------

class TelemetryInput(BaseModel):
    """Raw gameplay telemetry for one timestep."""
    deaths:           int   = Field(..., ge=0,   description="Deaths in current window")
    retries:          int   = Field(..., ge=0,   description="Retry count in current window")
    score_delta:      float = Field(...,          description="Score change this window")
    streak:           int   = Field(0,            description="Positive=win streak, negative=lose streak")
    reaction_time_ms: float = Field(..., gt=0,   description="Average reaction latency in ms")
    input_speed:      float = Field(..., ge=0,   description="Actions per second")


class NLPInput(BaseModel):
    """Player-generated text input (optional — may be absent in silent sessions)."""
    text:         str   = Field("", description="Raw player chat or voice-to-text")
    polarity:     float = Field(0.0, ge=-1.0, le=1.0,  description="Pre-computed polarity if available")
    intensity:    float = Field(0.0, ge=0.0,  le=1.0,  description="Pre-computed intensity if available")
    intent_gap:   float = Field(0.0, ge=0.0,  le=1.0,  description="Pre-computed intent-outcome gap")

    @field_validator("polarity")
    @classmethod
    def polarity_range(cls, v):
        return max(-1.0, min(1.0, v))


class SessionInput(BaseModel):
    """Full input snapshot for one timestep — sent to /process endpoint."""
    session_id:  str            = Field(..., description="Unique session identifier")
    timestamp:   float          = Field(..., description="Unix timestamp")
    telemetry:   TelemetryInput
    nlp:         Optional[NLPInput] = None


class BatchSessionInput(BaseModel):
    """Batch of timesteps for sequence-based LSTM prediction."""
    session_id:  str
    snapshots:   List[SessionInput] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# INTERNAL STATE SCHEMAS (also used as response sub-objects)
# ---------------------------------------------------------------------------

class EmotionalStateResponse(BaseModel):
    """E(t) components and derived signals."""
    polarity:        float
    intensity:       float
    performance_dev: float
    intent_gap:      float
    momentum:        float   # M(t) = P(t) * I(t)
    acceleration:    float   # A(t) = DeltaM / Delta_t
    cii:             float   # CII(t)


# ---------------------------------------------------------------------------
# RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Full system output for one processed timestep."""
    session_id:       str
    timestep:         int
    emotional_state:  EmotionalStateResponse
    player_state:     str   = Field(..., description="frustrated | bored | engaged")
    frustration_prob: float = Field(..., ge=0.0, le=1.0)
    boredom_prob:     float = Field(..., ge=0.0, le=1.0)
    engagement_prob:  float = Field(..., ge=0.0, le=1.0)
    cii:              float
    difficulty_action: str  = Field(..., description="decrease | maintain | increase")
    difficulty_level:  float
    difficulty_delta:  float


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    message: str = "Cognitive Regulation API is running"


class SessionSummaryResponse(BaseModel):
    """Aggregate statistics for a completed session."""
    session_id:          str
    total_steps:         int
    avg_cii:             float
    min_cii:             float
    max_cii:             float
    frustration_events:  int    # Steps where state == "frustrated"
    boredom_events:      int
    flow_rate:           float  # Fraction of steps in flow state
    final_difficulty:    float
    difficulty_changes:  int


# ---------------------------------------------------------------------------
# /predict endpoint — unified, service-layer schemas
# ---------------------------------------------------------------------------

class PredictTelemetry(BaseModel):
    """
    Gameplay telemetry for /predict.
    All fields optional — missing values get safe defaults.
    """
    deaths:           int   = Field(0,     ge=0,  description="Deaths this window")
    retries:          int   = Field(0,     ge=0,  description="Retry count this window")
    score_delta:      float = Field(0.0,          description="Score change (signed)")
    streak:           int   = Field(0,            description="+ve win streak, -ve lose streak")
    reaction_time_ms: float = Field(300.0, gt=0,  description="Avg reaction latency (ms)")
    input_speed:      float = Field(3.0,   ge=0,  description="Actions per second")

    model_config = {
        "json_schema_extra": {
            "example": {
                "deaths": 5, "retries": 4, "score_delta": -20.0,
                "streak": -3, "reaction_time_ms": 450, "input_speed": 2.1
            }
        }
    }


class PredictNLP(BaseModel):
    """
    NLP / affective signals for /predict.
    Provide either 'text' (processed here) or pre-computed polarity/intensity.
    """
    text:       str   = Field("",  description="Raw player chat or voice-to-text")
    polarity:   float = Field(0.0, ge=-1.0, le=1.0, description="Override polarity")
    intensity:  float = Field(0.0, ge=0.0,  le=1.0, description="Override intensity")
    intent_gap: float = Field(0.0, ge=0.0,  le=1.0, description="Intent-outcome gap G(t)")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "this is impossible", "intent_gap": 0.6}
        }
    }


class PredictRequest(BaseModel):
    """
    Request body for POST /predict.

    Minimal required fields: session_id + telemetry.
    All telemetry fields have safe defaults so partial payloads are valid.
    """
    session_id: str          = Field(..., description="Unique player / session identifier")
    telemetry:  PredictTelemetry = PredictTelemetry()
    nlp:        PredictNLP       = PredictNLP()

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "player-042-run-1",
                "telemetry": {
                    "deaths": 5, "retries": 4, "score_delta": -20.0,
                    "streak": -3, "reaction_time_ms": 450, "input_speed": 2.1
                },
                "nlp": {
                    "text": "this is impossible", "intent_gap": 0.6
                }
            }
        }
    }


class CIIDetail(BaseModel):
    """CII full decomposition in /predict response."""
    value:        float
    level:        str   = Field(..., description="acute_frustration | high_instability | mild_instability | stable_flow | boredom_risk")
    zone:         str   = Field(..., description="frustration | flow | boredom")
    momentum:     float
    acceleration: float
    perf_dev:     float
    intent_gap:   float
    polarity:     float
    intensity:    float
    contribution: dict  = Field(default_factory=dict, description="Per-component weighted contribution")


class PredictionDetail(BaseModel):
    """Prediction sub-object in /predict response."""
    state:             str   = Field(..., description="frustrated | engaged | bored")
    confidence:        float = Field(..., ge=0.0, le=1.0)
    frustration_score: float
    engagement_score:  float
    boredom_score:     float
    recommended_action: str
    reasoning:         List[str]


class DifficultyDetail(BaseModel):
    """Difficulty adjustment sub-object in /predict response."""
    action:      str   = Field(..., description="decrease | maintain | increase")
    level:       float = Field(..., description="New difficulty level [1–10]")
    delta:       float = Field(..., description="Signed change applied")
    prev_level:  float
    reward:      float = Field(..., description="RL reward signal r(t)")
    on_cooldown: bool  = Field(..., description="True if step was dampened by cooldown guard")


class PredictResponse(BaseModel):
    """
    Response body for POST /predict.

    Contains three sub-objects:
      cii        — full CII decomposition
      prediction — state label + confidence + per-class scores
      difficulty — action taken + new level + RL reward

    Plus a plain-English recommendation string.
    """
    session_id:     str
    timestep:       int
    timestamp:      float
    cii:            CIIDetail
    prediction:     PredictionDetail
    difficulty:     DifficultyDetail
    recommendation: str   = Field(..., description="Human-readable action summary")
    warnings:       List[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "player-042-run-1",
                "timestep": 3,
                "timestamp": 1711784444.0,
                "cii": {
                    "value": -0.604,
                    "level": "acute_frustration",
                    "zone": "frustration",
                    "momentum": -0.56,
                    "acceleration": -0.56,
                    "perf_dev": -1.2,
                    "intent_gap": 0.6,
                    "polarity": -0.7,
                    "intensity": 0.8,
                    "contribution": {
                        "momentum": -0.196, "acceleration": -0.168,
                        "performance_dev": -0.3, "intent_gap": 0.06
                    }
                },
                "prediction": {
                    "state": "frustrated",
                    "confidence": 0.923,
                    "frustration_score": 0.923,
                    "engagement_score": 0.054,
                    "boredom_score": 0.023,
                    "recommended_action": "DECREASE difficulty",
                    "reasoning": ["CII=-0.604 < -0.50 -> ACUTE frustration override"]
                },
                "difficulty": {
                    "action": "decrease",
                    "level": 4.25,
                    "delta": -0.75,
                    "prev_level": 5.0,
                    "reward": -0.4425,
                    "on_cooldown": False
                },
                "recommendation": "URGENT: Reduce difficulty to 4.2 — Frustrated detected (high confidence)",
                "warnings": []
            }
        }
    }


class SessionStatsResponse(BaseModel):
    """Response body for GET /session/{id}/stats (service-layer analytics)."""
    session_id:          str
    total_steps:         int
    avg_cii:             float
    min_cii:             float
    max_cii:             float
    std_cii:             float
    flow_rate:           float
    frustration_events:  int
    boredom_events:      int
    acute_events:        int
    engagement_rate:     float
    avg_confidence:      float
    avg_reward:          float
    total_reward:        float
    final_difficulty:    float
    avg_difficulty:      float
    interventions:       int
    difficulty_changes:  int
    dominant_cii_driver: str
    trend:               str = Field(..., description="worsening | recovering | stable")
