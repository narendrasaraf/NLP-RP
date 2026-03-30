"""
app/core/emotional_dynamics.py
-------------------------------
Emotional Dynamics Engine — the mathematical heart of the system.

Computes at each timestep t:
  E(t) = (P(t), I(t), D(t), G(t))     -- Emotional State vector
  M(t) = P(t) * I(t)                  -- Emotional Momentum
  A(t) = (M(t) - M(t-1)) / delta_t   -- Emotional Acceleration
  CII(t) = alpha*M + beta*A + gamma*D + delta*G  -- Instability Index

CII is the primary output fed to the predictor and RL controller.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import CIIConfig, settings


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EmotionalState:
    """
    Full emotional state at timestep t.
    All fields are ready for API serialization.
    """
    polarity:        float   # P(t)  ∈ [-1, +1]
    intensity:       float   # I(t)  ∈ [ 0,  1]
    performance_dev: float   # D(t)  ∈  R   (z-score)
    intent_gap:      float   # G(t)  ∈ [ 0,  1]
    momentum:        float   # M(t)  ∈ [-1, +1]
    acceleration:    float   # A(t)  ∈  R
    cii:             float   # CII(t) ∈ R

    def as_feature_vector(self) -> list:
        """Returns [CII, M, A, D, G] — input format for LSTM."""
        return [self.cii, self.momentum, self.acceleration,
                self.performance_dev, self.intent_gap]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EmotionalDynamicsEngine:
    """
    Stateful engine that processes feature inputs and produces
    the complete emotional state E(t) at each timestep.

    Maintains _prev_momentum for acceleration computation.
    One instance per session.
    """

    def __init__(self, cfg: CIIConfig = None, delta_t: float = None):
        self.cfg     = cfg     or settings.cii
        self.delta_t = delta_t or settings.delta_t
        self._prev_momentum: float = 0.0
        self._step: int = 0

    def process(
        self,
        polarity:        float,
        intensity:       float,
        performance_dev: float,
        intent_gap:      float,
    ) -> EmotionalState:
        """
        Compute full E(t) given the four component values.

        Args:
            polarity        : P(t) from NLP module
            intensity       : I(t) from NLP module
            performance_dev : D(t) from telemetry module (z-score)
            intent_gap      : G(t) from NLP module

        Returns:
            EmotionalState with all derived fields populated.
        """
        self._step += 1

        # ── Clamp inputs to valid ranges ────────────────────────────────────
        P = max(-1.0, min(1.0, polarity))
        I = max( 0.0, min(1.0, intensity))
        D = performance_dev            # Unbounded z-score
        G = max( 0.0, min(1.0, intent_gap))

        # ── Emotional Momentum  M(t) = P(t) * I(t) ─────────────────────────
        M = P * I

        # ── Emotional Acceleration  A(t) = ΔM / Δt ─────────────────────────
        A = (M - self._prev_momentum) / self.delta_t
        self._prev_momentum = M

        # ── Cognitive Instability Index ─────────────────────────────────────
        CII = (
            self.cfg.alpha * M +
            self.cfg.beta  * A +
            self.cfg.gamma * D +
            self.cfg.delta * G
        )

        return EmotionalState(
            polarity        = round(P,   4),
            intensity       = round(I,   4),
            performance_dev = round(D,   4),
            intent_gap      = round(G,   4),
            momentum        = round(M,   4),
            acceleration    = round(A,   4),
            cii             = round(CII, 4),
        )

    def reset(self):
        """Reset engine state for a new session."""
        self._prev_momentum = 0.0
        self._step = 0

    @property
    def timestep(self) -> int:
        return self._step


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

class DynamicsRegistry:
    """One EmotionalDynamicsEngine per active session."""

    def __init__(self):
        self._engines: dict = {}

    def get_or_create(self, session_id: str) -> EmotionalDynamicsEngine:
        if session_id not in self._engines:
            self._engines[session_id] = EmotionalDynamicsEngine()
        return self._engines[session_id]

    def drop_session(self, session_id: str):
        self._engines.pop(session_id, None)


dynamics_registry = DynamicsRegistry()
