"""
app/core/telemetry.py
---------------------
Gameplay telemetry processing module.

Responsibilites:
  - Maintain a rolling performance history per session
  - Compute composite performance score from raw telemetry
  - Compute performance deviation D(t) as a z-score relative
    to the player's own rolling baseline

D(t) = (P_obs(t) - mu_P) / sigma_P

This player-normalized metric is invariant to between-player skill
differences and captures subjective underperformance.
"""

import math
from collections import deque
from typing import Dict

import numpy as np

from app.api.schemas import TelemetryInput
from app.core.config import PerformanceTrackerConfig, settings


class PerformanceTracker:
    """
    Per-session rolling performance baseline tracker.

    Maintains a fixed-length deque of recent performance scores and
    computes the z-score of the latest observation relative to that history.
    """

    def __init__(self, cfg: PerformanceTrackerConfig = None):
        self.cfg = cfg or settings.perf
        self._history: deque = deque(maxlen=self.cfg.window_size)
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, telemetry: TelemetryInput) -> float:
        """
        Ingest one telemetry snapshot and return D(t).

        Returns 0.0 during warm-up (insufficient history).
        """
        score = self._composite_score(telemetry)
        self._history.append(score)
        self._step += 1

        if self._step < self.cfg.warmup_steps:
            return 0.0

        return self._z_score(score)

    def reset(self):
        """Reset tracker state for a new session."""
        self._history.clear()
        self._step = 0

    @property
    def history(self):
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_score(t: TelemetryInput) -> float:
        """
        Composite performance score from raw telemetry signals.

        Higher score = better performance:
          + score_delta  (positive score change is good)
          - deaths * 2   (each death is a strong negative signal)
          - retries * 1.5 (retries indicate difficulty / frustration)
          + streak * 0.5  (positive streak amplifies performance)
          - max(0, reaction_time_ms - 300) * 0.01  (penalize latency above 300ms)
        """
        latency_penalty = max(0.0, t.reaction_time_ms - 300.0) * 0.01
        return (
            t.score_delta
            - (t.deaths   * 2.0)
            - (t.retries  * 1.5)
            + (t.streak   * 0.5)
            - latency_penalty
        )

    def _z_score(self, score: float) -> float:
        """Z-score of score relative to rolling history."""
        if len(self._history) < 2:
            return 0.0
        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        std = math.sqrt(variance) if variance > 0 else 1.0
        return (score - mean) / std


# ---------------------------------------------------------------------------
# Session-level registry — one tracker per active session
# ---------------------------------------------------------------------------

class TelemetryRegistry:
    """
    Manages one PerformanceTracker per active game session.
    Allows multi-session concurrent API usage.
    """

    def __init__(self):
        self._trackers: Dict[str, PerformanceTracker] = {}

    def get_or_create(self, session_id: str) -> PerformanceTracker:
        if session_id not in self._trackers:
            self._trackers[session_id] = PerformanceTracker()
        return self._trackers[session_id]

    def reset_session(self, session_id: str):
        if session_id in self._trackers:
            self._trackers[session_id].reset()

    def drop_session(self, session_id: str):
        self._trackers.pop(session_id, None)

    def active_sessions(self):
        return list(self._trackers.keys())


# Module-level singleton
telemetry_registry = TelemetryRegistry()
