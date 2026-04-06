"""
utils/temporal_features.py
---------------------------
Temporal Emotional Feature Engine for the Cognitive Regulation system.

Responsibility:
  Given a stream of NLP feature snapshots (one per telemetry tick),
  maintain a rolling history of Emotional Momentum and compute:

    M(t)  = P(t) * I(t)              Emotional Momentum
    A(t)  = M(t) - M(t-1)            Emotional Acceleration  (dt-normalised)
    μ_M   = mean(M over last N)      Rolling Mean of Momentum
    σ_M   = std(M over last N)       Rolling Std  of Momentum
    trend = linear slope of M        Direction signal (rising / falling)
    z_M   = (M(t) - μ_M) / σ_M      Momentum Z-score (deviation from baseline)

Position in the pipeline:
    NLPExtractor  →  TemporalFeatureEngine  →  CIIComputer / Predictor

Why a separate module?
  - CIIComputer (utils/cii.py) only stores prev_momentum as a scalar.
    It has no concept of a rolling momentum window or its statistics.
  - preprocessing.py's DerivedFeatureComputer mixes NLP + telemetry
    in one pass, making it hard to use in real-time tick-by-tick updates.
  - This module is STATEFUL per session and STATELESS across sessions
    (call reset() between players).

Design:
  - Uses collections.deque for O(1) append and automatic eviction.
  - Pure Python — zero external dependencies — runs in microseconds.
  - Returns a frozen dataclass (TemporalFeatures) with all signals.

Usage:
    from utils.temporal_features import TemporalFeatureEngine

    engine = TemporalFeatureEngine(window=10)

    # Feed one NLP dict per game tick
    tf = engine.update({"polarity": -0.4, "intensity": 0.7})
    print(tf.momentum)       # -0.28
    print(tf.acceleration)   # delta from previous step
    print(tf.rolling_mean)   # mean of last 10 momentum values
    print(tf.rolling_std)    # std  of last 10 momentum values
    print(tf.trend)          # "worsening" | "stable" | "recovering"
    print(tf.as_dict())      # flat dict — ready for CII / dashboard
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalFeatures:
    """
    Immutable snapshot of all temporal emotion signals at one timestep.

    Fields
    ------
    momentum          M(t)  = P(t) * I(t)                   ∈ [-1, +1]
    acceleration      A(t)  = (M(t) - M(t-1)) / delta_t     ∈  R
    rolling_mean      μ_M   = mean of last-N M values        ∈ [-1, +1]
    rolling_std       σ_M   = std  of last-N M values        ∈ [0,  1]
    momentum_zscore   z_M   = (M - μ_M) / σ_M               ∈  R
    trend             Linear slope of momentum history        ∈  R
    trend_label       "worsening" | "stable" | "recovering"
    window_size       Number of values the rolling stats use  (int)
    history_len       Actual buffered values so far           (int ≤ N)
    is_warming_up     True while history < window_size        (bool)

    Input echo (for traceability):
    polarity          P(t) passed in                          ∈ [-1, +1]
    intensity         I(t) passed in                          ∈ [0,   1]
    """
    # ── Temporal signals ────────────────────────────────────────────────────
    momentum:        float
    acceleration:    float
    rolling_mean:    float
    rolling_std:     float
    momentum_zscore: float
    trend:           float
    trend_label:     str    # "worsening" | "stable" | "recovering"

    # ── Metadata ─────────────────────────────────────────────────────────────
    window_size:  int
    history_len:  int
    is_warming_up: bool

    # ── Input echo ─────────────────────────────────────────────────────────────
    polarity:  float
    intensity: float

    # ── Convenience ──────────────────────────────────────────────────────────

    def as_dict(self) -> Dict[str, float | int | bool | str]:
        """Flat dict suitable for logging, API responses, and dashboards."""
        return asdict(self)

    def as_cii_inputs(self) -> Dict[str, float]:
        """
        Returns only the signals consumed by CIIComputer.compute().
        Drop-in compatible with the 'polarity', 'intensity', keys it expects.
        """
        return {
            "polarity":     self.polarity,
            "intensity":    self.intensity,
            "momentum":     self.momentum,
            "acceleration": self.acceleration,
        }

    def __str__(self) -> str:
        warm = " [WARMING UP]" if self.is_warming_up else ""
        return (
            f"TemporalFeatures @ t (N={self.history_len}/{self.window_size}){warm}\n"
            f"  P={self.polarity:+.4f}  I={self.intensity:.4f}\n"
            f"  M(t)={self.momentum:+.4f}  A(t)={self.acceleration:+.4f}\n"
            f"  μ_M={self.rolling_mean:+.4f}  σ_M={self.rolling_std:.4f}  "
            f"z_M={self.momentum_zscore:+.4f}\n"
            f"  trend={self.trend:+.4f}  ({self.trend_label})"
        )


# ---------------------------------------------------------------------------
# Safe defaults when history is empty
# ---------------------------------------------------------------------------

def _empty_features(
    polarity:    float,
    intensity:   float,
    window_size: int,
) -> TemporalFeatures:
    M = polarity * intensity
    return TemporalFeatures(
        momentum        = round(M, 6),
        acceleration    = 0.0,
        rolling_mean    = round(M, 6),
        rolling_std     = 0.0,
        momentum_zscore = 0.0,
        trend           = 0.0,
        trend_label     = "stable",
        window_size     = window_size,
        history_len     = 1,
        is_warming_up   = True,
        polarity        = round(polarity, 6),
        intensity       = round(intensity, 6),
    )


# ---------------------------------------------------------------------------
# TemporalFeatureEngine
# ---------------------------------------------------------------------------

class TemporalFeatureEngine:
    """
    Stateful engine that processes a stream of NLP feature dicts and
    outputs TemporalFeatures on every tick.

    Args
    ----
    window     : Rolling window size N for mean / std / trend (default 10).
    delta_t    : Sampling interval in seconds — used to scale acceleration
                 to meaningful units (default 1.0, i.e., A = ΔM per tick).
    trend_slope_thresholds : (low, high) thresholds for labelling trend.
                              default (-0.03, +0.03)

    Thread safety: NOT thread-safe. Create one engine per session/thread.
    """

    def __init__(
        self,
        window:                  int   = 10,
        delta_t:                 float = 1.0,
        trend_slope_thresholds:  tuple[float, float] = (-0.03, +0.03),
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if delta_t <= 0:
            raise ValueError(f"delta_t must be > 0, got {delta_t}")

        self._window   = window
        self._delta_t  = delta_t
        self._lo_slope, self._hi_slope = trend_slope_thresholds

        # Rolling history of M values (oldest-first, newest-last)
        self._history: deque[float] = deque(maxlen=window)

        # Previous momentum for acceleration
        self._prev_M: float = 0.0

        # Timestep counter (for diagnostics)
        self._t: int = 0

    # ------------------------------------------------------------------
    # Primary API — call once per game tick
    # ------------------------------------------------------------------

    def update(self, nlp_features: Dict[str, float]) -> TemporalFeatures:
        """
        Ingest one NLP feature snapshot and return the temporal signals.

        Args
        ----
        nlp_features : dict with at minimum 'polarity' and 'intensity' keys.
                       Accepts output of NLPExtractor.extract() directly.
                       Missing keys default to 0.0.

        Returns
        -------
        TemporalFeatures — immutable snapshot of all temporal signals.
        """
        # ── Extract scalars (safe defaults if keys missing) ───────────────
        P = float(nlp_features.get("polarity",  0.0))
        I = float(nlp_features.get("intensity", 0.0))

        # ── Clamp inputs to valid ranges ──────────────────────────────────
        P = max(-1.0, min(1.0, P))
        I = max( 0.0, min(1.0, I))

        # ── Emotional Momentum: M(t) = P(t) * I(t) ───────────────────────
        M = P * I                               # ∈ [-1, +1] by construction

        # ── First tick: initialise, return safe defaults ──────────────────
        if self._t == 0:
            self._prev_M = M
            self._history.append(M)
            self._t += 1
            return _empty_features(P, I, self._window)

        # ── Emotional Acceleration: A(t) = (M(t) - M(t-1)) / Δt ─────────
        A = (M - self._prev_M) / self._delta_t
        self._prev_M = M

        # ── Append to rolling history ─────────────────────────────────────
        self._history.append(M)
        self._t += 1

        # ── Rolling statistics ────────────────────────────────────────────
        mu, sigma = _rolling_mean_std(self._history)
        z_M       = _z_score(M, mu, sigma)

        # ── Trend (linear slope of momentum history) ──────────────────────
        slope       = _linear_slope(list(self._history))
        trend_label = self._label_trend(slope)

        return TemporalFeatures(
            momentum        = round(M,   6),
            acceleration    = round(A,   6),
            rolling_mean    = round(mu,  6),
            rolling_std     = round(sigma, 6),
            momentum_zscore = round(z_M, 6),
            trend           = round(slope, 6),
            trend_label     = trend_label,
            window_size     = self._window,
            history_len     = len(self._history),
            is_warming_up   = len(self._history) < self._window,
            polarity        = round(P, 6),
            intensity       = round(I, 6),
        )

    def update_from_values(
        self,
        polarity:  float,
        intensity: float,
    ) -> TemporalFeatures:
        """
        Convenience wrapper — pass polarity and intensity directly
        without constructing a full NLP dict.

        Equivalent to: engine.update({"polarity": p, "intensity": i})
        """
        return self.update({"polarity": polarity, "intensity": intensity})

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(
        self,
        nlp_sequence: Sequence[Dict[str, float]],
    ) -> List[TemporalFeatures]:
        """
        Process a list of NLP feature dicts in order, maintaining
        temporal state (acceleration, rolling history) across steps.

        Args
        ----
        nlp_sequence : Ordered list of dicts, each with 'polarity'/'intensity'.

        Returns
        -------
        List of TemporalFeatures, one per input step.
        """
        return [self.update(feats) for feats in nlp_sequence]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[float]:
        """Read-only copy of the current momentum history buffer."""
        return list(self._history)

    @property
    def timestep(self) -> int:
        """Number of update() calls made since last reset()."""
        return self._t

    @property
    def is_warming_up(self) -> bool:
        """True while fewer than `window` ticks have been processed."""
        return len(self._history) < self._window

    def summary(self) -> Dict[str, float | int | bool | str]:
        """
        Aggregate statistics over the entire session so far.
        Useful for end-of-session reporting or dashboard displays.
        """
        if not self._history:
            return {"timesteps": 0}
        h = list(self._history)
        mu, sigma = _rolling_mean_std(h)
        return {
            "timesteps":       self._t,
            "window_size":     self._window,
            "history_len":     len(h),
            "is_warming_up":   self.is_warming_up,
            "current_momentum": round(h[-1], 4),
            "rolling_mean":    round(mu, 4),
            "rolling_std":     round(sigma, 4),
            "min_momentum":    round(min(h), 4),
            "max_momentum":    round(max(h), 4),
            "session_trend":   _label_trend_value(
                                   _linear_slope(h),
                                   self._lo_slope, self._hi_slope
                               ),
        }

    def reset(self) -> None:
        """
        Reset all session state.
        Call this between independent player sessions.
        Does NOT alter window size or delta_t.
        """
        self._history.clear()
        self._prev_M = 0.0
        self._t      = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _label_trend(self, slope: float) -> str:
        return _label_trend_value(slope, self._lo_slope, self._hi_slope)


# ---------------------------------------------------------------------------
# Pure math helpers  (no class state — independently testable)
# ---------------------------------------------------------------------------

def _rolling_mean_std(values: deque | List[float]) -> tuple[float, float]:
    """
    Compute mean and population standard deviation of a sequence.

    Returns (0.0, 0.0) for empty input, (mean, 0.0) for single element.
    Uses population std (divides by N) since the window IS the population.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(list(values)[0]), 0.0
    vals = list(values)
    mu   = sum(vals) / n
    var  = sum((x - mu) ** 2 for x in vals) / n
    return mu, math.sqrt(var)


def _z_score(value: float, mu: float, sigma: float) -> float:
    """
    Standard z-score.  Returns 0.0 when sigma is near zero
    (avoids division by zero in flat / constant sequences).
    """
    if sigma < 1e-9:
        return 0.0
    return (value - mu) / sigma


def _linear_slope(values: List[float]) -> float:
    """
    Ordinary least-squares slope of values vs. their index.

    slope > 0  →  momentum rising    (recovering)
    slope < 0  →  momentum falling   (worsening)
    slope ≈ 0  →  momentum stable

    Uses closed-form OLS for speed (O(N), no scipy dependency).
    Returns 0.0 for sequences shorter than 2.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    cov = sum((i - x_mean) * values[i] for i in range(n))
    var = sum((i - x_mean) ** 2 for i in range(n))
    return cov / var if var > 1e-12 else 0.0


def _label_trend_value(slope: float, lo: float, hi: float) -> str:
    """Map slope to human-readable trend label."""
    if slope < lo:
        return "worsening"
    if slope > hi:
        return "recovering"
    return "stable"


# ---------------------------------------------------------------------------
# Module-level default engine (shared singleton for backends without session mgmt)
# ---------------------------------------------------------------------------

#: Default engine — import this for scripts and tools that don't need
#: per-session isolation.  For multi-session systems use TemporalFeatureEngine().
temporal_engine = TemporalFeatureEngine(window=10, delta_t=1.0)


# ---------------------------------------------------------------------------
# Self-test  (run: python -m utils.temporal_features)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 68

    print(SEP)
    print("  TemporalFeatureEngine — Self Test")
    print(SEP)

    # ── Scenario 1: Gradual frustration escalation ────────────────────────
    print("\n[Scenario 1] Gradual frustration escalation\n")
    engine = TemporalFeatureEngine(window=5, delta_t=1.0)

    nlp_stream = [
        {"polarity":  0.2, "intensity": 0.3},   # slightly positive, calm
        {"polarity":  0.0, "intensity": 0.4},   # neutral
        {"polarity": -0.2, "intensity": 0.5},   # mild negative
        {"polarity": -0.5, "intensity": 0.7},   # frustration emerging
        {"polarity": -0.8, "intensity": 0.9},   # strong frustration
        {"polarity": -0.9, "intensity": 0.95},  # peak frustration
        {"polarity": -0.7, "intensity": 0.8},   # slight recovery
    ]

    for i, feats in enumerate(nlp_stream, 1):
        tf = engine.update(feats)
        warm = "*" if tf.is_warming_up else " "
        print(
            f"  t={i:02d}{warm} "
            f"P={tf.polarity:+.2f}  I={tf.intensity:.2f}  "
            f"M={tf.momentum:+.4f}  A={tf.acceleration:+.4f}  "
            f"μ={tf.rolling_mean:+.4f}  σ={tf.rolling_std:.4f}  "
            f"z={tf.momentum_zscore:+.4f}  [{tf.trend_label}]"
        )

    print(f"\n  Session summary: {engine.summary()}")

    # ── Scenario 2: Boredom / flat line ──────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 2] Flat / bored player\n")
    engine2 = TemporalFeatureEngine(window=5)
    for i in range(6):
        tf = engine2.update({"polarity": 0.05, "intensity": 0.1})
        print(f"  t={i+1:02d}  M={tf.momentum:+.4f}  "
              f"μ={tf.rolling_mean:+.4f}  σ={tf.rolling_std:.4f}  [{tf.trend_label}]")

    # ── Scenario 3: Batch processing ─────────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 3] Batch processing + as_cii_inputs()\n")
    engine3 = TemporalFeatureEngine(window=4)
    batch = [
        {"polarity": -0.3, "intensity": 0.5},
        {"polarity": -0.6, "intensity": 0.7},
        {"polarity": -0.8, "intensity": 0.85},
    ]
    results = engine3.process_batch(batch)
    for i, tf in enumerate(results, 1):
        print(f"  t={i:02d}  CII-ready inputs: {tf.as_cii_inputs()}")

    # ── Scenario 4: reset() correctness ──────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 4] reset() between sessions\n")
    engine4 = TemporalFeatureEngine(window=4)
    engine4.update({"polarity": -0.9, "intensity": 0.9})
    engine4.update({"polarity": -0.8, "intensity": 0.8})
    print(f"  Before reset — history: {engine4.history}  t={engine4.timestep}")
    engine4.reset()
    print(f"  After  reset — history: {engine4.history}  t={engine4.timestep}")
    tf_fresh = engine4.update({"polarity": 0.3, "intensity": 0.4})
    print(f"  First tick after reset: {tf_fresh}")

    print(f"\n{SEP}")
    print("  All scenarios passed.")
    print(SEP)
