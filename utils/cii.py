"""
utils/cii.py
-------------
Standalone Cognitive Instability Index (CII) computation module.

This module is the single authoritative source for CII math.
It has ZERO dependency on any other project module — import it
anywhere: notebooks, scripts, tests, training pipelines.

CII(t) = alpha * M(t) + beta * A(t) + gamma * D(t) + delta * G(t)

where:
    M(t) = P(t) * I(t)               -- Emotional Momentum
    A(t) = (M(t) - M(t-1)) / dt     -- Emotional Acceleration
    D(t) = (P_obs - mu_P) / sigma_P -- Performance Deviation (z-score)
    G(t) = intent-outcome gap        -- Cognitive Friction ∈ [0, 1]

Interpretation:
    CII < -0.50        →  Acute frustration
   -0.50 ≤ CII < -0.25 →  High instability (frustration risk)
   -0.25 ≤ CII <  0.00 →  Mild instability (monitor closely)
    0.00 ≤ CII ≤ +0.30  →  Stable flow (optimal)
    CII  > +0.30        →  Boredom risk (under-challenged)

Usage:
    from utils.cii import CIIComputer, CIIWeights

    computer = CIIComputer(weights=CIIWeights(alpha=0.35, beta=0.30,
                                               gamma=0.25, delta=0.10))
    result = computer.compute(polarity=-0.7, intensity=0.8,
                              performance_dev=-1.2, intent_gap=0.6)
    print(result)
    # CIIResult(value=-0.604, level='high_instability', zone='frustration', ...)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration: Configurable Weights
# ---------------------------------------------------------------------------

@dataclass
class CIIWeights:
    """
    Learnable weight vector for CII composition.

    Constraint: alpha + beta + gamma + delta == 1.0
    All weights must be positive.

    Default values from empirical calibration (Phase 3):
        alpha = 0.35  (Emotional Momentum — NLP-driven, highly predictive)
        beta  = 0.30  (Emotional Acceleration — key early-warning signal)
        gamma = 0.25  (Performance Deviation — behavioral grounding)
        delta = 0.10  (Intent-Outcome Gap — cognitive friction)
    """
    alpha: float = 0.35   # Emotional Momentum weight
    beta:  float = 0.30   # Emotional Acceleration weight
    gamma: float = 0.25   # Performance Deviation weight
    delta: float = 0.10   # Intent-Outcome Gap weight

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for name, val in [("alpha", self.alpha), ("beta", self.beta),
                          ("gamma", self.gamma), ("delta", self.delta)]:
            if val < 0:
                raise ValueError(f"Weight '{name}' must be >= 0, got {val}")
        total = self.alpha + self.beta + self.gamma + self.delta
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0 (got {total:.6f}). "
                f"Adjust values or use CIIWeights.normalized()."
            )

    @classmethod
    def normalized(cls, alpha: float, beta: float,
                   gamma: float, delta: float) -> "CIIWeights":
        """
        Factory: create weights from unnormalized values.
        Automatically normalizes so they sum to 1.0.

        Example:
            w = CIIWeights.normalized(3.5, 3.0, 2.5, 1.0)
            # -> alpha=0.35, beta=0.30, gamma=0.25, delta=0.10
        """
        total = alpha + beta + gamma + delta
        if total <= 0:
            raise ValueError("Sum of raw weights must be positive.")
        return cls(
            alpha = alpha / total,
            beta  = beta  / total,
            gamma = gamma / total,
            delta = delta / total,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CIIWeights":
        return cls(
            alpha = d.get("alpha", 0.35),
            beta  = d.get("beta",  0.30),
            gamma = d.get("gamma", 0.25),
            delta = d.get("delta", 0.10),
        )

    def to_dict(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "delta": self.delta}

    def __repr__(self) -> str:
        return (f"CIIWeights(alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, delta={self.delta})")


# ---------------------------------------------------------------------------
# Intermediate: CII Components
# ---------------------------------------------------------------------------

@dataclass
class CIIComponents:
    """
    All four CII input components at a single timestep.
    Useful for logging, visualization, and ablation studies.
    """
    polarity:        float   # P(t) ∈ [-1, +1]   raw NLP sentiment
    intensity:       float   # I(t) ∈ [ 0,  1]   emotion magnitude
    momentum:        float   # M(t) = P * I
    acceleration:    float   # A(t) = ΔM / Δt
    performance_dev: float   # D(t) z-score
    intent_gap:      float   # G(t) ∈ [0, 1]

    def as_dict(self) -> Dict[str, float]:
        return {
            "polarity":        self.polarity,
            "intensity":       self.intensity,
            "momentum":        self.momentum,
            "acceleration":    self.acceleration,
            "performance_dev": self.performance_dev,
            "intent_gap":      self.intent_gap,
        }


# ---------------------------------------------------------------------------
# Interpretation: 5-level system
# ---------------------------------------------------------------------------

class InstabilityLevel(str, Enum):
    """
    Five-tier cognitive instability classification.

    Maps CII ranges to actionable system states.
    """
    STABLE_FLOW       = "stable_flow"        # CII ∈ [0.00, +0.30]
    MILD_INSTABILITY  = "mild_instability"   # CII ∈ [-0.25, 0.00)
    HIGH_INSTABILITY  = "high_instability"   # CII ∈ [-0.50, -0.25)
    ACUTE_FRUSTRATION = "acute_frustration"  # CII < -0.50
    BOREDOM_RISK      = "boredom_risk"       # CII > +0.30


class CIIZone(str, Enum):
    """Coarse three-zone classification aligned with difficulty controller."""
    FRUSTRATION = "frustration"   # Decrease difficulty
    FLOW        = "flow"          # Maintain difficulty
    BOREDOM     = "boredom"       # Increase difficulty


# Thresholds (tunable without changing logic)
THRESHOLDS = {
    "acute_frustration": -0.50,
    "high_instability":  -0.25,
    "mild_instability":   0.00,
    "stable_flow":       +0.30,
    # Above +0.30 → boredom_risk
}

DIFFICULTY_ACTION = {
    InstabilityLevel.ACUTE_FRUSTRATION: "DECREASE (urgent)",
    InstabilityLevel.HIGH_INSTABILITY:  "DECREASE",
    InstabilityLevel.MILD_INSTABILITY:  "MAINTAIN (monitor)",
    InstabilityLevel.STABLE_FLOW:       "MAINTAIN",
    InstabilityLevel.BOREDOM_RISK:      "INCREASE",
}


def _classify(cii: float) -> Tuple[InstabilityLevel, CIIZone]:
    if cii < THRESHOLDS["acute_frustration"]:
        return InstabilityLevel.ACUTE_FRUSTRATION, CIIZone.FRUSTRATION
    if cii < THRESHOLDS["high_instability"]:
        return InstabilityLevel.HIGH_INSTABILITY, CIIZone.FRUSTRATION
    if cii < THRESHOLDS["mild_instability"]:
        return InstabilityLevel.MILD_INSTABILITY, CIIZone.FLOW
    if cii <= THRESHOLDS["stable_flow"]:
        return InstabilityLevel.STABLE_FLOW, CIIZone.FLOW
    return InstabilityLevel.BOREDOM_RISK, CIIZone.BOREDOM


# ---------------------------------------------------------------------------
# Output: CII Result
# ---------------------------------------------------------------------------

@dataclass
class CIIResult:
    """
    Full output of one CII computation step.

    Fields:
        value          : CII(t) scalar — the instability score
        level          : 5-tier InstabilityLevel classification
        zone           : 3-zone CIIZone (frustration / flow / boredom)
        action         : Recommended difficulty controller action
        components     : All four components for traceability
        weights        : Weights used for this computation
        contribution   : Per-component contribution to CII (α*M, β*A, etc.)
    """
    value:        float
    level:        InstabilityLevel
    zone:         CIIZone
    action:       str
    components:   CIIComponents
    weights:      CIIWeights
    contribution: Dict[str, float]   # {"momentum": α*M, "accel": β*A, ...}

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def is_frustrated(self) -> bool:
        return self.zone == CIIZone.FRUSTRATION

    @property
    def is_bored(self) -> bool:
        return self.zone == CIIZone.BOREDOM

    @property
    def is_flow(self) -> bool:
        return self.zone == CIIZone.FLOW

    @property
    def urgency(self) -> int:
        """0 (flow) to 4 (acute). Useful for priority queues."""
        ranking = {
            InstabilityLevel.STABLE_FLOW:       0,
            InstabilityLevel.BOREDOM_RISK:       1,
            InstabilityLevel.MILD_INSTABILITY:   2,
            InstabilityLevel.HIGH_INSTABILITY:   3,
            InstabilityLevel.ACUTE_FRUSTRATION:  4,
        }
        return ranking[self.level]

    def dominant_driver(self) -> str:
        """Returns the component with the largest absolute contribution."""
        return max(self.contribution, key=lambda k: abs(self.contribution[k]))

    def as_dict(self) -> dict:
        return {
            "cii":        self.value,
            "level":      self.level.value,
            "zone":       self.zone.value,
            "action":     self.action,
            "urgency":    self.urgency,
            "components": self.components.as_dict(),
            "contribution": self.contribution,
        }

    def __str__(self) -> str:
        c = self.contribution
        return (
            f"\nCIIResult\n"
            f"  CII    = {self.value:+.4f}\n"
            f"  Level  : {self.level.value}\n"
            f"  Zone   : {self.zone.value}\n"
            f"  Action : {self.action}\n"
            f"  Urgency: {self.urgency}/4\n"
            f"  Driver : {self.dominant_driver()}\n"
            f"  Components:\n"
            f"    M(t) = {self.components.momentum:+.4f}  "
            f"contrib = {c['momentum']:+.4f}  (alpha={self.weights.alpha})\n"
            f"    A(t) = {self.components.acceleration:+.4f}  "
            f"contrib = {c['acceleration']:+.4f}  (beta={self.weights.beta})\n"
            f"    D(t) = {self.components.performance_dev:+.4f}  "
            f"contrib = {c['performance_dev']:+.4f}  (gamma={self.weights.gamma})\n"
            f"    G(t) = {self.components.intent_gap:+.4f}  "
            f"contrib = {c['intent_gap']:+.4f}  (delta={self.weights.delta})\n"
        )


# ---------------------------------------------------------------------------
# Core: CII Computer
# ---------------------------------------------------------------------------

class CIIComputer:
    """
    Computes CII at each timestep. Stateful: maintains momentum history
    for acceleration and performance history for D(t) z-score.

    Args:
        weights      : CIIWeights instance (default values or custom)
        delta_t      : Sampling interval in seconds (default 1.0)
        perf_window  : Rolling window size for D(t) baseline (default 20)

    Example:
        computer = CIIComputer()
        r = computer.compute(polarity=-0.7, intensity=0.8,
                             performance_dev=-1.2, intent_gap=0.6)
        print(r)

    Or pass pre-computed performance deviation via `performance_dev` arg.
    If you want the computer to track D(t) internally, pass a raw
    performance_score instead and use compute_from_score().
    """

    def __init__(
        self,
        weights:     CIIWeights = None,
        delta_t:     float = 1.0,
        perf_window: int   = 20,
    ):
        self.weights     = weights or CIIWeights()
        self.delta_t     = delta_t
        self.perf_window = perf_window

        # Internal state
        self._prev_momentum:   float         = 0.0
        self._perf_history:    deque         = deque(maxlen=perf_window)
        self._timestep:        int           = 0

    # ------------------------------------------------------------------
    # Main compute — pre-computed D(t) provided by caller
    # ------------------------------------------------------------------

    def compute(
        self,
        polarity:        float,
        intensity:       float,
        performance_dev: float,
        intent_gap:      float,
    ) -> CIIResult:
        """
        Compute CII from four component values.

        Args:
            polarity        : Sentiment polarity P(t) ∈ [-1, +1]
            intensity       : Emotion intensity I(t) ∈ [0, 1]
            performance_dev : Pre-computed D(t) (z-score vs. player baseline)
            intent_gap      : Cosine distance G(t) ∈ [0, 1]

        Returns:
            CIIResult with value, level, zone, action, contributions
        """
        self._timestep += 1

        # ── Input clamping ────────────────────────────────────────────────
        P = max(-1.0, min(1.0, float(polarity)))
        I = max( 0.0, min(1.0, float(intensity)))
        D = float(performance_dev)
        G = max( 0.0, min(1.0, float(intent_gap)))

        # ── Emotional Momentum ────────────────────────────────────────────
        M = P * I

        # ── Emotional Acceleration ─────────────────────────────────────────
        A = (M - self._prev_momentum) / self.delta_t
        self._prev_momentum = M

        # ── CII ────────────────────────────────────────────────────────────
        w = self.weights
        contrib = {
            "momentum":        round(w.alpha * M, 5),
            "acceleration":    round(w.beta  * A, 5),
            "performance_dev": round(w.gamma * D, 5),
            "intent_gap":      round(w.delta * G, 5),
        }
        cii = sum(contrib.values())

        level, zone = _classify(cii)

        return CIIResult(
            value        = round(cii, 4),
            level        = level,
            zone         = zone,
            action       = DIFFICULTY_ACTION[level],
            components   = CIIComponents(
                polarity        = round(P, 4),
                intensity       = round(I, 4),
                momentum        = round(M, 4),
                acceleration    = round(A, 4),
                performance_dev = round(D, 4),
                intent_gap      = round(G, 4),
            ),
            weights      = self.weights,
            contribution = contrib,
        )

    # ------------------------------------------------------------------
    # Alternate entry: pass raw performance score; D(t) computed here
    # ------------------------------------------------------------------

    def compute_from_score(
        self,
        polarity:      float,
        intensity:     float,
        perf_score:    float,
        intent_gap:    float,
    ) -> CIIResult:
        """
        Same as compute() but accepts a raw composite performance score
        and computes the z-score D(t) internally using rolling history.

        Use this when you do NOT have a pre-normalized D(t) — for example
        when feeding raw game metrics directly without preprocessing.py.
        """
        D = self._rolling_z_score(perf_score)
        return self.compute(polarity, intensity, D, intent_gap)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        steps: List[Dict[str, float]],
    ) -> List[CIIResult]:
        """
        Process a list of timestep dicts in sequence.
        Each dict must have keys: polarity, intensity, performance_dev, intent_gap.

        Temporal state (acceleration, perf history) is maintained across steps.

        Example:
            results = computer.compute_batch([
                {"polarity": -0.1, "intensity": 0.3,
                 "performance_dev": 0.0, "intent_gap": 0.2},
                {"polarity": -0.6, "intensity": 0.7,
                 "performance_dev": -1.2, "intent_gap": 0.6},
            ])
        """
        return [
            self.compute(
                polarity        = s["polarity"],
                intensity       = s["intensity"],
                performance_dev = s["performance_dev"],
                intent_gap      = s["intent_gap"],
            )
            for s in steps
        ]

    # ------------------------------------------------------------------
    # Sensitivity analysis — what-if per component
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        polarity:        float,
        intensity:       float,
        performance_dev: float,
        intent_gap:      float,
        delta: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute ΔCII for ±delta perturbation of each input.
        Returns a dict showing how sensitive CII is to each component.

        Useful for feature importance analysis and ablation studies.

        Returns:
            {"d_polarity": ..., "d_intensity": ...,
             "d_performance_dev": ..., "d_intent_gap": ...}
        """
        base = self.compute(polarity, intensity, performance_dev, intent_gap)
        base_cii = base.value

        def perturb(field: str, val: float) -> float:
            args = {"polarity": polarity, "intensity": intensity,
                    "performance_dev": performance_dev, "intent_gap": intent_gap}
            args[field] = val
            # Revert acceleration state to keep comparison fair
            saved = self._prev_momentum
            r = self.compute(**args)
            self._prev_momentum = saved   # restore
            return abs(r.value - base_cii)

        return {
            "d_polarity":        round(perturb("polarity",        polarity        + delta), 5),
            "d_intensity":       round(perturb("intensity",       min(1.0, intensity + delta)), 5),
            "d_performance_dev": round(perturb("performance_dev", performance_dev + delta), 5),
            "d_intent_gap":      round(perturb("intent_gap",      min(1.0, intent_gap + delta)), 5),
        }

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> int:
        return self._timestep

    def reset(self) -> None:
        """Reset all temporal state. Call between independent sessions."""
        self._prev_momentum = 0.0
        self._perf_history.clear()
        self._timestep = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_z_score(self, score: float) -> float:
        self._perf_history.append(score)
        h = list(self._perf_history)
        if len(h) < 2:
            return 0.0
        mean  = sum(h) / len(h)
        var   = sum((x - mean) ** 2 for x in h) / len(h)
        sigma = math.sqrt(var) if var > 1e-9 else 1.0
        return (score - mean) / sigma


# ---------------------------------------------------------------------------
# Session-level CII History tracker
# ---------------------------------------------------------------------------

@dataclass
class CIIHistory:
    """
    Tracks CII results across a full session.
    Provides aggregate analytics and trend detection.

    Usage:
        history = CIIHistory()
        history.record(result)          # after each compute()
        summary = history.summary()
    """
    _records: List[CIIResult] = field(default_factory=list)

    def record(self, result: CIIResult) -> None:
        self._records.append(result)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def values(self) -> List[float]:
        return [r.value for r in self._records]

    def summary(self) -> Dict:
        """Aggregate session statistics."""
        if not self._records:
            return {}
        vals   = self.values
        levels = [r.level.value for r in self._records]
        n      = len(vals)
        return {
            "timesteps":          n,
            "mean_cii":           round(sum(vals) / n, 4),
            "min_cii":            round(min(vals), 4),
            "max_cii":            round(max(vals), 4),
            "std_cii":            round(math.sqrt(sum((v - sum(vals)/n)**2 for v in vals) / n), 4),
            "flow_rate":          round(sum(1 for r in self._records if r.is_flow) / n, 4),
            "frustration_events": sum(1 for r in self._records if r.is_frustrated),
            "boredom_events":     sum(1 for r in self._records if r.is_bored),
            "acute_events":       levels.count(InstabilityLevel.ACUTE_FRUSTRATION.value),
            "peak_urgency":       max(r.urgency for r in self._records),
            "dominant_driver":    max(
                set(r.dominant_driver() for r in self._records),
                key=lambda d: sum(1 for r in self._records if r.dominant_driver() == d)
            ),
        }

    def trend(self, window: int = 5) -> str:
        """
        Returns 'worsening', 'recovering', or 'stable'
        based on the linear slope of the last `window` CII values.
        """
        recent = self.values[-window:]
        if len(recent) < 2:
            return "stable"
        n      = len(recent)
        x_mean = (n - 1) / 2.0
        cov    = sum((i - x_mean) * recent[i] for i in range(n))
        var    = sum((i - x_mean) ** 2 for i in range(n))
        slope  = cov / var if var > 0 else 0.0
        if slope < -0.05:
            return "worsening"
        if slope >  0.05:
            return "recovering"
        return "stable"

    def clear(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 60

    # ── Test 1: Default weights ────────────────────────────────────────────
    print(SEP)
    print("  TEST 1: Default Weights — Single Compute")
    print(SEP)
    computer = CIIComputer()
    result = computer.compute(
        polarity=-0.7, intensity=0.8,
        performance_dev=-1.2, intent_gap=0.6
    )
    print(result)

    # ── Test 2: Custom weights via normalized factory ──────────────────────
    print(SEP)
    print("  TEST 2: Custom Weights (NLP-heavy configuration)")
    print(SEP)
    custom_weights = CIIWeights.normalized(alpha=5, beta=3, gamma=1, delta=1)
    print(f"  Weights: {custom_weights}")
    comp2  = CIIComputer(weights=custom_weights)
    result2 = comp2.compute(
        polarity=-0.7, intensity=0.8,
        performance_dev=-1.2, intent_gap=0.6
    )
    print(result2)

    # ── Test 3: Batch — Escalating Frustration ────────────────────────────
    print(SEP)
    print("  TEST 3: Batch — Escalating Frustration Scenario")
    print(SEP)
    computer3 = CIIComputer()
    history   = CIIHistory()
    steps = [
        {"polarity": -0.1, "intensity": 0.30, "performance_dev":  0.00, "intent_gap": 0.20},
        {"polarity": -0.3, "intensity": 0.50, "performance_dev": -0.80, "intent_gap": 0.40},
        {"polarity": -0.6, "intensity": 0.70, "performance_dev": -1.20, "intent_gap": 0.60},
        {"polarity": -0.8, "intensity": 0.90, "performance_dev": -1.50, "intent_gap": 0.75},
        {"polarity": -0.9, "intensity": 0.95, "performance_dev": -1.80, "intent_gap": 0.85},
    ]
    results = computer3.compute_batch(steps)
    for i, r in enumerate(results, 1):
        bar = "#" * int(abs(r.value) * 20)
        print(f"  t={i:02d}  CII={r.value:+.4f} [{bar:<20}]  "
              f"{r.level.value:<20}  action={r.action}")
        history.record(r)

    print(f"\n  Trend (last 5): {history.trend()}")
    print(f"\n  Session Summary:")
    for k, v in history.summary().items():
        print(f"    {k:<22}: {v}")

    # ── Test 4: Sensitivity analysis ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  TEST 4: Sensitivity Analysis at t=3 state")
    print(SEP)
    computer4 = CIIComputer()
    sens = computer4.sensitivity(
        polarity=-0.6, intensity=0.7,
        performance_dev=-1.2, intent_gap=0.6,
        delta=0.1
    )
    for comp, delta_cii in sorted(sens.items(), key=lambda x: -x[1]):
        bar = "#" * int(delta_cii * 100)
        print(f"  {comp:<22}: DCII={delta_cii:.5f}  {bar}")

    # ── Test 5: Validation ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TEST 5: Weight Validation")
    print(SEP)
    try:
        bad = CIIWeights(alpha=0.5, beta=0.5, gamma=0.1, delta=0.1)
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    try:
        neg = CIIWeights(alpha=-0.1, beta=0.5, gamma=0.35, delta=0.25)
    except ValueError as e:
        print(f"  Correctly raised: {e}")
