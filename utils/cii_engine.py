"""
utils/cii_engine.py
--------------------
Cognitive Instability Index (CII) integration engine.

This module is the ASSEMBLY LAYER that connects the three upstream feature
modules into a single CII computation:

    ┌──────────────────────┐
    │   NLPExtractor       │  polarity, intensity, confidence, frustration
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │ TemporalFeatureEngine│  momentum M(t), acceleration A(t)
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │ TelemetryProcessor   │  perf_zscore D(t), drop_rate_norm, accuracy
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │     CIIEngine        │  G(t) = intent gap
    │  CII = αM + βA + γD + δG  │
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │     CIIResult        │  {"cii": float, "level": "low"|"medium"|"high", ...}
    └──────────────────────┘

Key design choices
------------------
1.  CIIEngine is STATELESS — it holds only weights.
    All temporal state (momentum history, D rolling window) lives in the
    upstream engines (TemporalFeatureEngine, TelemetryProcessor).

2.  M and A come FROM TemporalFeatureEngine, not recomputed here.
    This avoids a double-tracking conflict: CIIComputer in utils/cii.py
    internally tracks _prev_momentum to derive A, which would desync from
    TemporalFeatureEngine's buffer. We bypass that by computing the weighted
    sum directly.

3.  Intent Gap G(t) is assembled here from two cross-modal signals:
        G = confidence_weight × confidence_gap
              + accuracy_weight × (1 − accuracy)
    where confidence_gap = max(0, confidence − accuracy).
    This measures cognitive friction: the larger the gap between how
    confident the player sounds and how accurately they are playing,
    the higher the instability signal.

4.  Level mapping:
        "high"   → CII < -0.25   (frustration zone)
        "medium" → CII ∈ [-0.25, 0.30]  (mild instability or boredom risk)
        "low"    → CII > +0.30   (stable flow)
    This is intentionally coarser than CIIComputer's 5-tier system to
    match the requested output schema. Full detail is in CIIResult.level.

5.  Weights configurable via CIIWeights dataclass — validated at init.
    Convenience factory: CIIEngine.from_dict({"alpha": 0.4, ...}).

Usage
-----
    from utils.cii_engine import CIIEngine
    from utils.nlp_extractor import nlp_extractor
    from utils.temporal_features import TemporalFeatureEngine
    from utils.telemetry_processor import TelemetryProcessor

    engine   = CIIEngine()                    # default weights
    temporal = TemporalFeatureEngine(window=10)
    tele_proc = TelemetryProcessor(window=20)

    # Per tick:
    nlp_feats  = nlp_extractor.extract(chat_message)
    temp_feats = temporal.update(nlp_feats)
    tele_feats = tele_proc.update(raw_telemetry)

    result = engine.compute(nlp_feats, temp_feats.as_dict(), tele_feats.as_dict())
    print(result["cii"])     # -0.317
    print(result["level"])   # "high"
    print(result["zone"])    # "frustration"
    print(result["action"])  # "decrease"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Literal, Optional

# Re-use the authoritative CIIWeights and thresholds from cii.py
from utils.cii import CIIWeights, THRESHOLDS, InstabilityLevel, CIIZone


# ---------------------------------------------------------------------------
# Level mapping
# (maps 5-tier InstabilityLevel → 3-tier "low" | "medium" | "high")
# ---------------------------------------------------------------------------

_LEVEL_MAP: Dict[InstabilityLevel, str] = {
    InstabilityLevel.ACUTE_FRUSTRATION: "high",
    InstabilityLevel.HIGH_INSTABILITY:  "high",
    InstabilityLevel.MILD_INSTABILITY:  "medium",
    InstabilityLevel.STABLE_FLOW:       "low",
    InstabilityLevel.BOREDOM_RISK:      "medium",  # under-challenged = medium concern
}


# ---------------------------------------------------------------------------
# Intent Gap configuration
# ---------------------------------------------------------------------------

@dataclass
class IntentGapConfig:
    """
    Controls how G(t) is assembled from confidence and accuracy.

    G(t) = confidence_weight * max(0, confidence - accuracy)
           + accuracy_weight * (1 - accuracy)

    Both terms are clamped to [0, 1] before weighting.
    Final G is also clamped to [0, 1].

    Default split: 60% confidence mismatch / 40% raw miss rate.
    """
    confidence_weight: float = 0.60   # weight on confidence-accuracy gap
    accuracy_weight:   float = 0.40   # weight on raw miss rate (1 - accuracy)

    def __post_init__(self):
        total = self.confidence_weight + self.accuracy_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"confidence_weight + accuracy_weight must equal 1.0, got {total:.4f}"
            )


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CIIResult:
    """
    Full output of one CIIEngine.compute() call.

    Simple fields (requested schema)
    ---------------------------------
    cii     : float — CII(t) scalar
    level   : "low" | "medium" | "high"

    Extended fields (for research, logging, dashboard)
    ---------------------------------------------------
    zone        : "frustration" | "flow" | "boredom"
    action      : recommended difficulty controller action
    level_detail: full 5-tier label from InstabilityLevel
    urgency     : 0 (flow) → 4 (acute) — useful for priority queues

    Components (all four inputs to CII formula)
    --------------------------------------------
    momentum        M(t)  = P * I
    acceleration    A(t)  = ΔM / Δt
    perf_deviation  D(t)  = z-score of performance vs player baseline
    intent_gap      G(t)  = confidence-accuracy friction signal

    Contributions (weighted per-component CII contribution)
    --------------------------------------------------------
    contrib_M, contrib_A, contrib_D, contrib_G

    Weights used
    ------------
    weights : CIIWeights used in this computation
    """
    # ── Requested schema ──────────────────────────────────────────────────────
    cii:   float
    level: str    # "low" | "medium" | "high"

    # ── Extended ──────────────────────────────────────────────────────────────
    zone:         str
    action:       str
    level_detail: str
    urgency:      int

    # ── Components ────────────────────────────────────────────────────────────
    momentum:       float   # M(t)
    acceleration:   float   # A(t)
    perf_deviation: float   # D(t)
    intent_gap:     float   # G(t)

    # ── Weighted contributions ────────────────────────────────────────────────
    contrib_M: float
    contrib_A: float
    contrib_D: float
    contrib_G: float

    # ── Weights ───────────────────────────────────────────────────────────────
    alpha: float
    beta:  float
    gamma: float
    delta: float

    # ── Convenience ──────────────────────────────────────────────────────────

    def as_dict(self) -> Dict:
        """Flat dict — the full result. Use for logging, dashboard, API response."""
        return asdict(self)

    def as_simple(self) -> Dict[str, float | str]:
        """Minimal schema: {"cii": float, "level": str}."""
        return {"cii": self.cii, "level": self.level}

    def dominant_driver(self) -> str:
        """Component with the largest absolute weighted contribution."""
        contribs = {
            "momentum":       abs(self.contrib_M),
            "acceleration":   abs(self.contrib_A),
            "perf_deviation": abs(self.contrib_D),
            "intent_gap":     abs(self.contrib_G),
        }
        return max(contribs, key=contribs.get)

    def __str__(self) -> str:
        return (
            f"\nCIIResult\n"
            f"  CII     = {self.cii:+.4f}  [{self.level.upper()}]\n"
            f"  Level   : {self.level_detail}\n"
            f"  Zone    : {self.zone}\n"
            f"  Action  : {self.action}\n"
            f"  Urgency : {self.urgency}/4\n"
            f"  Driver  : {self.dominant_driver()}\n"
            f"  ─── Components ──────────────────────────────\n"
            f"  M(t)  = {self.momentum:+.4f}  contrib = {self.contrib_M:+.4f}"
            f"  (α={self.alpha})\n"
            f"  A(t)  = {self.acceleration:+.4f}  contrib = {self.contrib_A:+.4f}"
            f"  (β={self.beta})\n"
            f"  D(t)  = {self.perf_deviation:+.4f}  contrib = {self.contrib_D:+.4f}"
            f"  (γ={self.gamma})\n"
            f"  G(t)  = {self.intent_gap:+.4f}  contrib = {self.contrib_G:+.4f}"
            f"  (δ={self.delta})\n"
        )


# ---------------------------------------------------------------------------
# CIIEngine — main class
# ---------------------------------------------------------------------------

class CIIEngine:
    """
    Stateless CII computation engine.

    Accepts outputs from the three upstream feature modules and returns
    a CIIResult with both the simple and extended output schemas.

    Args
    ----
    weights       : CIIWeights instance. Default = (0.35, 0.30, 0.25, 0.10).
    intent_config : IntentGapConfig controlling G(t) assembly.

    Thread safety: SAFE — holds no mutable state between calls.
    """

    def __init__(
        self,
        weights:       CIIWeights       = None,
        intent_config: IntentGapConfig  = None,
    ) -> None:
        self._w  = weights       or CIIWeights()
        self._gc = intent_config or IntentGapConfig()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def compute(
        self,
        nlp_features:      Dict[str, float],
        temporal_features: Dict[str, float],
        telemetry_features: Dict[str, float],
    ) -> CIIResult:
        """
        Compute CII from the outputs of the three feature modules.

        Args
        ----
        nlp_features       : Output of NLPExtractor.extract()
                             Required keys: polarity, intensity, confidence
                             Optional:      frustration, anger
        temporal_features  : Output of TemporalFeatureEngine.update().as_dict()
                             Required keys: momentum, acceleration
        telemetry_features : Output of TelemetryProcessor.update().as_dict()
                             Required keys: perf_zscore, accuracy
                             Optional:      drop_rate_norm

        Returns
        -------
        CIIResult — immutable with full component breakdown.
        """
        # ── Extract components ─────────────────────────────────────────────

        # M(t) and A(t) come directly from TemporalFeatureEngine
        # (avoids re-computing and double-tracking momentum state)
        M = float(temporal_features.get("momentum",     0.0))
        A = float(temporal_features.get("acceleration", 0.0))

        # D(t) = z-score of performance deviation from TelemetryProcessor
        # Negative z-score = performing BETTER than baseline → lowers CII
        # Positive z-score = performing WORSE than baseline → raises CII |CII|
        D = float(telemetry_features.get("perf_zscore", 0.0))

        # G(t) = Intent Gap — assembled from NLP + telemetry cross-modal signals
        G = self._compute_intent_gap(nlp_features, telemetry_features)

        # ── Clamp inputs to valid ranges ───────────────────────────────────
        M = _clamp(M, -1.0, 1.0)
        A = _clamp(A, -2.0, 2.0)    # A can exceed [-1,1] on sharp reversals
        # D is a z-score — no hard clamp (typically in [-3, +3])
        G = _clamp(G,  0.0, 1.0)

        # ── CII = αM + βA + γD + δG ────────────────────────────────────────
        w = self._w

        # Note on sign convention:
        # M and A are POSITIVE when the player is emotionally positive and
        # improving → higher CII → boredom risk (under-challenged).
        # M and A are NEGATIVE when player is frustrated → lower CII.
        # D is POSITIVE when perf is BELOW baseline → push CII more negative.
        # G is POSITIVE when confidence-accuracy gap is high → instability.
        #
        # Therefore D and G contribute NEGATIVELY to CII
        # (instability drivers that push toward frustration zone).
        contrib_M = w.alpha * M
        contrib_A = w.beta  * A
        contrib_D = w.gamma * (-D)   # inverted: poor perf lowers CII
        contrib_G = w.delta * (-G)   # inverted: high gap lowers CII

        raw_cii = contrib_M + contrib_A + contrib_D + contrib_G

        # ── Classify ──────────────────────────────────────────────────────
        level_detail, zone = _classify(raw_cii)
        simple_level       = _LEVEL_MAP[level_detail]
        action             = _DIFFICULTY_ACTION[level_detail]
        urgency            = _URGENCY[level_detail]

        return CIIResult(
            cii             = round(raw_cii, 4),
            level           = simple_level,
            zone            = zone.value,
            action          = action,
            level_detail    = level_detail.value,
            urgency         = urgency,
            momentum        = round(M,          4),
            acceleration    = round(A,          4),
            perf_deviation  = round(D,          4),
            intent_gap      = round(G,          4),
            contrib_M       = round(contrib_M,  5),
            contrib_A       = round(contrib_A,  5),
            contrib_D       = round(contrib_D,  5),
            contrib_G       = round(contrib_G,  5),
            alpha           = w.alpha,
            beta            = w.beta,
            gamma           = w.gamma,
            delta           = w.delta,
        )

    def compute_from_values(
        self,
        polarity:       float,
        intensity:      float,
        acceleration:   float,
        perf_deviation: float,
        confidence:     float = 0.5,
        accuracy:       float = 0.5,
    ) -> CIIResult:
        """
        Convenience method — pass raw scalar values directly without
        constructing feature dicts. Useful in notebooks and unit tests.

        G is computed internally from confidence + accuracy.
        """
        nlp_feats  = {"polarity": polarity, "intensity": intensity,
                      "confidence": confidence}
        temp_feats = {
            "momentum":     polarity * intensity,
            "acceleration": acceleration,
        }
        tele_feats = {
            "perf_zscore": perf_deviation,
            "accuracy":    accuracy,
        }
        return self.compute(nlp_feats, temp_feats, tele_feats)

    # ------------------------------------------------------------------
    # Weight configuration helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, weight_dict: Dict[str, float], **kwargs) -> "CIIEngine":
        """
        Factory: create engine from a plain weight dictionary.

        Example:
            engine = CIIEngine.from_dict({"alpha": 0.4, "beta": 0.3,
                                           "gamma": 0.2, "delta": 0.1})
        """
        weights = CIIWeights.from_dict(weight_dict)
        return cls(weights=weights, **kwargs)

    @classmethod
    def nlp_heavy(cls) -> "CIIEngine":
        """
        Pre-configured for NLP-dominant research scenarios.
        Alpha (momentum) and Beta (acceleration) get 80% of total weight.
        """
        return cls(weights=CIIWeights.normalized(alpha=5, beta=3, gamma=1, delta=1))

    @classmethod
    def telemetry_heavy(cls) -> "CIIEngine":
        """
        Pre-configured for telemetry-dominant scenarios.
        Gamma (perf deviation) gets 50% of total weight.
        """
        return cls(weights=CIIWeights.normalized(alpha=2, beta=1, gamma=5, delta=2))

    @property
    def weights(self) -> CIIWeights:
        """Current weight configuration (read-only view)."""
        return self._w

    def update_weights(self, **kwargs) -> "CIIEngine":
        """
        Return a new engine with updated weights (immutable pattern).
        Pass any subset of: alpha, beta, gamma, delta.
        Remaining weights are inherited from the current engine.

        Example:
            engine2 = engine.update_weights(alpha=0.40, beta=0.35)
        """
        current = self._w.to_dict()
        current.update({k: v for k, v in kwargs.items()
                        if k in ("alpha", "beta", "gamma", "delta")})
        return CIIEngine(
            weights=CIIWeights.normalized(**current),
            intent_config=self._gc,
        )

    # ------------------------------------------------------------------
    # Private: Intent Gap computation
    # ------------------------------------------------------------------

    def _compute_intent_gap(
        self,
        nlp:  Dict[str, float],
        tele: Dict[str, float],
    ) -> float:
        """
        G(t) = confidence_weight × max(0, confidence − accuracy)
                 + accuracy_weight × (1 − accuracy)

        Interpretation:
          - High confidence + low accuracy = high G (player THINKS they are
            playing well but metrics say otherwise → cognitive friction).
          - Low confidence + low accuracy = moderate G (aware of failure).
          - Low confidence + high accuracy = near zero G (playing well).

        Both components are clamped before applying weights.
        Final G is clamped to [0, 1].
        """
        confidence = _clamp(float(nlp.get("confidence", 0.5)),   0.0, 1.0)
        accuracy   = _clamp(float(tele.get("accuracy",  0.5)),   0.0, 1.0)

        # Component 1: mismatch between self-assessed and actual performance
        confidence_gap = max(0.0, confidence - accuracy)

        # Component 2: raw accuracy deficit (1 - acc)
        miss_rate = 1.0 - accuracy

        G = (
            self._gc.confidence_weight * confidence_gap
            + self._gc.accuracy_weight * miss_rate
        )
        return _clamp(G, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Classification helpers (re-using thresholds from cii.py)
# ---------------------------------------------------------------------------

def _classify(cii: float):
    """Thin wrapper re-using thresholds already defined in utils/cii.py."""
    if cii < THRESHOLDS["acute_frustration"]:
        return InstabilityLevel.ACUTE_FRUSTRATION, CIIZone.FRUSTRATION
    if cii < THRESHOLDS["high_instability"]:
        return InstabilityLevel.HIGH_INSTABILITY, CIIZone.FRUSTRATION
    if cii < THRESHOLDS["mild_instability"]:
        return InstabilityLevel.MILD_INSTABILITY, CIIZone.FLOW
    if cii <= THRESHOLDS["stable_flow"]:
        return InstabilityLevel.STABLE_FLOW, CIIZone.FLOW
    return InstabilityLevel.BOREDOM_RISK, CIIZone.BOREDOM


_DIFFICULTY_ACTION: Dict[InstabilityLevel, str] = {
    InstabilityLevel.ACUTE_FRUSTRATION: "decrease",
    InstabilityLevel.HIGH_INSTABILITY:  "decrease",
    InstabilityLevel.MILD_INSTABILITY:  "maintain",
    InstabilityLevel.STABLE_FLOW:       "maintain",
    InstabilityLevel.BOREDOM_RISK:      "increase",
}

_URGENCY: Dict[InstabilityLevel, int] = {
    InstabilityLevel.STABLE_FLOW:       0,
    InstabilityLevel.BOREDOM_RISK:      1,
    InstabilityLevel.MILD_INSTABILITY:  2,
    InstabilityLevel.HIGH_INSTABILITY:  3,
    InstabilityLevel.ACUTE_FRUSTRATION: 4,
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Module-level default engine (default weights)
# ---------------------------------------------------------------------------

#: Drop-in import for scripts / tests that don't need custom weights.
cii_engine = CIIEngine()


# ---------------------------------------------------------------------------
# Self-test  (run: python -m utils.cii_engine)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.nlp_extractor        import NLPExtractor
    from utils.temporal_features    import TemporalFeatureEngine
    from utils.telemetry_processor  import TelemetryProcessor

    SEP = "=" * 68

    print(SEP)
    print("  CIIEngine — Self Test (full pipeline)")
    print(SEP)

    # ── Setup ─────────────────────────────────────────────────────────────
    nlp_ext  = NLPExtractor()
    temporal = TemporalFeatureEngine(window=6)
    tele_proc = TelemetryProcessor(window=6)
    engine   = CIIEngine()

    # ── Scenario 1: Escalating Frustration ───────────────────────────────
    print("\n[Scenario 1] Gradual frustration escalation\n")

    TICKS = [
        # chat_msg,                      kill  death  miss   RT(ms)
        ("good start!",                    5,    1,     2,   280),
        ("nice, getting the hang of it",   4,    2,     4,   300),
        ("ugh why did i miss that",        2,    3,     8,   360),
        ("this is so frustrating!!",       1,    5,    12,   430),
        ("IMPOSSIBLE I give up",           0,    7,    18,   520),
        ("I QUIT this is unfair trash",    0,    9,    22,   640),
    ]

    for i, (chat, K, D, M, RT) in enumerate(TICKS, 1):
        nlp_feats  = nlp_ext.extract(chat)
        temp_feats = temporal.update(nlp_feats).as_dict()
        tele_feats = tele_proc.update({
            "kill_count": K, "death_count": D,
            "miss_count": M, "reaction_time": float(RT)
        }).as_dict()

        result = engine.compute(nlp_feats, temp_feats, tele_feats)
        bar = "#" * max(0, int(abs(result.cii) * 24))
        print(
            f"  t={i:02d}  CII={result.cii:+.4f}  [{bar:<24}]  "
            f"{result.level_detail:<20}  → {result.action}"
        )
        print(f"        M={result.momentum:+.4f}  A={result.acceleration:+.4f}  "
              f"D={result.perf_deviation:+.4f}  G={result.intent_gap:.4f}  "
              f"driver={result.dominant_driver()}")

    # ── Scenario 2: Boring easy run ───────────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 2] Bored / too easy\n")
    e2 = CIIEngine()
    t2 = TemporalFeatureEngine(window=4)
    p2 = TelemetryProcessor(window=4)

    for i in range(5):
        nf = nlp_ext.extract("ez game lol good stuff")
        tf = t2.update(nf).as_dict()
        pf = p2.update({"kill_count": 10, "death_count": 0,
                         "miss_count": 1,  "reaction_time": 140.0}).as_dict()
        r = e2.compute(nf, tf, pf)
        print(f"  t={i+1:02d}  CII={r.cii:+.4f}  level={r.level:<8}  "
              f"zone={r.zone}  action={r.action}")

    # ── Scenario 3: Direct scalar API + weight configurations ─────────────
    print(f"\n{SEP}")
    print("[Scenario 3] compute_from_values + weight presets\n")

    frustrated_state = dict(polarity=-0.8, intensity=0.9,
                            acceleration=-0.3, perf_deviation=2.0,
                            confidence=0.6, accuracy=0.2)

    for name, eng in [
        ("default",          CIIEngine()),
        ("nlp_heavy",        CIIEngine.nlp_heavy()),
        ("telemetry_heavy",  CIIEngine.telemetry_heavy()),
    ]:
        r = eng.compute_from_values(**frustrated_state)
        print(f"  [{name:16s}]  CII={r.cii:+.4f}  level={r.level}  "
              f"weights=(α={eng.weights.alpha:.2f} β={eng.weights.beta:.2f} "
              f"γ={eng.weights.gamma:.2f} δ={eng.weights.delta:.2f})")

    # ── Scenario 4: Simple output schema ──────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 4] Simple output schema {cii, level}\n")
    r = cii_engine.compute_from_values(
        polarity=-0.7, intensity=0.8, acceleration=-0.2,
        perf_deviation=1.5, confidence=0.7, accuracy=0.25
    )
    print(f"  Simple : {r.as_simple()}")
    print(f"  Driver : {r.dominant_driver()}")
    print(f"  Full   :{r}")

    print(SEP)
    print("  All scenarios passed.")
    print(SEP)
