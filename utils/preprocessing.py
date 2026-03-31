"""
utils/preprocessing.py
-----------------------
Standalone data pipeline module for the Cognitive Regulation System.

Responsibilities (4-stage pipeline):
  Stage 1 — Validate    : Type checks, range assertions, missing field defaults
  Stage 2 — Normalize   : Scale raw telemetry to [0, 1] / z-score
  Stage 3 — Extract     : Compute NLP features from text (or pass-through)
  Stage 4 — Derive      : Emotional Momentum, Acceleration, Performance Deviation
                          -> Return FeatureVector

Design principles:
  - Fully standalone: no FastAPI, no registries, no session state required
  - Reusable: call process() with any raw dict or pass typed objects
  - Modular: each stage is a pure function — easy to test independently
  - Compatible: FeatureVector feeds directly into EmotionalDynamicsEngine
    and CognitiveStatePredictor in the app/ backend

Usage:
    from utils.preprocessing import DataPipeline

    pipeline = DataPipeline()

    vector = pipeline.process(
        telemetry={"deaths": 5, "retries": 3, "score_delta": -15,
                   "streak": -2, "reaction_time_ms": 450, "input_speed": 2.1},
        nlp={"text": "this is impossible", "intent_gap": 0.6},
    )
    print(vector)
    # FeatureVector(polarity=-1.0, intensity=0.39, perf_dev=0.0,
    #               intent_gap=0.6, momentum=-0.39, acceleration=-0.39, cii=-0.25)
"""

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Output: Structured Feature Vector
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    The canonical output of the data pipeline.

    Contains all 7 features used by the predictor and LSTM:
      Raw NLP   : polarity, intensity, intent_gap
      Derived   : momentum (M), acceleration (A), perf_dev (D)
      Composite : cii (Cognitive Instability Index)

    Extras for traceability:
      raw_telemetry  : dict of the validated/normalized telemetry values
      raw_nlp        : dict of the validated NLP inputs
      warnings       : list of non-fatal validation warnings
    """
    # ── Core features ───────────────────────────────────────────────────────
    polarity:      float = 0.0   # P(t)  ∈ [-1, +1]
    intensity:     float = 0.0   # I(t)  ∈ [ 0,  1]
    intent_gap:    float = 0.0   # G(t)  ∈ [ 0,  1]
    perf_dev:      float = 0.0   # D(t)  ∈  R   (z-score)
    momentum:      float = 0.0   # M(t) = P * I
    acceleration:  float = 0.0   # A(t) = ΔM / Δt
    cii:           float = 0.0   # CII(t) = αM + βA + γD + δG

    # ── Traceability ─────────────────────────────────────────────────────────
    raw_telemetry: Dict[str, float] = field(default_factory=dict)
    raw_nlp:       Dict[str, Any]   = field(default_factory=dict)
    warnings:      List[str]        = field(default_factory=list)

    def as_lstm_input(self) -> List[float]:
        """Returns [CII, M, A, D, G] — the 5-dim LSTM input vector."""
        return [self.cii, self.momentum, self.acceleration,
                self.perf_dev, self.intent_gap]

    def as_dict(self) -> dict:
        """Flat dict representation (excludes nested raw fields)."""
        return {
            "polarity":     self.polarity,
            "intensity":    self.intensity,
            "intent_gap":   self.intent_gap,
            "perf_dev":     self.perf_dev,
            "momentum":     self.momentum,
            "acceleration": self.acceleration,
            "cii":          self.cii,
        }

    def __str__(self) -> str:
        w = f"  warnings={self.warnings}" if self.warnings else ""
        return (
            f"FeatureVector(\n"
            f"  polarity={self.polarity:+.4f}  intensity={self.intensity:.4f}  "
            f"intent_gap={self.intent_gap:.4f}\n"
            f"  perf_dev={self.perf_dev:+.4f}  momentum={self.momentum:+.4f}  "
            f"acceleration={self.acceleration:+.4f}\n"
            f"  cii={self.cii:+.4f}{w}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Telemetry field specifications (for validation + normalization)
# ---------------------------------------------------------------------------

TELEMETRY_SPEC: Dict[str, dict] = {
    #  field name         type     default   min     max      norm_max
    "deaths":          {"type": int,   "default": 0,    "min": 0,    "max": 50,   "norm": 50.0},
    "retries":         {"type": int,   "default": 0,    "min": 0,    "max": 50,   "norm": 50.0},
    "score_delta":     {"type": float, "default": 0.0,  "min": -200, "max": 200,  "norm": 200.0},
    "streak":          {"type": int,   "default": 0,    "min": -30,  "max": 30,   "norm": 30.0},
    "reaction_time_ms":{"type": float, "default": 300.0,"min": 50,   "max": 2000, "norm": 2000.0},
    "input_speed":     {"type": float, "default": 3.0,  "min": 0,    "max": 20,   "norm": 20.0},
}

NLP_SPEC: Dict[str, dict] = {
    "polarity":  {"type": float, "default": 0.0,  "min": -1.0, "max": 1.0},
    "intensity": {"type": float, "default": 0.0,  "min":  0.0, "max": 1.0},
    "intent_gap":{"type": float, "default": 0.0,  "min":  0.0, "max": 1.0},
}

# ---------------------------------------------------------------------------
# NLP lexicon (standalone — no dependency on app/core/nlp.py)
# ---------------------------------------------------------------------------

_NEG_WORDS = {
    "impossible", "broken", "trash", "quit", "rage", "garbage", "unfair",
    "stupid", "hate", "awful", "terrible", "worst", "useless", "bug",
    "hard", "difficult", "frustrating", "annoying", "stuck", "died",
}
_POS_WORDS = {
    "nice", "cool", "great", "love", "amazing", "good", "win", "yes",
    "awesome", "easy", "fun", "challenge", "better", "finally", "close",
}
_STRONG_NEG = {"trash", "quit", "garbage", "broken", "hate", "worst"}
_STRONG_POS = {"amazing", "perfect", "awesome", "love"}


# ---------------------------------------------------------------------------
# Stage 1 — Validation
# ---------------------------------------------------------------------------

class Validator:
    """
    Validates raw input dicts against TELEMETRY_SPEC and NLP_SPEC.

    - Missing fields receive their default value (non-fatal, adds warning)
    - Wrong type is coerced if safely possible (non-fatal, adds warning)
    - Out-of-range values are clamped (non-fatal, adds warning)
    - Unrecognised keys are ignored silently

    Returns: (validated_dict, warnings: List[str])
    """

    def validate_telemetry(self, raw: Dict) -> Tuple[Dict[str, float], List[str]]:
        return self._validate(raw, TELEMETRY_SPEC, "telemetry")

    def validate_nlp(self, raw: Dict) -> Tuple[Dict[str, Any], List[str]]:
        base, warnings = self._validate(raw, NLP_SPEC, "nlp")
        # Text is handled separately — pass through as-is
        base["text"] = str(raw.get("text", "")).strip()
        return base, warnings

    @staticmethod
    def _validate(
        raw:  Dict,
        spec: Dict[str, dict],
        name: str,
    ) -> Tuple[Dict, List[str]]:
        out      = {}
        warnings = []
        for field_name, rules in spec.items():
            raw_val  = raw.get(field_name)
            expected = rules["type"]

            # ── Missing field ─────────────────────────────────────────────
            if raw_val is None:
                out[field_name] = rules["default"]
                warnings.append(
                    f"[{name}] Missing field '{field_name}', "
                    f"using default={rules['default']}"
                )
                continue

            # ── Type coercion ─────────────────────────────────────────────
            try:
                val = expected(raw_val)
            except (TypeError, ValueError):
                out[field_name] = rules["default"]
                warnings.append(
                    f"[{name}] Could not cast '{field_name}'={raw_val!r} "
                    f"to {expected.__name__}, using default={rules['default']}"
                )
                continue

            # ── Range clamp ───────────────────────────────────────────────
            lo, hi = rules.get("min"), rules.get("max")
            if lo is not None and val < lo:
                warnings.append(
                    f"[{name}] '{field_name}'={val} < min={lo}, clamping."
                )
                val = type(val)(lo)
            if hi is not None and val > hi:
                warnings.append(
                    f"[{name}] '{field_name}'={val} > max={hi}, clamping."
                )
                val = type(val)(hi)

            out[field_name] = val
        return out, warnings


# ---------------------------------------------------------------------------
# Stage 2 — Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Normalizes validated telemetry values to [0, 1] using
    field-specific norm_max from TELEMETRY_SPEC.

    NLP features (polarity, intensity, intent_gap) are already
    in their natural ranges and need no further normalization.

    Reaction time is inverted: higher RT → lower normalized value
    (to represent performance degradation naturally).
    """

    @staticmethod
    def normalize_telemetry(validated: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for field_name, rules in TELEMETRY_SPEC.items():
            val      = validated[field_name]
            norm_max = rules["norm"]
            norm_val = val / norm_max if norm_max != 0 else 0.0

            # Invert reaction_time — high latency is bad
            if field_name == "reaction_time_ms":
                norm_val = 1.0 - norm_val

            # Signed normalization for signed fields (score_delta, streak)
            if field_name in ("score_delta", "streak"):
                lo   = rules["min"]
                span = rules["max"] - lo
                norm_val = (val - lo) / span if span > 0 else 0.5

            normalized[field_name] = round(max(0.0, min(1.0, norm_val)), 4)

        return normalized


# ---------------------------------------------------------------------------
# Stage 3 — NLP Feature Extraction (standalone)
# ---------------------------------------------------------------------------

class StandaloneNLPExtractor:
    """
    Lightweight NLP extractor that does NOT depend on app/core/nlp.py.

    If text is present, computes polarity and intensity from a
    rule-based lexicon. If polarity/intensity are provided by the caller
    (e.g., from a real transformer model), those take priority.
    """

    def extract(self, validated_nlp: Dict) -> Dict[str, float]:
        """
        Merge caller-provided values with text-derived values.
        Caller-provided values always win (they may come from a real model).
        """
        text      = validated_nlp.get("text", "")
        polarity  = validated_nlp["polarity"]
        intensity = validated_nlp["intensity"]
        intent_gap = validated_nlp["intent_gap"]

        # Only use analyzer if no real polarity was provided
        if text and polarity == 0.0:
            # Integrate the real-time EmotionAnalyzer
            from utils.emotion_analyzer import emotion_analyzer
            analysis = emotion_analyzer.analyze(text)
            polarity  = analysis["polarity"]
            intensity = analysis["intensity"]
            # Expose new probabilities to the raw_nlp payload for downstream visibility
            validated_nlp["anger_prob"] = analysis["anger"]
            validated_nlp["frustration_prob"] = analysis["frustration"]
            validated_nlp["confidence_prob"] = analysis["confidence"]

        return {
            "polarity":  round(max(-1.0, min(1.0, polarity)),  4),
            "intensity": round(max( 0.0, min(1.0, intensity)), 4),
            "intent_gap": round(max(0.0, min(1.0, intent_gap)), 4),
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _polarity(self, text: str) -> float:
        tokens    = self._tokenize(text)
        neg = sum(1 for t in tokens if t in _NEG_WORDS)
        pos = sum(1 for t in tokens if t in _POS_WORDS)
        total = neg + pos
        if total == 0:
            return 0.0
        raw = (pos - neg) / total
        if any(t in _STRONG_NEG for t in tokens):
            raw = min(raw - 0.35, -0.35)
        if any(t in _STRONG_POS for t in tokens):
            raw = max(raw + 0.20,  0.20)
        return max(-1.0, min(1.0, raw))

    def _intensity(self, text: str, polarity: float) -> float:
        if not text:
            return 0.0
        alpha = [c for c in text if c.isalpha()]
        caps  = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
        excl  = min(text.count("!") / max(len(text.split()), 1), 1.0)
        rep   = 1.0 if re.search(r"(.)\1{2,}", text) else 0.0
        strong = 1.0 if any(
            t in _STRONG_NEG or t in _STRONG_POS
            for t in self._tokenize(text)
        ) else 0.0
        raw = 0.35 * caps + 0.25 * excl + 0.20 * rep + 0.20 * strong
        raw *= 0.3 + 0.7 * abs(polarity)
        return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Stage 4 — Derived Feature Computation
# ---------------------------------------------------------------------------

class DerivedFeatureComputer:
    """
    Computes the three derived temporal features from NLP + telemetry:

      M(t) = P(t) * I(t)                    Emotional Momentum
      A(t) = (M(t) - M(t-1)) / delta_t     Emotional Acceleration
      D(t) = z-score of composite perf.     Performance Deviation

    Maintains prev_momentum and rolling perf history for statefulness.
    Call reset() between independent sessions.
    """

    # CII weights (αM + βA + γD + δG)
    ALPHA = 0.35
    BETA  = 0.30
    GAMMA = 0.25
    DELTA = 0.10

    def __init__(self, delta_t: float = 1.0, perf_window: int = 20):
        self._delta_t       = delta_t
        self._prev_momentum = 0.0
        self._perf_history: List[float] = []
        self._perf_window   = perf_window

    def compute(
        self,
        nlp_feats:   Dict[str, float],
        norm_telem:  Dict[str, float],
    ) -> Tuple[float, float, float, float]:
        """
        Returns (momentum, acceleration, perf_dev, cii).
        """
        P = nlp_feats["polarity"]
        I = nlp_feats["intensity"]
        G = nlp_feats["intent_gap"]

        # ── Emotional Momentum ────────────────────────────────────────────
        M = P * I

        # ── Emotional Acceleration ─────────────────────────────────────────
        A = (M - self._prev_momentum) / self._delta_t
        self._prev_momentum = M

        # ── Performance Deviation (z-score) ────────────────────────────────
        perf_score = self._composite_perf(norm_telem)
        self._perf_history.append(perf_score)
        if len(self._perf_history) > self._perf_window:
            self._perf_history.pop(0)
        D = self._z_score(perf_score)

        # ── CII ────────────────────────────────────────────────────────────
        CII = (
            self.ALPHA * M +
            self.BETA  * A +
            self.GAMMA * D +
            self.DELTA * G
        )

        return (
            round(M,   4),
            round(A,   4),
            round(D,   4),
            round(CII, 4),
        )

    @staticmethod
    def _composite_perf(norm: Dict[str, float]) -> float:
        """
        Composite performance score from normalized telemetry.
        All inputs are ∈ [0, 1] after normalization stage.

        Higher = better performance:
          + score_delta_norm  (gain is positive)
          - deaths_norm       (deaths penalize performance)
          - retries_norm      (retries penalize performance)
          + streak_norm       (positive streak boosts)
          + reaction_norm     (already inverted: high RT → low score)
        """
        return (
              norm.get("score_delta",      0.5) * 1.0
            - norm.get("deaths",           0.0) * 1.5
            - norm.get("retries",          0.0) * 1.0
            + (norm.get("streak",          0.5) - 0.5) * 0.8
            + norm.get("reaction_time_ms", 0.5) * 0.5
        )

    def _z_score(self, score: float) -> float:
        h = self._perf_history
        if len(h) < 2:
            return 0.0
        mean = sum(h) / len(h)
        var  = sum((x - mean) ** 2 for x in h) / len(h)
        std  = math.sqrt(var) if var > 1e-9 else 1.0
        return (score - mean) / std

    def reset(self):
        self._prev_momentum = 0.0
        self._perf_history.clear()


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Orchestrates all 4 stages into a single call.

    Instantiate once per session (maintains acceleration + perf state).
    Call reset() to start a new session without creating a new object.

    Example:
        pipeline = DataPipeline()

        # Step-by-step
        v1 = pipeline.process(
            telemetry={"deaths": 1, "retries": 1, "score_delta": -5,
                       "reaction_time_ms": 320, "input_speed": 2.5},
            nlp={"text": "almost had it", "intent_gap": 0.2},
        )

        v2 = pipeline.process(
            telemetry={"deaths": 5, "retries": 4, "score_delta": -20,
                       "reaction_time_ms": 450, "input_speed": 2.1},
            nlp={"text": "this is impossible", "intent_gap": 0.65},
        )

        # Batch processing
        vectors = pipeline.process_batch([
            {"telemetry": {...}, "nlp": {...}},
            ...
        ])
    """

    def __init__(self, delta_t: float = 1.0, perf_window: int = 20):
        self._validator  = Validator()
        self._normalizer = Normalizer()
        self._nlp        = StandaloneNLPExtractor()
        self._deriver    = DerivedFeatureComputer(delta_t, perf_window)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(
        self,
        telemetry: Dict,
        nlp:       Dict,
    ) -> FeatureVector:
        """
        Process one timestep. Returns FeatureVector.

        Args:
            telemetry : dict with any subset of TELEMETRY_SPEC keys
            nlp       : dict with optional 'text', 'polarity', 'intensity',
                        'intent_gap' keys
        """
        all_warnings: List[str] = []

        # ── Stage 1: Validate ─────────────────────────────────────────────
        val_telem, w1 = self._validator.validate_telemetry(telemetry)
        val_nlp,   w2 = self._validator.validate_nlp(nlp)
        all_warnings.extend(w1 + w2)

        # ── Stage 2: Normalize telemetry ──────────────────────────────────
        norm_telem = self._normalizer.normalize_telemetry(val_telem)

        # ── Stage 3: Extract NLP features ─────────────────────────────────
        nlp_feats = self._nlp.extract(val_nlp)

        # ── Stage 4: Compute derived features ────────────────────────────
        M, A, D, CII = self._deriver.compute(nlp_feats, norm_telem)

        return FeatureVector(
            polarity      = nlp_feats["polarity"],
            intensity     = nlp_feats["intensity"],
            intent_gap    = nlp_feats["intent_gap"],
            perf_dev      = D,
            momentum      = M,
            acceleration  = A,
            cii           = CII,
            raw_telemetry = norm_telem,
            raw_nlp       = val_nlp,
            warnings      = all_warnings,
        )

    def process_batch(self, steps: List[Dict]) -> List[FeatureVector]:
        """
        Process a list of {'telemetry': {...}, 'nlp': {...}} dicts in order.
        Maintains temporal state (acceleration, perf history) across steps.
        """
        return [
            self.process(
                telemetry = step.get("telemetry", {}),
                nlp       = step.get("nlp", {}),
            )
            for step in steps
        ]

    def reset(self):
        """Reset all temporal state for a new session."""
        self._deriver.reset()


# ---------------------------------------------------------------------------
# Quick self-test (run: python -m utils.preprocessing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Data Pipeline — Self Test")
    print("=" * 60)

    pipeline = DataPipeline()

    steps = [
        {
            "telemetry": {"deaths": 1, "retries": 1, "score_delta": -5,
                          "reaction_time_ms": 320, "input_speed": 2.5},
            "nlp":       {"text": "almost had it", "intent_gap": 0.2},
        },
        {
            "telemetry": {"deaths": 3, "retries": 2, "score_delta": -12,
                          "reaction_time_ms": 380, "input_speed": 2.1},
            "nlp":       {"text": "why does this keep happening", "intent_gap": 0.4},
        },
        {
            "telemetry": {"deaths": 5, "retries": 4, "score_delta": -20,
                          "reaction_time_ms": 450, "input_speed": 1.9},
            "nlp":       {"text": "this is impossible", "intent_gap": 0.6},
        },
        {
            "telemetry": {"deaths": 8, "retries": 7, "score_delta": -30,
                          "reaction_time_ms": 510, "input_speed": 1.5},
            "nlp":       {"text": "I QUIT this is trash", "intent_gap": 0.8},
        },
    ]

    print("\n[Batch processing — Escalating Frustration scenario]\n")
    vectors = pipeline.process_batch(steps)

    for i, v in enumerate(vectors, 1):
        label = (
            "[!!] FRUSTRATED" if v.cii < -0.35
            else "[~~] BORED" if v.cii > 0.30
            else "[OK] ENGAGED"
        )
        print(f"t={i:02d}  CII={v.cii:+.4f}  M={v.momentum:+.4f}  "
              f"A={v.acceleration:+.4f}  D={v.perf_dev:+.4f}  -> {label}")
        if v.warnings:
            for w in v.warnings:
                print(f"       WARN: {w}")

    print("\n[LSTM input vector for t=03]:")
    print(" ", vectors[2].as_lstm_input())

    print("\n[Validation test — missing fields + bad types]\n")
    pipeline.reset()
    bad_input = pipeline.process(
        telemetry={"deaths": "five", "score_delta": -99999},
        nlp={"polarity": 2.5, "text": ""},
    )
    print(bad_input)
