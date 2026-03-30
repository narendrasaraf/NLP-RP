"""
utils/predictor.py
-------------------
Standalone player cognitive state prediction module.

Input  : CIIResult (from utils/cii.py) or raw (cii, momentum, acceleration)
Output : PredictionResult (state, confidence, per-class scores, reasoning)

Architecture (designed for ML upgrade):
  ┌──────────────────────────────────────────────────┐
  │           StatePredictor  (facade)               │
  │  .predict(cii_result) -> PredictionResult        │
  └──────┬─────────────────────────────┬─────────────┘
         │                             │
  ┌──────▼──────────┐        ┌─────────▼──────────────┐
  │ RuleBasedPredictor│      │ MLPredictor (stub)      │
  │  (default, fast) │      │  sklearn / PyTorch drop-in│
  └──────────────────┘      └────────────────────────────┘

Usage:
    from utils.cii import CIIComputer
    from utils.predictor import StatePredictor

    cii_computer = CIIComputer()
    predictor    = StatePredictor()

    cii_result = cii_computer.compute(
        polarity=-0.7, intensity=0.8,
        performance_dev=-1.2, intent_gap=0.6
    )
    result = predictor.predict(cii_result)
    print(result)
    # PredictionResult(state=FRUSTRATED, confidence=0.87, ...)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# CIIResult is the primary input type (optional import — falls back gracefully)
try:
    from utils.cii import CIIResult, CIIComponents, InstabilityLevel, CIIZone
    _CII_AVAILABLE = True
except ImportError:
    _CII_AVAILABLE = False
    CIIResult = object   # type: ignore


# ---------------------------------------------------------------------------
# Player State Enum
# ---------------------------------------------------------------------------

class PlayerState(str, Enum):
    """
    Three cognitive state labels used for difficulty regulation.

    Maps to difficulty controller actions:
        FRUSTRATED -> decrease difficulty
        ENGAGED    -> maintain difficulty
        BORED      -> increase difficulty
    """
    FRUSTRATED = "frustrated"
    ENGAGED    = "engaged"
    BORED      = "bored"

    @property
    def action(self) -> str:
        return {
            PlayerState.FRUSTRATED: "DECREASE difficulty",
            PlayerState.ENGAGED:    "MAINTAIN difficulty",
            PlayerState.BORED:      "INCREASE difficulty",
        }[self]

    @property
    def urgency_label(self) -> str:
        return {
            PlayerState.FRUSTRATED: "[URGENT]",
            PlayerState.ENGAGED:    "[OK]",
            PlayerState.BORED:      "[MONITOR]",
        }[self]


# ---------------------------------------------------------------------------
# Prediction Result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Output of one prediction call.

    Fields:
        state            : Winning PlayerState label
        confidence       : Score of winning class ∈ [0, 1]
        scores           : Full { state: score } distribution (sums to 1)
        reasoning        : Human-readable explanation of the decision
        input_cii        : CII value used for prediction
        input_momentum   : M(t) used
        input_acceleration : A(t) used (acceleration key signal)
        predictor_type   : "rule_based" | "ml" | "ensemble"
    """
    state:              PlayerState
    confidence:         float
    scores:             Dict[str, float]           # {"frustrated": 0.8, ...}
    reasoning:          List[str]                  # Interpretable rules fired
    input_cii:          float
    input_momentum:     float
    input_acceleration: float
    predictor_type:     str = "rule_based"

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def is_frustrated(self) -> bool:
        return self.state == PlayerState.FRUSTRATED

    @property
    def is_bored(self) -> bool:
        return self.state == PlayerState.BORED

    @property
    def is_engaged(self) -> bool:
        return self.state == PlayerState.ENGAGED

    @property
    def needs_intervention(self) -> bool:
        """True if difficulty should be changed."""
        return self.state != PlayerState.ENGAGED

    @property
    def recommended_action(self) -> str:
        return self.state.action

    def runner_up(self) -> Tuple[str, float]:
        """Second highest scoring state."""
        sorted_scores = sorted(self.scores.items(), key=lambda x: -x[1])
        return sorted_scores[1] if len(sorted_scores) > 1 else ("none", 0.0)

    def as_dict(self) -> dict:
        return {
            "state":              self.state.value,
            "confidence":         self.confidence,
            "scores":             self.scores,
            "action":             self.recommended_action,
            "needs_intervention": self.needs_intervention,
            "cii":                self.input_cii,
            "predictor":          self.predictor_type,
        }

    def __str__(self) -> str:
        bar_len  = int(self.confidence * 30)
        conf_bar = "#" * bar_len + "-" * (30 - bar_len)
        lines = [
            f"\nPredictionResult",
            f"  State      : {self.state.urgency_label} {self.state.value.upper()}",
            f"  Confidence : [{conf_bar}] {self.confidence:.1%}",
            f"  Action     : {self.recommended_action}",
            f"  Scores     : frustrated={self.scores['frustrated']:.3f} "
                           f"| engaged={self.scores['engaged']:.3f} "
                           f"| bored={self.scores['bored']:.3f}",
            f"  CII={self.input_cii:+.4f}  M={self.input_momentum:+.4f}  "
                           f"A={self.input_acceleration:+.4f}",
            f"  Reasoning  :",
        ]
        for r in self.reasoning:
            lines.append(f"    - {r}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Base class for all predictors
# ---------------------------------------------------------------------------

class BasePredictor(ABC):
    """
    Abstract base class all predictors must implement.

    ML plug-in contract:
        1. Subclass BasePredictor
        2. Implement predict_raw(cii, momentum, acceleration, **kwargs)
        3. Pass instance to StatePredictor(backend=YourPredictor())
    """

    @abstractmethod
    def predict_raw(
        self,
        cii:          float,
        momentum:     float,
        acceleration: float,
        **kwargs,
    ) -> PredictionResult:
        """Predict from raw float inputs."""
        ...

    def predict(self, cii_result: "CIIResult") -> PredictionResult:
        """Predict from a CIIResult object (convenience wrapper)."""
        if _CII_AVAILABLE and isinstance(cii_result, CIIResult):
            c = cii_result.components
            return self.predict_raw(
                cii          = cii_result.value,
                momentum     = c.momentum,
                acceleration = c.acceleration,
                intent_gap   = c.intent_gap,
                perf_dev     = c.performance_dev,
            )
        # Fallback if CIIResult not available
        return self.predict_raw(cii=float(cii_result), momentum=0.0, acceleration=0.0)


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax over a dict of raw scores."""
    vals   = list(scores.values())
    keys   = list(scores.keys())
    max_v  = max(vals)
    exps   = [math.exp(v - max_v) for v in vals]
    total  = sum(exps)
    return {k: round(e / total, 4) for k, e in zip(keys, exps)}


# ---------------------------------------------------------------------------
# Rule-Based Predictor (production default)
# ---------------------------------------------------------------------------

class RuleBasedPredictor(BasePredictor):
    """
    Multi-signal threshold predictor with softmax confidence scoring.

    Decision logic (in priority order):

    1. ACUTE FRUSTRATION override:
       CII < -0.50  ->  Frustrated (urgency 4, bypass soft voting)

    2. ACCELERATION EARLY WARNING:
       A(t) < -0.30 AND CII < -0.15  ->  Frustrated boost (+0.4 raw score)
       Catches frustration BEFORE CII has fully collapsed.

    3. SOFT ZONE SCORING:
       Raw score per class = gaussian_score(CII) tuned per class center
       Produces smooth probability distribution instead of hard thresholds.

    4. BOREDOM:
       CII > +0.30  ->  Bored
       Low acceleration + high CII = stagnation signal

    5. Softmax normalization -> confidence scores

    Tunable parameters (all keyword args to __init__):
        frus_center       : CII center of frustration gaussian  (default -0.60)
        bore_center       : CII center of boredom gaussian      (default +0.45)
        eng_center        : CII center of engagement gaussian   (default  0.00)
        sigma             : Width of gaussians                  (default  0.40)
        accel_weight      : Acceleration signal amplifier       (default  0.40)
        acute_threshold   : CII below this -> hard frustration  (default -0.50)
    """

    def __init__(
        self,
        frus_center:     float = -0.60,
        bore_center:     float = +0.45,
        eng_center:      float =  0.00,
        sigma:           float =  0.40,
        accel_weight:    float =  0.40,
        acute_threshold: float = -0.50,
        boredom_threshold: float = +0.30,
    ):
        self.frus_center       = frus_center
        self.bore_center       = bore_center
        self.eng_center        = eng_center
        self.sigma             = sigma
        self.accel_weight      = accel_weight
        self.acute_threshold   = acute_threshold
        self.boredom_threshold = boredom_threshold

    # ------------------------------------------------------------------

    def predict_raw(
        self,
        cii:          float,
        momentum:     float,
        acceleration: float,
        intent_gap:   float = 0.0,
        perf_dev:     float = 0.0,
        **kwargs,
    ) -> PredictionResult:

        reasoning: List[str] = []

        # ── Stage 1: Acute override ───────────────────────────────────────
        if cii < self.acute_threshold:
            reasoning.append(
                f"CII={cii:+.4f} < threshold {self.acute_threshold} "
                f"-> ACUTE frustration override"
            )
            if acceleration < -0.2:
                reasoning.append(
                    f"Acceleration={acceleration:+.4f} confirms rapid collapse"
                )
            raw = {"frustrated": 4.0, "engaged": 0.2, "bored": 0.1}
            probs = _softmax(raw)
            return PredictionResult(
                state              = PlayerState.FRUSTRATED,
                confidence         = probs["frustrated"],
                scores             = probs,
                reasoning          = reasoning,
                input_cii          = cii,
                input_momentum     = momentum,
                input_acceleration = acceleration,
            )

        # ── Stage 2: Gaussian scoring ─────────────────────────────────────
        def gauss(center: float) -> float:
            return math.exp(-((cii - center) ** 2) / (2 * self.sigma ** 2))

        raw = {
            "frustrated": gauss(self.frus_center),
            "engaged":    gauss(self.eng_center),
            "bored":      gauss(self.bore_center),
        }

        # ── Stage 3: Acceleration boost ───────────────────────────────────
        if acceleration < -0.30 and cii < -0.15:
            boost = abs(acceleration) * self.accel_weight
            raw["frustrated"] += boost
            reasoning.append(
                f"Acceleration={acceleration:+.4f} < -0.30 with CII={cii:+.4f} "
                f"-> frustration boost +{boost:.3f} (early warning)"
            )

        # ── Stage 4: Boredom signal ───────────────────────────────────────
        if cii > self.boredom_threshold:
            raw["bored"] += 0.5
            reasoning.append(
                f"CII={cii:+.4f} > boredom_threshold={self.boredom_threshold} "
                f"-> boredom boost"
            )
            if abs(acceleration) < 0.05:
                raw["bored"] += 0.3
                reasoning.append(
                    f"Stable low acceleration ({acceleration:+.4f}) "
                    f"confirms stagnation / boredom"
                )

        # ── Stage 5: Intent gap amplification ─────────────────────────────
        if intent_gap > 0.60 and cii < 0.0:
            gap_boost = intent_gap * 0.3
            raw["frustrated"] += gap_boost
            reasoning.append(
                f"High intent-outcome gap G={intent_gap:.3f} + negative CII "
                f"-> frustration boost +{gap_boost:.3f}"
            )

        # ── Stage 6: Performance deviation amplification ──────────────────
        if perf_dev < -1.5:
            raw["frustrated"] += 0.2
            reasoning.append(
                f"Performance deviation D={perf_dev:.3f} < -1.5 "
                f"-> frustration signal from behavioral layer"
            )

        # ── Stage 7: Softmax + decision ───────────────────────────────────
        probs  = _softmax(raw)
        winner = max(probs, key=probs.get)
        state  = PlayerState(winner)

        # Add primary decision reason if none yet
        if not reasoning:
            reasoning.append(
                f"Gaussian scoring: CII={cii:+.4f} closest to "
                f"'{winner}' center ({self._center(winner):+.3f})"
            )

        return PredictionResult(
            state              = state,
            confidence         = probs[winner],
            scores             = probs,
            reasoning          = reasoning,
            input_cii          = cii,
            input_momentum     = momentum,
            input_acceleration = acceleration,
        )

    def _center(self, state: str) -> float:
        return {
            "frustrated": self.frus_center,
            "engaged":    self.eng_center,
            "bored":      self.bore_center,
        }[state]


# ---------------------------------------------------------------------------
# ML Predictor stub (sklearn / PyTorch upgrade path)
# ---------------------------------------------------------------------------

class MLPredictor(BasePredictor):
    """
    ML model drop-in replacement for RuleBasedPredictor.

    Shows exactly what to implement when upgrading from rule-based to ML.

    Plug in any of:
        sklearn : RandomForest, SVM, GradientBoosting
        PyTorch : MLP, fine-tuned LSTM head
        ONNX    : any exported model

    Steps to activate:
        1. Train your model with features [CII, M, A, D, G]
        2. Save it (joblib for sklearn, torch.save for PyTorch)
        3. Implement _load_model() and _infer()
        4. Pass MLPredictor() to StatePredictor(backend=MLPredictor())
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model     = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """
        Load a trained model from disk.

        Example (sklearn):
            import joblib
            self._model = joblib.load(path)

        Example (PyTorch):
            from app.models.lstm import CIILSTMPredictor
            self._model = CIILSTMPredictor.load(path)
        """
        raise NotImplementedError(
            "Implement _load_model() with your model loading logic. "
            "See docstring for sklearn and PyTorch examples."
        )

    def _infer(
        self,
        cii: float, momentum: float, acceleration: float,
        intent_gap: float, perf_dev: float,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Run model inference.

        Must return: (winning_class_str, {class: probability_float})

        Example (sklearn):
            X = [[cii, momentum, acceleration, intent_gap, perf_dev]]
            proba = self._model.predict_proba(X)[0]
            classes = self._model.classes_   # e.g. ['bored','engaged','frustrated']
            probs = dict(zip(classes, proba.tolist()))
            winner = max(probs, key=probs.get)
            return winner, probs

        Example (PyTorch LSTM — 2-output head):
            import torch
            seq = [[cii, momentum, acceleration, perf_dev, intent_gap]]
            p_frus, p_bore = self._model.predict_single(seq)
            p_eng = max(0.0, 1.0 - p_frus - p_bore)
            probs = {"frustrated": p_frus, "bored": p_bore, "engaged": p_eng}
            winner = max(probs, key=probs.get)
            return winner, probs
        """
        raise NotImplementedError("Implement _infer() with your inference logic.")

    def predict_raw(
        self,
        cii: float,
        momentum: float,
        acceleration: float,
        intent_gap: float = 0.0,
        perf_dev:   float = 0.0,
        **kwargs,
    ) -> PredictionResult:
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call MLPredictor(model_path='path/to/model')."
            )
        winner, probs = self._infer(cii, momentum, acceleration, intent_gap, perf_dev)
        return PredictionResult(
            state              = PlayerState(winner),
            confidence         = probs[winner],
            scores             = probs,
            reasoning          = [f"ML model inference: {type(self._model).__name__}"],
            input_cii          = cii,
            input_momentum     = momentum,
            input_acceleration = acceleration,
            predictor_type     = "ml",
        )


# ---------------------------------------------------------------------------
# Ensemble predictor (rule + ML blend)
# ---------------------------------------------------------------------------

class EnsemblePredictor(BasePredictor):
    """
    Weighted ensemble of RuleBasedPredictor and MLPredictor.

    Combines rule-based stability with ML accuracy.
    Particularly useful when the ML model has limited training data:
    rule-based component prevents catastrophic mispredictions.

    Default blend: 50% rule / 50% ML. Adjust via rule_weight.
    """

    def __init__(
        self,
        ml_predictor:  MLPredictor,
        rule_predictor: RuleBasedPredictor = None,
        rule_weight:   float = 0.50,
    ):
        self.rule = rule_predictor or RuleBasedPredictor()
        self.ml   = ml_predictor
        self.rule_weight = rule_weight
        self.ml_weight   = 1.0 - rule_weight

    def predict_raw(self, cii, momentum, acceleration, **kwargs) -> PredictionResult:
        rule_res = self.rule.predict_raw(cii, momentum, acceleration, **kwargs)
        ml_res   = self.ml.predict_raw(cii, momentum, acceleration, **kwargs)

        # Blend scores
        blended = {}
        for state in ("frustrated", "engaged", "bored"):
            blended[state] = round(
                self.rule_weight * rule_res.scores[state]
                + self.ml_weight * ml_res.scores[state],
                4,
            )

        winner    = max(blended, key=blended.get)
        reasoning = (
            [f"Ensemble blend (rule={self.rule_weight}, ml={self.ml_weight})"]
            + rule_res.reasoning[:2]
        )
        return PredictionResult(
            state              = PlayerState(winner),
            confidence         = blended[winner],
            scores             = blended,
            reasoning          = reasoning,
            input_cii          = cii,
            input_momentum     = momentum,
            input_acceleration = acceleration,
            predictor_type     = "ensemble",
        )


# ---------------------------------------------------------------------------
# Public Facade: StatePredictor
# ---------------------------------------------------------------------------

class StatePredictor:
    """
    Main entry point. Wraps any BasePredictor backend.

    Defaults to RuleBasedPredictor. Swap to MLPredictor or EnsemblePredictor
    without changing any downstream code.

    Usage:
        # Rule-based (default, no dependencies)
        predictor = StatePredictor()

        # ML-backed (when model is trained)
        predictor = StatePredictor(backend=MLPredictor("models/checkpoints/clf.pkl"))

        # Ensemble
        predictor = StatePredictor(
            backend=EnsemblePredictor(ml_predictor=MLPredictor("clf.pkl"))
        )

        # Predict from CIIResult (main usage)
        result = predictor.predict(cii_result)

        # Predict from raw values (direct usage)
        result = predictor.predict_raw(cii=-0.6, momentum=-0.48,
                                       acceleration=-0.3)
    """

    def __init__(self, backend: BasePredictor = None):
        self._backend = backend or RuleBasedPredictor()

    def predict(self, cii_result) -> PredictionResult:
        """Predict from a CIIResult object."""
        return self._backend.predict(cii_result)

    def predict_raw(
        self,
        cii:          float,
        momentum:     float     = 0.0,
        acceleration: float     = 0.0,
        intent_gap:   float     = 0.0,
        perf_dev:     float     = 0.0,
    ) -> PredictionResult:
        """Predict from raw scalar values."""
        return self._backend.predict_raw(
            cii=cii, momentum=momentum, acceleration=acceleration,
            intent_gap=intent_gap, perf_dev=perf_dev,
        )

    def predict_batch(self, cii_results: list) -> List[PredictionResult]:
        """Predict over a list of CIIResult objects."""
        return [self.predict(r) for r in cii_results]

    @property
    def backend_type(self) -> str:
        return type(self._backend).__name__


# ---------------------------------------------------------------------------
# Session history tracker
# ---------------------------------------------------------------------------

@dataclass
class PredictionHistory:
    """
    Tracks prediction results across a session.
    Provides intervention timing, state transition counts, and trend.
    """
    _records: List[PredictionResult] = field(default_factory=list)

    def record(self, result: PredictionResult) -> None:
        self._records.append(result)

    def __len__(self) -> int:
        return len(self._records)

    def state_sequence(self) -> List[str]:
        return [r.state.value for r in self._records]

    def interventions(self) -> List[int]:
        """Timesteps (1-indexed) where state != ENGAGED."""
        return [i + 1 for i, r in enumerate(self._records) if r.needs_intervention]

    def transitions(self) -> List[Tuple[str, str, int]]:
        """List of (from_state, to_state, timestep) for state changes."""
        seq = self.state_sequence()
        result = []
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                result.append((seq[i - 1], seq[i], i + 1))
        return result

    def summary(self) -> Dict:
        if not self._records:
            return {}
        seq = self.state_sequence()
        n   = len(seq)
        return {
            "timesteps":          n,
            "frustrated_steps":   seq.count("frustrated"),
            "engaged_steps":      seq.count("engaged"),
            "bored_steps":        seq.count("bored"),
            "engagement_rate":    round(seq.count("engaged") / n, 4),
            "interventions":      len(self.interventions()),
            "state_transitions":  len(self.transitions()),
            "avg_confidence":     round(
                sum(r.confidence for r in self._records) / n, 4
            ),
            "avg_cii":            round(
                sum(r.input_cii for r in self._records) / n, 4
            ),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 62

    pred_hist = PredictionHistory()   # always initialized

    # Check if utils.cii is available for integrated test
    if _CII_AVAILABLE:
        from utils.cii import CIIComputer, CIIHistory

        computer  = CIIComputer()
        predictor = StatePredictor()
        pred_hist = PredictionHistory()

        SCENARIOS = {
            "Escalating Frustration": [
                (-0.1, 0.30, 0.00, 0.20),
                (-0.3, 0.50, 0.80, 0.40),
                (-0.6, 0.70, 1.20, 0.60),
                (-0.8, 0.90, 1.50, 0.75),
                (-0.9, 0.95, 1.80, 0.85),
            ],
            "Boredom Onset": [
                ( 0.8, 0.60, 0.00, 0.05),
                ( 0.7, 0.40, 0.00, 0.05),
                ( 0.5, 0.20, 0.00, 0.02),
                ( 0.3, 0.10, 0.00, 0.01),
            ],
            "Stable Flow": [
                ( 0.4, 0.50, 0.50, 0.20),
                ( 0.5, 0.55, 0.60, 0.15),
                ( 0.35,0.50, 1.00, 0.25),
                ( 0.45,0.60, 0.80, 0.20),
            ],
        }

        for scenario, steps in SCENARIOS.items():
            print(f"\n{SEP}")
            print(f"  SCENARIO: {scenario}")
            print(SEP)
            computer.reset()

            for i, (pol, inten, pdev, gap) in enumerate(steps, 1):
                cii_res  = computer.compute(pol, inten, pdev, gap)
                pred_res = predictor.predict(cii_res)
                pred_hist.record(pred_res)

                conf_bar = "#" * int(pred_res.confidence * 20)
                print(
                    f"  t={i:02d}  CII={cii_res.value:+.4f}  "
                    f"{pred_res.state.urgency_label} {pred_res.state.value:<11}  "
                    f"conf=[{conf_bar:<20}] {pred_res.confidence:.1%}  "
                    f"-> {pred_res.recommended_action}"
                )
                if pred_res.reasoning:
                    print(f"         reason: {pred_res.reasoning[0]}")

    else:
        # Raw-value fallback test (no cii.py)
        print(f"\n{SEP}")
        print("  RAW VALUE TEST (utils.cii not available)")
        print(SEP)
        predictor = StatePredictor()
        test_cases = [
            (-0.60, -0.48, -0.40, "Acute frustration"),
            (-0.22, -0.18, -0.10, "Mild instability"),
            ( 0.10,  0.08,  0.01, "Stable flow"),
            ( 0.45,  0.36,  0.02, "Boredom risk"),
        ]
        for cii, mom, acc, label in test_cases:
            r = predictor.predict_raw(cii=cii, momentum=mom, acceleration=acc)
            print(f"  [{label:20s}]  CII={cii:+.2f}  "
                  f"-> {r.state.value:<11}  conf={r.confidence:.1%}")

    # Summary
    print(f"\n{SEP}")
    print("  SESSION SUMMARY")
    print(SEP)
    for k, v in pred_hist.summary().items():
        print(f"  {k:<22}: {v}")
    print(f"\n  State transitions: {pred_hist.transitions()}")
    print(f"  Interventions at : {pred_hist.interventions()}")
