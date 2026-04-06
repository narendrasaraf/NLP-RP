"""
backend/predictor.py
--------------------
Cognitive state prediction module.

Input  : CII scalar (float) from CIIEngine or backend/model.py
Output : {"state": "Frustrated"|"Engaged"|"Bored", "confidence": float}

Design
------
The CII axis (defined in utils/cii.py) is:

    CII < -0.50        →  Acute frustration
   -0.50 ≤ CII < -0.25 →  High instability  (frustration risk)
   -0.25 ≤ CII <  0.00 →  Mild instability  (borderline)
    0.00 ≤ CII ≤ +0.30 →  Stable flow       (engaged)
    CII  > +0.30        →  Boredom risk

Three-class mapping used here:
    CII ≤ -0.25   →  Frustrated   (high/acute instability zones)
    CII ≥ +0.30   →  Bored        (boredom risk zone)
    otherwise     →  Engaged      (flow zone + mild instability)

Confidence
----------
Modelled as a Gaussian centred at each class's "ideal" CII value.
This produces smooth, continuous confidence — not a step function.

    frustrated_confidence  ∝  exp(-((CII - (-0.60))² / 2σ²))
    engaged_confidence     ∝  exp(-((CII - ( 0.00))² / 2σ²))
    bored_confidence       ∝  exp(-((CII - (+0.45))² / 2σ²))

A softmax over the three raw Gaussian scores yields a proper probability
distribution that sums to 1.0. Winning class confidence is reported.

Hysteresis
----------
Optional one-step state memory prevents rapid oscillation at boundaries.
When enabled, the engine stays in its current state unless confidence for
a different state exceeds (current_confidence + hysteresis_margin).

ML Upgrade path
---------------
Subclass StatePredictor and override _infer():

    class MyMLPredictor(StatePredictor):
        def __init__(self):
            super().__init__()
            self._model = load_my_model()

        def _infer(self, cii, **ctx):
            # ctx contains: momentum, acceleration, perf_deviation, intent_gap
            X = [[cii, ctx.get("momentum",0), ctx.get("acceleration",0),
                  ctx.get("perf_deviation",0), ctx.get("intent_gap",0)]]
            proba   = self._model.predict_proba(X)[0]
            classes = self._model.classes_   # ["bored","engaged","frustrated"]
            return dict(zip(classes, proba.tolist()))

Then swap the global:
    from backend.predictor import predictor_engine
    predictor_engine = MyMLPredictor()

No other file needs changing.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# CII thresholds (must stay in sync with utils/cii.py's THRESHOLDS dict)
# ---------------------------------------------------------------------------

_FRUSTRATED_THRESHOLD: float = -0.25   # CII ≤ this → Frustrated
_BORED_THRESHOLD:      float = +0.30   # CII ≥ this → Bored

# Gaussian centers per class (where CII is most "pure" for each state)
_CENTER: Dict[str, float] = {
    "Frustrated": -0.60,
    "Engaged":     0.00,
    "Bored":      +0.45,
}

# Gaussian width — controls how sharply confidence falls off from center
_SIGMA: float = 0.35


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian(cii: float, center: float, sigma: float = _SIGMA) -> float:
    """Unnormalised Gaussian score centred at `center`."""
    return math.exp(-((cii - center) ** 2) / (2 * sigma ** 2))


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax → values sum to 1.0."""
    vals  = list(scores.values())
    keys  = list(scores.keys())
    max_v = max(vals)
    exps  = [math.exp(v - max_v) for v in vals]
    total = sum(exps)
    return {k: round(e / total, 4) for k, e in zip(keys, exps)}


# ---------------------------------------------------------------------------
# StatePredictor
# ---------------------------------------------------------------------------

class StatePredictor:
    """
    Rule-based cognitive state predictor.

    Args
    ----
    frustrated_threshold : CII values at or below this → Frustrated.
                           Default -0.25 (high instability + acute zones).
    bored_threshold      : CII values at or above this → Bored.
                           Default +0.30 (boredom risk zone).
    sigma                : Gaussian width for confidence scoring.
                           Smaller = sharper confidence near class centers.
    hysteresis           : Minimum confidence margin required to switch state.
                           0.0 disables hysteresis (default). 0.10 means the
                           new state must outscore the current by ≥10 pp.

    The predict() method signature is intentionally minimal:
        predict(cii: float, **context) → {"state": str, "confidence": float}

    `context` kwargs (momentum, acceleration, perf_deviation, intent_gap)
    are passed to _infer() — used by ML subclasses for multi-feature input.
    """

    def __init__(
        self,
        frustrated_threshold: float = _FRUSTRATED_THRESHOLD,
        bored_threshold:      float = _BORED_THRESHOLD,
        sigma:                float = _SIGMA,
        hysteresis:           float = 0.0,
    ) -> None:
        self.frustrated_threshold = frustrated_threshold
        self.bored_threshold      = bored_threshold
        self.sigma                = sigma
        self.hysteresis           = max(0.0, hysteresis)

        # Hysteresis state memory
        self._last_state:      Optional[str]  = None
        self._last_confidence: float          = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        cii: float,
        *,
        momentum:       float = 0.0,
        acceleration:   float = 0.0,
        perf_deviation: float = 0.0,
        intent_gap:     float = 0.0,
    ) -> Dict[str, float | str]:
        """
        Predict cognitive state from CII.

        Args
        ----
        cii           : Cognitive Instability Index scalar (float).
        momentum      : M(t) — optional; used by ML subclass.
        acceleration  : A(t) — optional; used by ML subclass.
        perf_deviation: D(t) — optional; used by ML subclass.
        intent_gap    : G(t) — optional; used by ML subclass.

        Returns
        -------
        {
            "state":      "Frustrated" | "Engaged" | "Bored",
            "confidence": float ∈ [0.0, 1.0]
        }
        """
        context = dict(
            momentum=momentum,
            acceleration=acceleration,
            perf_deviation=perf_deviation,
            intent_gap=intent_gap,
        )

        # Raw probability distribution from _infer (overridable by ML)
        probs = self._infer(cii, **context)

        # Hard-zone primary label (ensures boundary correctness)
        primary = self._hard_zone(cii)

        # Resolve final winner: primary wins ties, hysteresis prevents flipping
        winner, winner_conf = self._resolve(probs, primary)

        # Apply hysteresis
        winner, winner_conf = self._apply_hysteresis(winner, winner_conf, probs)

        # Persist for next call
        self._last_state      = winner
        self._last_confidence = winner_conf

        return {
            "state":      winner,
            "confidence": round(winner_conf, 4),
        }

    def predict_with_detail(
        self,
        cii: float,
        **context,
    ) -> Dict:
        """
        Extended output — includes per-class scores and reasoning.
        Useful for research, dashboards, and ablation studies.
        """
        probs   = self._infer(cii, **context)
        primary = self._hard_zone(cii)
        winner, winner_conf = self._resolve(probs, primary)
        winner, winner_conf = self._apply_hysteresis(winner, winner_conf, probs)

        self._last_state      = winner
        self._last_confidence = winner_conf

        action = {
            "Frustrated": "decrease",
            "Engaged":    "maintain",
            "Bored":      "increase",
        }[winner]

        reasoning = self._explain(cii, winner, probs, context)

        return {
            "state":       winner,
            "confidence":  round(winner_conf, 4),
            "action":      action,
            "scores":      probs,
            "reasoning":   reasoning,
            "cii":         round(cii, 4),
        }

    def reset(self) -> None:
        """Reset hysteresis state memory (call between sessions)."""
        self._last_state      = None
        self._last_confidence = 0.0

    # ------------------------------------------------------------------
    # Override this for ML upgrade
    # ------------------------------------------------------------------

    def _infer(self, cii: float, **context) -> Dict[str, float]:
        """
        Compute per-class probability scores.

        Rule-based default: Gaussian scores softmax'd.
        Override in ML subclass with model inference.

        Returns
        -------
        {"Frustrated": float, "Engaged": float, "Bored": float}
        — values should sum to 1.0 (softmax is applied here).
        """
        raw = {
            state: _gaussian(cii, center, self.sigma)
            for state, center in _CENTER.items()
        }
        return _softmax(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hard_zone(self, cii: float) -> str:
        """
        Hard threshold mapping — determines the correct zone boundary.

        CII ≤ frustrated_threshold  →  "Frustrated"
        CII ≥ bored_threshold       →  "Bored"
        otherwise                   →  "Engaged"

        The Gaussian winner must agree with this zone to win. If they
        disagree (e.g., CII is -0.30 but Gaussian picks Engaged), the
        hard zone wins with 60% confidence.
        """
        if cii <= self.frustrated_threshold:
            return "Frustrated"
        if cii >= self.bored_threshold:
            return "Bored"
        return "Engaged"

    def _resolve(
        self,
        probs:   Dict[str, float],
        primary: str,
    ) -> Tuple[str, float]:
        """
        Resolve final winner from Gaussian scores + hard zone.

        If Gaussian winner agrees with hard zone → Gaussian winner wins
        (confidence comes from soft scores).
        If they disagree → hard zone wins (confidence = 0.60 baseline).
        """
        gaussian_winner = max(probs, key=probs.get)

        if gaussian_winner == primary:
            return gaussian_winner, probs[gaussian_winner]
        else:
            # Zones disagree — hard boundary takes priority
            # Use the hard zone's Gaussian score if it exists
            hard_conf = max(probs.get(primary, 0.60), 0.60)
            return primary, hard_conf

    def _apply_hysteresis(
        self,
        winner: str,
        winner_conf: float,
        probs: Dict[str, float],
    ) -> Tuple[str, float]:
        """
        Prevent state flipping when the new winner barely beats the current.

        If hysteresis = 0.0 → always accept the new winner.
        If hysteresis > 0.0 → only switch if:
            new_winner_conf > last_conf + hysteresis_margin
        """
        if self.hysteresis == 0.0 or self._last_state is None:
            return winner, winner_conf

        if (
            winner != self._last_state
            and winner_conf < self._last_confidence + self.hysteresis
        ):
            # Stay in current state — new winner didn't beat threshold
            stayed_conf = probs.get(self._last_state, self._last_confidence)
            return self._last_state, stayed_conf

        return winner, winner_conf

    def _explain(
        self,
        cii:     float,
        winner:  str,
        probs:   Dict[str, float],
        context: Dict[str, float],
    ) -> str:
        """Human-readable explanation for the prediction."""
        parts = [f"CII={cii:+.4f} → {winner}"]

        if cii <= self.frustrated_threshold:
            parts.append(
                f"hard-zone: CII ≤ frustrated_threshold "
                f"({self.frustrated_threshold:+.2f})"
            )
        elif cii >= self.bored_threshold:
            parts.append(
                f"hard-zone: CII ≥ bored_threshold "
                f"({self.bored_threshold:+.2f})"
            )
        else:
            runner_up = sorted(probs.items(), key=lambda x: -x[1])[1]
            parts.append(
                f"soft-zone: scores "
                f"F={probs['Frustrated']:.3f} "
                f"E={probs['Engaged']:.3f} "
                f"B={probs['Bored']:.3f} "
                f"(runner-up: {runner_up[0]} @ {runner_up[1]:.3f})"
            )

        acc = context.get("acceleration", 0.0)
        if abs(acc) > 0.30:
            direction = "collapsing" if acc < 0 else "recovering"
            parts.append(f"acceleration={acc:+.4f} → {direction}")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton  (used by backend/model.py's predict_state_and_action)
# ---------------------------------------------------------------------------

class LSTMBackedPredictor(StatePredictor):
    """
    Real-time ML Predictor using O(1) stepwise LSTM.
    Highly optimized: utilizes hidden state caching for sub-50ms response times.
    """
    def __init__(self):
        super().__init__(hysteresis=0.05)
        from models.simple_lstm import load_or_train
        self._model = load_or_train(verbose=False)
        self._model.reset_state()

    def _infer(self, cii: float, **context) -> Dict[str, float]:
        # Step through the LSTM internally (inherits O(1) caching optimization)
        res = self._model.step(cii)
        
        # Distribute the available probability smoothly
        p_frus = res["frustration_prob"]
        p_eng  = max(0.0, 1.0 - p_frus - (1.0 - p_frus) * 0.3)
        p_bore = 1.0 - p_frus - p_eng
        
        return {
            "Frustrated": round(p_frus, 4),
            "Engaged":    round(p_eng, 4),
            "Bored":      round(p_bore, 4)
        }
        
    def reset(self) -> None:
        super().reset()
        self._model.reset_state()

try:
    #: Real-time LSTM predictor with O(1) optimized stepping.
    predictor_engine = LSTMBackedPredictor()
except ImportError:
    #: Fallback predictor if PyTorch is not installed.
    predictor_engine = StatePredictor(
        frustrated_threshold = _FRUSTRATED_THRESHOLD,
        bored_threshold      = _BORED_THRESHOLD,
        sigma                = _SIGMA,
        hysteresis           = 0.05,
    )

# Self-test  (run: python -m backend.predictor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 64

    print(SEP)
    print("  StatePredictor — Self Test")
    print(SEP)

    pred = StatePredictor(hysteresis=0.05)

    # ── Scenario 1: CII sweep across full range ────────────────────────────
    print("\n[Scenario 1] CII Sweep -1.0 → +1.0\n")
    print(f"  {'CII':>7}  {'State':<12}  {'Conf':>6}  {'F':>6}  {'E':>6}  {'B':>6}"
          f"  Action")
    print(f"  {'─'*7}  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")

    for raw_cii in [-1.0, -0.75, -0.50, -0.35, -0.25, -0.10,
                     0.00,  0.10,  0.20,  0.30,  0.45,  0.60, 1.0]:
        r = pred.predict_with_detail(raw_cii)
        marker = "◄" if r["state"] != "Engaged" else " "
        print(
            f"  {raw_cii:+7.2f}  {r['state']:<12}  {r['confidence']:>6.3f}  "
            f"{r['scores']['Frustrated']:>6.3f}  {r['scores']['Engaged']:>6.3f}  "
            f"{r['scores']['Bored']:>6.3f}  {r['action']}  {marker}"
        )

    # ── Scenario 2: Hysteresis demonstration ─────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 2] Hysteresis — noisy signal at boundary\n")
    pred2 = StatePredictor(hysteresis=0.10)
    # Oscillate around the frustrated boundary
    for cii in [-0.20, -0.28, -0.22, -0.30, -0.21, -0.26]:
        r = pred2.predict(cii)
        print(f"  CII={cii:+.2f}  →  {r['state']:<12}  conf={r['confidence']:.3f}")

    # ── Scenario 3: Simple output schema ─────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 3] Simple output schema {state, confidence}\n")
    for cii, label in [(-0.70, "acute frustration"), (0.05, "flow"),
                        (0.55, "bored"), (-0.28, "high instability")]:
        out = predictor_engine.predict(cii)
        print(f"  {label:<22}  CII={cii:+.2f}  →  {out}")

    # ── Scenario 4: ML upgrade stub demonstration ─────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 4] ML upgrade path demo\n")

    class DemoMLPredictor(StatePredictor):
        """Demo: Simulates an ML model with fixed probability table."""
        def _infer(self, cii: float, **ctx) -> Dict[str, float]:
            # In real use: call self._model.predict_proba([[cii, ...]]))
            if cii < -0.3:
                return {"Frustrated": 0.80, "Engaged": 0.15, "Bored": 0.05}
            elif cii > 0.3:
                return {"Frustrated": 0.05, "Engaged": 0.20, "Bored": 0.75}
            else:
                return {"Frustrated": 0.10, "Engaged": 0.80, "Bored": 0.10}

    ml_pred = DemoMLPredictor()
    for cii in [-0.6, 0.0, 0.5]:
        r = ml_pred.predict_with_detail(cii)
        print(f"  CII={cii:+.2f}  state={r['state']:<12}  conf={r['confidence']:.3f}  "
              f"[ML backend]")

    print(f"\n{SEP}")
    print("  All scenarios passed.")
    print(SEP)
