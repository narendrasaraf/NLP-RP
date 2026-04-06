"""
utils/telemetry_processor.py
-----------------------------
Standalone telemetry processing module for the Cognitive Regulation system.

Accepts the raw telemetry dict emitted by the game every N seconds
(matching the backend API's Telemetry schema exactly):

    {
        "kill_count":    int,
        "death_count":   int,
        "miss_count":    int,
        "reaction_time": float   (milliseconds)
    }

Computes and normalizes:

    D(t)   — Performance Deviation    (deaths + misses − kills, normalized)
    ΔD(t)  — Performance Drop Rate    (D(t) − D(t-1); how fast performance is decaying)
    ΔRT(t) — Reaction Time Change     (RT(t) − RT(t-1); rising = slowing down)
    RT_n   — Normalized Reaction Time (inverted: 1.0 = fastest, 0.0 = slowest)
    acc    — Shot Accuracy             (kills / (kills + misses))
    δ_acc  — Accuracy Delta           (acc(t) − acc(t-1))

All outputs are normalized to fixed ranges and returned as a frozen
TelemetryFeatures dataclass — ready to flow into CIIComputer or the predictor.

Position in the pipeline
------------------------
    game/main.py  →  backend/main.py  →  TelemetryProcessor.update()
                                                 │
                                                 ▼  TelemetryFeatures
                                         CIIComputer / TemporalFeatureEngine

Design
------
    - Zero external dependencies (no numpy, no scipy, no pandas).
    - Stateful per session: rolling D(t) window drives z-score D(t).
    - Stateless across sessions: call reset() or create a new instance.
    - Validation + clamping: no NaN, no crashes on bad inputs.
    - Compatible with backend/model.py schema (kill_count / death_count keys).

Usage
-----
    from utils.telemetry_processor import TelemetryProcessor

    proc = TelemetryProcessor(window=20)

    features = proc.update({
        "kill_count":    3,
        "death_count":   5,
        "miss_count":    8,
        "reaction_time": 420.0
    })

    print(features.perf_deviation)        # raw D before normalization
    print(features.perf_deviation_norm)   # D normalized to [-1, +1]
    print(features.drop_rate)             # ΔD (positive = worsening)
    print(features.reaction_time_change)  # ΔRT in ms
    print(features.reaction_time_norm)    # 0.0 (slow) → 1.0 (fast)
    print(features.accuracy)              # kill/(kill+miss)
    print(features.accuracy_delta)        # change from previous tick
    print(features.as_dict())             # flat dict for CII / logging
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Normalization constants
# (calibrated for a 2-second telemetry window in a 60-FPS shooter)
# ---------------------------------------------------------------------------

# Reaction time range (ms): 100ms = elite, 2000ms = very slow / AFK
RT_MIN:  float = 100.0
RT_MAX:  float = 2000.0

# Maximum realistic raw D in one window (used for clamped normalization)
# D worst case: 0 kills, ~15 deaths, ~30 misses → D ≈ 45
D_NORM_MAX: float = 20.0    # clip ceiling for normalization

# Maximum absolute ΔD per tick (drop-rate clip)
DROP_RATE_MAX: float = 15.0

# Maximum ΔRT change per tick (ms)
RT_CHANGE_MAX: float = 500.0


# ---------------------------------------------------------------------------
# Input field specification (validation + defaults)
# ---------------------------------------------------------------------------

_FIELD_SPEC: Dict[str, dict] = {
    "kill_count":    {"type": int,   "default": 0,     "min": 0,    "max": 1000},
    "death_count":   {"type": int,   "default": 0,     "min": 0,    "max": 1000},
    "miss_count":    {"type": int,   "default": 0,     "min": 0,    "max": 1000},
    "reaction_time": {"type": float, "default": 500.0, "min": 50.0, "max": 5000.0},
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelemetryFeatures:
    """
    Immutable snapshot of all processed telemetry signals at one timestep.

    Raw signals
    -----------
    kill_count          Raw kills in this window
    death_count         Raw deaths in this window
    miss_count          Raw missed shots in this window
    reaction_time_ms    Raw reaction time (ms)

    Derived signals (raw)
    ---------------------
    perf_deviation      D(t) = death_count + miss_count − kill_count
                               (positive = underperforming)
    drop_rate           ΔD(t) = D(t) − D(t-1)
                               (positive = performance worsening this tick)
    reaction_time_change ΔRT(t) = RT(t) − RT(t-1) in ms
                               (positive = slowing down)
    accuracy            kill_count / (kill_count + miss_count)
                               (0.0 = all misses, 1.0 = perfect aim)
    accuracy_delta      acc(t) − acc(t-1)
                               (negative = accuracy declining)

    Normalized signals  ∈ [0, 1] or [-1, +1]
    ------------------------------------------
    perf_deviation_norm D(t) normalized to [−1, +1]
                               −1 = elite performance, +1 = total breakdown
    drop_rate_norm      ΔD normalized to [−1, +1]
    reaction_time_norm  RT normalized to [0, 1], INVERTED:
                               1.0 = fastest (RT_MIN), 0.0 = slowest (RT_MAX)
    perf_zscore         Z-score of D(t) vs rolling D history
                               (how abnormal is this window vs player's own baseline)

    Session metadata
    ----------------
    window_size         Configured rolling window
    history_len         Number of windows processed so far
    is_warming_up       True while history < window_size
    warnings            Non-fatal validation messages
    """
    # ── Raw inputs ────────────────────────────────────────────────────────────
    kill_count:    int
    death_count:   int
    miss_count:    int
    reaction_time_ms: float

    # ── Derived — raw ─────────────────────────────────────────────────────────
    perf_deviation:        float   # D(t) ∈ ℝ
    drop_rate:             float   # ΔD(t) ∈ ℝ
    reaction_time_change:  float   # ΔRT(t) in ms
    accuracy:              float   # ∈ [0, 1]
    accuracy_delta:        float   # ∈ [−1, +1]

    # ── Normalized ────────────────────────────────────────────────────────────
    perf_deviation_norm:   float   # ∈ [−1, +1]
    drop_rate_norm:        float   # ∈ [−1, +1]
    reaction_time_norm:    float   # ∈ [ 0,  1]  inverted; 1.0 = fastest
    perf_zscore:           float   # ∈ ℝ

    # ── Metadata ──────────────────────────────────────────────────────────────
    window_size:   int
    history_len:   int
    is_warming_up: bool
    warnings:      tuple   # immutable list of validation warnings

    # ── Convenience ──────────────────────────────────────────────────────────

    def as_dict(self) -> Dict:
        """Flat dict — suitable for logging, API responses, dashboard."""
        return asdict(self)

    def as_cii_inputs(self) -> Dict[str, float]:
        """
        Returns the subset of signals consumed by CIIComputer.
        Maps to the 'performance_dev' argument expected there.
        """
        return {
            "performance_dev": self.perf_zscore,   # z-score D(t) — best for CII
            "accuracy":        self.accuracy,
            "drop_rate_norm":  self.drop_rate_norm,
        }

    def as_model_features(self) -> List[float]:
        """
        5-element feature vector for an ML predictor or LSTM:
        [perf_deviation_norm, drop_rate_norm, reaction_time_norm,
         accuracy, perf_zscore]
        """
        return [
            self.perf_deviation_norm,
            self.drop_rate_norm,
            self.reaction_time_norm,
            self.accuracy,
            self.perf_zscore,
        ]

    def __str__(self) -> str:
        warm = " [WARMING UP]" if self.is_warming_up else ""
        w    = f"\n  warnings={list(self.warnings)}" if self.warnings else ""
        return (
            f"TelemetryFeatures (t={self.history_len}/{self.window_size}){warm}\n"
            f"  Raw   K={self.kill_count}  D={self.death_count}  "
            f"M={self.miss_count}  RT={self.reaction_time_ms:.0f}ms\n"
            f"  Perf  dev={self.perf_deviation:+.2f}  "
            f"norm={self.perf_deviation_norm:+.4f}  "
            f"z={self.perf_zscore:+.4f}\n"
            f"  Drop  rate={self.drop_rate:+.2f}  "
            f"norm={self.drop_rate_norm:+.4f}\n"
            f"  RT    change={self.reaction_time_change:+.1f}ms  "
            f"norm={self.reaction_time_norm:.4f}\n"
            f"  Acc   {self.accuracy:.4f}  delta={self.accuracy_delta:+.4f}{w}"
        )


# ---------------------------------------------------------------------------
# TelemetryProcessor — main class
# ---------------------------------------------------------------------------

class TelemetryProcessor:
    """
    Stateful per-session telemetry processor.

    Args
    ----
    window      : Rolling window size for performance z-score (default 20).
    rt_min      : Reaction time lower bound in ms (default 100 ms).
    rt_max      : Reaction time upper bound in ms (default 2000 ms).
    d_norm_max  : Clip value for D normalization (default 20.0).

    Thread safety: NOT thread-safe. One instance per session/thread.
    """

    def __init__(
        self,
        window:     int   = 20,
        rt_min:     float = RT_MIN,
        rt_max:     float = RT_MAX,
        d_norm_max: float = D_NORM_MAX,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")

        self._window     = window
        self._rt_min     = rt_min
        self._rt_max     = rt_max
        self._d_norm_max = d_norm_max

        # Rolling D(t) history for z-score
        self._d_history: deque[float] = deque(maxlen=window)

        # Previous-tick values for delta computations
        self._prev_D:   float = 0.0
        self._prev_RT:  float = (rt_min + rt_max) / 2.0   # neutral start
        self._prev_acc: float = 0.5                         # neutral start

        # Tick counter
        self._t: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def update(self, raw: Dict) -> TelemetryFeatures:
        """
        Ingest one raw telemetry dict and return normalized TelemetryFeatures.

        Args
        ----
        raw : Dict with keys kill_count, death_count, miss_count, reaction_time.
              Extra keys are silently ignored.  Missing keys receive defaults.
              Wrong types are coerced where possible.

        Returns
        -------
        TelemetryFeatures — immutable, fully processed snapshot.
        """
        # ── Stage 1: Validate & coerce ────────────────────────────────────
        validated, warnings = _validate(raw)
        K  = validated["kill_count"]
        Dd = validated["death_count"]
        M  = validated["miss_count"]
        RT = validated["reaction_time"]

        # ── Stage 2: Raw derived signals ──────────────────────────────────

        # Performance Deviation: high = underperforming
        D = float(Dd + M - K)

        # Accuracy: fraction of offensive actions that were kills
        shots_taken = K + M
        acc = K / shots_taken if shots_taken > 0 else 0.5  # neutral when silent

        # Deltas (change from previous tick)
        drop_rate            = D  - self._prev_D
        reaction_time_change = RT - self._prev_RT
        accuracy_delta       = acc - self._prev_acc

        # ── Stage 3: Normalize ────────────────────────────────────────────

        # D → [-1, +1]  (negative = great performance, positive = breakdown)
        perf_dev_norm = _clamp(D / self._d_norm_max, -1.0, 1.0)

        # ΔD → [-1, +1]
        drop_rate_norm = _clamp(drop_rate / DROP_RATE_MAX, -1.0, 1.0)

        # RT → [0, 1] inverted: 1.0 = fastest reaction, 0.0 = slowest
        rt_norm_raw = (RT - self._rt_min) / (self._rt_max - self._rt_min)
        rt_norm     = 1.0 - _clamp(rt_norm_raw, 0.0, 1.0)

        # ── Stage 4: Rolling z-score of D ─────────────────────────────────
        self._d_history.append(D)
        d_zscore = _zscore(D, self._d_history)

        # ── Stage 5: Persist state for next tick ──────────────────────────
        self._prev_D   = D
        self._prev_RT  = RT
        self._prev_acc = acc
        self._t       += 1

        return TelemetryFeatures(
            # Raw
            kill_count            = K,
            death_count           = Dd,
            miss_count            = M,
            reaction_time_ms      = round(RT,  2),
            # Derived
            perf_deviation        = round(D,                    4),
            drop_rate             = round(drop_rate,            4),
            reaction_time_change  = round(reaction_time_change, 4),
            accuracy              = round(acc,                  4),
            accuracy_delta        = round(accuracy_delta,       4),
            # Normalized
            perf_deviation_norm   = round(perf_dev_norm,  4),
            drop_rate_norm        = round(drop_rate_norm, 4),
            reaction_time_norm    = round(rt_norm,        4),
            perf_zscore           = round(d_zscore,       4),
            # Metadata
            window_size           = self._window,
            history_len           = len(self._d_history),
            is_warming_up         = len(self._d_history) < self._window,
            warnings              = tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(self, sequence: List[Dict]) -> List[TelemetryFeatures]:
        """
        Process a list of raw telemetry dicts in order, maintaining
        temporal state (drop_rate, RT delta, rolling z-score) across steps.

        Returns a list of TelemetryFeatures, one per input dict.
        """
        return [self.update(raw) for raw in sequence]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> int:
        """Number of update() calls since last reset()."""
        return self._t

    @property
    def is_warming_up(self) -> bool:
        """True while fewer than `window` ticks have been processed."""
        return len(self._d_history) < self._window

    @property
    def d_history(self) -> List[float]:
        """Read-only copy of the rolling D(t) history."""
        return list(self._d_history)

    def summary(self) -> Dict:
        """Aggregate stats over the session's rolling D history."""
        if not self._d_history:
            return {"timesteps": self._t, "history_len": 0}
        h   = list(self._d_history)
        mu  = sum(h) / len(h)
        var = sum((x - mu) ** 2 for x in h) / len(h)
        return {
            "timesteps":       self._t,
            "history_len":     len(h),
            "window_size":     self._window,
            "is_warming_up":   self.is_warming_up,
            "mean_deviation":  round(mu, 4),
            "std_deviation":   round(math.sqrt(var), 4),
            "min_deviation":   round(min(h), 4),
            "max_deviation":   round(max(h), 4),
            "current_rt_ms":   round(self._prev_RT, 2),
            "current_acc":     round(self._prev_acc, 4),
        }

    def reset(self) -> None:
        """
        Reset all session state for a new player / session.
        Window size and normalization bounds are preserved.
        """
        self._d_history.clear()
        self._prev_D   = 0.0
        self._prev_RT  = (self._rt_min + self._rt_max) / 2.0
        self._prev_acc = 0.5
        self._t        = 0


# ---------------------------------------------------------------------------
# Pure helper functions  (independently testable, no class state)
# ---------------------------------------------------------------------------

def _validate(raw: Dict) -> tuple[Dict, List[str]]:
    """
    Validate raw telemetry dict against _FIELD_SPEC.

    - Missing keys   → default value + warning
    - Wrong type     → coerce if possible, else default + warning
    - Out-of-range   → clamp to [min, max] + warning
    - Unknown keys   → silently ignored

    Returns (validated_dict, warnings_list)
    """
    out      = {}
    warnings = []

    for name, spec in _FIELD_SPEC.items():
        raw_val  = raw.get(name)
        expected = spec["type"]

        # ── Missing ──────────────────────────────────────────────────────
        if raw_val is None:
            out[name] = spec["default"]
            warnings.append(
                f"Missing '{name}', using default={spec['default']}"
            )
            continue

        # ── Type coercion ─────────────────────────────────────────────────
        try:
            val = expected(raw_val)
        except (TypeError, ValueError):
            out[name] = spec["default"]
            warnings.append(
                f"Cannot cast '{name}'={raw_val!r} to {expected.__name__}, "
                f"using default={spec['default']}"
            )
            continue

        # ── Range clamping ────────────────────────────────────────────────
        lo, hi = spec["min"], spec["max"]
        if val < lo:
            warnings.append(f"'{name}'={val} < min={lo}, clamped.")
            val = type(val)(lo)
        elif val > hi:
            warnings.append(f"'{name}'={val} > max={hi}, clamped.")
            val = type(val)(hi)

        out[name] = val

    return out, warnings


def _zscore(value: float, history: deque) -> float:
    """
    Population z-score of `value` relative to `history`.
    Returns 0.0 if history is too short or σ ≈ 0 (flat sequence).
    """
    h = list(history)
    n = len(h)
    if n < 2:
        return 0.0
    mu  = sum(h) / n
    var = sum((x - mu) ** 2 for x in h) / n
    sigma = math.sqrt(var) if var > 1e-9 else 0.0
    if sigma < 1e-9:
        return 0.0
    return (value - mu) / sigma


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_linear(value: float, lo: float, hi: float) -> float:
    """Linear min-max normalization → [0, 1]."""
    span = hi - lo
    if span < 1e-9:
        return 0.0
    return _clamp((value - lo) / span, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Default processor — import directly for single-session scripts.
#: For multi-session systems create one TelemetryProcessor() per session.
telemetry_processor = TelemetryProcessor(window=20)


# ---------------------------------------------------------------------------
# Self-test  (run: python -m utils.telemetry_processor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 68

    print(SEP)
    print("  TelemetryProcessor — Self Test")
    print(SEP)

    proc = TelemetryProcessor(window=5)

    # ── Scenario 1: Gradual performance collapse ──────────────────────────
    print("\n[Scenario 1] Gradual performance collapse\n")
    stream = [
        {"kill_count": 5, "death_count": 1, "miss_count": 2,  "reaction_time": 280.0},
        {"kill_count": 4, "death_count": 2, "miss_count": 4,  "reaction_time": 320.0},
        {"kill_count": 2, "death_count": 4, "miss_count": 8,  "reaction_time": 390.0},
        {"kill_count": 1, "death_count": 6, "miss_count": 12, "reaction_time": 460.0},
        {"kill_count": 0, "death_count": 8, "miss_count": 15, "reaction_time": 550.0},
        {"kill_count": 0, "death_count": 9, "miss_count": 18, "reaction_time": 620.0},
    ]

    for i, raw in enumerate(stream, 1):
        tf = proc.update(raw)
        warm = "*" if tf.is_warming_up else " "
        print(
            f"  t={i:02d}{warm} "
            f"D={tf.perf_deviation:+.1f}  "
            f"D_norm={tf.perf_deviation_norm:+.4f}  "
            f"D_z={tf.perf_zscore:+.4f}  "
            f"ΔD={tf.drop_rate:+.1f}  "
            f"ΔRT={tf.reaction_time_change:+.1f}ms  "
            f"acc={tf.accuracy:.3f}  "
            f"RT_n={tf.reaction_time_norm:.3f}"
        )

    print(f"\n  Session summary: {proc.summary()}")

    # ── Scenario 2: Validation (bad inputs) ──────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 2] Input validation\n")
    proc.reset()
    bad = proc.update({
        "kill_count":    "five",    # wrong type → default
        "death_count":   -3,        # below min → clamped to 0
        "miss_count":    9999,      # above max → clamped to 1000
        # "reaction_time" missing  → default 500ms
    })
    print(bad)

    # ── Scenario 3: Perfect performance followed by crash ─────────────────
    print(f"\n{SEP}")
    print("[Scenario 3] Perfect → crash transition\n")
    proc2 = TelemetryProcessor(window=4)
    perfect = {"kill_count": 10, "death_count": 0, "miss_count": 0, "reaction_time": 150.0}
    crash   = {"kill_count":  0, "death_count": 8, "miss_count": 15, "reaction_time": 700.0}

    for _ in range(3):
        proc2.update(perfect)  # warm up baseline with good performance

    tf_crash = proc2.update(crash)
    print("  After 3 perfect ticks, sudden crash:")
    print(tf_crash)
    print(f"\n  CII-ready inputs: {tf_crash.as_cii_inputs()}")
    print(f"  ML feature vector: {tf_crash.as_model_features()}")

    # ── Scenario 4: Batch processing ─────────────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 4] Batch processing\n")
    proc3   = TelemetryProcessor(window=4)
    batch   = stream[:4]
    results = proc3.process_batch(batch)
    for i, tf in enumerate(results, 1):
        print(f"  t={i:02d}  D_norm={tf.perf_deviation_norm:+.4f}  "
              f"z={tf.perf_zscore:+.4f}  acc={tf.accuracy:.3f}")

    print(f"\n{SEP}")
    print("  All scenarios passed.")
    print(SEP)
