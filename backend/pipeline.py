"""
backend/pipeline.py
--------------------
Full real-time cognitive state prediction pipeline.

Wires all five specialist modules into a single callable:

    chat ──► NLPExtractor ──► TemporalFeatureEngine ──► CIIEngine ──► StatePredictor
                                                    ▲
    telemetry ──────────── TelemetryProcessor ──────┘

Public API
----------
    predict_state(input_data)       → {"cii", "state", "confidence"}
    predict_state_full(input_data)  → above + all intermediate features

    Both are thin wrappers around:
        pipeline.run(input_data)
        pipeline.run_full(input_data)

    where `pipeline` is the module-level singleton instance.

Input schema
------------
    {
        "chat":      str | None,          # player chat message (optional)
        "telemetry": {                    # raw game telemetry (optional)
            "kill_count":    int,
            "death_count":   int,
            "miss_count":    int,
            "reaction_time": float        # milliseconds
        }
    }

Minimal output
--------------
    {"cii": float, "state": str, "confidence": float}

Full output (run_full)
----------------------
    {
        "cii":        float,
        "state":      str,           # "Frustrated" | "Engaged" | "Bored"
        "confidence": float,
        "action":     str,           # "decrease" | "maintain" | "increase"
        "level":      str,           # "low" | "medium" | "high"
        "features": {
            "nlp":       dict,       # 8 NLP signals
            "temporal":  dict,       # M, A, rolling mean/std, trend
            "telemetry": dict,       # D, drop_rate, RT_norm, accuracy, z-score
            "cii_detail": dict       # per-component breakdown
        }
    }

Session state
-------------
CognitivePipeline is STATEFUL — temporal momentum history and telemetry
z-score windows are maintained across calls. Call pipeline.reset() between
independent player sessions to clear history.

Integration with backend/main.py
---------------------------------
Replace the scattered NLP + model logic in backend/main.py with:

    from backend.pipeline import predict_state
    result = predict_state({"chat": payload.chat,
                             "telemetry": payload.telemetry.model_dump()})

The returned dict is a superset of what backend/main.py currently builds.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

# ── Step 1: NLP extraction ───────────────────────────────────────────────────
from utils.nlp_extractor       import NLPExtractor

# ── Step 2: Temporal features ─────────────────────────────────────────────────
from utils.temporal_features   import TemporalFeatureEngine

# ── Step 3: Telemetry processing ─────────────────────────────────────────────
from utils.telemetry_processor import TelemetryProcessor

# ── Step 4: CII computation ───────────────────────────────────────────────────
from utils.cii_engine          import CIIEngine

# ── Step 5: State prediction ──────────────────────────────────────────────────
from backend.predictor         import StatePredictor


# ---------------------------------------------------------------------------
# Safe empty defaults
# ---------------------------------------------------------------------------

_EMPTY_TELEMETRY: Dict[str, Any] = {
    "kill_count":    0,
    "death_count":   0,
    "miss_count":    0,
    "reaction_time": 500.0,
}


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class CognitivePipeline:
    """
    Stateful end-to-end pipeline for real-time cognitive state prediction.

    Instantiate ONCE per player session. Maintains:
      - TemporalFeatureEngine history  (momentum rolling window)
      - TelemetryProcessor   history  (performance z-score window)

    Args
    ----
    nlp_mode          : "rule_based" (default) or "pretrained".
    temporal_window   : Rolling window size for momentum stats (default 10).
    telemetry_window  : Rolling window for performance z-score (default 20).
    hysteresis        : Confidence margin to suppress state oscillation (0.05).
    """

    def __init__(
        self,
        nlp_mode:         str   = "rule_based",
        temporal_window:  int   = 10,
        telemetry_window: int   = 20,
        hysteresis:       float = 0.05,
    ) -> None:
        # ── Module instances (created once, reused per tick) ──────────────
        self._nlp       = NLPExtractor(mode=nlp_mode)
        self._temporal  = TemporalFeatureEngine(window=temporal_window)
        self._telemetry = TelemetryProcessor(window=telemetry_window)
        self._cii       = CIIEngine()
        self._predictor = StatePredictor(hysteresis=hysteresis)

        # ── Tick counter for diagnostics ──────────────────────────────────
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Primary public method — minimal output
    # ------------------------------------------------------------------

    def run(self, input_data: Dict) -> Dict[str, Any]:
        """
        Run the full pipeline and return the minimal output schema.

        Args
        ----
        input_data : dict with optional keys "chat" and "telemetry".
                     Missing or None values receive safe defaults.

        Returns
        -------
        {"cii": float, "state": str, "confidence": float}
        """
        result = self._execute(input_data)
        return {
            "cii":        result["cii"],
            "state":      result["state"],
            "confidence": result["confidence"],
        }

    # ------------------------------------------------------------------
    # Extended output — all intermediate features included
    # ------------------------------------------------------------------

    def run_full(self, input_data: Dict) -> Dict[str, Any]:
        """
        Run the full pipeline and return all intermediate features.

        Returns
        -------
        {
            "cii":        float,
            "state":      str,
            "confidence": float,
            "action":     str,
            "level":      str,
            "features": {
                "nlp":       dict,
                "temporal":  dict,
                "telemetry": dict,
                "cii_detail": dict
            }
        }
        """
        return self._execute(input_data)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all temporal state for a new player session.
        Module configs (windows, weights) are preserved.
        """
        self._temporal.reset()
        self._telemetry.reset()
        self._predictor.reset()
        self._tick = 0

    @property
    def tick(self) -> int:
        """Number of pipeline ticks since last reset."""
        return self._tick

    @property
    def is_warming_up(self) -> bool:
        """True while either history buffer is below its window size."""
        return self._temporal.is_warming_up or self._telemetry.is_warming_up

    def session_summary(self) -> Dict:
        """Aggregate stats for the current session."""
        return {
            "tick":                     self._tick,
            "is_warming_up":            self.is_warming_up,
            "temporal_summary":         self._temporal.summary(),
            "telemetry_summary":        self._telemetry.summary(),
        }

    # ------------------------------------------------------------------
    # Core execution — internal, single source of truth
    # ------------------------------------------------------------------

    def _execute(self, input_data: Dict) -> Dict[str, Any]:
        """
        Run all five pipeline stages in sequence.
        Returns the full result dict; callers trim as needed.
        """
        self._tick += 1

        # ── Unpack input safely ───────────────────────────────────────────
        chat      = _safe_str(input_data.get("chat"))
        raw_telem = _safe_dict(input_data.get("telemetry"), _EMPTY_TELEMETRY)

        # ════ Stage 1: NLP extraction ════════════════════════════════════
        # Input : chat string (or "" if silent)
        # Output: 8 NLP feature scalars including polarity, intensity,
        #         confidence, anger, frustration, etc.
        nlp_feats = self._nlp.extract(chat)

        # ════ Stage 2: Temporal features ════════════════════════════════
        # Input : NLP features (polarity + intensity)
        # Output: M(t), A(t), rolling mean/std, momentum z-score, trend
        temporal_obj  = self._temporal.update(nlp_feats)
        temporal_feats = temporal_obj.as_dict()

        # ════ Stage 3: Telemetry processing ════════════════════════════
        # Input : raw game telemetry dict
        # Output: D(t) z-score, drop_rate, RT_norm, accuracy, etc.
        telem_obj  = self._telemetry.update(raw_telem)
        telem_feats = telem_obj.as_dict()

        # ════ Stage 4: CII computation ══════════════════════════════════
        # Input : NLP feats + temporal feats + telemetry feats
        # Output: CII scalar, zone, action, per-component breakdown
        cii_result = self._cii.compute(nlp_feats, temporal_feats, telem_feats)

        # ════ Stage 5: State prediction ═════════════════════════════════
        # Input : CII scalar (+ optional M, A for context)
        # Output: state label + confidence
        pred = self._predictor.predict_with_detail(
            cii_result.cii,
            momentum      = cii_result.momentum,
            acceleration  = cii_result.acceleration,
            perf_deviation = cii_result.perf_deviation,
            intent_gap    = cii_result.intent_gap,
        )

        # ── Assemble full output ──────────────────────────────────────────
        return {
            # Minimal schema (always present)
            "cii":        cii_result.cii,
            "state":      pred["state"],
            "confidence": pred["confidence"],

            # Extended schema
            "action": pred["action"],       # "decrease" | "maintain" | "increase"
            "level":  cii_result.level,     # "low" | "medium" | "high"

            # All intermediate feature layers
            "features": {
                "nlp": {
                    "chat":                chat,
                    "polarity":            nlp_feats["polarity"],
                    "intensity":           nlp_feats["intensity"],
                    "anger":               nlp_feats["anger"],
                    "frustration":         nlp_feats["frustration"],
                    "confidence":          nlp_feats["confidence"],
                    "exclamation_count":   nlp_feats["exclamation_count"],
                    "uppercase_ratio":     nlp_feats["uppercase_ratio"],
                    "negative_word_ratio": nlp_feats["negative_word_ratio"],
                },
                "temporal": {
                    "momentum":         temporal_feats["momentum"],
                    "acceleration":     temporal_feats["acceleration"],
                    "rolling_mean":     temporal_feats["rolling_mean"],
                    "rolling_std":      temporal_feats["rolling_std"],
                    "momentum_zscore":  temporal_feats["momentum_zscore"],
                    "trend":            temporal_feats["trend"],
                    "trend_label":      temporal_feats["trend_label"],
                    "is_warming_up":    temporal_feats["is_warming_up"],
                },
                "telemetry": {
                    "kill_count":           telem_feats["kill_count"],
                    "death_count":          telem_feats["death_count"],
                    "miss_count":           telem_feats["miss_count"],
                    "reaction_time_ms":     telem_feats["reaction_time_ms"],
                    "perf_deviation":       telem_feats["perf_deviation"],
                    "perf_deviation_norm":  telem_feats["perf_deviation_norm"],
                    "perf_zscore":          telem_feats["perf_zscore"],
                    "drop_rate":            telem_feats["drop_rate"],
                    "drop_rate_norm":       telem_feats["drop_rate_norm"],
                    "reaction_time_norm":   telem_feats["reaction_time_norm"],
                    "accuracy":             telem_feats["accuracy"],
                    "accuracy_delta":       telem_feats["accuracy_delta"],
                    "is_warming_up":        telem_feats["is_warming_up"],
                },
                "cii_detail": {
                    "cii":           cii_result.cii,
                    "momentum_M":    cii_result.momentum,
                    "acceleration_A": cii_result.acceleration,
                    "perf_dev_D":    cii_result.perf_deviation,
                    "intent_gap_G":  cii_result.intent_gap,
                    "contrib_M":     cii_result.contrib_M,
                    "contrib_A":     cii_result.contrib_A,
                    "contrib_D":     cii_result.contrib_D,
                    "contrib_G":     cii_result.contrib_G,
                    "level_detail":  cii_result.level_detail,
                    "zone":          cii_result.zone,
                    "urgency":       cii_result.urgency,
                    "driver":        cii_result.dominant_driver(),
                },
            },
        }


# ---------------------------------------------------------------------------
# Input sanitisers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """Return value as str, or '' if None/non-string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_dict(value: Any, default: Dict) -> Dict:
    """Return value if it is a non-empty dict, else the default."""
    if isinstance(value, dict) and value:
        return value
    return dict(default)


# ---------------------------------------------------------------------------
# Module-level singleton  (one shared session — suitable for single-player API)
# ---------------------------------------------------------------------------

#: Default pipeline — import and call directly in backend/main.py.
#: For multi-session systems, create one CognitivePipeline() per session_id.
pipeline = CognitivePipeline()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def predict_state(input_data: Dict) -> Dict[str, Any]:
    """
    Run the full pipeline and return the minimal output.

    Args
    ----
    input_data : {
        "chat":      str | None,
        "telemetry": {"kill_count", "death_count", "miss_count",
                      "reaction_time"} | None
    }

    Returns
    -------
    {"cii": float, "state": str, "confidence": float}
    """
    return pipeline.run(input_data)


def predict_state_full(input_data: Dict) -> Dict[str, Any]:
    """
    Run the full pipeline and return all intermediate features.

    Returns
    -------
    {"cii", "state", "confidence", "action", "level", "features": {...}}
    """
    return pipeline.run_full(input_data)


# ---------------------------------------------------------------------------
# Self-test  (run: python -m backend.pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 68

    print(SEP)
    print("  CognitivePipeline — Full Integration Self Test")
    print(SEP)

    # ── Scenario 1: Escalating frustration ───────────────────────────────
    print("\n[Scenario 1] Escalating frustration — 6 ticks\n")
    test_pipeline = CognitivePipeline(temporal_window=5, telemetry_window=5)

    TICKS = [
        # chat,                              kill  death  miss   RT(ms)
        ("starting warm, feeling good",        5,    1,    2,    280),
        ("nice play, getting there",           4,    2,    5,    310),
        ("ugh missed that one",                2,    3,    9,    370),
        ("WHY IS THIS SO HARD!!",              1,    5,   13,    450),
        ("IMPOSSIBLE I give up this is trash", 0,    7,   18,    540),
        ("I QUIT unfair garbage game!!",       0,    9,   22,    640),
    ]

    print(f"  {'t':>2}  {'Chat':<34}  {'CII':>7}  {'State':<12}  "
          f"{'Conf':>6}  Action")
    print(f"  {'─'*2}  {'─'*34}  {'─'*7}  {'─'*12}  {'─'*6}  {'─'*8}")

    for i, (chat, K, D, M, RT) in enumerate(TICKS, 1):
        result = test_pipeline.run({
            "chat":      chat,
            "telemetry": {"kill_count": K, "death_count": D,
                          "miss_count": M, "reaction_time": float(RT)},
        })
        warm = "*" if test_pipeline.is_warming_up else " "
        print(
            f"  {i:02d}{warm} {chat[:33]:<33}  "
            f"  {result['cii']:+.4f}  {result['state']:<12}  "
            f"{result['confidence']:>6.3f}  {result.get('action', '-')}"
        )

    # ── Scenario 2: Full output inspection ───────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 2] Full output inspection (single tick)\n")
    p2 = CognitivePipeline()
    full = p2.run_full({
        "chat":      "this is so frustrating I keep dying!!",
        "telemetry": {"kill_count": 0, "death_count": 6,
                      "miss_count": 14, "reaction_time": 510.0},
    })

    print(f"  cii        = {full['cii']:+.4f}")
    print(f"  state      = {full['state']}")
    print(f"  confidence = {full['confidence']:.4f}")
    print(f"  action     = {full['action']}")
    print(f"  level      = {full['level']}")
    print(f"\n  NLP features:")
    for k, v in full["features"]["nlp"].items():
        if k != "chat":
            print(f"    {k:<24} = {v}")
    print(f"\n  Temporal features:")
    for k, v in full["features"]["temporal"].items():
        print(f"    {k:<24} = {v}")
    print(f"\n  Telemetry features:")
    for k, v in full["features"]["telemetry"].items():
        print(f"    {k:<24} = {v}")
    print(f"\n  CII breakdown:")
    for k, v in full["features"]["cii_detail"].items():
        print(f"    {k:<24} = {v}")

    # ── Scenario 3: Silent player (no chat, no telemetry) ─────────────────
    print(f"\n{SEP}")
    print("[Scenario 3] Silent player — no chat, default telemetry\n")
    p3 = CognitivePipeline()
    for _ in range(3):
        r = p3.run({})   # completely empty input
        print(f"  CII={r['cii']:+.4f}  state={r['state']:<12}  conf={r['confidence']:.3f}")

    # ── Scenario 4: Module-level function API ─────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 4] Module-level predict_state() function\n")
    # Uses the module-level `pipeline` singleton
    for chat, telem in [
        ("gg ez win!",
         {"kill_count": 8, "death_count": 0, "miss_count": 1, "reaction_time": 145.0}),
        ("why do i keep missing!!",
         {"kill_count": 1, "death_count": 5, "miss_count": 16, "reaction_time": 480.0}),
        (None,
         {"kill_count": 3, "death_count": 2, "miss_count": 4, "reaction_time": 300.0}),
    ]:
        r = predict_state({"chat": chat, "telemetry": telem})
        label = repr(chat[:25]) if chat else "None"
        print(f"  chat={label:<28}  →  {r}")

    # ── Scenario 5: reset() between sessions ──────────────────────────────
    print(f"\n{SEP}")
    print("[Scenario 5] reset() clears session state\n")
    p5 = CognitivePipeline(temporal_window=4)
    # Run 4 frustrated ticks to build up history
    for _ in range(4):
        p5.run({"chat": "I QUIT", "telemetry": {
            "kill_count": 0, "death_count": 8,
            "miss_count": 20, "reaction_time": 600.0
        }})
    print(f"  Before reset: t={p5.tick}  warming={p5.is_warming_up}  "
          f"summary={p5.session_summary()['temporal_summary']}")
    p5.reset()
    print(f"  After  reset: t={p5.tick}  warming={p5.is_warming_up}  "
          f"summary={p5.session_summary()['temporal_summary']}")

    print(f"\n{SEP}")
    print("  All scenarios passed.")
    print(SEP)
