"""
pipeline/game_pipeline.py
--------------------------
Unified NLP + RL Adaptive Game Pipeline.

Connects:
  nlp/transformer_sentiment.py  ->  Sentiment & emotion detection (BERT/rule)
  rl/q_learning.py              ->  Q-learning difficulty agent
  pipeline/logger.py            ->  Structured step logging

Full message-to-action flow:

  Raw chat message
       |
       v
  [NLP Stage]
    1. Preprocess: slang normalise, collapse repeats, detect sarcasm
    2. Sentiment: BERT score -> [-1, +1]
    3. Emotion:   anger / frustration / joy / neutral
       |
       v
  [Frustration Computation]
    4. frustration_score = f(sentiment, emotion_scores, sarcasm)
       |
       v
  [State Mapping]
    5. FrustrationState = {low | moderate | high}
       |
       v
  [Q-Learning Stage]
    6. Epsilon-greedy action selection using Q[state]
    7. Apply action -> new difficulty
    8. Bellman Q-table update
       |
       v
  [Output]
    JSON with: sentiment_score, emotion_label, frustration_level,
               frustration_state, action, new_difficulty, reward,
               q_table (current), updated_q_table

Output schema:
  {
    "timestamp":           str   ISO-8601,
    "step":                int,
    "player_id":           str | None,

    -- NLP --
    "message":             str   original,
    "cleaned_message":     str,
    "sentiment_score":     float [-1, +1],
    "sentiment_label":     str   {negative, neutral, positive},
    "emotion_label":       str   {anger, frustration, joy, neutral},
    "emotion_scores":      dict  {anger, frustration, joy, neutral},
    "sarcasm_detected":    bool,
    "sarcasm_score":       float,
    "model_used":          str,

    -- RL --
    "frustration_score":   float [0, 1],
    "frustration_state":   str   {low_frustration, moderate_frustration, high_frustration},
    "prev_difficulty":     float,
    "action":              str   {increase_difficulty, decrease_difficulty, no_change},
    "new_difficulty":      float,
    "difficulty_delta":    float,
    "reward":              float,
    "td_error":            float,
    "epsilon":             float,

    -- Q-Table --
    "q_table":             dict  Q[state][action] -> float,
    "optimal_policy":      dict  state -> best_action,
  }

Usage:
    from pipeline.game_pipeline import GamePipeline

    pipe = GamePipeline()
    result = pipe.process("omg this noob is SOOOO bad wtf!!")
    print(result)   # full JSON-serialisable dict

    # Personalized (player-specific Q-tables)
    pipe = GamePipeline(personalized=True)
    result = pipe.process("I QUIT this trash!!", player_id="player_001")
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
from copy    import deepcopy
from typing  import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap (works whether run as module or script)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nlp.transformer_sentiment import TransformerSentimentAnalyzer
from rl.q_learning             import (QLearningAgent, FrustrationState,
                                        DifficultyAction)
from pipeline.logger           import PipelineLogger


# =============================================================================
# Frustration Score Computer
# =============================================================================

class FrustrationComputer:
    """
    Derives a single continuous frustration_score in [0, 1] from NLP output.

    Formula:
        frustration_score =
            w_sentiment  * negative_sentiment_component
          + w_emotion     * weighted_negative_emotion
          + w_sarcasm     * sarcasm_score_component

    Negative sentiment component:
        max(0, -sentiment_score)   (negates polarity so positive = low frus)

    Weighted negative emotion:
        anger_weight * anger + frustration_weight * frustration
        (joy & neutral suppressed from contributing)

    Sarcasm component:
        sarcasm_score if sarcasm_detected else 0
        (sarcasm often masks frustration as feigned positivity)
    """

    def __init__(
        self,
        w_sentiment:   float = 0.40,
        w_emotion:     float = 0.45,
        w_sarcasm:     float = 0.15,
        anger_weight:  float = 0.55,   # anger heavier than frustration
        frust_weight:  float = 0.45,
    ):
        self.w_sentiment  = w_sentiment
        self.w_emotion    = w_emotion
        self.w_sarcasm    = w_sarcasm
        self.anger_weight = anger_weight
        self.frust_weight = frust_weight

    def compute(self, nlp_result: Dict) -> float:
        """
        Compute frustration score from a TransformerSentimentAnalyzer result dict.

        Args:
            nlp_result: Dict as returned by TransformerSentimentAnalyzer.analyze()

        Returns:
            frustration_score: float in [0.0, 1.0]
        """
        # --- Sentiment component: how negative is the message? ---
        sent_score  = float(nlp_result.get("sentiment_score", 0.0))
        neg_sent    = max(0.0, -sent_score)          # [0,1] — 0=positive, 1=fully negative

        # --- Emotion component: anger + frustration weighted ---
        emo = nlp_result.get("emotion_scores", {})
        anger_v = float(emo.get("anger",       0.0))
        frust_v = float(emo.get("frustration", 0.0))
        emo_component = (self.anger_weight * anger_v
                         + self.frust_weight * frust_v)
        # Normalise (max possible = anger_weight + frust_weight)
        emo_norm = min(1.0, emo_component / (self.anger_weight + self.frust_weight))

        # --- Sarcasm component ---
        sarcasm_v = float(nlp_result.get("sarcasm_score", 0.0))
        # Only count sarcasm if detected — undetected sarcasm is not frustration
        sarcasm_component = sarcasm_v if nlp_result.get("sarcasm_detected", False) else 0.0

        # --- Weighted combination ---
        score = (self.w_sentiment * neg_sent
               + self.w_emotion   * emo_norm
               + self.w_sarcasm   * sarcasm_component)

        return round(min(1.0, max(0.0, score)), 4)


# =============================================================================
# Temporal Frustration Tracker  (moving average + EMA over last N messages)
# =============================================================================

class TemporalFrustrationTracker:
    """
    Maintains a rolling window of the last N frustration scores and
    computes two smoothed signals:

      1. Moving Average (MA)  — simple arithmetic mean over the window.
      2. EMA                  — exponential moving average, more responsive
                                to recent changes.

    Both signals reduce noise from single outlier messages (e.g., a single
    joy message in a generally frustrated session doesn't drop difficulty).

    The smoothed EMA value is the primary signal fed to FrustrationState
    mapping for Q-learning when temporal=True on the pipeline.

    Args:
        window_size:  Number of past messages kept (default 5).
        ema_alpha:    EMA weight for newest observation (default 0.35).
                      Higher = more responsive, lower = more smooth.
    """

    def __init__(self, window_size: int = 5, ema_alpha: float = 0.35):
        self.window_size  = window_size
        self.ema_alpha    = ema_alpha
        self._window:     List[float] = []   # rolling window
        self._ema:        Optional[float] = None   # current EMA value

    def update(self, frustration_score: float) -> dict:
        """
        Add a new raw frustration score and return updated smoothed values.

        Args:
            frustration_score: Raw [0,1] frustration score for this step.

        Returns:
            dict with keys:
              moving_avg  : simple mean over last window_size scores
              ema         : exponential moving average
              window      : copy of current window (list of floats)
              n           : number of messages seen so far
        """
        # Rolling window
        self._window.append(frustration_score)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        # Moving average
        moving_avg = sum(self._window) / len(self._window)

        # EMA: initialise with first value, then apply formula
        if self._ema is None:
            self._ema = frustration_score
        else:
            self._ema = (self.ema_alpha * frustration_score
                         + (1.0 - self.ema_alpha) * self._ema)

        return {
            "moving_avg":  round(moving_avg, 4),
            "ema":         round(self._ema,  4),
            "window":      list(self._window),
            "n":           len(self._window),
        }

    def reset(self):
        self._window.clear()
        self._ema = None

    @property
    def ema(self) -> Optional[float]:
        return round(self._ema, 4) if self._ema is not None else None

    @property
    def moving_avg(self) -> Optional[float]:
        if not self._window:
            return None
        return round(sum(self._window) / len(self._window), 4)


# =============================================================================
# Performance Estimator (infers performance from NLP signals)
# =============================================================================

class PerformanceEstimator:
    """
    Infers a proxy performance_score [0,1] from the NLP result when no
    explicit game telemetry is available.

    Logic:
        - High joy emotion  => likely performing well
        - High anger/frus   => likely performing poorly
        - Positive sentiment => infer slightly above average performance
        - Sarcasm detected  => deflate (ironically positive = actually bad)
    """

    def estimate(self, nlp_result: Dict, frustration_score: float) -> float:
        """Estimate performance from NLP output."""
        emo = nlp_result.get("emotion_scores", {})
        joy_v  = float(emo.get("joy",         0.0))
        frus_v = float(emo.get("frustration", 0.0))
        ang_v  = float(emo.get("anger",       0.0))
        sent   = float(nlp_result.get("sentiment_score", 0.0))

        # Base: map sentiment [-1,+1] onto [0.1, 0.9]
        base = 0.5 + sent * 0.4

        # Joy boost, frustration/anger drag
        base += 0.15 * joy_v
        base -= 0.15 * frus_v
        base -= 0.10 * ang_v

        # Sarcasm deflation: "great job (not really)" -> performance likely low
        if nlp_result.get("sarcasm_detected", False):
            base -= 0.15 * nlp_result.get("sarcasm_score", 0.0)

        return round(max(0.05, min(0.95, base)), 4)


# =============================================================================
# Main Pipeline
# =============================================================================

class GamePipeline:
    """
    Real-time NLP + RL adaptive game pipeline.

    Combines TransformerSentimentAnalyzer (NLP) and QLearningAgent (RL)
    into a single process() call that ingests a chat message and returns
    a full structured output dict with sentiment, emotion, frustration,
    action, and updated Q-table.

    Args:
        nlp_model:        HuggingFace model for sentiment (default nlptown BERT).
                          Set to None to use rule-based fallback (no download).
        nlp_preprocess:   Apply gaming preprocessor (slang/repeats/sarcasm).
        sarcasm_adjust:   Invert sentiment if sarcasm detected.
        use_bert:         If False, skip model loading (pure rule-based NLP).
        alpha:            Q-learning rate.
        gamma:            Q-learning discount factor.
        epsilon_start:    Initial exploration rate.
        epsilon_end:      Minimum exploration rate.
        epsilon_decay:    Per-step multiplicative epsilon decay.
        initial_difficulty: Starting game difficulty (1-10).
        difficulty_step:  How much difficulty changes per action.
        personalized:     If True, each player_id gets an independent Q-table.
        log_dir:          Directory for JSON log files. None = no file logging.
        seed:             RNG seed.
    """

    DEFAULT_NLP_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(
        self,
        nlp_model:           str   = DEFAULT_NLP_MODEL,
        nlp_preprocess:      bool  = True,
        sarcasm_adjust:      bool  = True,
        use_bert:            bool  = False,    # False = rule-based, no download
        alpha:               float = 0.15,
        gamma:               float = 0.90,
        epsilon_start:       float = 0.30,    # lower default for inference mode
        epsilon_end:         float = 0.05,
        epsilon_decay:       float = 0.99,
        initial_difficulty:  float = 5.0,
        difficulty_step:     float = 0.8,
        personalized:        bool  = False,
        log_dir:             Optional[str] = None,
        seed:                Optional[int] = None,
    ):
        # --- NLP ---
        self._nlp = TransformerSentimentAnalyzer(
            model_name    = nlp_model,
            emotion_mode  = "rule",
            preprocess    = nlp_preprocess,
            sarcasm_adjust= sarcasm_adjust,
        )
        if use_bert:
            # Trigger model load immediately so first process() is fast
            self._nlp._load_models()
        else:
            self._nlp._loaded = True           # skip model download
            self._nlp._sentiment_pipe = None   # force rule-based polarity

        # --- Frustration & performance helpers ---
        self._frus_computer  = FrustrationComputer()
        self._perf_estimator = PerformanceEstimator()

        # --- RL ---
        self._agent = QLearningAgent(
            alpha         = alpha,
            gamma         = gamma,
            epsilon_start = epsilon_start,
            epsilon_end   = epsilon_end,
            epsilon_decay = epsilon_decay,
            personalized  = personalized,
            seed          = seed,
        )
        self._difficulty      = initial_difficulty
        self._difficulty_step = difficulty_step
        self._prev_difficulty: Optional[float] = None
        self._step_count      = 0

        # --- Temporal tracker ---
        self._temporal = TemporalFrustrationTracker(window_size=5, ema_alpha=0.35)

        # --- Logger ---
        self._logger = PipelineLogger(log_dir=log_dir)

        print(f"[GamePipeline] Ready | NLP={'BERT' if use_bert else 'rule-based'} "
              f"| RL=Q-learning(alpha={alpha}, gamma={gamma}) "
              f"| difficulty={initial_difficulty} | temporal_window=5")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        message:          str,
        player_id:        Optional[str] = None,
        performance_score: Optional[float] = None,
    ) -> Dict:
        """
        Full pipeline: NLP -> frustration -> RL -> output.

        Args:
            message:          Raw player chat string.
            player_id:        Player identifier (for personalised Q-tables).
            performance_score: Explicit [0,1] game performance if available.
                               If None, it is inferred from NLP signals.

        Returns:
            Full result dict (JSON-serialisable). See module docstring.
        """
        self._step_count += 1
        ts = datetime.datetime.now().isoformat(timespec="milliseconds")

        # ── Step 1 & 2: NLP — sentiment + emotion ──────────────────────────
        nlp_result = self._nlp.analyze(message)

        # ── Step 3: Compute frustration score ─────────────────────────────
        frustration_score = self._frus_computer.compute(nlp_result)

        # ── Performance inference (if not provided externally) ─────────────
        if performance_score is None:
            performance_score = self._perf_estimator.estimate(nlp_result, frustration_score)

        # ── Step 5: Temporal smoothing (EMA + moving average) ────────────────
        temporal = self._temporal.update(frustration_score)
        smoothed_frustration = temporal["ema"]       # primary smoothed signal

        # ── Step 6: Map SMOOTHED frustration to discrete state ───────────────
        #    Using EMA keeps the state stable across short positive messages
        frustration_state = FrustrationState.from_float(smoothed_frustration)

        # ── Step 5 & 6: Q-learning — action selection & Q-table update ─────
        prev_difficulty = self._difficulty

        rl_result = self._agent.step(
            state             = frustration_state,
            performance_score = performance_score,
            frustration_score = frustration_score,
            difficulty        = self._difficulty,
            next_state        = frustration_state,    # same-step approximation
            prev_difficulty   = self._prev_difficulty,
            player_id         = player_id,
        )

        # ── Step 7: Apply action — update difficulty ────────────────────────
        action  = DifficultyAction(rl_result["action"])
        new_d   = self._difficulty + action.direction * self._difficulty_step
        new_d   = round(max(1.0, min(10.0, new_d)), 3)

        self._prev_difficulty = self._difficulty
        self._difficulty      = new_d
        difficulty_delta      = round(new_d - prev_difficulty, 3)

        # ── Build structured output ─────────────────────────────────────────
        q_table  = deepcopy(self._agent._q_table(player_id))
        opt_pol  = self._agent.optimal_policy(player_id)

        result: Dict = {
            # Meta
            "timestamp":          ts,
            "step":               self._step_count,
            "player_id":          player_id,

            # NLP
            "message":            message,
            "cleaned_message":    nlp_result["preprocessing"]["cleaned_text"],
            "sentiment_score":    nlp_result["sentiment_score"],
            "sentiment_label":    nlp_result["sentiment_label"],
            "emotion_label":      nlp_result["emotion_label"],
            "emotion_scores":     nlp_result["emotion_scores"],
            "sarcasm_detected":   nlp_result["sarcasm_detected"],
            "sarcasm_score":      nlp_result["sarcasm_score"],
            "model_used":         nlp_result["model_used"],
            "slang_replacements": nlp_result["preprocessing"]["slang_replacements"],
            "normalizations":     nlp_result["preprocessing"]["normalizations"],

            # Frustration computation + temporal
            "frustration_score":      frustration_score,
            "smoothed_frustration":   smoothed_frustration,
            "moving_avg_frustration": temporal["moving_avg"],
            "frustration_window":     temporal["window"],
            "frustration_state":      frustration_state.value,
            "performance_score":      performance_score,

            # RL
            "prev_difficulty":    prev_difficulty,
            "action":             action.value,
            "action_short":       action.short,
            "new_difficulty":     new_d,
            "difficulty_delta":   difficulty_delta,
            "reward":             rl_result["reward"],
            "td_error":           rl_result["td_error"],
            "epsilon":            rl_result["epsilon"],

            # Q-table
            "q_table":            q_table,
            "optimal_policy":     opt_pol,
        }

        # ── Log ────────────────────────────────────────────────────────────
        self._logger.log(result)

        return result

    def process_batch(
        self,
        messages:   List[str],
        player_id:  Optional[str] = None,
    ) -> List[Dict]:
        """Process a list of messages in sequence."""
        return [self.process(msg, player_id=player_id) for msg in messages]

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def q_table(self) -> Dict:
        return deepcopy(self._agent._global_q)

    @property
    def temporal_tracker(self) -> TemporalFrustrationTracker:
        """Access the temporal frustration tracker."""
        return self._temporal

    def reset(self, player_id: Optional[str] = None):
        """Reset difficulty and context window (call between game sessions)."""
        self._difficulty      = 5.0
        self._prev_difficulty = None
        self._temporal.reset()
        if self._nlp._preprocessor:
            self._nlp._preprocessor.reset_context()

    def set_difficulty(self, value: float):
        """Manually set current difficulty (for game integration)."""
        self._difficulty = max(1.0, min(10.0, float(value)))

    def get_agent(self) -> QLearningAgent:
        """Access underlying Q-learning agent."""
        return self._agent

    # ------------------------------------------------------------------
    # Pretty output
    # ------------------------------------------------------------------

    def print_result(self, result: Dict, compact: bool = False) -> None:
        """Print a formatted pipeline result to stdout."""
        sep = "=" * 64
        mid = "-" * 64

        print(f"\n{sep}")
        print(f"  STEP {result['step']:>4}  |  {result['timestamp']}")
        if result.get("player_id"):
            print(f"  Player: {result['player_id']}")
        print(mid)

        # Message
        msg = result["message"]
        cln = result["cleaned_message"]
        print(f"  MSG     : {msg!r}")
        if cln != msg.lower().strip():
            print(f"  CLEANED : {cln!r}")
        if result.get("slang_replacements"):
            sl = dict(list(result["slang_replacements"].items())[:4])
            print(f"  SLANG   : {sl}")
        print(mid)

        # NLP
        s_score = result["sentiment_score"]
        s_label = result["sentiment_label"].upper()
        s_bar   = self._mini_bar(s_score, lo=-1, hi=1, width=20)
        print(f"  SENTIMENT  {s_label:<10} {s_bar}  {s_score:+.3f}")

        emo_label = result["emotion_label"].upper()
        emo_scores = result["emotion_scores"]
        top_emo_v  = emo_scores[result["emotion_label"]]
        print(f"  EMOTION    {emo_label:<10} conf={top_emo_v:.2f}")

        if not compact:
            for emo, val in emo_scores.items():
                bar = "#" * int(val * 20)
                print(f"    {emo:<14} [{bar:<20}] {val:.3f}")

        frus_score = result["frustration_score"]
        frus_state = result["frustration_state"].replace("_frustration", "").upper()
        frus_bar   = "#" * int(frus_score * 20)
        print(f"  FRUSTRATION {frus_state:<8} [{frus_bar:<20}] {frus_score:.3f}")

        if result.get("sarcasm_detected"):
            print(f"  SARCASM  [!] score={result['sarcasm_score']:.2f}  (sentiment adjusted)")

        print(mid)

        # RL
        action = result["action"].replace("_difficulty", "").upper()
        a_sym  = {"INCREASE": "[^]", "DECREASE": "[v]", "NO_CHANGE": "[=]"}.get(action, "[ ]")
        print(f"  RL ACTION  {a_sym} {action}")
        print(f"  DIFFICULTY {result['prev_difficulty']:.1f}  ->  "
              f"{result['new_difficulty']:.1f}  "
              f"(delta={result['difficulty_delta']:+.1f})")
        d_bar = "#" * int(result["new_difficulty"] * 2)
        print(f"  DIFF BAR   [{d_bar:<20}] {result['new_difficulty']:.1f}/10")
        print(f"  REWARD     {result['reward']:+.4f}   "
              f"td_err={result['td_error']:.4f}   "
              f"eps={result['epsilon']:.3f}")

        if not compact:
            print(mid)
            print("  Q-TABLE (current):")
            q = result["q_table"]
            print(f"  {'STATE':<22}  {'INC':>10}  {'DEC':>10}  {'NOP':>10}")
            for s_val, a_dict in q.items():
                s_short = s_val.replace("_frustration", "").upper()
                best    = max(a_dict, key=a_dict.get)
                best_s  = DifficultyAction(best).short
                vals    = f"  {a_dict['increase_difficulty']:>+10.4f}  {a_dict['decrease_difficulty']:>+10.4f}  {a_dict['no_change']:>+10.4f}"
                print(f"  {s_short:<22}{vals}   <- {best_s}")
            print(mid)
            print("  OPTIMAL POLICY:")
            for s_val, a_val in result["optimal_policy"].items():
                s_short = s_val.replace("_frustration", "").upper()
                print(f"    {s_short:<12} -> {a_val}")

        print(sep)

    @staticmethod
    def _mini_bar(v: float, lo: float, hi: float, width: int = 20) -> str:
        """Produce a small ASCII bar for a value in [lo, hi]."""
        pct    = (v - lo) / (hi - lo)
        filled = int(pct * width)
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def to_json(self, message: str, player_id: Optional[str] = None,
                indent: int = 2) -> str:
        """Process and return JSON string."""
        return json.dumps(self.process(message, player_id=player_id),
                          indent=indent, ensure_ascii=False)
