"""
utils/difficulty_engine.py
---------------------------
Standalone Difficulty Adaptation Engine.

Translates predicted player state -> difficulty action -> new difficulty level.

Architecture (RL-ready policy abstraction):

  ┌─────────────────────────────────────────────────────────────┐
  │               DifficultyEngine  (orchestrator)              │
  │   .adapt(prediction, cii) -> AdaptationResult               │
  └──────────────────────┬──────────────────────────────────────┘
                         │ delegates to
          ┌──────────────┴──────────────────┐
          │                                 │
  ┌───────▼──────────┐            ┌─────────▼───────────────┐
  │  RulePolicy      │            │  RLPolicy  (stub)        │
  │  (default, now)  │            │  PPO / DQN drop-in       │
  └──────────────────┘            └──────────────────────────┘

Key features:
  - Confidence-weighted step sizing (high-confidence -> larger step)
  - Cooldown guard (prevents rapid oscillation between states)
  - Differential step size (urgent states get amplified adjustment)
  - Momentum smoothing (exponential moving average of difficulty)
  - RL reward signal computed at every step
  - Full session history for analytics and RL replay buffer
  - Clean RL state vector and episode management hooks

Usage:
    from utils.cii       import CIIComputer
    from utils.predictor import StatePredictor
    from utils.difficulty_engine import DifficultyEngine

    computer  = CIIComputer()
    predictor = StatePredictor()
    engine    = DifficultyEngine()

    cii_res  = computer.compute(-0.7, 0.8, -1.2, 0.6)
    pred_res = predictor.predict(cii_res)
    result   = engine.adapt(pred_res, cii_res.value)

    print(result)
    # AdaptationResult(action=DECREASE, difficulty=4.25, delta=-0.75, reward=-0.58)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Optional imports — degrade gracefully if utils modules not on path
try:
    from utils.predictor import PredictionResult, PlayerState
    _PREDICTOR_AVAILABLE = True
except ImportError:
    _PREDICTOR_AVAILABLE = False
    PredictionResult = object  # type: ignore


# ---------------------------------------------------------------------------
# Action Enum
# ---------------------------------------------------------------------------

class DifficultyAction(str, Enum):
    """
    Three-action discrete space aligned with RL formulation.
    MDP: A = {a⁻, a⁰, a⁺}
    """
    DECREASE = "decrease"   # a⁻  frustration intervention
    MAINTAIN = "maintain"   # a⁰  flow preservation
    INCREASE = "increase"   # a⁺  boredom intervention

    @property
    def direction(self) -> int:
        """Returns -1, 0, or +1 — useful for vectorized RL environments."""
        return {"decrease": -1, "maintain": 0, "increase": 1}[self.value]

    @property
    def symbol(self) -> str:
        return {"decrease": "[v]", "maintain": "[=]", "increase": "[^]"}[self.value]


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """
    All tunable parameters for the difficulty engine.
    Single source of truth — change here, affects the whole system.
    """
    # Difficulty bounds
    difficulty_init: float = 5.0
    difficulty_min:  float = 1.0
    difficulty_max:  float = 10.0

    # Step sizing
    base_step:       float = 0.5    # Base difficulty change per step
    confidence_scale: float = 1.2   # Multiplier when confidence is high
    urgency_scale:   float = 1.5    # Extra multiplier for acute frustration
    max_step:        float = 2.0    # Hard cap on single-step change

    # Cooldown (prevents rapid oscillation)
    cooldown_steps:  int   = 2      # Steps to wait before changing direction

    # Momentum smoothing (EMA of difficulty)
    ema_alpha:       float = 0.3    # EMA weight for difficulty smoothing
    use_ema:         bool  = False  # Toggle — raw or smoothed difficulty

    # RL reward function weights
    reward_cii_weight:    float = 0.50   # Penalise |CII| deviation from 0
    reward_frus_weight:   float = 0.30   # Penalise frustration probability
    reward_bore_weight:   float = 0.15   # Penalise boredom probability
    reward_flow_bonus:    float = 0.10   # Bonus for flow state
    flow_cii_range:       Tuple[float, float] = (-0.20, +0.20)

    # RL episode parameters
    episode_length:  int   = 100    # Steps per RL training episode


# ---------------------------------------------------------------------------
# Adaptation Result
# ---------------------------------------------------------------------------

@dataclass
class AdaptationResult:
    """
    Full output of one adaptation step.

    Designed to serve as:
      - API response field
      - RL environment step() return value
      - Analytics record
    """
    # Core output
    action:           DifficultyAction
    difficulty_level: float          # New difficulty after this step
    difficulty_delta: float          # Change applied (signed)
    prev_difficulty:  float          # Difficulty before this step

    # RL signals
    reward:           float          # r(t) for RL training
    rl_state_vector:  List[float]    # s(t) = [CII, p_frus, p_bore, D_norm]

    # Explanation
    state_label:      str            # "frustrated" | "engaged" | "bored"
    confidence:       float
    step_size:        float          # Actual step applied before clamping
    reasoning:        List[str]      # Why this action was taken
    on_cooldown:      bool           # Was full step suppressed by cooldown?

    @property
    def is_intervention(self) -> bool:
        return self.action != DifficultyAction.MAINTAIN

    @property
    def rl_done(self) -> bool:
        """True if difficulty hit a boundary (episode terminal signal)."""
        return (self.difficulty_level <= 0.0 or
                self.difficulty_level >= 10.0)

    def as_dict(self) -> dict:
        return {
            "action":           self.action.value,
            "difficulty_level": self.difficulty_level,
            "difficulty_delta": self.difficulty_delta,
            "reward":           self.reward,
            "state":            self.state_label,
            "confidence":       self.confidence,
            "on_cooldown":      self.on_cooldown,
        }

    def __str__(self) -> str:
        delta_sign = f"+{self.difficulty_delta:.3f}" if self.difficulty_delta >= 0 \
                     else f"{self.difficulty_delta:.3f}"
        bar = "#" * int(self.difficulty_level * 2)
        lines = [
            f"\nAdaptationResult",
            f"  Action     : {self.action.symbol} {self.action.value.upper()}",
            f"  Difficulty : {self.prev_difficulty:.2f} -> "
                           f"{self.difficulty_level:.2f} ({delta_sign})",
            f"  Level      : [{bar:<20}] {self.difficulty_level:.1f}/10",
            f"  State      : {self.state_label.upper()} "
                           f"(conf={self.confidence:.1%})",
            f"  Reward     : {self.reward:+.4f}",
            f"  Cooldown   : {'YES (step suppressed)' if self.on_cooldown else 'no'}",
            f"  Reasoning  :",
        ]
        for r in self.reasoning:
            lines.append(f"    - {r}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Policy abstraction
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    """
    Abstract policy — computes action and step size from prediction signals.

    RL Upgrade contract:
        1. Subclass BasePolicy
        2. Implement select_action(state_vector, prediction) -> (action, step)
        3. Pass to DifficultyEngine(policy=YourPolicy())

    The DifficultyEngine handles all bounds-checking, cooldown, and
    reward computation — the policy ONLY decides (action, step_size).
    """

    @abstractmethod
    def select_action(
        self,
        state_vector: List[float],
        state_label:  str,
        confidence:   float,
        cii:          float,
    ) -> Tuple[DifficultyAction, float]:
        """
        Returns (action, step_size).
        Step size is a positive float — direction is implied by action.
        """
        ...


# ---------------------------------------------------------------------------
# Rule-Based Policy (production default)
# ---------------------------------------------------------------------------

class RulePolicy(BasePolicy):
    """
    Deterministic rule policy with confidence-weighted step sizing.

    Step size computation:
        base_step
        x confidence_scale  (if confidence > 0.70)
        x urgency_scale     (if state == acute_frustration, CII < -0.50)
        capped at max_step

    This ensures:
        - High-confidence, urgent predictions -> aggressive intervention
        - Low-confidence predictions -> conservative, small adjustments
        - Reduces oscillation in ambiguous states
    """

    def __init__(self, cfg: EngineConfig = None):
        self.cfg = cfg or EngineConfig()

    def select_action(
        self,
        state_vector: List[float],
        state_label:  str,
        confidence:   float,
        cii:          float,
    ) -> Tuple[DifficultyAction, float]:

        # ── Action ───────────────────────────────────────────────────────
        action = {
            "frustrated": DifficultyAction.DECREASE,
            "bored":      DifficultyAction.INCREASE,
            "engaged":    DifficultyAction.MAINTAIN,
        }.get(state_label, DifficultyAction.MAINTAIN)

        # ── Step size ─────────────────────────────────────────────────────
        step = self.cfg.base_step

        # Confidence scaling
        if confidence > 0.70:
            step *= self.cfg.confidence_scale

        # Urgency scaling (acute frustration)
        if state_label == "frustrated" and cii < -0.50:
            step *= self.cfg.urgency_scale

        # CII magnitude scaling — larger instability = bigger correction
        step *= (1.0 + abs(cii) * 0.5)

        # Cap
        step = min(step, self.cfg.max_step)

        if action == DifficultyAction.MAINTAIN:
            step = 0.0

        return action, round(step, 4)


# ---------------------------------------------------------------------------
# RL Policy stub (PPO / DQN upgrade path)
# ---------------------------------------------------------------------------

class RLPolicy(BasePolicy):
    """
    Reinforcement Learning policy drop-in.

    Wraps a trained agent (PPO, DQN, SAC, etc.) to select actions.

    State vector fed to the agent (4-dimensional):
        s(t) = [CII(t), p_frus(t), p_bore(t), D_norm(t)]
        where D_norm = difficulty / difficulty_max

    Action space (discrete, 3 actions):
        0 -> DECREASE
        1 -> MAINTAIN
        2 -> INCREASE

    Reward function (computed by DifficultyEngine, NOT this class):
        r(t) = -alpha * CII^2
               - beta  * p_frus
               - gamma * p_bore
               + delta * [CII in flow range]

    To integrate Stable-Baselines3 (SB3):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        # 1. Wrap DifficultyEngine as a gym.Env (see notebooks/04_rl_simulation.ipynb)
        # 2. Train: model = PPO("MlpPolicy", env, verbose=1); model.learn(100_000)
        # 3. Load here: self._agent = PPO.load("models/checkpoints/ppo_agent")
        # 4. Infer: action, _ = self._agent.predict(state_vector)

    To integrate custom PyTorch DQN:
        from torch import tensor
        q_values = self._net(tensor(state_vector).unsqueeze(0))
        action_idx = q_values.argmax().item()
    """

    ACTION_MAP = {0: DifficultyAction.DECREASE,
                  1: DifficultyAction.MAINTAIN,
                  2: DifficultyAction.INCREASE}

    def __init__(self, agent_path: Optional[str] = None, base_step: float = 0.5):
        self.base_step   = base_step
        self._agent      = None
        if agent_path:
            self._load_agent(agent_path)

    def _load_agent(self, path: str):
        """
        Load a trained RL agent.

        SB3 example:
            from stable_baselines3 import PPO
            self._agent = PPO.load(path)

        Custom PyTorch:
            self._agent = torch.load(path)
        """
        raise NotImplementedError(
            "Implement _load_agent() with your RL framework. "
            "See class docstring for SB3 and PyTorch examples."
        )

    def select_action(
        self,
        state_vector: List[float],
        state_label:  str,
        confidence:   float,
        cii:          float,
    ) -> Tuple[DifficultyAction, float]:
        if self._agent is None:
            raise RuntimeError(
                "No RL agent loaded. Train an agent and call "
                "RLPolicy(agent_path='models/checkpoints/ppo_agent')."
            )

        # SB3-style inference:
        # action_idx, _ = self._agent.predict(state_vector, deterministic=True)

        # PyTorch DQN-style:
        # import torch
        # q_vals = self._agent(torch.tensor(state_vector).float().unsqueeze(0))
        # action_idx = q_vals.argmax().item()

        raise NotImplementedError("Implement RL inference in select_action().")


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------

class CooldownTracker:
    """
    Prevents rapid oscillation by enforcing a minimum number of steps
    between direction changes (e.g., DECREASE -> INCREASE).

    A 'direction change' is when the action flips from DECREASE to INCREASE
    or vice versa. MAINTAIN does not reset the cooldown clock.
    """

    def __init__(self, cooldown_steps: int = 2):
        self.cooldown_steps = cooldown_steps
        self._last_action:  Optional[DifficultyAction] = None
        self._steps_since_change: int = 0

    def check(self, action: DifficultyAction) -> bool:
        """Returns True if action is allowed (not on cooldown)."""
        if action == DifficultyAction.MAINTAIN:
            return True
        if self._last_action is None:
            return True

        # Check if this is a direction reversal
        is_reversal = (
            self._last_action == DifficultyAction.DECREASE and
            action == DifficultyAction.INCREASE
        ) or (
            self._last_action == DifficultyAction.INCREASE and
            action == DifficultyAction.DECREASE
        )

        if is_reversal and self._steps_since_change < self.cooldown_steps:
            return False  # On cooldown
        return True

    def update(self, action: DifficultyAction):
        if action != DifficultyAction.MAINTAIN:
            if self._last_action != action:
                self._steps_since_change = 0
            else:
                self._steps_since_change += 1
            self._last_action = action
        else:
            self._steps_since_change += 1

    def reset(self):
        self._last_action = None
        self._steps_since_change = 0


# ---------------------------------------------------------------------------
# Reward function (standalone, usable for RL training)
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    RL reward function: r(t) = -alpha*CII^2 - beta*p_frus - gamma*p_bore + delta*[flow]

    Objective: push CII toward 0 (flow state) while penalising both
    frustration and boredom. Positive reward for maintaining flow.

    All weights configurable via EngineConfig.
    """

    def __init__(self, cfg: EngineConfig = None):
        self.cfg = cfg or EngineConfig()

    def compute(
        self,
        cii:    float,
        p_frus: float,
        p_bore: float,
    ) -> float:
        lo, hi     = self.cfg.flow_cii_range
        in_flow    = lo <= cii <= hi
        reward = (
            - self.cfg.reward_cii_weight  * cii ** 2
            - self.cfg.reward_frus_weight * p_frus
            - self.cfg.reward_bore_weight * p_bore
            + self.cfg.reward_flow_bonus  * (1.0 if in_flow else 0.0)
        )
        return round(reward, 5)

    def theoretical_max(self) -> float:
        """Maximum achievable reward (CII=0, p_frus=0, p_bore=0, in_flow=True)."""
        return self.cfg.reward_flow_bonus

    def theoretical_min(self) -> float:
        """Approximate minimum (CII=-1, p_frus=1, p_bore=1, not in flow)."""
        return -(self.cfg.reward_cii_weight + self.cfg.reward_frus_weight +
                 self.cfg.reward_bore_weight)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class DifficultyEngine:
    """
    Orchestrates policy -> step sizing -> cooldown -> bounds -> reward.

    One instance per game session. Call adapt() at each timestep.
    Call reset() between sessions.

    RL interface:
        state_vector() -> s(t)   environment observation
        adapt(...)     -> step() environment transition
        result.reward  -> r(t)   reward signal
        result.rl_done -> done   episode terminal condition
    """

    def __init__(
        self,
        policy: BasePolicy        = None,
        cfg:    EngineConfig      = None,
    ):
        self.cfg      = cfg    or EngineConfig()
        self.policy   = policy or RulePolicy(self.cfg)
        self._reward_fn = RewardFunction(self.cfg)
        self._cooldown  = CooldownTracker(self.cfg.cooldown_steps)

        # State
        self._difficulty:   float       = self.cfg.difficulty_init
        self._ema_diff:     float       = self.cfg.difficulty_init
        self._timestep:     int         = 0
        self._history:      List[AdaptationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self,
        prediction,           # PredictionResult or dict
        cii:        float,
        p_frus:     float = None,
        p_bore:     float = None,
    ) -> AdaptationResult:
        """
        Run one adaptation step.

        Args:
            prediction : PredictionResult from utils.predictor (preferred)
                         OR dict with keys: state, confidence, frustration_prob,
                                            boredom_prob
            cii        : CII(t) scalar
            p_frus     : Override frustration probability (optional)
            p_bore     : Override boredom probability (optional)

        Returns AdaptationResult.
        """
        self._timestep += 1

        # ── Unpack prediction ─────────────────────────────────────────────
        state_label, confidence, p_f, p_b = self._unpack(prediction)
        p_frus = p_frus if p_frus is not None else p_f
        p_bore = p_bore if p_bore is not None else p_b

        # ── RL state vector  s(t) = [CII, p_frus, p_bore, D_norm] ─────────
        d_norm       = self._difficulty / self.cfg.difficulty_max
        state_vector = [
            round(cii,        5),
            round(p_frus,     5),
            round(p_bore,     5),
            round(d_norm,     5),
        ]

        # ── Policy: select action + step ──────────────────────────────────
        action, step = self.policy.select_action(
            state_vector = state_vector,
            state_label  = state_label,
            confidence   = confidence,
            cii          = cii,
        )

        # ── Cooldown guard ────────────────────────────────────────────────
        on_cooldown = False
        reasoning:  List[str] = []

        if not self._cooldown.check(action):
            on_cooldown = True
            step        = self.cfg.base_step * 0.3   # damped step on cooldown
            reasoning.append(
                f"Cooldown active: direction reversal suppressed "
                f"(only {self._cooldown._steps_since_change}/"
                f"{self.cfg.cooldown_steps} cooldown steps elapsed)"
            )

        self._cooldown.update(action)

        # ── Apply action ──────────────────────────────────────────────────
        prev_diff = self._difficulty
        delta     = 0.0

        if action == DifficultyAction.DECREASE:
            delta             = -step
            self._difficulty  = max(self.cfg.difficulty_min,
                                    self._difficulty + delta)
            delta             = self._difficulty - prev_diff
            reasoning.append(
                f"State=FRUSTRATED (conf={confidence:.1%}) -> "
                f"difficulty reduced by {abs(delta):.3f}"
            )

        elif action == DifficultyAction.INCREASE:
            delta             = +step
            self._difficulty  = min(self.cfg.difficulty_max,
                                    self._difficulty + delta)
            delta             = self._difficulty - prev_diff
            reasoning.append(
                f"State=BORED (conf={confidence:.1%}) -> "
                f"difficulty increased by {delta:.3f}"
            )

        else:
            reasoning.append(
                f"State=ENGAGED (conf={confidence:.1%}) -> "
                f"difficulty maintained at {self._difficulty:.2f}"
            )

        # ── EMA smoothing (optional) ────────────────────────────────────
        self._ema_diff = (
            self.cfg.ema_alpha * self._difficulty +
            (1.0 - self.cfg.ema_alpha) * self._ema_diff
        )
        effective_diff = (
            round(self._ema_diff, 3) if self.cfg.use_ema
            else round(self._difficulty, 3)
        )

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._reward_fn.compute(cii, p_frus, p_bore)

        # ── Build result ──────────────────────────────────────────────────
        result = AdaptationResult(
            action           = action,
            difficulty_level = effective_diff,
            difficulty_delta = round(delta, 3),
            prev_difficulty  = round(prev_diff, 3),
            reward           = reward,
            rl_state_vector  = state_vector,
            state_label      = state_label,
            confidence       = round(confidence, 4),
            step_size        = round(step, 4),
            reasoning        = reasoning,
            on_cooldown      = on_cooldown,
        )
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # RL environment interface
    # ------------------------------------------------------------------

    def rl_state(self) -> List[float]:
        """Current RL state observation s(t) = [CII*, p_frus*, p_bore*, D_norm]."""
        d_norm = self._difficulty / self.cfg.difficulty_max
        return [0.0, 0.0, 0.0, round(d_norm, 5)]   # Filled by adapt()

    def rl_reset(self) -> List[float]:
        """Reset engine for a new RL episode. Returns initial state."""
        self.reset()
        return self.rl_state()

    # ------------------------------------------------------------------
    # Session utilities
    # ------------------------------------------------------------------

    @property
    def difficulty(self) -> float:
        return round(self._difficulty, 3)

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def history(self) -> List[AdaptationResult]:
        return self._history

    def reset(self):
        self._difficulty = self.cfg.difficulty_init
        self._ema_diff   = self.cfg.difficulty_init
        self._timestep   = 0
        self._history.clear()
        self._cooldown.reset()

    def session_summary(self) -> Dict:
        """Aggregate session analytics."""
        if not self._history:
            return {}
        h   = self._history
        n   = len(h)
        actions      = [r.action.value for r in h]
        rewards      = [r.reward for r in h]
        difficulties = [r.difficulty_level for r in h]
        interventions = sum(1 for r in h if r.is_intervention)

        return {
            "timesteps":          n,
            "final_difficulty":   round(self._difficulty, 3),
            "avg_difficulty":     round(sum(difficulties) / n, 3),
            "min_difficulty":     round(min(difficulties), 3),
            "max_difficulty":     round(max(difficulties), 3),
            "difficulty_range":   round(max(difficulties) - min(difficulties), 3),
            "decreases":          actions.count("decrease"),
            "maintains":          actions.count("maintain"),
            "increases":          actions.count("increase"),
            "interventions":      interventions,
            "intervention_rate":  round(interventions / n, 4),
            "avg_reward":         round(sum(rewards) / n, 5),
            "total_reward":       round(sum(rewards), 4),
            "cooldown_events":    sum(1 for r in h if r.on_cooldown),
        }

    def action_history(self) -> List[str]:
        return [r.action.symbol for r in self._history]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack(prediction) -> Tuple[str, float, float, float]:
        """Extract (state_label, confidence, p_frus, p_bore) from any input."""
        if _PREDICTOR_AVAILABLE and isinstance(prediction, PredictionResult):
            return (
                prediction.state.value,
                prediction.confidence,
                prediction.scores.get("frustrated", 0.0),
                prediction.scores.get("bored", 0.0),
            )
        elif isinstance(prediction, dict):
            return (
                prediction.get("state", "engaged"),
                prediction.get("confidence", 0.5),
                prediction.get("frustration_prob", prediction.get("p_frus", 0.0)),
                prediction.get("boredom_prob",     prediction.get("p_bore", 0.0)),
            )
        raise TypeError(
            f"prediction must be PredictionResult or dict, got {type(prediction).__name__}"
        )


# ---------------------------------------------------------------------------
# Session registry (multi-player support)
# ---------------------------------------------------------------------------

class DifficultyEngineRegistry:
    """One DifficultyEngine per active session — multi-session safe."""

    def __init__(self, policy_factory=None):
        self._engines: Dict[str, DifficultyEngine] = {}
        self._policy_factory = policy_factory   # callable -> BasePolicy

    def get_or_create(self, session_id: str) -> DifficultyEngine:
        if session_id not in self._engines:
            policy = self._policy_factory() if self._policy_factory else None
            self._engines[session_id] = DifficultyEngine(policy=policy)
        return self._engines[session_id]

    def drop(self, session_id: str):
        self._engines.pop(session_id, None)

    def summary_all(self) -> Dict[str, dict]:
        return {sid: e.session_summary() for sid, e in self._engines.items()}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "=" * 64

    # ── Full integrated test (cii + predictor + engine) ───────────────────
    try:
        from utils.cii       import CIIComputer
        from utils.predictor import StatePredictor
        FULL = True
    except ImportError:
        FULL = False

    SCENARIOS = {
        "Escalating Frustration": [
            (-0.1, 0.30, 0.00, 0.20),
            (-0.3, 0.50, -0.80, 0.40),
            (-0.6, 0.70, -1.20, 0.60),
            (-0.8, 0.90, -1.50, 0.75),
            (-0.9, 0.95, -1.80, 0.85),
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
            ( 0.35, 0.50, 1.00, 0.25),
            ( 0.45, 0.60, 0.80, 0.20),
        ],
    }

    for scenario, steps in SCENARIOS.items():
        print(f"\n{SEP}")
        print(f"  SCENARIO: {scenario}")
        print(SEP)
        print(f"  {'t':>3}  {'CII':>7}  {'State':<12}  {'Action':<10}  "
              f"{'Difficulty':>10}  {'Delta':>7}  {'Reward':>8}")
        print(f"  {'-'*3}  {'-'*7}  {'-'*12}  {'-'*10}  "
              f"{'-'*10}  {'-'*7}  {'-'*8}")

        if FULL:
            computer  = CIIComputer()
            predictor = StatePredictor()
        engine = DifficultyEngine()

        for i, (pol, inten, pdev, gap) in enumerate(steps, 1):
            if FULL:
                cii_res  = computer.compute(pol, inten, pdev, gap)
                pred_res = predictor.predict(cii_res)
                result   = engine.adapt(pred_res, cii_res.value)
                cii_val  = cii_res.value
            else:
                # Fallback: dict input
                cii_val  = round(0.35 * pol * inten + 0.30 * 0 + 0.25 * pdev + 0.10 * gap, 4)
                pred_dict = {
                    "state":            "frustrated" if cii_val < -0.25 else
                                        "bored"      if cii_val > 0.30  else "engaged",
                    "confidence":       min(0.5 + abs(cii_val), 0.99),
                    "frustration_prob": max(0, -cii_val),
                    "boredom_prob":     max(0, cii_val),
                }
                result = engine.adapt(pred_dict, cii_val)

            cooldown_flag = " [cooldown]" if result.on_cooldown else ""
            print(f"  {i:>3}  {cii_val:>+.4f}  "
                  f"{result.state_label:<12}  "
                  f"{result.action.symbol} {result.action.value:<8}  "
                  f"{result.prev_difficulty:>5.2f} -> {result.difficulty_level:<5.2f}  "
                  f"{result.difficulty_delta:>+.3f}  "
                  f"{result.reward:>+.4f}"
                  f"{cooldown_flag}")

        print(f"\n  Action trail: {' '.join(engine.action_history())}")
        print(f"  Summary:")
        for k, v in engine.session_summary().items():
            print(f"    {k:<22}: {v}")

    # ── Cooldown test ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COOLDOWN TEST (rapid frustration -> boredom flip)")
    print(SEP)
    engine2 = DifficultyEngine(cfg=EngineConfig(cooldown_steps=3))
    test_preds = [
        {"state": "frustrated", "confidence": 0.90, "frustration_prob": 0.9, "boredom_prob": 0.05},
        {"state": "frustrated", "confidence": 0.88, "frustration_prob": 0.88,"boredom_prob": 0.05},
        {"state": "bored",      "confidence": 0.85, "frustration_prob": 0.05,"boredom_prob": 0.85},
        {"state": "bored",      "confidence": 0.80, "frustration_prob": 0.05,"boredom_prob": 0.80},
        {"state": "bored",      "confidence": 0.82, "frustration_prob": 0.05,"boredom_prob": 0.82},
        {"state": "bored",      "confidence": 0.88, "frustration_prob": 0.05,"boredom_prob": 0.88},
    ]
    for i, p in enumerate(test_preds, 1):
        r = engine2.adapt(p, cii=0.35 if p["state"] == "bored" else -0.55)
        flag = " <-- COOLDOWN suppressed" if r.on_cooldown else ""
        print(f"  t={i}  pred={p['state']:<12}  "
              f"action={r.action.value:<10}  "
              f"diff={r.difficulty_level:.2f}{flag}")

    # ── Reward function standalone test ───────────────────────────────────
    print(f"\n{SEP}")
    print("  REWARD FUNCTION (theoretical range)")
    print(SEP)
    rf = RewardFunction()
    print(f"  Theoretical MAX : {rf.theoretical_max():+.4f}  (perfect flow)")
    print(f"  Theoretical MIN : {rf.theoretical_min():+.4f}  (worst case)")
    test_rewards = [
        (-0.60, 0.85, 0.05, "Acute frustration"),
        (-0.10, 0.20, 0.10, "Flow state"),
        ( 0.00, 0.10, 0.10, "Perfect flow"),
        ( 0.45, 0.05, 0.80, "Boredom risk"),
    ]
    for cii, pf, pb, label in test_rewards:
        r = rf.compute(cii, pf, pb)
        print(f"  [{label:<20}]  CII={cii:+.2f}  "
              f"pf={pf:.2f}  pb={pb:.2f}  reward={r:+.5f}")
