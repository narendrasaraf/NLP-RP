"""
rl/q_learning.py
-----------------
Tabular Q-Learning Adaptive Difficulty Agent.

Implements the exact formulation requested:

  States  : {low_frustration, moderate_frustration, high_frustration}
  Actions : {increase_difficulty, decrease_difficulty, no_change}
  Q-table : Q[state][action]  -- initialised to zero

  Reward  : reward = player_performance_score - frustration_penalty

  Update  : Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a'(Q(s',a')) - Q(s,a))

Personalization layer:
  - PlayerProfile tracks per-player tolerance to difficulty
  - Each player gets an INDEPENDENT Q-table
  - Reward function includes player-specific frustration-sensitivity weight
  - Tolerance metric updated after every episode

Design principles:
  - Zero external dependencies  (pure Python + stdlib)
  - Fully deterministic replay when seeded
  - All learning curves and Q-table data returned as plain dicts for easy
    plotting (matplotlib) or JSON export
  - Plugs into DifficultyEngine via the QLearningPolicy wrapper

Usage:
    from rl.q_learning import QLearningAgent, simulate_training

    agent = QLearningAgent()
    history = simulate_training(agent, n_episodes=300, seed=42)
    agent.print_q_table()
    agent.print_optimal_policy()
"""

from __future__ import annotations

import json
import math
import random
from copy      import deepcopy
from dataclasses import dataclass, field
from enum        import Enum
from typing      import Dict, List, Optional, Tuple


# =============================================================================
# 1.  DISCRETE STATE & ACTION SPACES
# =============================================================================

class FrustrationState(str, Enum):
    """Three-bucket frustration level — the MDP state space."""
    LOW      = "low_frustration"
    MODERATE = "moderate_frustration"
    HIGH     = "high_frustration"

    @property
    def short(self) -> str:
        return {"low_frustration": "LOW", "moderate_frustration": "MOD",
                "high_frustration": "HIGH"}[self.value]

    @staticmethod
    def from_float(frustration_score: float) -> "FrustrationState":
        """Discretize a continuous [0,1] frustration score into 3 states."""
        if frustration_score < 0.33:
            return FrustrationState.LOW
        if frustration_score < 0.67:
            return FrustrationState.MODERATE
        return FrustrationState.HIGH

    @staticmethod
    def all() -> List["FrustrationState"]:
        return [FrustrationState.LOW, FrustrationState.MODERATE, FrustrationState.HIGH]


class DifficultyAction(str, Enum):
    """Three-action discrete action space."""
    INCREASE = "increase_difficulty"
    DECREASE = "decrease_difficulty"
    NO_CHANGE = "no_change"

    @property
    def direction(self) -> int:
        return {"increase_difficulty": +1,
                "decrease_difficulty": -1,
                "no_change":            0}[self.value]

    @property
    def short(self) -> str:
        return {"increase_difficulty": "INC",
                "decrease_difficulty": "DEC",
                "no_change":           "NOP"}[self.value]

    @staticmethod
    def all() -> List["DifficultyAction"]:
        return [DifficultyAction.INCREASE,
                DifficultyAction.DECREASE,
                DifficultyAction.NO_CHANGE]


# =============================================================================
# 2.  Q-TABLE  (pure dict — no numpy required)
# =============================================================================

def make_q_table() -> Dict[str, Dict[str, float]]:
    """
    Initialise Q[state][action] = 0.0 for all (s, a) pairs.

    Returns:
        Nested dict  Q[state_value][action_value] -> float
    """
    return {
        s.value: {a.value: 0.0 for a in DifficultyAction.all()}
        for s in FrustrationState.all()
    }


# =============================================================================
# 3.  REWARD FUNCTION
# =============================================================================

@dataclass
class RewardConfig:
    """
    Reward = player_performance_score - frustration_penalty

    frustration_penalty = frustration_weight * frustration_score
                        + difficulty_mismatch_weight * |difficulty - ideal_difficulty|

    Optional bonuses:
      + flow_bonus        if frustration_score in flow zone
      - rapid_change_penalty if difficulty oscillates too fast
    """
    # Weights
    performance_weight:       float = 1.0   # scale raw performance score
    frustration_weight:       float = 0.8   # penalise frustration
    difficulty_mismatch_w:    float = 0.3   # penalise wrong difficulty level
    flow_bonus:               float = 0.2   # bonus for being in flow zone
    rapid_change_penalty:     float = 0.1   # penalise oscillation

    # Flow zone: frustration score considered optimal
    flow_lo:  float = 0.20
    flow_hi:  float = 0.45


class RewardFunction:
    """
    Computes r(t) = performance - frustration_penalty.

    The 'player_frustration_sensitivity' parameter (from PlayerProfile)
    scales the frustration_weight — high-sensitivity players suffer more
    from the same objective frustration level.
    """

    def __init__(self, cfg: RewardConfig = None):
        self.cfg = cfg or RewardConfig()

    def compute(
        self,
        performance_score:    float,
        frustration_score:    float,
        difficulty:           float,
        ideal_difficulty:     float,
        prev_difficulty:      float = None,
        frustration_sensitivity: float = 1.0,
    ) -> float:
        """
        Args:
            performance_score:    Raw [0,1] player performance this step.
            frustration_score:    Raw [0,1] frustration level.
            difficulty:           Current difficulty (0-10 normalised to [0,1]).
            ideal_difficulty:     Player's estimated optimal difficulty.
            prev_difficulty:      Difficulty at previous step (for oscillation check).
            frustration_sensitivity: Player-specific scalar from PlayerProfile.

        Returns:
            reward  float  (can be negative)
        """
        cfg = self.cfg

        # Core reward
        performance_term = cfg.performance_weight * performance_score

        # Frustration penalty (player-sensitivity weighted)
        frus_penalty = (cfg.frustration_weight
                        * frustration_sensitivity
                        * frustration_score)

        # Difficulty mismatch penalty
        mismatch = cfg.difficulty_mismatch_w * abs(difficulty - ideal_difficulty)

        # Flow bonus
        in_flow = cfg.flow_lo <= frustration_score <= cfg.flow_hi
        flow_b  = cfg.flow_bonus if in_flow else 0.0

        # Rapid-change penalty
        osc_penalty = 0.0
        if prev_difficulty is not None:
            osc_penalty = cfg.rapid_change_penalty * abs(difficulty - prev_difficulty)

        reward = (performance_term
                  - frus_penalty
                  - mismatch
                  + flow_b
                  - osc_penalty)

        return round(reward, 6)


# =============================================================================
# 4.  PLAYER PROFILE  (personalization)
# =============================================================================

@dataclass
class PlayerProfile:
    """
    Per-player persistent state for personalised Q-learning.

    Tracks:
      - frustration_sensitivity  : how much this player reacts to frustration
      - preferred_difficulty     : estimated comfort zone (0-10)
      - skill_level              : estimated skill (0-1, updated via EMA)
      - q_table                  : player-specific Q-table
      - episode_rewards          : full learning history
    """
    player_id:              str
    frustration_sensitivity: float  = 1.0     # 0.5=tolerant, 1.5=sensitive
    preferred_difficulty:   float  = 5.0     # comfort zone (1-10)
    skill_level:            float  = 0.5     # [0,1], updated via EMA
    skill_ema_alpha:        float  = 0.1     # EMA weight for skill updates

    # Learning history
    episode_rewards:   List[float] = field(default_factory=list)
    episode_lengths:   List[int]   = field(default_factory=list)
    difficulty_trace:  List[float] = field(default_factory=list)
    action_counts:     Dict[str, int] = field(default_factory=lambda: {
        a.value: 0 for a in DifficultyAction.all()
    })

    # Private Q-table (initialised lazily)
    _q_table: Optional[Dict[str, Dict[str, float]]] = field(
        default=None, repr=False
    )

    @property
    def q_table(self) -> Dict[str, Dict[str, float]]:
        if self._q_table is None:
            self._q_table = make_q_table()
        return self._q_table

    def update_skill(self, performance_score: float):
        """EMA update of estimated skill level."""
        self.skill_level = round(
            self.skill_ema_alpha * performance_score
            + (1 - self.skill_ema_alpha) * self.skill_level,
            4,
        )

    def update_tolerance(self, avg_frustration: float, avg_reward: float):
        """
        Adapt frustration_sensitivity after each episode.

        If player consistently under-performs with high frustration ->
        increase sensitivity (make system more protective).

        If player performs well even at high frustration ->
        decrease sensitivity (more tolerant profile).
        """
        if avg_reward > 0.2 and avg_frustration > 0.5:
            # Doing fine despite frustration -> less sensitive
            self.frustration_sensitivity = max(
                0.5, self.frustration_sensitivity - 0.05
            )
        elif avg_reward < -0.1 and avg_frustration > 0.4:
            # Struggling AND frustrated -> more sensitive
            self.frustration_sensitivity = min(
                2.0, self.frustration_sensitivity + 0.05
            )

    def update_preferred_difficulty(self, difficulty_at_best_reward: float):
        """Nudge preferred_difficulty toward where rewards were highest."""
        self.preferred_difficulty = round(
            0.9 * self.preferred_difficulty + 0.1 * difficulty_at_best_reward,
            3,
        )

    def record_action(self, action: DifficultyAction):
        self.action_counts[action.value] = self.action_counts.get(action.value, 0) + 1

    def to_dict(self) -> dict:
        return {
            "player_id":               self.player_id,
            "frustration_sensitivity": self.frustration_sensitivity,
            "preferred_difficulty":    self.preferred_difficulty,
            "skill_level":             self.skill_level,
            "total_episodes":          len(self.episode_rewards),
            "avg_reward":              round(sum(self.episode_rewards) / max(1, len(self.episode_rewards)), 4),
            "action_counts":           self.action_counts,
        }


# =============================================================================
# 5.  Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """
    Tabular Q-learning agent for adaptive difficulty control.

    Supports two modes:
      A. Global mode  (single shared Q-table, default)
         QLearningAgent()
         agent.step(state, performance, frustration, difficulty)

      B. Personalized mode  (per-player Q-tables)
         agent = QLearningAgent(personalized=True)
         agent.step(state, ..., player_id="player_001")

    Epsilon-greedy exploration schedule:
      epsilon decays from epsilon_start -> epsilon_end over epsilon_decay steps.
      eps(t) = max(epsilon_end, epsilon_start * decay_rate^t)

    Args:
        alpha:          Learning rate               (default 0.15)
        gamma:          Discount factor             (default 0.90)
        epsilon_start:  Initial exploration rate    (default 1.00)
        epsilon_end:    Minimum exploration rate    (default 0.05)
        epsilon_decay:  Steps to reach epsilon_end  (default 500)
        personalized:   Enable per-player Q-tables  (default False)
        seed:           RNG seed for reproducibility
    """

    def __init__(
        self,
        alpha:          float = 0.15,
        gamma:          float = 0.90,
        epsilon_start:  float = 1.00,
        epsilon_end:    float = 0.05,
        epsilon_decay:  float = 0.995,     # multiplicative decay per step
        personalized:   bool  = False,
        reward_cfg:     RewardConfig = None,
        seed:           Optional[int] = None,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.personalized  = personalized
        self._rng          = random.Random(seed)

        self._reward_fn    = RewardFunction(reward_cfg or RewardConfig())
        self._global_q     = make_q_table()   # used in non-personalized mode

        # Personalization registry
        self._profiles: Dict[str, PlayerProfile] = {}

        # Training telemetry (global)
        self.step_count:   int          = 0
        self.episode_count: int         = 0
        self._q_snapshots: List[dict]   = []   # Q-table snapshots over time
        self._reward_log:  List[float]  = []   # reward per step
        self._epsilon_log: List[float]  = []   # epsilon over steps
        self._action_log:  List[str]    = []   # action taken each step
        self._state_log:   List[str]    = []   # state observed each step

    # ------------------------------------------------------------------
    # Profile management (personalization)
    # ------------------------------------------------------------------

    def get_profile(self, player_id: str) -> PlayerProfile:
        """Get or create a PlayerProfile for the given player_id."""
        if player_id not in self._profiles:
            self._profiles[player_id] = PlayerProfile(player_id=player_id)
        return self._profiles[player_id]

    def _q_table(self, player_id: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Return the appropriate Q-table (global or player-specific)."""
        if self.personalized and player_id:
            return self.get_profile(player_id).q_table
        return self._global_q

    # ------------------------------------------------------------------
    # Epsilon-greedy policy
    # ------------------------------------------------------------------

    def _select_action(
        self,
        state: FrustrationState,
        q:     Dict[str, Dict[str, float]],
        force_greedy: bool = False,
    ) -> DifficultyAction:
        """
        Epsilon-greedy action selection.

        With probability epsilon:  random action (exploration)
        With probability 1-eps:    argmax_a Q[s][a]  (exploitation)
        """
        if not force_greedy and self._rng.random() < self.epsilon:
            # Explore
            return self._rng.choice(DifficultyAction.all())
        else:
            # Exploit — argmax with random tie-breaking
            q_vals = q[state.value]
            max_q  = max(q_vals.values())
            best   = [a for a, v in q_vals.items() if v == max_q]
            return DifficultyAction(self._rng.choice(best))

    # ------------------------------------------------------------------
    # Q-table update (Bellman equation)
    # ------------------------------------------------------------------

    def _update_q(
        self,
        q:          Dict[str, Dict[str, float]],
        state:      FrustrationState,
        action:     DifficultyAction,
        reward:     float,
        next_state: FrustrationState,
    ) -> float:
        """
        Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a'(Q(s',a')) - Q(s,a))

        Returns:
            td_error: |target - current|  (useful for logging)
        """
        current_q   = q[state.value][action.value]
        max_next_q  = max(q[next_state.value].values())
        td_target   = reward + self.gamma * max_next_q
        td_error    = td_target - current_q
        q[state.value][action.value] = current_q + self.alpha * td_error
        return abs(td_error)

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def _decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    # ------------------------------------------------------------------
    # Main step interface
    # ------------------------------------------------------------------

    def step(
        self,
        state:             FrustrationState,
        performance_score: float,
        frustration_score: float,
        difficulty:        float,
        next_state:        FrustrationState,
        prev_difficulty:   float = None,
        player_id:         Optional[str] = None,
        force_greedy:      bool  = False,
    ) -> dict:
        """
        Execute one Q-learning step.

        Args:
            state:             Current frustration state s(t).
            performance_score: Player's performance [0,1] this timestep.
            frustration_score: Continuous frustration value [0,1].
            difficulty:        Current difficulty level [1,10].
            next_state:        Next frustration state s(t+1) after env transition.
            prev_difficulty:   Difficulty at t-1 (for oscillation penalty).
            player_id:         Player identity for personalized mode.
            force_greedy:      Skip exploration (for evaluation).

        Returns:
            Step result dict with action, reward, q_update info.
        """
        q = self._q_table(player_id)

        # Get player profile (for personalization)
        profile = None
        ideal_difficulty = 5.0
        frus_sensitivity = 1.0
        if player_id:
            profile = self.get_profile(player_id)
            ideal_difficulty = profile.preferred_difficulty
            frus_sensitivity = profile.frustration_sensitivity

        # 1. Observe state  (already provided as FrustrationState)
        # 2. Choose action using epsilon-greedy
        action = self._select_action(state, q, force_greedy=force_greedy)

        # 3. Compute reward
        d_norm       = difficulty / 10.0        # normalise to [0,1]
        ideal_norm   = ideal_difficulty / 10.0
        prev_d_norm  = (prev_difficulty / 10.0) if prev_difficulty is not None else None

        reward = self._reward_fn.compute(
            performance_score    = performance_score,
            frustration_score    = frustration_score,
            difficulty           = d_norm,
            ideal_difficulty     = ideal_norm,
            prev_difficulty      = prev_d_norm,
            frustration_sensitivity = frus_sensitivity,
        )

        # 4. Update Q-table  (Bellman)
        td_error = self._update_q(q, state, action, reward, next_state)

        # 5. Decay epsilon
        self._decay_epsilon()
        self.step_count += 1

        # Logging
        self._reward_log.append(reward)
        self._epsilon_log.append(round(self.epsilon, 5))
        self._action_log.append(action.value)
        self._state_log.append(state.value)

        if profile:
            profile.update_skill(performance_score)
            profile.record_action(action)

        return {
            "step":              self.step_count,
            "state":             state.value,
            "action":            action.value,
            "action_short":      action.short,
            "reward":            reward,
            "td_error":          round(td_error, 6),
            "epsilon":           round(self.epsilon, 4),
            "next_state":        next_state.value,
            "player_id":         player_id,
            "q_snapshot":        deepcopy(q),
        }

    # ------------------------------------------------------------------
    # Episode-level API
    # ------------------------------------------------------------------

    def end_episode(
        self,
        player_id:         Optional[str] = None,
        difficulty_trace:  List[float] = None,
        frustration_trace: List[float] = None,
        reward_trace:      List[float] = None,
    ) -> dict:
        """
        Called at the end of each training episode.

        Updates player profile tolerance and preferred_difficulty.
        Records episode summary.
        """
        self.episode_count += 1

        # Take a snapshot of the Q-table every 10 episodes
        if self.episode_count % 10 == 0:
            self._q_snapshots.append({
                "episode": self.episode_count,
                "q_table": deepcopy(self._q_table(player_id)),
            })

        summary: dict = {"episode": self.episode_count}

        if reward_trace:
            ep_reward = sum(reward_trace)
            summary["episode_reward"] = round(ep_reward, 4)

            if player_id:
                profile = self.get_profile(player_id)
                profile.episode_rewards.append(ep_reward)
                profile.episode_lengths.append(len(reward_trace))
                if difficulty_trace:
                    profile.difficulty_trace.extend(difficulty_trace)

                if frustration_trace and reward_trace:
                    avg_frus   = sum(frustration_trace) / len(frustration_trace)
                    avg_reward = sum(reward_trace) / len(reward_trace)
                    profile.update_tolerance(avg_frus, avg_reward)

                if difficulty_trace and reward_trace:
                    # Find difficulty that corresponded to highest reward
                    best_idx = reward_trace.index(max(reward_trace))
                    if best_idx < len(difficulty_trace):
                        profile.update_preferred_difficulty(difficulty_trace[best_idx])

        return summary

    # ------------------------------------------------------------------
    # Inference / policy extraction
    # ------------------------------------------------------------------

    def optimal_action(
        self,
        state: FrustrationState,
        player_id: Optional[str] = None,
    ) -> DifficultyAction:
        """Return greedy-optimal action for a state (pure exploitation)."""
        return self._select_action(state, self._q_table(player_id), force_greedy=True)

    def optimal_policy(self, player_id: Optional[str] = None) -> Dict[str, str]:
        """Return optimal action for every state."""
        return {
            s.value: self.optimal_action(s, player_id).value
            for s in FrustrationState.all()
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_q_table(self, player_id: Optional[str] = None, title: str = ""):
        """Pretty-print the Q-table."""
        q   = self._q_table(player_id)
        sep = "-" * 70
        hdr = f"  Q-TABLE{' -- ' + title if title else ''}"
        if player_id:
            hdr += f"  [Player: {player_id}]"

        print("\n" + "=" * 70)
        print(hdr)
        print("=" * 70)

        # Header row
        actions = DifficultyAction.all()
        col_w   = 14
        header  = f"  {'STATE':<22}" + "".join(f"{a.short:>{col_w}}" for a in actions)
        print(header)
        print(sep)

        for state in FrustrationState.all():
            vals = q[state.value]
            row  = f"  {state.short:<22}"
            row += "".join(f"{vals[a.value]:>{col_w}.4f}" for a in actions)
            # Star best action
            best_a = max(vals, key=vals.get)
            best_v = vals[best_a]
            row   += f"   <- best: {DifficultyAction(best_a).short}={best_v:.4f}"
            print(row)

        print(sep)

    def print_optimal_policy(self, player_id: Optional[str] = None):
        """Print the greedy optimal action for each state."""
        print("\n  OPTIMAL POLICY:")
        print("  " + "-" * 40)
        for s in FrustrationState.all():
            a = self.optimal_action(s, player_id)
            print(f"  {s.short:<8}  ->  {a.value}")
        print()

    # ------------------------------------------------------------------
    # Data export (for visualisation)
    # ------------------------------------------------------------------

    def get_learning_curves(self) -> dict:
        """
        Return all logged learning data as a plain dict.

        Keys:
            steps           : step indices
            rewards         : reward at each step
            rewards_smooth  : moving-average smoothed (window=20)
            epsilon         : epsilon at each step
            actions         : action taken at each step
            states          : state observed at each step
            q_snapshots     : Q-table snapshots (every 10 episodes)
        """
        n       = len(self._reward_log)
        steps   = list(range(1, n + 1))
        window  = 20
        smooth  = [
            sum(self._reward_log[max(0, i - window):i + 1])
            / len(self._reward_log[max(0, i - window):i + 1])
            for i in range(n)
        ]
        return {
            "steps":          steps,
            "rewards":        list(self._reward_log),
            "rewards_smooth": [round(v, 5) for v in smooth],
            "epsilon":        list(self._epsilon_log),
            "actions":        list(self._action_log),
            "states":         list(self._state_log),
            "q_snapshots":    self._q_snapshots,
        }

    def to_json(self, player_id: Optional[str] = None, indent: int = 2) -> str:
        """Export Q-table and optimal policy as JSON."""
        return json.dumps({
            "q_table":       self._q_table(player_id),
            "optimal_policy": self.optimal_policy(player_id),
            "hyperparams": {
                "alpha": self.alpha, "gamma": self.gamma,
                "epsilon": round(self.epsilon, 4),
                "step_count": self.step_count,
                "episode_count": self.episode_count,
            },
        }, indent=indent)
