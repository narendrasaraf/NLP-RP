"""
rl/player_simulator.py
-----------------------
Synthetic player behaviour simulator for Q-learning training & testing.

Models three distinct player archetypes:
  1. CASUAL   : low skill, high frustration sensitivity, quits when overwhelmed
  2. AVERAGE  : baseline player, moderate tolerance
  3. HARDCORE : high skill, embraces difficulty, gets bored easily

For each archetype, the simulator:
  - Generates performance_score  f(difficulty, skill, fatigue)
  - Generates frustration_score  f(difficulty, failures, recent_performance)
  - Models learning curves        skill slowly improves over time
  - Models fatigue                performance degrades after long sessions

The environment transition follows:
  next_frustration = f(current_frustration, difficulty_delta, performance)

Usage:
    from rl.player_simulator import PlayerSimulator

    sim = PlayerSimulator(archetype="casual", seed=42)
    for t in range(200):
        obs = sim.step(difficulty=5.0)
        print(obs["frustration_score"], obs["performance_score"])
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum        import Enum
from typing      import Dict, List, Optional

from rl.q_learning import FrustrationState


# =============================================================================
# Player Archetypes
# =============================================================================

class Archetype(str, Enum):
    CASUAL    = "casual"
    AVERAGE   = "average"
    HARDCORE  = "hardcore"


@dataclass
class ArchetypeConfig:
    """Numerical parameters defining each archetype."""
    # Skill characteristics
    base_skill:          float    # Starting ability [0,1]
    skill_growth_rate:   float    # Per-step skill gain (learning)
    max_skill:           float    # Ceiling

    # Tolerance
    frustration_sensitivity: float   # How fast frustration rises
    frustration_recovery:    float   # How fast frustration drops when doing well
    boredom_threshold:       float   # Difficulty below which player gets bored
    quit_threshold:          float   # Frustration above which simulate AFK/quit

    # Performance model parameters
    performance_noise:   float   # Stdev of random noise on performance
    fatigue_rate:        float   # Per-step performance degradation
    fatigue_recovery:    float   # Fatigue recovered per rest step


ARCHETYPE_CONFIGS: Dict[Archetype, ArchetypeConfig] = {
    Archetype.CASUAL: ArchetypeConfig(
        base_skill=0.30, skill_growth_rate=0.0005, max_skill=0.65,
        frustration_sensitivity=1.50, frustration_recovery=0.08,
        boredom_threshold=2.0, quit_threshold=0.85,
        performance_noise=0.15, fatigue_rate=0.002, fatigue_recovery=0.01,
    ),
    Archetype.AVERAGE: ArchetypeConfig(
        base_skill=0.55, skill_growth_rate=0.0008, max_skill=0.80,
        frustration_sensitivity=1.00, frustration_recovery=0.10,
        boredom_threshold=3.5, quit_threshold=0.90,
        performance_noise=0.10, fatigue_rate=0.0015, fatigue_recovery=0.008,
    ),
    Archetype.HARDCORE: ArchetypeConfig(
        base_skill=0.80, skill_growth_rate=0.001, max_skill=0.98,
        frustration_sensitivity=0.55, frustration_recovery=0.18,
        boredom_threshold=6.0, quit_threshold=0.95,
        performance_noise=0.06, fatigue_rate=0.001, fatigue_recovery=0.006,
    ),
}


# =============================================================================
# Observation returned by each step
# =============================================================================

@dataclass
class StepObservation:
    t:                  int
    archetype:          str
    difficulty:         float
    performance_score:  float    # [0,1]
    frustration_score:  float    # [0,1]
    frustration_state:  FrustrationState
    skill_level:        float
    fatigue:            float
    quit_signal:        bool     # Player about to rage-quit


# =============================================================================
# Player Simulator
# =============================================================================

class PlayerSimulator:
    """
    Simulates a synthetic player's response to changing difficulty.

    Performance model:
        base_perf = skill - (difficulty/10 - skill)^2 * challenge_factor
        perf      = clip(base_perf + noise, 0, 1)

    Frustration model:
        If perf < 0.35  : frustration += sensitivity * difficulty_factor
        If perf > 0.55  : frustration -= recovery * success_factor
        frustration is EMA-smoothed over a 3-step window.

    Args:
        archetype   : "casual" | "average" | "hardcore"
        seed        : RNG seed for reproducibility
        custom_cfg  : Override archetype config fields
    """

    def __init__(
        self,
        archetype:  str  = "average",
        seed:       Optional[int] = None,
        custom_cfg: Optional[dict] = None,
    ):
        self.archetype  = Archetype(archetype)
        self._cfg       = ARCHETYPE_CONFIGS[self.archetype]
        self._rng       = random.Random(seed)

        # Override any config fields
        if custom_cfg:
            for k, v in custom_cfg.items():
                if hasattr(self._cfg, k):
                    setattr(self._cfg, k, v)

        # Internal state
        self._t:           int   = 0
        self._skill:       float = self._cfg.base_skill
        self._fatigue:     float = 0.0
        self._frustration: float = 0.3        # starting frustration
        self._frus_window: List[float] = []   # short-term EMA window

        # History for visualisation
        self.history: List[StepObservation] = []

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, difficulty: float) -> StepObservation:
        """
        Simulate one timestep at the given difficulty.

        Args:
            difficulty: current difficulty level [1, 10]

        Returns:
            StepObservation with all player signals
        """
        cfg = self._cfg
        self._t += 1

        # Normalise difficulty to [0,1]
        d_norm = difficulty / 10.0

        # ── Performance ────────────────────────────────────────────────────
        # A player's performance is best near their skill level.
        # Too easy  (d << skill): boring but still fine
        # Too hard  (d >> skill): failure, performance drops steeply
        challenge_gap = d_norm - self._skill
        if challenge_gap <= 0:
            # Under-challenged: some easy wins, mild performance
            base_perf = self._skill + 0.1 * abs(challenge_gap)
        else:
            # Over-challenged: quadratic drop-off
            dropoff   = 1.5 * (challenge_gap ** 1.5)
            base_perf = self._skill - dropoff

        # Fatigue penalty
        base_perf = base_perf - self._fatigue * 0.4

        # Add Gaussian noise
        noise     = self._rng.gauss(0, cfg.performance_noise)
        perf      = float(max(0.0, min(1.0, base_perf + noise)))

        # ── Skill growth (learning curve) ─────────────────────────────────
        # Skill grows faster when near flow point (challenged but not overwhelmed)
        if 0.0 < challenge_gap < 0.30 and perf > 0.35:
            self._skill = min(cfg.max_skill,
                              self._skill + cfg.skill_growth_rate)

        # ── Fatigue ───────────────────────────────────────────────────────
        if perf < 0.4:
            # Struggling sessions build fatigue
            self._fatigue = min(1.0, self._fatigue + cfg.fatigue_rate * 2)
        else:
            self._fatigue = max(0.0, self._fatigue - cfg.fatigue_recovery)

        # ── Frustration ───────────────────────────────────────────────────
        if perf < 0.35:
            # Failure increases frustration
            rise = cfg.frustration_sensitivity * (0.35 - perf) * (1 + d_norm)
            self._frustration = min(1.0, self._frustration + rise * 0.15)
        elif perf > 0.60:
            # Success reduces frustration
            drop = cfg.frustration_recovery * (perf - 0.60)
            self._frustration = max(0.0, self._frustration - drop * 0.10)

        # Boredom: very low difficulty raises frustration (boredom-type)
        if difficulty < cfg.boredom_threshold and self._frustration < 0.5:
            self._frustration = min(1.0, self._frustration + 0.005)

        # EMA smoothing of frustration (3-step window)
        self._frus_window.append(self._frustration)
        if len(self._frus_window) > 3:
            self._frus_window.pop(0)
        smooth_frus = sum(self._frus_window) / len(self._frus_window)
        smooth_frus = float(max(0.0, min(1.0, smooth_frus)))

        # Discretize
        frus_state = FrustrationState.from_float(smooth_frus)

        # Quit signal
        quit_signal = smooth_frus >= cfg.quit_threshold

        obs = StepObservation(
            t                  = self._t,
            archetype          = self.archetype.value,
            difficulty         = difficulty,
            performance_score  = round(perf, 4),
            frustration_score  = round(smooth_frus, 4),
            frustration_state  = frus_state,
            skill_level        = round(self._skill, 4),
            fatigue            = round(self._fatigue, 4),
            quit_signal        = quit_signal,
        )
        self.history.append(obs)
        return obs

    # ------------------------------------------------------------------
    # Reset between episodes
    # ------------------------------------------------------------------

    def reset(self, keep_skill: bool = True):
        """Reset simulator state. Optionally retain accumulated skill."""
        saved_skill = self._skill if keep_skill else self._cfg.base_skill
        self._t           = 0
        self._skill       = saved_skill
        self._fatigue     = 0.0
        self._frustration = 0.3
        self._frus_window = []
        self.history.clear()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self.history:
            return {}
        perfs  = [o.performance_score for o in self.history]
        fruss  = [o.frustration_score  for o in self.history]
        diffs  = [o.difficulty         for o in self.history]
        quit_t = next((o.t for o in self.history if o.quit_signal), None)
        return {
            "archetype":        self.archetype.value,
            "total_steps":      self._t,
            "avg_performance":  round(sum(perfs)/len(perfs), 4),
            "avg_frustration":  round(sum(fruss)/len(fruss), 4),
            "avg_difficulty":   round(sum(diffs)/len(diffs), 4),
            "final_skill":      round(self._skill, 4),
            "quit_signal_at":   quit_t,
        }
