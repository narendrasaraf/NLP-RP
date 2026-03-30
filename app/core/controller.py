"""
app/core/controller.py
-----------------------
Adaptive Difficulty Controller — RL-inspired difficulty adjustment policy.

Policy:
  p_frus > theta_frus  ->  decrease difficulty
  p_bore > theta_bore  ->  increase difficulty
  otherwise            ->  maintain (flow state)

The step size scales with |CII| so that larger instability triggers
proportionally larger corrections. This is the RL policy's action executor;
the reward signal is -CII^2, driving the system toward CII = 0.

Also maintains per-session history for analytics and the Streamlit dashboard.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from app.core.config import ControllerConfig, PredictorConfig, settings
from app.core.predictor import PredictionResult
from app.core.emotional_dynamics import EmotionalState


# ---------------------------------------------------------------------------
# Controller step output
# ---------------------------------------------------------------------------

@dataclass
class ControllerOutput:
    action:           str    # "decrease" | "maintain" | "increase"
    difficulty_level: float
    difficulty_delta: float
    reward:           float  # Instantaneous RL reward


# ---------------------------------------------------------------------------
# Adaptive controller
# ---------------------------------------------------------------------------

class AdaptiveDifficultyController:
    """
    Proportional difficulty adjustment controller.

    Implements the RL policy's action layer:
      State    : [CII, p_frus, p_bore, D(t)]
      Action   : {decrease, maintain, increase}
      Reward   : -alpha*CII^2 - beta*p_frus - gamma*p_bore + delta*[flow]

    The controller is stateful per session, tracking difficulty history.
    """

    # Reward weights (mirror RL reward function)
    R_CII_ALPHA  = 0.50
    R_FRUS_BETA  = 0.30
    R_BORE_GAMMA = 0.20
    R_FLOW_DELTA = 0.10
    FLOW_WINDOW  = (-0.20, +0.20)   # CII range considered "flow"

    def __init__(
        self,
        ctrl_cfg: ControllerConfig  = None,
        pred_cfg: PredictorConfig   = None,
    ):
        self.ctrl = ctrl_cfg or settings.controller
        self.pred = pred_cfg or settings.predictor

        self._difficulty: float = self.ctrl.difficulty_init
        self._history: List[dict] = []
        self._changes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        prediction: PredictionResult,
        state:      EmotionalState,
    ) -> ControllerOutput:
        """
        Given the prediction and emotional state, select and apply
        a difficulty action. Returns ControllerOutput.
        """
        cii    = state.cii
        p_frus = prediction.frustration_prob
        p_bore = prediction.boredom_prob

        # ── Action selection ────────────────────────────────────────────────
        prev_difficulty = self._difficulty
        scaled_step = self.ctrl.base_step * (1.0 + abs(cii))

        if prediction.player_state == "frustrated" or p_frus > self.pred.frustration_prob_min:
            action = "decrease"
            self._difficulty = max(
                self.ctrl.difficulty_min,
                self._difficulty - scaled_step,
            )
        elif prediction.player_state == "bored" or p_bore > self.pred.boredom_prob_min:
            action = "increase"
            self._difficulty = min(
                self.ctrl.difficulty_max,
                self._difficulty + scaled_step,
            )
        else:
            action  = "maintain"
            scaled_step = 0.0

        delta = self._difficulty - prev_difficulty
        if delta != 0.0:
            self._changes += 1

        # ── Reward computation ─────────────────────────────────────────────
        in_flow = self.FLOW_WINDOW[0] <= cii <= self.FLOW_WINDOW[1]
        reward  = (
            - self.R_CII_ALPHA  * cii ** 2
            - self.R_FRUS_BETA  * p_frus
            - self.R_BORE_GAMMA * p_bore
            + self.R_FLOW_DELTA * (1.0 if in_flow else 0.0)
        )

        output = ControllerOutput(
            action           = action,
            difficulty_level = round(self._difficulty, 3),
            difficulty_delta = round(delta,             3),
            reward           = round(reward,            4),
        )

        # ── History logging ────────────────────────────────────────────────
        self._history.append({
            "cii":        cii,
            "state":      prediction.player_state,
            "action":     action,
            "difficulty": self._difficulty,
            "reward":     reward,
        })

        return output

    def reset(self):
        self._difficulty = self.ctrl.difficulty_init
        self._history.clear()
        self._changes = 0

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def history(self) -> List[dict]:
        return self._history

    @property
    def difficulty_changes(self) -> int:
        return self._changes

    def session_summary(self) -> dict:
        """Compute aggregate session statistics."""
        if not self._history:
            return {}
        ciis       = [h["cii"] for h in self._history]
        states     = [h["state"] for h in self._history]
        flow_count = sum(
            1 for h in self._history
            if self.FLOW_WINDOW[0] <= h["cii"] <= self.FLOW_WINDOW[1]
        )
        return {
            "total_steps":        len(self._history),
            "avg_cii":            round(sum(ciis) / len(ciis), 4),
            "min_cii":            round(min(ciis), 4),
            "max_cii":            round(max(ciis), 4),
            "frustration_events": states.count("frustrated"),
            "boredom_events":     states.count("bored"),
            "flow_rate":          round(flow_count / len(self._history), 4),
            "final_difficulty":   round(self._difficulty, 3),
            "difficulty_changes": self._changes,
        }


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

class ControllerRegistry:
    def __init__(self):
        self._controllers: Dict[str, AdaptiveDifficultyController] = {}

    def get_or_create(self, session_id: str) -> AdaptiveDifficultyController:
        if session_id not in self._controllers:
            self._controllers[session_id] = AdaptiveDifficultyController()
        return self._controllers[session_id]

    def drop_session(self, session_id: str):
        self._controllers.pop(session_id, None)

    def get_summary(self, session_id: str) -> dict:
        ctrl = self._controllers.get(session_id)
        return ctrl.session_summary() if ctrl else {}


controller_registry = ControllerRegistry()
