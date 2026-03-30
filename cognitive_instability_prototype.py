"""
=============================================================================
Predictive Cognitive State Modeling — Prototype Implementation
=============================================================================
Project : Adaptive Game Difficulty Regulation via CII
Phase   : 4 — Python Prototype (Demo-Ready)
Author  : Research Prototype
=============================================================================

Pipeline:
  Raw Features → Emotional State E(t) → Momentum M(t) → Acceleration A(t)
  → CII(t) → State Prediction → Difficulty Adjustment

Temporal Extension:
  Sequence of CII(t) → LSTM → Frustration / Boredom Probability
=============================================================================
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerInput:
    """
    Raw input snapshot at a single timestep t.
    In production, these come from the game engine + NLP pipeline.
    """
    timestamp:        float   # Unix timestamp
    polarity:         float   # P(t) ∈ [-1, +1]  — from NLP sentiment model
    intensity:        float   # I(t) ∈ [ 0,  1]  — from NLP + telemetry
    retries:          int     # r(t) ≥ 0          — retry count in window
    deaths:           int     # d(t) ≥ 0          — death count in window
    reaction_time_ms: float   # τ(t) > 0          — avg reaction latency
    score_delta:      float   # Δs(t) ∈ ℝ         — score change this window
    intent_gap:       float   # G(t) ∈ [0, 1]     — intent-outcome cosine dist
    text_input:       str     = ""                 # raw player text (optional)


@dataclass
class EmotionalState:
    """
    E(t) = (P(t), I(t), D(t), G(t)) — the 4D affective state vector.
    """
    polarity:            float   # P(t)
    intensity:           float   # I(t)
    performance_dev:     float   # D(t) — z-score relative to player baseline
    intent_gap:          float   # G(t)
    momentum:            float   = 0.0   # M(t) = P(t) * I(t)
    acceleration:        float   = 0.0   # A(t) = ΔM / Δt
    cii:                 float   = 0.0   # CII(t)


@dataclass
class PredictionOutput:
    """
    Output of the cognitive state predictor at time t.
    """
    player_state:         str     # "frustrated" | "bored" | "engaged"
    frustration_prob:     float   # p̂_frus ∈ [0, 1]
    boredom_prob:         float   # p̂_bore ∈ [0, 1]
    engagement_prob:      float   # p̂_eng  ∈ [0, 1]
    cii:                  float   # raw CII value
    difficulty_action:    str     # "decrease" | "increase" | "maintain"
    difficulty_delta:     float   # Δd applied


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PERFORMANCE BASELINE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Maintains a rolling window of performance scores to compute
    the z-score normalization for D(t).

    D(t) = (P_obs(t) - μ_P) / σ_P
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def _performance_score(self, inp: PlayerInput) -> float:
        """
        Composite performance score from telemetry.
        Lower deaths + lower retries + higher score delta → higher performance.
        """
        return inp.score_delta - (inp.deaths * 2.0) - (inp.retries * 1.5)

    def update_and_get_deviation(self, inp: PlayerInput) -> float:
        """
        Update history and return D(t) — z-scored performance deviation.
        Returns 0.0 during warm-up (insufficient history).
        """
        p_obs = self._performance_score(inp)
        self.history.append(p_obs)

        if len(self.history) < 3:
            return 0.0   # warm-up period — not enough data

        mean = sum(self.history) / len(self.history)
        variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        std = math.sqrt(variance) if variance > 0 else 1.0

        return (p_obs - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: EMOTIONAL DYNAMICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EmotionalDynamicsEngine:
    """
    Computes the full emotional state E(t) at each timestep:

        M(t)   = P(t) · I(t)
        A(t)   = [M(t) - M(t-1)] / Δt
        CII(t) = α·M(t) + β·A(t) + γ·D(t) + δ·G(t)

    CII Weights (α, β, γ, δ):
        α = 0.35  — emotional momentum (primary affective signal)
        β = 0.30  — acceleration (early-warning / directional change)
        γ = 0.25  — performance deviation (behavioral grounding)
        δ = 0.10  — intent-outcome gap (cognitive friction)
    """

    def __init__(self,
                 alpha: float = 0.35,
                 beta:  float = 0.30,
                 gamma: float = 0.25,
                 delta: float = 0.10,
                 delta_t: float = 1.0):

        assert abs(alpha + beta + gamma + delta - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.delta   = delta
        self.delta_t = delta_t          # time interval between steps

        self._prev_momentum: float = 0.0

    def compute(self, inp: PlayerInput, performance_dev: float) -> EmotionalState:
        """
        Given raw input and performance deviation, return full EmotionalState.
        """
        P = inp.polarity
        I = inp.intensity
        D = performance_dev
        G = inp.intent_gap

        # ── Emotional Momentum ──────────────────────────────────────────────
        M = P * I                                            # M(t) = P(t)·I(t)

        # ── Emotional Acceleration ──────────────────────────────────────────
        A = (M - self._prev_momentum) / self.delta_t        # A(t) = ΔM / Δt
        self._prev_momentum = M

        # ── Cognitive Instability Index ──────────────────────────────────────
        CII = (self.alpha * M) + \
              (self.beta  * A) + \
              (self.gamma * D) + \
              (self.delta * G)

        return EmotionalState(
            polarity        = P,
            intensity       = I,
            performance_dev = D,
            intent_gap      = G,
            momentum        = M,
            acceleration    = A,
            cii             = CII
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STATE PREDICTOR (RULE-BASED THRESHOLD ENGINE)
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedStatePredictor:
    """
    Threshold-based cognitive state predictor using CII.

    Decision boundaries (calibrated from initial weights):
        CII < -0.40  → Frustrated
        CII > +0.30  → Bored
        Otherwise    → Engaged (flow state)

    Also uses acceleration as a secondary signal for early-warning.
    """

    FRUSTRATION_THRESHOLD = -0.40
    BOREDOM_THRESHOLD     = +0.30
    ACCEL_WARNING         = -0.25   # negative acceleration amplifies frustration risk

    def predict(self, state: EmotionalState) -> Tuple[str, float, float, float]:
        """
        Returns (player_state, p_frustrated, p_bored, p_engaged).
        Probabilities are heuristic approximations (softmax-like over CII).
        """

        cii = state.cii
        accel = state.acceleration

        # ── Raw scores (higher = more likely for that state) ─────────────────
        # Frustration score: negative CII + rapid downward acceleration
        frus_score = max(0.0, -cii + max(0.0, -accel))

        # Boredom score: positive CII + near-zero acceleration
        bore_score = max(0.0, cii - abs(accel) * 0.5)

        # Engagement score: CII near zero, low magnitude
        eng_score  = max(0.0, 1.0 - abs(cii) - 0.3 * abs(accel))

        # ── Softmax normalization ────────────────────────────────────────────
        total = frus_score + bore_score + eng_score + 1e-9
        p_frus = frus_score / total
        p_bore = bore_score / total
        p_eng  = eng_score  / total

        # ── Hard decision ────────────────────────────────────────────────────
        # Acceleration warning: even if CII is borderline, rapid decline → frus
        if cii < self.FRUSTRATION_THRESHOLD or \
           (cii < -0.20 and accel < self.ACCEL_WARNING):
            state_label = "frustrated"

        elif cii > self.BOREDOM_THRESHOLD:
            state_label = "bored"

        else:
            state_label = "engaged"

        return state_label, p_frus, p_bore, p_eng


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: ADAPTIVE DIFFICULTY CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDifficultyController:
    """
    Adjusts game difficulty D ∈ [D_min, D_max] based on predicted player state.

    Policy:
        p_frus > θ_frus  → decrease difficulty by Δd
        p_bore > θ_bore  → increase difficulty by Δd
        Otherwise        → maintain difficulty

    Δd is scaled by the magnitude of CII to make the response proportional.
    """

    def __init__(self,
                 difficulty_init: float = 5.0,
                 d_min: float = 1.0,
                 d_max: float = 10.0,
                 base_delta: float = 0.5,
                 theta_frus: float = 0.45,
                 theta_bore: float = 0.45):

        self.difficulty  = difficulty_init
        self.d_min       = d_min
        self.d_max       = d_max
        self.base_delta  = base_delta
        self.theta_frus  = theta_frus
        self.theta_bore  = theta_bore

    def adjust(self, state_label: str, p_frus: float,
               p_bore: float, cii: float) -> Tuple[str, float]:
        """
        Returns (action, delta_applied).
        """
        # Scale adjustment by CII magnitude (larger instability → larger step)
        scaled_delta = self.base_delta * (1.0 + abs(cii))

        if state_label == "frustrated" or p_frus > self.theta_frus:
            action = "decrease"
            self.difficulty = max(self.d_min, self.difficulty - scaled_delta)

        elif state_label == "bored" or p_bore > self.theta_bore:
            action = "increase"
            self.difficulty = min(self.d_max, self.difficulty + scaled_delta)

        else:
            action = "maintain"
            scaled_delta = 0.0

        return action, scaled_delta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: SIMULATED LSTM PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedLSTMPredictor:
    """
    Simulates the LSTM temporal predictor that would operate on a
    sequence of past CII values to predict future cognitive state.

    In a full implementation, this would be a trained PyTorch/Keras LSTM:

        Input  : Tensor of shape (batch, seq_len, input_dim)
                 where input_dim = [CII, M, A, D, G] at each step
        Output : (p̂_frus, p̂_bore)  — future state probabilities

    Here we use a weighted exponential moving average over the CII
    history as a proxy for the sequential pattern detection of LSTM.
    This captures the core insight: recent CII values are weighted
    more heavily than older ones (analogous to LSTM's learned memory gate).
    """

    def __init__(self, window: int = 10, decay: float = 0.75):
        """
        window : number of past CII steps used as context
        decay  : exponential decay weight for older steps (0 < decay < 1)
        """
        self.window = window
        self.decay  = decay
        self.cii_history: deque = deque(maxlen=window)

    def update(self, cii: float):
        self.cii_history.append(cii)

    def predict_future_state(self) -> Tuple[float, float]:
        """
        Returns (p̂_frus, p̂_bore) based on exponentially-weighted CII history.

        Logic:
          - A sustained downward trend in CII → high frustration probability
          - A sustained upward trend in CII   → high boredom probability
          - Stable near-zero CII              → high engagement probability
        """
        if len(self.cii_history) < 2:
            return 0.0, 0.0   # insufficient history

        # Exponentially weighted average: recent steps weighted more
        weights = [self.decay ** (len(self.cii_history) - 1 - i)
                   for i in range(len(self.cii_history))]
        w_sum   = sum(weights)
        ewa_cii = sum(w * c for w, c in zip(weights, self.cii_history)) / w_sum

        # Trend: linear slope over history (proxy for LSTM's temporal gradient)
        n = len(self.cii_history)
        x_mean = (n - 1) / 2.0
        history_list = list(self.cii_history)
        cov = sum((i - x_mean) * history_list[i] for i in range(n))
        var = sum((i - x_mean) ** 2 for i in range(n))
        trend = cov / var if var > 0 else 0.0   # slope (negative = declining)

        # Map EWA-CII and trend to probabilities
        p_frus = _sigmoid(-ewa_cii * 3.0 - trend * 2.0)
        p_bore = _sigmoid(ewa_cii * 3.0 + trend * 2.0)

        # Normalize so they don't jointly exceed 1
        total  = p_frus + p_bore + 1e-9
        if total > 1.0:
            p_frus /= total
            p_bore /= total

        return round(p_frus, 4), round(p_bore, 4)


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid: σ(x) = 1 / (1 + e^{-x})"""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN SYSTEM ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveRegulationSystem:
    """
    Top-level orchestrator that wires all components together.

    Per-timestep pipeline:
        PlayerInput → PerformanceTracker → EmotionalDynamicsEngine
        → RuleBasedStatePredictor → SimulatedLSTMPredictor
        → AdaptiveDifficultyController → PredictionOutput
    """

    def __init__(self):
        self.perf_tracker  = PerformanceTracker(window_size=20)
        self.dynamics      = EmotionalDynamicsEngine()
        self.predictor     = RuleBasedStatePredictor()
        self.lstm          = SimulatedLSTMPredictor(window=10, decay=0.75)
        self.controller    = AdaptiveDifficultyController()
        self.timestep      = 0

    def process(self, inp: PlayerInput) -> PredictionOutput:
        self.timestep += 1

        # Step 1: Compute performance deviation D(t)
        D = self.perf_tracker.update_and_get_deviation(inp)

        # Step 2: Compute full emotional state E(t) and CII(t)
        state = self.dynamics.compute(inp, D)

        # Step 3: Rule-based state prediction
        state_label, p_frus, p_bore, p_eng = self.predictor.predict(state)

        # Step 4: LSTM temporal prediction (augments rule-based)
        self.lstm.update(state.cii)
        lstm_p_frus, lstm_p_bore = self.lstm.predict_future_state()

        # Blend rule-based and LSTM predictions (60% rule, 40% LSTM)
        final_p_frus = 0.60 * p_frus + 0.40 * lstm_p_frus
        final_p_bore = 0.60 * p_bore + 0.40 * lstm_p_bore
        final_p_eng  = max(0.0, 1.0 - final_p_frus - final_p_bore)

        # Re-evaluate state label from blended probabilities
        probs = {"frustrated": final_p_frus,
                 "bored":      final_p_bore,
                 "engaged":    final_p_eng}
        final_label = max(probs, key=probs.get)

        # Step 5: Adaptive difficulty adjustment
        action, delta = self.controller.adjust(
            final_label, final_p_frus, final_p_bore, state.cii)

        return PredictionOutput(
            player_state      = final_label,
            frustration_prob  = round(final_p_frus, 4),
            boredom_prob      = round(final_p_bore, 4),
            engagement_prob   = round(final_p_eng,  4),
            cii               = round(state.cii, 4),
            difficulty_action = action,
            difficulty_delta  = round(delta, 3)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: DEMO RUNNER WITH SIMULATED SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def print_header():
    print("\n" + "=" * 70)
    print("  PREDICTIVE COGNITIVE STATE MODELING — LIVE DEMO")
    print("  Project: Adaptive Game Difficulty Regulation via CII")
    print("=" * 70)


def print_output(t: int, inp: PlayerInput, out: PredictionOutput,
                 difficulty: float):
    """Pretty-print one timestep result."""
    state_icon = {"frustrated": "[!!]", "bored": "[~~]", "engaged": "[OK]"}
    action_icon = {"decrease": "[v]", "increase": "[^]", "maintain": "[=]"}

    print(f"\n[t={t:02d}] Text: \"{inp.text_input}\"")
    print(f"       Polarity={inp.polarity:+.2f} | Intensity={inp.intensity:.2f} |"
          f" Deaths={inp.deaths} | Retries={inp.retries}")
    print(f"       CII = {out.cii:+.4f}")
    print(f"       Probabilities -> Frustrated: {out.frustration_prob:.2%} | "
          f"Bored: {out.boredom_prob:.2%} | Engaged: {out.engagement_prob:.2%}")
    print(f"       State  : {state_icon[out.player_state]} {out.player_state.upper()}")
    print(f"       Action : {action_icon[out.difficulty_action]}"
          f" {out.difficulty_action.upper()} difficulty"
          f"  (delta={out.difficulty_delta:+.3f}  ->  D={difficulty:.2f})")
    print("       " + "-" * 60)


def run_demo():
    """
    Three scenario simulation:
      Scenario A: Escalating frustration (player struggles repeatedly)
      Scenario B: Boredom onset (player dominates, no challenge)
      Scenario C: Stable flow (healthy engagement)
    """
    print_header()
    system = CognitiveRegulationSystem()

    # ── Scenario definitions ──────────────────────────────────────────────
    scenarios = [

        # ——— Scenario A: Escalating Frustration ——————————————————————————
        ("SCENARIO A — ESCALATING FRUSTRATION", [
            PlayerInput(time.time(), polarity=-0.1, intensity=0.3, retries=1,
                        deaths=1, reaction_time_ms=320, score_delta=-5,
                        intent_gap=0.2, text_input="almost had it"),
            PlayerInput(time.time(), polarity=-0.3, intensity=0.5, retries=2,
                        deaths=3, reaction_time_ms=380, score_delta=-12,
                        intent_gap=0.4, text_input="why does this keep happening"),
            PlayerInput(time.time(), polarity=-0.6, intensity=0.7, retries=4,
                        deaths=5, reaction_time_ms=450, score_delta=-20,
                        intent_gap=0.6, text_input="this is impossible"),
            PlayerInput(time.time(), polarity=-0.8, intensity=0.9, retries=7,
                        deaths=8, reaction_time_ms=510, score_delta=-30,
                        intent_gap=0.8, text_input="I QUIT this is trash"),
            PlayerInput(time.time(), polarity=-0.9, intensity=0.95, retries=9,
                        deaths=10, reaction_time_ms=600, score_delta=-35,
                        intent_gap=0.9, text_input="completely broken game"),
        ]),

        # ——— Scenario B: Boredom Onset ————————————————————————————————————
        ("SCENARIO B — BOREDOM ONSET", [
            PlayerInput(time.time(), polarity=0.8, intensity=0.6, retries=0,
                        deaths=0, reaction_time_ms=180, score_delta=25,
                        intent_gap=0.05, text_input="easy win"),
            PlayerInput(time.time(), polarity=0.7, intensity=0.4, retries=0,
                        deaths=0, reaction_time_ms=170, score_delta=30,
                        intent_gap=0.05, text_input="too easy"),
            PlayerInput(time.time(), polarity=0.5, intensity=0.2, retries=0,
                        deaths=0, reaction_time_ms=160, score_delta=28,
                        intent_gap=0.02, text_input="..."),
            PlayerInput(time.time(), polarity=0.3, intensity=0.1, retries=0,
                        deaths=0, reaction_time_ms=155, score_delta=22,
                        intent_gap=0.01, text_input="this is boring ngl"),
            PlayerInput(time.time(), polarity=0.2, intensity=0.05, retries=0,
                        deaths=0, reaction_time_ms=150, score_delta=18,
                        intent_gap=0.01, text_input="give me something harder"),
        ]),

        # ——— Scenario C: Stable Flow State ————————————————————————————————
        ("SCENARIO C — STABLE FLOW STATE", [
            PlayerInput(time.time(), polarity=0.4, intensity=0.5, retries=1,
                        deaths=1, reaction_time_ms=240, score_delta=10,
                        intent_gap=0.2, text_input="nice challenge"),
            PlayerInput(time.time(), polarity=0.5, intensity=0.55, retries=1,
                        deaths=0, reaction_time_ms=230, score_delta=12,
                        intent_gap=0.15, text_input="getting better"),
            PlayerInput(time.time(), polarity=0.35, intensity=0.5, retries=2,
                        deaths=1, reaction_time_ms=250, score_delta=8,
                        intent_gap=0.25, text_input="tough but fair"),
            PlayerInput(time.time(), polarity=0.45, intensity=0.6, retries=1,
                        deaths=1, reaction_time_ms=235, score_delta=11,
                        intent_gap=0.2, text_input="close one"),
            PlayerInput(time.time(), polarity=0.5, intensity=0.5, retries=1,
                        deaths=0, reaction_time_ms=220, score_delta=14,
                        intent_gap=0.18, text_input="love this game"),
        ]),
    ]

    # ── Run each scenario ──────────────────────────────────────────────────
    global_t = 0
    for scenario_name, inputs in scenarios:
        # Fresh system per scenario for clarity
        system = CognitiveRegulationSystem()

        print(f"\n" + "-" * 70)
        print(f"  {scenario_name}")
        print("-" * 70)

        for local_t, inp in enumerate(inputs, start=1):
            global_t += 1
            output = system.process(inp)
            print_output(local_t, inp, output,
                         system.controller.difficulty)
            time.sleep(0.05)   # small delay for readability in live demo

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("  For LSTM-based temporal prediction, replace SimulatedLSTMPredictor")
    print("  with a trained PyTorch model (see lstm_model_design.py)")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: PYTORCH LSTM DESIGN (REFERENCE — SHOWS PRODUCTION ARCHITECTURE)
# ─────────────────────────────────────────────────────────────────────────────

LSTM_ARCHITECTURE_REFERENCE = """
# Production LSTM Implementation Reference (PyTorch)
# ---------------------------------------------------
# Install: pip install torch

import torch
import torch.nn as nn

class CIILSTMPredictor(nn.Module):
    '''
    Temporal CII Predictor using LSTM.

    Input  : Tensor (batch_size, seq_len, input_dim)
             input_dim = 5 → [CII, M, A, D, G] at each timestep

    Output : (p_frus, p_bore) — future state probabilities

    Architecture:
        LSTM (2 layers, hidden=64) → Dropout(0.3) → Linear(64→2) → Sigmoid
    '''

    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim, 2)   # → [p_frus, p_bore]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden)
        last_hidden  = lstm_out[:, -1, :]  # take final timestep
        out = self.dropout(last_hidden)
        out = self.head(out)             # (batch, 2)
        return self.sigmoid(out)         # probabilities in [0,1]

# --- Training Loop ---
# model     = CIILSTMPredictor()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.BCELoss()
#
# for epoch in range(epochs):
#     for X_batch, y_batch in dataloader:
#         # X_batch: (B, seq_len, 5)
#         # y_batch: (B, 2) — [label_frus, label_bore]
#         preds = model(X_batch)
#         loss  = criterion(preds, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
"""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demo()
    # LSTM_ARCHITECTURE_REFERENCE is a developer code reference block.
    # Uncomment below to print it; it requires UTF-8 compatible terminal.
    # print(LSTM_ARCHITECTURE_REFERENCE)
