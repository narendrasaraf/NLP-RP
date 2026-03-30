"""
app/core/config.py
------------------
Central configuration for the Cognitive Regulation System.
All tunable parameters live here — no magic numbers in the codebase.
"""

from dataclasses import dataclass, field


@dataclass
class CIIConfig:
    """Weights for Cognitive Instability Index (must sum to 1.0)."""
    alpha: float = 0.35   # Emotional Momentum weight
    beta:  float = 0.30   # Emotional Acceleration weight
    gamma: float = 0.25   # Performance Deviation weight
    delta: float = 0.10   # Intent-Outcome Gap weight

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma + self.delta
        assert abs(total - 1.0) < 1e-6, f"CII weights must sum to 1.0 (got {total})"


@dataclass
class PredictorConfig:
    """Thresholds for state prediction."""
    frustration_threshold: float = -0.40   # CII below this -> Frustrated
    boredom_threshold:     float = +0.30   # CII above this -> Bored
    accel_warning:         float = -0.25   # Negative accel amplifies frus risk
    frustration_prob_min:  float =  0.45   # p_frus above this -> trigger
    boredom_prob_min:      float =  0.45   # p_bore above this -> trigger


@dataclass
class ControllerConfig:
    """Difficulty adjustment controller parameters."""
    difficulty_init:  float = 5.0
    difficulty_min:   float = 1.0
    difficulty_max:   float = 10.0
    base_step:        float = 0.5    # Base difficulty adjustment step


@dataclass
class LSTMConfig:
    """LSTM temporal predictor hyperparameters."""
    input_dim:   int   = 5         # [CII, M, A, D, G]
    hidden_dim:  int   = 64
    num_layers:  int   = 2
    dropout:     float = 0.3
    seq_len:     int   = 10        # Sliding window length W
    lr:          float = 1e-3
    batch_size:  int   = 32


@dataclass
class PerformanceTrackerConfig:
    """Rolling performance baseline tracker."""
    window_size:   int   = 20      # Steps to keep in rolling history
    warmup_steps:  int   = 3       # Steps before D(t) is meaningful


@dataclass
class SystemConfig:
    """Top-level system configuration — aggregates all sub-configs."""
    cii:         CIIConfig              = field(default_factory=CIIConfig)
    predictor:   PredictorConfig        = field(default_factory=PredictorConfig)
    controller:  ControllerConfig       = field(default_factory=ControllerConfig)
    lstm:        LSTMConfig             = field(default_factory=LSTMConfig)
    perf:        PerformanceTrackerConfig = field(default_factory=PerformanceTrackerConfig)
    delta_t:     float                  = 1.0   # Seconds per timestep
    api_host:    str                    = "0.0.0.0"
    api_port:    int                    = 8000
    debug:       bool                   = True


# Singleton instance used across the application
settings = SystemConfig()
