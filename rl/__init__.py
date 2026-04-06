"""
rl/__init__.py
--------------
Q-Learning adaptive difficulty package.

Quick usage:
    from rl.q_learning import QLearningAgent
    from rl.training   import simulate_training
    from rl.player_simulator import PlayerSimulator
"""

from rl.q_learning       import (QLearningAgent, FrustrationState,
                                   DifficultyAction, PlayerProfile,
                                   RewardFunction, RewardConfig, make_q_table)
from rl.player_simulator import PlayerSimulator, Archetype
from rl.training         import (simulate_training, simulate_personalized,
                                  print_training_summary,
                                  print_ascii_learning_curves,
                                  visualize, visualize_personalized)

__all__ = [
    "QLearningAgent", "FrustrationState", "DifficultyAction",
    "PlayerProfile", "RewardFunction", "RewardConfig", "make_q_table",
    "PlayerSimulator", "Archetype",
    "simulate_training", "simulate_personalized",
    "print_training_summary", "print_ascii_learning_curves",
    "visualize", "visualize_personalized",
]
