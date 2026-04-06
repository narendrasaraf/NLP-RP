"""
rl/demo.py
-----------
End-to-end demo for the Q-learning adaptive difficulty system.

Runs three demonstrations:
  DEMO 1 — Basic Q-learning (single average player, 300 episodes)
  DEMO 2 — Personalized training (3 archetypes, independent Q-tables)
  DEMO 3 — Policy comparison (show before/after improvement)

Run:
    python -m rl.demo                    # text output only (no matplotlib)
    python -m rl.demo --viz              # save matplotlib figures
    python -m rl.demo --episodes 500     # longer training
    python -m rl.demo --archetype casual # single archetype
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.q_learning       import QLearningAgent, FrustrationState, DifficultyAction
from rl.player_simulator import PlayerSimulator
from rl.training         import (simulate_training, simulate_personalized,
                                  print_training_summary, print_ascii_learning_curves,
                                  visualize, visualize_personalized)


# ---------------------------------------------------------------------------
# Helper: show Q-table evolution (early vs late snapshots)
# ---------------------------------------------------------------------------

def print_q_evolution(history: dict) -> None:
    """Show how Q-values changed from start to finish of training."""
    snaps = history.get("q_snapshots", [])
    if len(snaps) < 2:
        return

    first = snaps[0]
    last  = snaps[-1]
    sep   = "-" * 64

    print(f"\n  Q-TABLE EVOLUTION  (Episode {first['episode']} -> {last['episode']})")
    print(sep)
    print(f"  {'STATE-ACTION':<28}  {'EARLY':>10}  {'FINAL':>10}  {'DELTA':>10}")
    print(sep)

    for st in FrustrationState.all():
        for ac in DifficultyAction.all():
            early_v = first["q_table"][st.value][ac.value]
            late_v  = last["q_table"][st.value][ac.value]
            delta   = late_v - early_v
            label   = f"{st.short}->{ac.short}"
            marker  = " <-*" if abs(delta) > 0.05 else ""
            print(f"  {label:<28}  {early_v:>+10.4f}  {late_v:>+10.4f}  {delta:>+10.4f}{marker}")
    print(sep)


# ---------------------------------------------------------------------------
# Helper: show how system learns better actions over time
# ---------------------------------------------------------------------------

def print_policy_comparison(history: dict, agent: QLearningAgent) -> None:
    """Show how the optimal policy evolved across training."""
    snaps = history.get("q_snapshots", [])
    if not snaps:
        return

    print(f"\n  POLICY EVOLUTION OVER TRAINING")
    print("=" * 64)
    print(f"  {'STATE':<12}", end="")
    # Show policy at 5 snapshots
    indices = [0, len(snaps)//4, len(snaps)//2, 3*len(snaps)//4, len(snaps)-1]
    for idx in indices:
        ep = snaps[idx]["episode"]
        print(f"  {'Ep'+str(ep):>10}", end="")
    print(f"  {'FINAL':>10}")
    print("-" * 64)

    for st in FrustrationState.all():
        print(f"  {st.short:<12}", end="")
        for idx in indices:
            q = snaps[idx]["q_table"]
            best = max(q[st.value], key=q[st.value].get)
            print(f"  {DifficultyAction(best).short:>10}", end="")
        # Final policy
        final_ac = agent.optimal_action(st)
        print(f"  {final_ac.short:>10}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Helper: JSON output
# ---------------------------------------------------------------------------

def export_results(agent: QLearningAgent, history: dict, path: str) -> None:
    payload = {
        "q_table":       agent._q_table(),
        "optimal_policy": agent.optimal_policy(),
        "hyperparams": {
            "alpha": agent.alpha, "gamma": agent.gamma,
            "epsilon_final": round(agent.epsilon, 4),
            "total_steps": agent.step_count,
            "episodes": history["n_episodes"],
        },
        "episode_rewards": history["episode_rewards"],
        "archetype": history["archetype"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results exported to: {path}")


# ---------------------------------------------------------------------------
# DEMO 1: Basic Q-Learning
# ---------------------------------------------------------------------------

def run_demo1(n_episodes: int, archetype: str, viz: bool = False) -> None:
    print("\n" + "=" * 64)
    print(f"  DEMO 1: Basic Q-Learning  [{archetype.upper()} player]")
    print("=" * 64)

    agent   = QLearningAgent(
        alpha=0.15, gamma=0.90,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        seed=42,
    )
    history = simulate_training(
        agent, n_episodes=n_episodes, steps_per_ep=50,
        archetype=archetype, initial_diff=5.0, seed=42, verbose=True,
    )

    print_training_summary(agent, history)
    print_q_evolution(history)
    print_policy_comparison(history, agent)
    print_ascii_learning_curves(history)

    # JSON output
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"q_results_{archetype}.json"
    )
    export_results(agent, history, out_path)

    if viz:
        fig_path = out_path.replace(".json", ".png")
        visualize(history, agent, save_path=fig_path)

    return agent, history


# ---------------------------------------------------------------------------
# DEMO 2: Personalized Training
# ---------------------------------------------------------------------------

def run_demo2(n_episodes: int, viz: bool = False) -> None:
    print("\n" + "=" * 64)
    print("  DEMO 2: Personalized Q-Learning (3 Player Archetypes)")
    print("=" * 64)

    histories, agent = simulate_personalized(
        n_episodes   = n_episodes,
        steps_per_ep = 40,
        seed         = 0,
    )

    print("\n  PERSONALIZED RESULTS:")
    for pid, hist in histories.items():
        print(f"\n  -- Player: {pid} --")
        agent.print_q_table(player_id=pid, title=f"After {n_episodes} eps")
        agent.print_optimal_policy(player_id=pid)
        profile = agent.get_profile(pid)
        print(f"  Profile: sensitivity={profile.frustration_sensitivity:.2f}  "
              f"pref_diff={profile.preferred_difficulty:.1f}  "
              f"skill={profile.skill_level:.3f}")

    # ASCII learning curves for the average player
    if "average_01" in histories:
        print_ascii_learning_curves(histories["average_01"])

    if viz:
        fig_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "q_personalized.png"
        )
        visualize_personalized(histories, agent, save_path=fig_path)

    return histories, agent


# ---------------------------------------------------------------------------
# DEMO 3: Before vs After (show learning improvement numerically)
# ---------------------------------------------------------------------------

def run_demo3(archetype: str = "average") -> None:
    print("\n" + "=" * 64)
    print("  DEMO 3: System Learning Improvement — Before vs After")
    print("=" * 64)

    sim = PlayerSimulator(archetype=archetype, seed=99)

    # --- UNTRAINED agent (0 episodes) ---
    untrained = QLearningAgent(
        alpha=0.15, gamma=0.90, epsilon_start=0.0, seed=99  # greedy but random init
    )

    print(f"\n  [UNTRAINED] Random Q-Table:")
    untrained.print_q_table(title="Untrained (all zeros)")
    print("  Untrained optimal policy (all ties -> random):")
    untrained.print_optimal_policy()

    # --- TRAINED agent ---
    trained = QLearningAgent(alpha=0.15, gamma=0.90, epsilon_start=1.0,
                              epsilon_end=0.05, epsilon_decay=0.994, seed=99)
    hist = simulate_training(trained, n_episodes=400, steps_per_ep=50,
                              archetype=archetype, seed=99, verbose=False)

    print(f"\n  [TRAINED — 400 episodes] Learned Q-Table:")
    trained.print_q_table(title="Trained")
    trained.print_optimal_policy()

    # --- Side-by-side test: run 20 episodes, compare total rewards ---
    print(f"\n  EVALUATION: 20 episodes comparison (epsilon=0, pure greedy)")
    print("  " + "-" * 50)

    for label, agent in [("UNTRAINED", untrained), ("TRAINED", trained)]:
        sim.reset(keep_skill=False)
        ep_rewards = []
        difficulty  = 5.0

        obs = sim.step(difficulty)
        for t in range(50):
            state  = obs.frustration_state
            action = agent.optimal_action(state)   # greedy, no exploration
            from rl.training import _apply_action
            difficulty = _apply_action(difficulty, action)
            obs        = sim.step(difficulty)
            reward = agent._reward_fn.compute(
                obs.performance_score, obs.frustration_score,
                difficulty/10, 5.0/10,
            )
            ep_rewards.append(reward)

        tot = sum(ep_rewards)
        avg_frus = sum(o.frustration_score for o in sim.history) / max(1, len(sim.history))
        print(f"  {label:<12}  total_reward={tot:+.3f}  avg_frustration={avg_frus:.3f}")

    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning Adaptive Difficulty Demo"
    )
    parser.add_argument("--episodes",  type=int,  default=300,
                        help="Training episodes per demo (default 300)")
    parser.add_argument("--archetype", type=str,  default="average",
                        choices=["casual", "average", "hardcore"],
                        help="Player archetype for Demo 1 & 3")
    parser.add_argument("--viz",       action="store_true",
                        help="Save matplotlib figures (requires matplotlib)")
    parser.add_argument("--demo",      type=int,  default=0,
                        help="Run only demo N (1=basic, 2=personalized, 3=before-after). 0=all")
    args = parser.parse_args()

    if args.demo in (0, 1):
        run_demo1(args.episodes, args.archetype, viz=args.viz)
    if args.demo in (0, 2):
        run_demo2(args.episodes // 2, viz=args.viz)
    if args.demo in (0, 3):
        run_demo3(args.archetype)

    print("\n  Done.")
    print("  JSON results saved to: rl/q_results_*.json")
    if args.viz:
        print("  Figures saved to: rl/*.png")
    else:
        print("  For matplotlib figures, run with: --viz")


if __name__ == "__main__":
    main()
