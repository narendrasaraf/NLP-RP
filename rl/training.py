"""
rl/training.py
---------------
Training loop, simulation harness, and visualisation for the Q-learning
adaptive difficulty system.

Three entry points:

  1. simulate_training(agent, n_episodes, archetype, seed)
     -- Run a single player training loop. Returns full history.

  2. simulate_personalized(agent, player_configs, n_episodes)
     -- Train one agent across multiple players simultaneously.
        Each player has an independent Q-table and profile.

  3. visualize(history)  / visualize_personalized(histories)
     -- Matplotlib plots:
        (a) Q-value evolution over time (per state-action pair)
        (b) Action selection trends (stacked area)
        (c) Reward learning curve (raw + smoothed)
        (d) Epsilon decay schedule
        (e) Per-player Q-table heatmaps  (personalized mode)
        (f) Difficulty trace vs frustration trace

Usage (no-viz fast demo):
    from rl.training import simulate_training, print_training_summary
    from rl.q_learning import QLearningAgent

    agent   = QLearningAgent(seed=42)
    history = simulate_training(agent, n_episodes=300)
    print_training_summary(agent, history)

Usage (with matplotlib viz):
    from rl.training import simulate_training, visualize
    agent   = QLearningAgent(seed=42)
    history = simulate_training(agent, n_episodes=300)
    visualize(history, agent)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from rl.q_learning       import (QLearningAgent, FrustrationState,
                                   DifficultyAction, make_q_table)
from rl.player_simulator import PlayerSimulator, Archetype


# =============================================================================
# Difficulty environment helper
# =============================================================================

def _apply_action(
    difficulty:    float,
    action:        DifficultyAction,
    step_size:     float = 0.8,
    d_min:         float = 1.0,
    d_max:         float = 10.0,
) -> float:
    """Apply a Q-learning action to the current difficulty level."""
    new_d = difficulty + action.direction * step_size
    return round(max(d_min, min(d_max, new_d)), 3)


# =============================================================================
# 1. Single-player training loop
# =============================================================================

def simulate_training(
    agent:        QLearningAgent,
    n_episodes:   int  = 300,
    steps_per_ep: int  = 50,
    archetype:    str  = "average",
    initial_diff: float = 5.0,
    step_size:    float = 0.8,
    seed:         Optional[int] = None,
    player_id:    Optional[str] = None,
    verbose:      bool  = False,
) -> dict:
    """
    Full Q-learning training simulation.

    Each episode:
      1. Reset simulator (retain skill)
      2. For each step:
         a. Observe frustration state s(t)
         b. Agent selects action via epsilon-greedy
         c. Apply action  -> new difficulty
         d. Simulate player response at new difficulty  -> s(t+1), perf, frus
         e. Compute reward
         f. Update Q-table
      3. End of episode -> update player profile, store metrics

    Args:
        agent:        QLearningAgent instance
        n_episodes:   Number of training episodes
        steps_per_ep: Steps per episode
        archetype:    Player archetype string
        initial_diff: Starting difficulty level
        step_size:    How much difficulty changes per action
        seed:         Random seed for simulator
        player_id:    Player ID for personalized mode
        verbose:      Print per-episode stats

    Returns:
        Training history dict with all metrics for visualisation.
    """
    sim = PlayerSimulator(archetype=archetype, seed=seed)

    # Tracking
    episode_rewards:   List[float] = []
    episode_avg_frus:  List[float] = []
    difficulty_traces: List[List[float]] = []
    frus_traces:       List[List[float]] = []
    action_counts_ep:  List[Dict[str, int]] = []

    for ep in range(1, n_episodes + 1):
        sim.reset(keep_skill=True)
        difficulty      = initial_diff
        prev_difficulty = None

        ep_rewards:    List[float] = []
        ep_frus:       List[float] = []
        ep_diffs:      List[float] = []
        ep_actions:    Dict[str, int] = {a.value: 0 for a in DifficultyAction.all()}

        # Warm up first observation
        obs = sim.step(difficulty)
        state = obs.frustration_state

        for t in range(steps_per_ep):
            # Agent step: choose action, update Q
            result = agent.step(
                state             = state,
                performance_score = obs.performance_score,
                frustration_score = obs.frustration_score,
                difficulty        = difficulty,
                next_state        = state,          # will be updated below
                prev_difficulty   = prev_difficulty,
                player_id         = player_id,
            )

            # Apply action to difficulty
            action         = DifficultyAction(result["action"])
            prev_difficulty = difficulty
            difficulty      = _apply_action(difficulty, action, step_size)

            # Simulate player at new difficulty -> get next state
            next_obs   = sim.step(difficulty)
            next_state = next_obs.frustration_state

            # Re-run the Q-update with the correct next_state
            # (first call above was exploratory, use next_state for Bellman)
            agent._update_q(
                agent._q_table(player_id),
                state, action,
                result["reward"],
                next_state,
            )

            # Track
            ep_rewards.append(result["reward"])
            ep_frus.append(obs.frustration_score)
            ep_diffs.append(difficulty)
            ep_actions[action.value] = ep_actions.get(action.value, 0) + 1

            # Advance
            state = next_state
            obs   = next_obs

            if obs.quit_signal:
                if verbose:
                    print(f"  [ep {ep:>3}] Player quit at t={t+1}")
                break

        # End of episode
        ep_summary = agent.end_episode(
            player_id         = player_id,
            difficulty_trace  = ep_diffs,
            frustration_trace = ep_frus,
            reward_trace      = ep_rewards,
        )

        ep_total  = sum(ep_rewards)
        ep_avg_f  = sum(ep_frus) / max(len(ep_frus), 1)

        episode_rewards.append(ep_total)
        episode_avg_frus.append(ep_avg_f)
        difficulty_traces.append(ep_diffs)
        frus_traces.append(ep_frus)
        action_counts_ep.append(ep_actions)

        if verbose and ep % 50 == 0:
            print(f"  Episode {ep:>3}/{n_episodes}  "
                  f"reward={ep_total:+.3f}  "
                  f"avg_frus={ep_avg_f:.3f}  "
                  f"eps={agent.epsilon:.3f}")

    # Learning curve data from agent
    curves = agent.get_learning_curves()

    return {
        "archetype":         archetype,
        "n_episodes":        n_episodes,
        "steps_per_ep":      steps_per_ep,
        "player_id":         player_id,
        "episode_rewards":   episode_rewards,
        "episode_avg_frus":  episode_avg_frus,
        "difficulty_traces": difficulty_traces,
        "frus_traces":       frus_traces,
        "action_counts_ep":  action_counts_ep,
        "step_rewards":      curves["rewards"],
        "step_rewards_smooth": curves["rewards_smooth"],
        "epsilon_trace":     curves["epsilon"],
        "step_actions":      curves["actions"],
        "step_states":       curves["states"],
        "q_snapshots":       curves["q_snapshots"],
        "final_q_table":     agent._q_table(player_id),
        "optimal_policy":    agent.optimal_policy(player_id),
        "sim_summary":       sim.summary(),
    }


# =============================================================================
# 2. Multi-player personalized training
# =============================================================================

def simulate_personalized(
    n_episodes:   int  = 200,
    steps_per_ep: int  = 40,
    player_configs: Optional[List[dict]] = None,
    seed:         Optional[int] = 0,
) -> Dict[str, dict]:
    """
    Train a personalized Q-learning agent across multiple players.

    Each player gets an independent Q-table and PlayerProfile.
    Returns per-player training histories.

    Args:
        n_episodes:     Episodes per player
        steps_per_ep:   Steps per episode
        player_configs: List of {player_id, archetype, initial_diff} dicts.
                        Defaults to 3 archetype representatives.
        seed:           Base RNG seed (incremented per player)

    Returns:
        {player_id: training_history_dict}
    """
    if player_configs is None:
        player_configs = [
            {"player_id": "casual_01",   "archetype": "casual",   "initial_diff": 3.0},
            {"player_id": "average_01",  "archetype": "average",  "initial_diff": 5.0},
            {"player_id": "hardcore_01", "archetype": "hardcore",  "initial_diff": 7.0},
        ]

    agent = QLearningAgent(
        alpha=0.15, gamma=0.90, epsilon_start=1.0,
        epsilon_end=0.05, epsilon_decay=0.997,
        personalized=True, seed=seed,
    )

    histories: Dict[str, dict] = {}

    for i, cfg in enumerate(player_configs):
        pid       = cfg["player_id"]
        archetype = cfg.get("archetype", "average")
        init_d    = cfg.get("initial_diff", 5.0)

        print(f"\n  [Personalized] Training player: {pid}  "
              f"archetype={archetype}  init_diff={init_d}")

        # Reset epsilon for each player so each has a warm exploration phase
        agent.epsilon = agent.epsilon_start * (0.8 ** i)  # slightly faster decay per player

        history = simulate_training(
            agent         = agent,
            n_episodes    = n_episodes,
            steps_per_ep  = steps_per_ep,
            archetype     = archetype,
            initial_diff  = init_d,
            seed          = (seed or 0) + i * 100,
            player_id     = pid,
            verbose       = True,
        )

        profile_dict = agent.get_profile(pid).to_dict()
        history["player_profile"] = profile_dict
        histories[pid] = history

    return histories, agent


# =============================================================================
# 3. Print summaries
# =============================================================================

def print_training_summary(
    agent:   QLearningAgent,
    history: dict,
    player_id: Optional[str] = None,
) -> None:
    """Print a concise training summary to stdout."""
    sep    = "=" * 64
    n_ep   = history["n_episodes"]
    ep_r   = history["episode_rewards"]
    ep_f   = history["episode_avg_frus"]

    # Compute learning progress (first 10% vs last 10% of episodes)
    n10 = max(1, n_ep // 10)
    early_r = sum(ep_r[:n10]) / n10
    late_r  = sum(ep_r[-n10:]) / n10
    early_f = sum(ep_f[:n10]) / n10
    late_f  = sum(ep_f[-n10:]) / n10

    print(f"\n{sep}")
    print(f"  TRAINING SUMMARY  [{history['archetype'].upper()} player]")
    print(sep)
    print(f"  Episodes         : {n_ep}")
    print(f"  Steps/episode    : {history['steps_per_ep']}")
    print(f"  Total steps      : {agent.step_count}")
    print(f"  Final epsilon    : {agent.epsilon:.4f}")
    print()
    print(f"  Avg reward  (early)  : {early_r:+.4f}")
    print(f"  Avg reward  (late)   : {late_r:+.4f}")
    print(f"  Improvement          : {late_r - early_r:+.4f}  "
          f"({'improved' if late_r > early_r else 'degraded'})")
    print()
    print(f"  Avg frustration (early)  : {early_f:.4f}")
    print(f"  Avg frustration (late)   : {late_f:.4f}")
    print(f"  Frustration reduction    : {early_f - late_f:+.4f}  "
          f"({'better' if late_f < early_f else 'worse'})")
    print()

    # Action distribution (last 20 episodes)
    last_20 = history["action_counts_ep"][-20:]
    total_last = {a: sum(ep.get(a, 0) for ep in last_20) for a in
                  [a.value for a in DifficultyAction.all()]}
    total_n = max(1, sum(total_last.values()))
    print(f"  Action distribution (last 20 eps):")
    for a_val, cnt in total_last.items():
        pct = cnt / total_n * 100
        bar = "#" * int(pct / 3)
        print(f"    {a_val:<25} {bar:<33} {pct:5.1f}%")
    print()

    agent.print_q_table(player_id=player_id, title=f"After {n_ep} episodes")
    agent.print_optimal_policy(player_id=player_id)

    print(f"  Simulator summary: {history['sim_summary']}")
    print(sep)


# =============================================================================
# 4. Text-based visualisation (no matplotlib dependency)
# =============================================================================

def print_ascii_learning_curves(history: dict, width: int = 60) -> None:
    """
    Print ASCII approximations of the key learning curves.
    Works without matplotlib.
    """
    SEP = "-" * (width + 10)

    def _sparkline(values: List[float], w: int = width) -> str:
        """Map values to a single-line sparkline using block chars."""
        chars = " ._-=+#@"
        mn, mx = min(values), max(values)
        rng = max(mx - mn, 1e-9)
        buckets = [int((v - mn) / rng * (len(chars) - 1)) for v in values]
        # Downsample to width
        step = max(1, len(buckets) // w)
        sampled = [buckets[i] for i in range(0, len(buckets), step)][:w]
        return "".join(chars[b] for b in sampled)

    def _bar_chart(values_dict: Dict[str, float], w: int = 40) -> List[str]:
        """Horizontal bar chart."""
        lines = []
        mx = max(values_dict.values()) if values_dict else 1
        for label, val in values_dict.items():
            bar_len = int(val / max(mx, 1) * w)
            lines.append(f"  {label:<25} {'#'*bar_len:<{w}} {val:>6.1f}%")
        return lines

    print(f"\n  ASCII LEARNING CURVES")
    print("=" * (width + 14))

    # Reward curve
    ep_r = history["episode_rewards"]
    print(f"\n  Episode Reward (smooth window=10):")
    window = 10
    smooth = [sum(ep_r[max(0,i-window):i+1]) / len(ep_r[max(0,i-window):i+1])
              for i in range(len(ep_r))]
    print(f"  [{_sparkline(smooth)}]")
    print(f"  min={min(smooth):+.3f}  max={max(smooth):+.3f}  "
          f"final={smooth[-1]:+.3f}")

    # Frustration curve
    frus = history["episode_avg_frus"]
    print(f"\n  Avg Frustration per Episode:")
    print(f"  [{_sparkline(frus)}]")
    print(f"  start={frus[0]:.3f}  end={frus[-1]:.3f}  "
          f"delta={frus[-1]-frus[0]:+.3f}")

    # Epsilon
    eps = history["epsilon_trace"]
    print(f"\n  Epsilon Decay:")
    print(f"  [{_sparkline(eps)}]")
    print(f"  {eps[0]:.3f} -> {eps[-1]:.4f}")

    # Action distribution over time (every 25% of training)
    step_actions = history["step_actions"]
    n = len(step_actions)
    quarters = [
        ("  0-25%",   step_actions[:n//4]),
        (" 25-50%",   step_actions[n//4:n//2]),
        (" 50-75%",   step_actions[n//2:3*n//4]),
        (" 75-100%",  step_actions[3*n//4:]),
    ]
    print(f"\n  Action Selection Trends:")
    print(f"  {'Quarter':<10}  {'INC%':>8}  {'DEC%':>8}  {'NOP%':>8}")
    print(f"  {SEP[:40]}")
    for label, chunk in quarters:
        if not chunk:
            continue
        total = len(chunk)
        inc = chunk.count("increase_difficulty") / total * 100
        dec = chunk.count("decrease_difficulty") / total * 100
        nop = chunk.count("no_change") / total * 100
        print(f"  {label:<10}  {inc:>7.1f}%  {dec:>7.1f}%  {nop:>7.1f}%")

    print()


# =============================================================================
# 5. Matplotlib visualisation (optional)
# =============================================================================

def visualize(
    history:    dict,
    agent:      QLearningAgent,
    player_id:  Optional[str] = None,
    save_path:  Optional[str] = None,
) -> None:
    """
    Generate a 6-panel matplotlib figure showing:
      1. Q-value convergence over training (per state-action)
      2. Episode reward learning curve
      3. Epsilon decay
      4. Action selection trend (stacked area)
      5. Last-episode: difficulty vs frustration trace
      6. Q-table heatmap (final)

    Requires: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[visualize] matplotlib not installed. "
              "Run: pip install matplotlib  or use print_ascii_learning_curves()")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    DARK  = "#0d1117"
    PANEL = "#161b22"
    GRID  = "#30363d"
    TEXT  = "#e6edf3"
    ACC   = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]

    def _ax(row, col, colspan=1, rowspan=1):
        ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        return ax

    n_ep = history["n_episodes"]

    # ── Panel 1: Episode reward learning curve ────────────────────────────
    ax1 = _ax(0, 0, colspan=2)
    ep_r = history["episode_rewards"]
    episodes = list(range(1, len(ep_r) + 1))
    window = max(1, n_ep // 20)
    smooth_ep = [
        sum(ep_r[max(0, i-window):i+1]) / len(ep_r[max(0, i-window):i+1])
        for i in range(len(ep_r))
    ]
    ax1.plot(episodes, ep_r, color=ACC[0], alpha=0.25, linewidth=0.7, label="Raw")
    ax1.plot(episodes, smooth_ep, color=ACC[0], linewidth=2.0, label=f"Smooth (w={window})")
    ax1.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax1.set_title("Episode Reward — Learning Curve", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
    ax1.grid(color=GRID, linewidth=0.4)

    # ── Panel 2: Epsilon decay ────────────────────────────────────────────
    ax2 = _ax(0, 2)
    steps_eps = list(range(1, len(history["epsilon_trace"]) + 1))
    ax2.plot(steps_eps, history["epsilon_trace"], color=ACC[3], linewidth=1.5)
    ax2.set_title("Epsilon Decay (Exploration Rate)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Epsilon")
    ax2.grid(color=GRID, linewidth=0.4)

    # ── Panel 3: Q-value convergence ─────────────────────────────────────
    ax3 = _ax(1, 0, colspan=2)
    snapshots = history["q_snapshots"]
    if snapshots:
        ep_snaps = [s["episode"] for s in snapshots]
        state_action_pairs = [
            (FrustrationState.HIGH,    DifficultyAction.DECREASE, "HIGH->DEC"),
            (FrustrationState.HIGH,    DifficultyAction.NO_CHANGE, "HIGH->NOP"),
            (FrustrationState.LOW,     DifficultyAction.INCREASE, "LOW->INC"),
            (FrustrationState.MODERATE,DifficultyAction.NO_CHANGE, "MOD->NOP"),
        ]
        for i, (st, ac, lbl) in enumerate(state_action_pairs):
            qvals = [s["q_table"][st.value][ac.value] for s in snapshots]
            ax3.plot(ep_snaps, qvals, color=ACC[i % len(ACC)],
                     linewidth=1.8, marker="o", markersize=3, label=lbl)
        ax3.set_title("Q-Value Convergence (Key State-Action Pairs)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Q-Value")
        ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
        ax3.grid(color=GRID, linewidth=0.4)
        ax3.axhline(0, color=GRID, linewidth=0.8, linestyle="--")

    # ── Panel 4: Action selection trend (stacked area) ────────────────────
    ax4 = _ax(1, 2)
    step_actions = history["step_actions"]
    n_steps = len(step_actions)
    seg_w   = max(1, n_steps // 40)
    segs    = [step_actions[i:i+seg_w] for i in range(0, n_steps, seg_w)]

    def _pct(seg, val):
        return sum(1 for s in seg if s == val) / max(1, len(seg)) * 100

    xs    = [i * seg_w for i in range(len(segs))]
    inc_y = [_pct(s, "increase_difficulty") for s in segs]
    dec_y = [_pct(s, "decrease_difficulty") for s in segs]
    nop_y = [_pct(s, "no_change")           for s in segs]

    ax4.stackplot(xs, inc_y, dec_y, nop_y,
                  labels=["Increase", "Decrease", "No Change"],
                  colors=[ACC[0], ACC[2], ACC[1]], alpha=0.80)
    ax4.set_title("Action Selection Trend", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Action % per Window")
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, loc="upper right")
    ax4.grid(color=GRID, linewidth=0.4)

    # ── Panel 5: Last episode difficulty vs frustration trace ─────────────
    ax5 = _ax(2, 0, colspan=2)
    last_diffs = history["difficulty_traces"][-1] if history["difficulty_traces"] else []
    last_frus  = history["frus_traces"][-1]        if history["frus_traces"]  else []
    t_steps    = list(range(1, len(last_diffs) + 1))
    ax5l = ax5
    ax5r = ax5.twinx()
    ax5r.set_facecolor(PANEL)
    ax5r.spines[:].set_color(GRID)
    ax5r.tick_params(colors=TEXT, labelsize=8)

    if last_diffs:
        ax5l.plot(t_steps, last_diffs, color=ACC[0], linewidth=2.0, label="Difficulty")
    if last_frus:
        ax5r.plot(t_steps[:len(last_frus)], last_frus,
                  color=ACC[2], linewidth=1.5, linestyle="--", label="Frustration")
    ax5l.set_title("Last Episode: Difficulty vs Frustration Trace", fontsize=10, fontweight="bold")
    ax5l.set_xlabel("Step")
    ax5l.set_ylabel("Difficulty", color=ACC[0])
    ax5r.set_ylabel("Frustration", color=ACC[2])
    ax5l.tick_params(axis="y", labelcolor=ACC[0])
    ax5r.tick_params(axis="y", labelcolor=ACC[2])
    ax5l.grid(color=GRID, linewidth=0.4)
    lines1, labs1 = ax5l.get_legend_handles_labels()
    lines2, labs2 = ax5r.get_legend_handles_labels()
    ax5l.legend(lines1 + lines2, labs1 + labs2, fontsize=8, facecolor=PANEL, labelcolor=TEXT)

    # ── Panel 6: Q-table heatmap ─────────────────────────────────────────
    ax6 = _ax(2, 2)
    q   = history["final_q_table"]
    states  = [s.value for s in FrustrationState.all()]
    actions = [a.value for a in DifficultyAction.all()]
    data    = [[q[s][a] for a in actions] for s in states]

    mn_v = min(min(row) for row in data) - 0.01
    mx_v = max(max(row) for row in data) + 0.01

    def lerp_color(v, lo, hi):
        t = (v - lo) / (hi - lo + 1e-9)
        # Red -> Yellow -> Green
        if t < 0.5:
            r, g, b = 1.0, t * 2, 0.0
        else:
            r, g, b = 1.0 - (t - 0.5) * 2, 1.0, 0.0
        return r, g, b

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            color = lerp_color(val, mn_v, mx_v)
            rect  = plt.Rectangle([j, i], 1, 1, color=color)
            ax6.add_patch(rect)
            ax6.text(j + 0.5, i + 0.5, f"{val:.3f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if val < (mn_v + mx_v) / 2 else "black",
                     fontweight="bold")

    ax6.set_xlim(0, len(actions))
    ax6.set_ylim(0, len(states))
    ax6.set_xticks([j + 0.5 for j in range(len(actions))])
    ax6.set_xticklabels(["INC", "DEC", "NOP"], fontsize=8, color=TEXT)
    ax6.set_yticks([i + 0.5 for i in range(len(states))])
    ax6.set_yticklabels(["LOW", "MOD", "HIGH"], fontsize=8, color=TEXT)
    ax6.set_title("Final Q-Table Heatmap", fontsize=10, fontweight="bold")

    # Title
    pid_str = f" | Player: {player_id}" if player_id else ""
    fig.suptitle(
        f"Q-Learning Adaptive Difficulty — "
        f"{history['archetype'].upper()} Player{pid_str}  "
        f"({n_ep} episodes, {agent.step_count} steps)",
        fontsize=13, fontweight="bold", color=TEXT, y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK, edgecolor="none")
        print(f"  [visualize] Saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_personalized(
    histories: Dict[str, dict],
    agent:     QLearningAgent,
    save_path: Optional[str] = None,
) -> None:
    """
    Generate per-player comparison plots for personalized training.
    Shows Q-table heatmaps + learning curves side by side per player.

    Requires: matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[visualize_personalized] matplotlib not installed.")
        return

    n_players = len(histories)
    if n_players == 0:
        return

    DARK  = "#0d1117"
    PANEL = "#161b22"
    GRID  = "#30363d"
    TEXT  = "#e6edf3"
    ACC   = ["#58a6ff", "#3fb950", "#f78166"]

    fig, axes = plt.subplots(
        n_players, 3, figsize=(18, 5 * n_players),
        facecolor=DARK
    )
    if n_players == 1:
        axes = [axes]

    for row_idx, (pid, history) in enumerate(histories.items()):
        axrow = axes[row_idx]
        for ax in axrow:
            ax.set_facecolor(PANEL)
            ax.spines[:].set_color(GRID)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.title.set_color(TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)

        # Col 0: Episode reward curve
        ax0 = axrow[0]
        ep_r = history["episode_rewards"]
        n    = len(ep_r)
        w    = max(1, n // 20)
        smooth = [sum(ep_r[max(0,i-w):i+1])/len(ep_r[max(0,i-w):i+1]) for i in range(n)]
        ax0.plot(range(n), ep_r, color=ACC[row_idx % len(ACC)], alpha=0.25, linewidth=0.6)
        ax0.plot(range(n), smooth, color=ACC[row_idx % len(ACC)], linewidth=2.0)
        ax0.axhline(0, color=GRID, linewidth=0.7, linestyle="--")
        ax0.set_title(f"{pid} — Episode Reward", fontsize=9, fontweight="bold")
        ax0.set_xlabel("Episode")
        ax0.set_ylabel("Reward")
        ax0.grid(color=GRID, linewidth=0.4)

        # Col 1: Average frustration per episode
        ax1 = axrow[1]
        frus = history["episode_avg_frus"]
        ax1.plot(range(len(frus)), frus, color="#ffa657", linewidth=1.8)
        ax1.axhline(0.33, color=GRID, linewidth=0.6, linestyle=":")
        ax1.axhline(0.67, color=GRID, linewidth=0.6, linestyle=":")
        ax1.fill_between(range(len(frus)), 0.33, 0.67, alpha=0.1, color="#3fb950",
                         label="Moderate zone")
        ax1.set_title(f"{pid} — Avg Frustration", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Frustration")
        ax1.set_ylim(0, 1)
        ax1.grid(color=GRID, linewidth=0.4)

        # Col 2: Q-table heatmap
        ax2  = axrow[2]
        q    = agent.get_profile(pid).q_table
        states  = [s.value for s in FrustrationState.all()]
        actions = [a.value for a in DifficultyAction.all()]
        data    = [[q[s][a] for a in actions] for s in states]
        mn_v = min(min(r) for r in data) - 0.01
        mx_v = max(max(r) for r in data) + 0.01

        import numpy as np
        mat = [[data[i][j] for j in range(len(actions))] for i in range(len(states))]
        im  = ax2.imshow(mat, cmap="RdYlGn", vmin=mn_v, vmax=mx_v, aspect="auto")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        for i in range(len(states)):
            for j in range(len(actions)):
                ax2.text(j, i, f"{data[i][j]:.3f}",
                         ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax2.set_xticks(range(len(actions)))
        ax2.set_xticklabels(["INC", "DEC", "NOP"], fontsize=8, color=TEXT)
        ax2.set_yticks(range(len(states)))
        ax2.set_yticklabels(["LOW", "MOD", "HIGH"], fontsize=8, color=TEXT)
        ax2.set_title(f"{pid} — Q-Table ({history['archetype'].upper()})", fontsize=9, fontweight="bold")

        # Profile annotation
        profile = history.get("player_profile", {})
        if profile:
            ax2.set_xlabel(
                f"sens={profile.get('frustration_sensitivity', '-'):.2f}  "
                f"pref_diff={profile.get('preferred_difficulty', '-'):.1f}  "
                f"skill={profile.get('skill_level', '-'):.2f}",
                fontsize=7, color="#8b949e",
            )

    fig.suptitle("Personalized Q-Learning — Per-Player Comparison",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK, edgecolor="none")
        print(f"  [visualize_personalized] Saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)
