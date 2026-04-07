"""
pipeline/demo.py
-----------------
End-to-end demo for the NLP + RL Game Pipeline.

Demonstrates:
  DEMO 1 — Single-player session (10 gaming chat messages)
            Shows full formatted output per message + session summary
  DEMO 2 — Multi-player session (3 players with personalised Q-tables)
            Each player sends the same messages, observe how the system
            learns different policies for different players
  DEMO 3 — Real-time simulation loop
            Shows how difficulty converges over 20 messages for a
            frustrated player vs a happy player

Run:
    python -m pipeline.demo
    python -m pipeline.demo --demo 1
    python -m pipeline.demo --demo 2
    python -m pipeline.demo --demo 3
    python -m pipeline.demo --bert      # use BERT (downloads ~400MB first run)
    python -m pipeline.demo --json      # JSON output for demo 1
    python -m pipeline.demo --log logs/ # save JSONL log files
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.game_pipeline import GamePipeline
from rl.q_learning          import FrustrationState, DifficultyAction


# ---------------------------------------------------------------------------
# Test message suites
# ---------------------------------------------------------------------------

FRUSTRATED_MSGS = [
    "omg this noob is SOOOO bad wtf!!",
    "I QUIT this trash rigged game af!!!",
    "noooo why does this keep happening fml",
    "lag again are you kidding me smh",
    "yeah right so skilled lol get rekt",
    "IMPOSSIBLE this game is BROKEN",
    "I can't even this is tilted af",
    "feeder every single game wtf",
    "ugh stuck again same spot",
    "why am I even playing this garbage",
]

MIXED_MSGS = [
    "gg nice clutch everyone!",
    "omg this noob is SOOOO bad wtf!!",
    "not bad, almost had it",
    "why does this always happen fml",
    "oh great another lag spike smh",
    "yeah right so skilled lol",
    "clutch play nice one!",
    "this is IMPOSSIBLE ugh",
    "gg wp that was fun",
    "I QUIT this broken game!!",
]

POSITIVE_MSGS = [
    "gg wp amazing game everyone!",
    "clutch play nice one pro move!",
    "that was a fun round let's go!",
    "finally figured it out lol skill issue fixed",
    "gg good game team carried",
    "omg that clutch was INSANE",
    "love this game so much",
    "yes finally got them!",
    "great round everyone glhf",
    "nailed it perfect strategy",
]


# ---------------------------------------------------------------------------
# Demo 1 — Single player, full formatted output
# ---------------------------------------------------------------------------

def run_demo1(use_bert: bool = False, json_out: bool = False,
              log_dir: Optional[str] = None) -> None:
    print("\n" + "=" * 64)
    print("  DEMO 1: Single-Player NLP + RL Pipeline")
    print("=" * 64)

    pipe = GamePipeline(
        use_bert     = use_bert,
        epsilon_start= 0.30,
        log_dir      = log_dir,
        seed         = 42,
    )

    messages = MIXED_MSGS

    for i, msg in enumerate(messages):
        result = pipe.process(msg, player_id="player_001")

        if json_out:
            # JSON output (compact)
            compact = {
                "step":              result["step"],
                "message":           result["message"],
                "sentiment_score":   result["sentiment_score"],
                "sentiment_label":   result["sentiment_label"],
                "emotion_label":     result["emotion_label"],
                "frustration_score": result["frustration_score"],
                "frustration_state": result["frustration_state"],
                "action":            result["action"],
                "new_difficulty":    result["new_difficulty"],
                "reward":            result["reward"],
                "q_table":           result["q_table"],
                "optimal_policy":    result["optimal_policy"],
            }
            print(json.dumps(compact, indent=2))
        else:
            pipe.print_result(result, compact=(i > 0))

    # Session summary
    pipe._logger.session_summary()
    pipe._agent.print_q_table(title="After 10 messages")
    pipe._agent.print_optimal_policy()


# ---------------------------------------------------------------------------
# Demo 2 — Multi-player personalised
# ---------------------------------------------------------------------------

def run_demo2(use_bert: bool = False, log_dir: Optional[str] = None) -> None:
    print("\n" + "=" * 64)
    print("  DEMO 2: Personalised Multi-Player Pipeline")
    print("=" * 64)

    pipe = GamePipeline(
        use_bert     = use_bert,
        personalized = True,
        epsilon_start= 0.20,
        log_dir      = log_dir,
        seed         = 0,
    )

    players = {
        "frustrated_player": FRUSTRATED_MSGS,
        "mixed_player":      MIXED_MSGS,
        "positive_player":   POSITIVE_MSGS,
    }

    for player_id, msgs in players.items():
        print(f"\n  --- {player_id} ---")
        # Reset difficulty and NLP context between players
        pipe.reset()
        pipe.set_difficulty(5.0)

        for msg in msgs:
            pipe.process(msg, player_id=player_id)

        print(f"\n  Q-table for {player_id}:")
        pipe._agent.print_q_table(player_id=player_id)
        pipe._agent.print_optimal_policy(player_id=player_id)

        profile = pipe._agent.get_profile(player_id)
        print(f"  Profile: sensitivity={profile.frustration_sensitivity:.2f}  "
              f"pref_diff={profile.preferred_difficulty:.1f}  "
              f"skill={profile.skill_level:.3f}")

    pipe._logger.session_summary()


# ---------------------------------------------------------------------------
# Demo 3 — Real-time convergence simulation
# ---------------------------------------------------------------------------

def run_demo3(use_bert: bool = False) -> None:
    print("\n" + "=" * 64)
    print("  DEMO 3: Real-Time Difficulty Convergence Simulation")
    print("=" * 64)

    # Two pipelines with same initial state — different player chat patterns
    frustrated_pipe = GamePipeline(
        use_bert=use_bert, epsilon_start=0.20, seed=1,
    )
    positive_pipe = GamePipeline(
        use_bert=use_bert, epsilon_start=0.20, seed=2,
    )

    print("\n  Simulating 10 messages per player type...")
    print(f"  {'Step':<5}  {'Frustrated Diff':>16}  {'Frus State':<18}  "
          f"{'Positive Diff':>14}  {'Pos State':<18}")
    print("  " + "-" * 80)

    for i in range(10):
        f_msg = FRUSTRATED_MSGS[i]
        p_msg = POSITIVE_MSGS[i]

        f_result = frustrated_pipe.process(f_msg, player_id="frus")
        p_result = positive_pipe.process(p_msg, player_id="pos")

        f_diff  = f_result["new_difficulty"]
        f_state = f_result["frustration_state"].split("_")[0].upper()
        f_action= f_result["action_short"]

        p_diff  = p_result["new_difficulty"]
        p_state = p_result["frustration_state"].split("_")[0].upper()
        p_action= p_result["action_short"]

        print(f"  {i+1:<5}  "
              f"{f_diff:>6.1f} [{f_action}] ({f_state}){'':<8}  "
              f"{p_diff:>6.1f} [{p_action}] ({p_state})")

    print()
    print("  Expected: Frustrated player -> difficulty DECREASES over time")
    print("            Positive player   -> difficulty INCREASES over time")

    print("\n  Final Q-Tables:")
    print("\n  [Frustrated Player]")
    frustrated_pipe._agent.print_q_table(title="Frustrated")
    frustrated_pipe._agent.print_optimal_policy()

    print("\n  [Positive Player]")
    positive_pipe._agent.print_q_table(title="Positive")
    positive_pipe._agent.print_optimal_policy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

try:
    from typing import Optional
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="NLP + RL Game Pipeline Demo"
    )
    parser.add_argument("--demo",  type=int, default=0,
                        help="Demo to run: 1=single, 2=multi, 3=convergence. 0=all")
    parser.add_argument("--bert",  action="store_true",
                        help="Use BERT model (downloads ~400MB on first run)")
    parser.add_argument("--json",  action="store_true",
                        help="Output JSON for demo 1")
    parser.add_argument("--log",   type=str, default=None,
                        help="Directory for JSONL log files (e.g. logs/)")
    args = parser.parse_args()

    if args.demo in (0, 1):
        run_demo1(use_bert=args.bert, json_out=args.json, log_dir=args.log)
    if args.demo in (0, 2):
        run_demo2(use_bert=args.bert, log_dir=args.log)
    if args.demo in (0, 3):
        run_demo3(use_bert=args.bert)

    print("\n  [pipeline.demo] Done.")


if __name__ == "__main__":
    main()
