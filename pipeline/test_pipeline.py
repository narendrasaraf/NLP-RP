"""
pipeline/test_pipeline.py
--------------------------
Complete integration test for the NLP + RL Adaptive Game System.

Runs the 7 specified test messages through the full pipeline:

  1. "This level is impossible"
  2. "I keep losing again and again"
  3. "Okay this is getting better"
  4. "Nice, I finally won"
  5. "Game is fun but still hard"
  6. "Bro this is so annoying"
  7. "Alright I think I got it now"

For each message, prints:
  - Chat message
  - Preprocessing result (cleaned text, slang)
  - Sentiment score + label
  - Emotion label + breakdown
  - Raw frustration score
  - EMA-smoothed frustration (temporal modeling, window=5)
  - Moving average frustration
  - Frustration level (Low / Moderate / High)
  - Q-learning action (Increase / Decrease / No Change) with symbol
  - Game difficulty change
  - Reward signal + td-error

After all messages:
  - Frustration evolution table (raw vs EMA over time)
  - Action timeline with difficulty change highlights
  - ASCII frustration chart showing temporal smoothing
  - Final Q-table
  - Optimal policy per state
  - Session summary

Run:
    python pipeline/test_pipeline.py
    python pipeline/test_pipeline.py --bert    # use BERT (downloads ~400MB)
    python pipeline/test_pipeline.py --json    # JSON output
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
from rl.q_learning import FrustrationState, DifficultyAction


# ============================================================================
# Test messages (exact as specified)
# ============================================================================

TEST_MESSAGES = [
    "This level is impossible",
    "I keep losing again and again",
    "Okay this is getting better",
    "Nice, I finally won",
    "Game is fun but still hard",
    "Bro this is so annoying",
    "Alright I think I got it now",
]


# ============================================================================
# Display helpers
# ============================================================================

SEP  = "=" * 78
MID  = "-" * 78
THIN = "." * 78

def _bar(val: float, width: int = 24, lo: float = 0.0, hi: float = 1.0,
         fill: str = "#", empty: str = ".") -> str:
    """ASCII progress bar."""
    pct    = max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))
    filled = int(pct * width)
    return "[" + fill * filled + empty * (width - filled) + f"] {val:.3f}"

def _sentiment_bar(score: float, width: int = 22) -> str:
    """Bipolar ASCII bar: negative left / positive right, zero at center."""
    half = width // 2
    if score >= 0:
        pos = int(score * half)
        return "-" * half + "#" * pos + "." * (half - pos) + f"  {score:+.3f}"
    else:
        neg = int(-score * half)
        return "." * (half - neg) + "#" * neg + "-" * half + f"  {score:+.3f}"

def _level_badge(state: str) -> str:
    """Return padded frustration level label."""
    label = state.replace("_frustration", "").upper()
    badges = {"LOW": "[ LOW  ]", "MODERATE": "[MODERATE]", "HIGH": "[ HIGH ]"}
    return badges.get(label, f"[{label}]")

def _action_symbol(action: str) -> str:
    if "increase" in action: return "[^] INCREASE"
    if "decrease" in action: return "[v] DECREASE  <-- DIFFICULTY REDUCED"
    return "[=] NO CHANGE"

def _difficulty_bar(val: float, width: int = 20) -> str:
    filled = int(val / 10.0 * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {val:.1f}/10"


# ============================================================================
# Per-step detailed print
# ============================================================================

def print_step(result: dict, idx: int) -> None:
    """Print full structured output for one pipeline step."""
    print(f"\n{SEP}")
    print(f"  STEP {idx}  |  Message {idx} of {len(TEST_MESSAGES)}")
    print(SEP)

    # --- Message ---
    msg = result["message"]
    cln = result["cleaned_message"]
    print(f"  Chat Message   : {msg!r}")
    if cln and cln != msg.lower().strip():
        print(f"  Cleaned        : {cln!r}")
    slang = result.get("slang_replacements", {})
    if slang:
        slang_preview = {k: v for k, v in list(slang.items())[:5]}
        print(f"  Slang Fixes    : {slang_preview}")
    norms = result.get("normalizations", [])
    if norms:
        print(f"  Normalizations : {norms}")

    print(MID)

    # --- Sentiment ---
    s  = result["sentiment_score"]
    sl = result["sentiment_label"].upper()
    print(f"  Sentiment Score : {_sentiment_bar(s)}")
    print(f"  Sentiment Label : {sl}")

    # --- Emotion ---
    emo_label  = result["emotion_label"].upper()
    emo_scores = result["emotion_scores"]
    print(f"\n  Emotion Label   : {emo_label}")
    print(f"  Emotion Breakdown:")
    for emo, val in emo_scores.items():
        marker = " <--" if emo == result["emotion_label"] else ""
        print(f"    {emo:<14} {_bar(val, width=20)}{marker}")

    print(MID)

    # --- Frustration ---
    raw_f  = result["frustration_score"]
    ema_f  = result["smoothed_frustration"]
    ma_f   = result["moving_avg_frustration"]
    state  = result["frustration_state"]
    window = result.get("frustration_window", [])

    print(f"  Frustration Score (raw)     : {_bar(raw_f)}")
    print(f"  Smoothed Frustration (EMA)  : {_bar(ema_f)}")
    print(f"  Moving Average  (window={len(window)})  : {_bar(ma_f)}")
    print(f"  Temporal Window             : "
          + "  ".join(f"{v:.3f}" for v in window))
    print(f"\n  Frustration Level : {_level_badge(state)}")

    if result.get("sarcasm_detected"):
        print(f"  Sarcasm Detected  : YES (score={result['sarcasm_score']:.2f})  "
              f"-- sentiment adjusted")

    print(MID)

    # --- RL Action ---
    action = result["action"]
    a_str  = _action_symbol(action)
    print(f"  Q-Learning Action : {a_str}")
    print(f"  Difficulty        : {result['prev_difficulty']:.1f}  ->"
          f"  {result['new_difficulty']:.1f}  "
          f"(delta={result['difficulty_delta']:+.1f})")
    print(f"  Difficulty Bar    : {_difficulty_bar(result['new_difficulty'])}")
    print(f"  Reward            : {result['reward']:+.4f}")
    print(f"  TD-Error          : {result['td_error']:.5f}")
    print(f"  Epsilon           : {result['epsilon']:.4f}")

    print(MID)

    # --- Q-Table ---
    print("  Q-TABLE (current state):")
    q = result["q_table"]
    print(f"  {'State':<12} {'INC':>12} {'DEC':>12} {'NOP':>12} {'Best':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for s_val, a_dict in q.items():
        s_short = s_val.replace("_frustration", "").upper()
        best    = max(a_dict, key=a_dict.get)
        print(f"  {s_short:<12} "
              f"{a_dict['increase_difficulty']:>+12.4f} "
              f"{a_dict['decrease_difficulty']:>+12.4f} "
              f"{a_dict['no_change']:>+12.4f} "
              f"{DifficultyAction(best).short:>8}")

    print(f"\n  Optimal Policy:")
    for s_val, a_val in result["optimal_policy"].items():
        s_short = s_val.replace("_frustration", "").upper()
        print(f"    {s_short:<10} ->  {a_val}")

    print(SEP)


# ============================================================================
# Summary tables
# ============================================================================

def print_evolution_table(results: list) -> None:
    """Print frustration evolution over time."""
    print(f"\n{SEP}")
    print("  FRUSTRATION EVOLUTION OVER TIME")
    print(SEP)
    print(f"  {'#':<3} {'Message':<38} {'Raw':>7} {'EMA':>7} {'MA':>7} {'Level':<12} {'Action':<15} {'Diff':>6}")
    print(f"  {'-'*3} {'-'*38} {'-'*7} {'-'*7} {'-'*7} {'-'*12} {'-'*15} {'-'*6}")

    for i, r in enumerate(results, 1):
        msg     = r["message"][:36]
        raw     = r["frustration_score"]
        ema     = r["smoothed_frustration"]
        ma      = r["moving_avg_frustration"]
        level   = r["frustration_state"].replace("_frustration", "").upper()[:8]
        action  = r["action"].replace("_difficulty", "").replace("no_change", "no change").upper()[:12]
        diff    = r["new_difficulty"]
        reduced = " [v]" if r["difficulty_delta"] < 0 else ("  [^]" if r["difficulty_delta"] > 0 else "  [=]")
        print(f"  {i:<3} {msg:<38} {raw:>7.3f} {ema:>7.3f} {ma:>7.3f} {level:<12} {action:<15} {diff:>5.1f}{reduced}")

    print(SEP)


def print_ascii_frustration_chart(results: list) -> None:
    """Print two-row ASCII chart: raw frustration vs EMA."""
    print(f"\n{SEP}")
    print("  FRUSTRATION CHART  (raw vs EMA smoothed)")
    print(f"  Steps: {len(results)}    Window: 5 messages    EMA alpha: 0.35")
    print(SEP)

    width = 50

    def _sparkrow(values: list, lo: float = 0.0, hi: float = 1.0, fill: str = "#") -> str:
        chars = " ._-=+#@"
        rng   = max(hi - lo, 1e-9)
        line  = ""
        col_w = max(1, width // len(values))
        for v in values:
            idx   = int(max(0.0, min(1.0, (v - lo) / rng)) * (len(chars) - 1))
            block = chars[idx] * col_w
            label = f"{v:.2f}"
            # Centre label inside block
            if col_w >= 6:
                block = label.center(col_w)
            line += block
        return line

    raw_vals = [r["frustration_score"]     for r in results]
    ema_vals = [r["smoothed_frustration"]  for r in results]
    ma_vals  = [r["moving_avg_frustration"] for r in results]

    # Draw horizontal scale lines
    hi_line = "  1.0 |"
    mod_up  = "  0.67|"
    mod_lo  = "  0.33|"
    lo_line = "  0.0 |"

    def _plot_row(values: list, fill: str = "#") -> str:
        rows = [""] * 10
        col_w = max(2, width // max(len(values), 1))
        for v in values:
            bucket = int(v * 9)
            for r_idx in range(10):
                if r_idx <= bucket:
                    rows[9 - r_idx] += fill * col_w
                else:
                    rows[9 - r_idx] += " " * col_w
        return rows

    print(f"\n  RAW frustration (per message):")
    raw_str = "  ".join(f"{v:.3f}" for v in raw_vals)
    raw_bar = "  ".join(_bar(v, width=8, fill="#", empty=".") for v in raw_vals)

    # Compact inline bar display
    for v in raw_vals:
        filled = int(v * 30)
        print(f"    {_bar(v, width=30, fill='#', empty='.')} ", end="")
    print()

    print(f"\n  EMA smoothed (alpha=0.35, window visible):")
    for v in ema_vals:
        print(f"    {_bar(v, width=30, fill='*', empty='.')} ", end="")
    print()

    print(f"\n  Moving Average (window=5):")
    for v in ma_vals:
        print(f"    {_bar(v, width=30, fill='+', empty='.')} ", end="")
    print()

    # Horizontal table with step labels
    print(f"\n  {'Step':<8}", end="")
    for i in range(1, len(results)+1):
        print(f" {i:>6}", end="")
    print()

    print(f"  {'Raw':<8}", end="")
    for v in raw_vals:
        print(f" {v:>6.3f}", end="")
    print()

    print(f"  {'EMA':<8}", end="")
    for v in ema_vals:
        print(f" {v:>6.3f}", end="")
    print()

    print(f"  {'MA':<8}", end="")
    for v in ma_vals:
        print(f" {v:>6.3f}", end="")
    print()

    print(f"\n  Level thresholds:  LOW < 0.33 <= MODERATE < 0.67 <= HIGH")

    # Show frustration state per step
    print(f"\n  {'State':<8}", end="")
    for r in results:
        s = r["frustration_state"].replace("_frustration","").upper()[:3]
        print(f" {s:>6}", end="")
    print()

    # Show action per step
    print(f"  {'Action':<8}", end="")
    for r in results:
        a = r["action_short"]
        print(f" {a:>6}", end="")
    print()

    print(SEP)


def print_action_timeline(results: list) -> None:
    """Print action timeline highlighting difficulty reduction events."""
    print(f"\n{SEP}")
    print("  ACTION TIMELINE  -- Difficulty Evolution")
    print(SEP)
    print(f"  {'Step':<5} {'Action':<12} {'Symbol':<8} {'Prev':>6} {'New':>6} {'Delta':>7} {'Note'}")
    print(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*30}")

    for i, r in enumerate(results, 1):
        action  = r["action"].replace("_difficulty", "").replace("no_change", "no_change")
        sym     = {"increase_difficulty": "[^]", "decrease_difficulty": "[v]",
                   "no_change": "[=]"}.get(r["action"], "[ ]")
        delta   = r["difficulty_delta"]
        note    = ""
        if delta < 0:
            note = "** DIFFICULTY REDUCED  (high frustration)"
        elif delta > 0:
            note = "   Difficulty increased (low frustration)"
        msg_short = r["message"][:28]
        print(f"  {i:<5} {action:<12} {sym:<8} {r['prev_difficulty']:>6.1f} "
              f"{r['new_difficulty']:>6.1f} {delta:>+7.1f}  {note}")

    # Sum up
    n_dec = sum(1 for r in results if r["difficulty_delta"] < 0)
    n_inc = sum(1 for r in results if r["difficulty_delta"] > 0)
    n_nop = sum(1 for r in results if r["difficulty_delta"] == 0)
    print(MID)
    print(f"  Decreases (difficulty reduced)  : {n_dec}")
    print(f"  Increases (difficulty raised)   : {n_inc}")
    print(f"  No change (maintained)          : {n_nop}")
    diffs = [r["new_difficulty"] for r in results]
    print(f"  Difficulty range                : {min(diffs):.1f} -- {max(diffs):.1f}")
    print(SEP)


def print_final_q_table(result: dict) -> None:
    """Print the final Q-table and optimal policy."""
    print(f"\n{SEP}")
    print("  FINAL Q-TABLE  (after all 7 messages)")
    print(SEP)
    q = result["q_table"]
    print(f"  {'State':<12} {'INC':>14} {'DEC':>14} {'NOP':>14} {'Best Action'}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*20}")
    for s_val, a_dict in q.items():
        s_short = s_val.replace("_frustration", "").upper()
        best    = max(a_dict, key=a_dict.get)
        best_str = DifficultyAction(best).value
        print(f"  {s_short:<12} "
              f"{a_dict['increase_difficulty']:>+14.5f} "
              f"{a_dict['decrease_difficulty']:>+14.5f} "
              f"{a_dict['no_change']:>+14.5f}  "
              f"{best_str}")

    print(f"\n  OPTIMAL POLICY:")
    print(f"  {'Frustration State':<22} {'Best Action'}")
    print(f"  {'-'*22} {'-'*30}")
    for s_val, a_val in result["optimal_policy"].items():
        s_short = s_val.replace("_frustration", "").upper()
        sym     = {"increase_difficulty": "[^]", "decrease_difficulty": "[v]",
                   "no_change": "[=]"}.get(a_val, "[ ]")
        note    = ("(reduce difficulty)" if "decrease" in a_val
                   else "(raise difficulty)" if "increase" in a_val
                   else "(keep stable)")
        print(f"  {s_short:<22} {sym} {a_val}  {note}")
    print(SEP)


def print_json_output(results: list) -> None:
    """Print compact JSON for all results."""
    output = []
    for r in results:
        output.append({
            "step":                 r["step"],
            "message":              r["message"],
            "sentiment_score":      r["sentiment_score"],
            "sentiment_label":      r["sentiment_label"],
            "emotion_label":        r["emotion_label"],
            "emotion_scores":       r["emotion_scores"],
            "frustration_score":    r["frustration_score"],
            "smoothed_frustration": r["smoothed_frustration"],
            "moving_avg":           r["moving_avg_frustration"],
            "frustration_state":    r["frustration_state"],
            "action":               r["action"],
            "prev_difficulty":      r["prev_difficulty"],
            "new_difficulty":       r["new_difficulty"],
            "difficulty_delta":     r["difficulty_delta"],
            "reward":               r["reward"],
            "q_table":              r["q_table"],
            "optimal_policy":       r["optimal_policy"],
        })
    print(json.dumps(output, indent=2))


# ============================================================================
# Main test runner
# ============================================================================

def main(use_bert: bool = False, json_out: bool = False) -> None:

    if not json_out:
        print(f"\n{SEP}")
        print("  NLP + RL ADAPTIVE GAME SYSTEM — INTEGRATION TEST")
        print(f"  {len(TEST_MESSAGES)} test messages  |  "
              f"NLP={'BERT' if use_bert else 'Rule-Based'}  |  "
              f"RL=Q-Learning  |  Temporal=EMA(alpha=0.35, window=5)")
        print(SEP)

    # Initialise pipeline
    pipe = GamePipeline(
        use_bert      = use_bert,
        epsilon_start = 0.20,   # mostly greedy — 20% explore
        epsilon_end   = 0.05,
        epsilon_decay = 0.98,
        initial_difficulty = 5.0,
        difficulty_step    = 1.0,   # clear 1-point steps for readability
        seed          = 42,
    )

    # Process all messages
    results = []
    for i, msg in enumerate(TEST_MESSAGES, 1):
        result = pipe.process(msg)
        results.append(result)

        if json_out:
            pass   # collect then dump together
        else:
            print_step(result, i)

    # JSON output
    if json_out:
        print_json_output(results)
        return

    # --- Summary outputs ---
    print_evolution_table(results)
    print_ascii_frustration_chart(results)
    print_action_timeline(results)
    print_final_q_table(results[-1])
    pipe._logger.session_summary()

    print(f"\n{SEP}")
    print("  TEST COMPLETE")
    print(f"  All {len(TEST_MESSAGES)} messages processed successfully.")
    print(f"  Pipeline NLP model : {results[-1]['model_used']}")
    print(f"  Final difficulty   : {results[-1]['new_difficulty']:.1f}/10")
    print(f"  Total RL steps     : {pipe.step_count}")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP + RL Game Pipeline Test")
    parser.add_argument("--bert", action="store_true",
                        help="Use BERT (downloads ~400MB on first run)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of formatted text")
    args = parser.parse_args()
    main(use_bert=args.bert, json_out=args.json)
