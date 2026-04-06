"""
pipeline/logger.py
-------------------
Structured logging for the Game Pipeline.

Provides two output streams:
  1. Console log  — colour-coded, compact per-step summary
  2. File log     — JSON-Lines (.jsonl) for offline analysis

Each log entry is a full pipeline result dict (JSON-serialisable).

Log levels:
  DEBUG   - every field including full Q-table
  INFO    - sentiment, frustration, action, difficulty, reward  (default)
  MINIMAL - action + difficulty only

Usage:
    from pipeline.logger import PipelineLogger

    logger = PipelineLogger(log_dir="logs/", level="INFO")
    logger.log(result_dict)
    logger.session_summary()
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Dict, List, Optional


# =============================================================================
# ANSI colour codes (degrades gracefully on Windows without colour support)
# =============================================================================

_USE_COLOUR = os.name != "nt" or "WT_SESSION" in os.environ  # Windows Terminal OK

def _c(text: str, code: str) -> str:
    if not _USE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"

RED    = "31"
YELLOW = "33"
GREEN  = "32"
CYAN   = "36"
BOLD   = "1"
DIM    = "2"


# =============================================================================
# Session Tracker (in-memory analytics)
# =============================================================================

class SessionTracker:
    """Accumulates per-step metrics for session-level reporting."""

    def __init__(self):
        self.steps:              List[int]   = []
        self.rewards:            List[float] = []
        self.frustration_scores: List[float] = []
        self.frustration_states: List[str]   = []
        self.actions:            List[str]   = []
        self.difficulties:       List[float] = []
        self.sentiment_scores:   List[float] = []
        self.emotion_labels:     List[str]   = []
        self.sarcasm_count:      int         = 0
        self.start_time:         str         = datetime.datetime.now().isoformat()

    def record(self, result: Dict):
        self.steps.append(result["step"])
        self.rewards.append(result.get("reward", 0.0))
        self.frustration_scores.append(result.get("frustration_score", 0.0))
        self.frustration_states.append(result.get("frustration_state", ""))
        self.actions.append(result.get("action", ""))
        self.difficulties.append(result.get("new_difficulty", 5.0))
        self.sentiment_scores.append(result.get("sentiment_score", 0.0))
        self.emotion_labels.append(result.get("emotion_label", "neutral"))
        if result.get("sarcasm_detected"):
            self.sarcasm_count += 1

    def summary(self) -> Dict:
        n = max(len(self.steps), 1)
        action_counts = {
            "increase_difficulty": self.actions.count("increase_difficulty"),
            "decrease_difficulty": self.actions.count("decrease_difficulty"),
            "no_change":           self.actions.count("no_change"),
        }
        emotion_counts: Dict[str, int] = {}
        for e in self.emotion_labels:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        state_counts: Dict[str, int] = {}
        for s in self.frustration_states:
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "session_start":         self.start_time,
            "total_steps":           n,
            "avg_reward":            round(sum(self.rewards) / n, 5),
            "total_reward":          round(sum(self.rewards), 4),
            "avg_frustration":       round(sum(self.frustration_scores) / n, 4),
            "avg_sentiment":         round(sum(self.sentiment_scores) / n, 4),
            "final_difficulty":      round(self.difficulties[-1], 3) if self.difficulties else 5.0,
            "difficulty_range":      round(max(self.difficulties, default=5) - min(self.difficulties, default=5), 3),
            "sarcasm_events":        self.sarcasm_count,
            "action_counts":         action_counts,
            "emotion_distribution":  emotion_counts,
            "frustration_state_counts": state_counts,
        }


# =============================================================================
# Pipeline Logger
# =============================================================================

class PipelineLogger:
    """
    Structured step-by-step logger for the GamePipeline.

    Args:
        log_dir:  Directory to write .jsonl log files.
                  None = console only.
        level:    "DEBUG" | "INFO" | "MINIMAL"
        prefix:   Prefix for log filenames (e.g. player_id).
    """

    LEVELS = {"MINIMAL": 0, "INFO": 1, "DEBUG": 2}

    def __init__(
        self,
        log_dir: Optional[str] = None,
        level:   str = "INFO",
        prefix:  str = "session",
    ):
        self.level   = level.upper()
        self._lv     = self.LEVELS.get(self.level, 1)
        self._tracker = SessionTracker()

        # File output
        self._file_handle = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(log_dir, f"{prefix}_{ts}.jsonl")
            self._file_handle = open(filename, "w", encoding="utf-8")
            print(f"[Logger] Writing to: {filename}")

    # ------------------------------------------------------------------
    # Main log call
    # ------------------------------------------------------------------

    def log(self, result: Dict) -> None:
        """Log one pipeline step result."""
        self._tracker.record(result)
        self._console_log(result)
        self._file_log(result)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _console_log(self, result: Dict) -> None:
        step = result["step"]
        ts   = result["timestamp"].split("T")[1][:12]   # HH:MM:SS.mmm

        # --- Sentiment colour ---
        s   = result.get("sentiment_score", 0.0)
        if s > 0.15:
            s_str = _c(f"{s:+.3f}", GREEN)
        elif s < -0.15:
            s_str = _c(f"{s:+.3f}", RED)
        else:
            s_str = _c(f"{s:+.3f}", YELLOW)

        # --- Frustration colour ---
        fs = result.get("frustration_score", 0.0)
        if fs >= 0.67:
            f_str = _c(f"{fs:.3f}", RED)
        elif fs >= 0.33:
            f_str = _c(f"{fs:.3f}", YELLOW)
        else:
            f_str = _c(f"{fs:.3f}", GREEN)

        # --- Action colour ---
        action = result.get("action", "")
        if "increase" in action:
            a_str = _c("[^] INC", CYAN)
        elif "decrease" in action:
            a_str = _c("[v] DEC", RED)
        else:
            a_str = _c("[=] NOP", DIM)

        # --- Sarcasm marker ---
        sarc = " [SARCASM]" if result.get("sarcasm_detected") else ""

        msg_preview = result.get("message", "")[:48]

        if self._lv >= 1:    # INFO
            print(
                f"[{ts}] #{step:>3}  "
                f"sent={s_str}  "
                f"em={result.get('emotion_label','?'):<12}  "
                f"frus={f_str} ({result.get('frustration_state','?').split('_')[0].upper():<4})  "
                f"{a_str}  "
                f"diff={result.get('prev_difficulty',5):.1f}->"
                f"{result.get('new_difficulty',5):.1f}  "
                f"rew={result.get('reward',0):+.3f}"
                f"{sarc}"
            )
            if self._lv >= 2:  # DEBUG
                print(f"         MSG: {msg_preview!r}")
                q = result.get("q_table", {})
                q_strs = []
                for s_val, a_dict in q.items():
                    s_sh = s_val.split("_")[0].upper()[:3]
                    vals = ",".join(f"{v:+.3f}" for v in a_dict.values())
                    q_strs.append(f"{s_sh}:[{vals}]")
                print(f"         Q  : {' | '.join(q_strs)}")
        else:    # MINIMAL
            print(
                f"#{step:>3}  {a_str}  "
                f"{result.get('prev_difficulty',5):.1f}->"
                f"{result.get('new_difficulty',5):.1f}"
            )

    # ------------------------------------------------------------------
    # File output (JSON-Lines)
    # ------------------------------------------------------------------

    def _file_log(self, result: Dict) -> None:
        if self._file_handle is None:
            return
        try:
            # Exclude q_table from file log unless DEBUG (can be large)
            if self._lv < 2:
                log_entry = {k: v for k, v in result.items()
                             if k not in ("q_table",)}
            else:
                log_entry = result
            self._file_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self._file_handle.flush()
        except Exception as e:
            print(f"[Logger] File write error: {e}")

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def session_summary(self, print_output: bool = True) -> Dict:
        """Return and optionally print session analytics."""
        summary = self._tracker.summary()
        if print_output:
            sep = "=" * 64
            print(f"\n{sep}")
            print(f"  SESSION SUMMARY")
            print(sep)
            print(f"  Total steps       : {summary['total_steps']}")
            print(f"  Avg reward        : {summary['avg_reward']:+.4f}")
            print(f"  Total reward      : {summary['total_reward']:+.4f}")
            print(f"  Avg frustration   : {summary['avg_frustration']:.4f}")
            print(f"  Avg sentiment     : {summary['avg_sentiment']:+.4f}")
            print(f"  Final difficulty  : {summary['final_difficulty']:.1f}")
            print(f"  Difficulty range  : {summary['difficulty_range']:.2f}")
            print(f"  Sarcasm events    : {summary['sarcasm_events']}")
            print(f"  Action breakdown  :")
            for action, cnt in summary["action_counts"].items():
                pct = cnt / max(summary["total_steps"], 1) * 100
                bar = "#" * int(pct / 3)
                print(f"    {action:<28} {bar:<33} {cnt:>3}  ({pct:.0f}%)")
            print(f"  Emotion breakdown :")
            for emo, cnt in sorted(summary["emotion_distribution"].items(),
                                    key=lambda x: -x[1]):
                print(f"    {emo:<16} {cnt}")
            print(sep)
        return summary

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()
