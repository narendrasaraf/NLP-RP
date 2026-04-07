#!/usr/bin/env python
"""
nlp/demo_inference.py
----------------------
End-to-end demo for the full Sentiment + Emotion pipeline.

Demonstrates all three modes:
  MODE 1  — Rule-based only (zero deps, instant)
  MODE 2  — Preprocessor + rule-based  (slang, sarcasm, repeats handled)
  MODE 3  — Preprocessor + BERT/RoBERTa transformer (requires transformers library)

And the comparison report.

Run:
    # Fast demo (no transformers required):
    python -m nlp.demo_inference

    # Full BERT demo (downloads model ~400 MB on first run):
    python -m nlp.demo_inference --bert

    # JSON output only:
    python -m nlp.demo_inference --json
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.gaming_preprocessor   import GamingPreprocessor
from nlp.transformer_sentiment  import TransformerSentimentAnalyzer
from nlp.sentiment_comparison   import SentimentComparison


# ---------------------------------------------------------------------------
# Test suite — covers all the interesting gaming chat edge cases
# ---------------------------------------------------------------------------
DEMO_MESSAGES = [
    # ── Slang normalization ────────────────────────────────────────────────
    ("omg this noob is SOOOO bad wtf",                 "slang + repeated chars + anger"),
    ("gg ez lmao nice clutch bruh",                    "slang → positive gaming expression"),
    ("lmfao this lag is killing me af",                "slang + frustration"),

    # ── Repeated characters ────────────────────────────────────────────────
    ("noooooo why does this always happen!!",           "repeated chars + frustration"),
    ("yesssss finally got it!!!",                      "repeated chars + joy"),
    ("this is SOOOOO unfair I can't even!!",           "repeated + all-caps + frustration"),

    # ── Sarcasm detection ─────────────────────────────────────────────────
    ("oh great another lag spike smh",                 "sarcasm (oh great pattern)"),
    ("yeah right like that was a fair fight lol",      "sarcasm (yeah right pattern)"),
    ("nice job feeding the enemy the whole game lmao", "sarcasm (contextual)"),
    ("definitely not tilted rn lol",                   "sarcasm + negation + gaming"),

    # ── Negation handling ─────────────────────────────────────────────────
    ("not bad, almost got them",                       "negated negative → mild positive"),
    ("this game is not fun at all anymore",            "negated positive → negative"),
    ("I don't hate this strategy",                     "double neg → mildly positive"),

    # ── Clear positive ────────────────────────────────────────────────────
    ("clutch play nice one team amazing",              "joy/positive"),
    ("ggwp that was incredible, glhf next round",      "joy + gaming slang"),

    # ── Clear negative / anger ────────────────────────────────────────────
    ("I QUIT this trash rigged garbage game!!",        "high anger + all caps"),
    ("this feeder is ruining everything wtf!",         "anger at teammate"),

    # ── Strategic / neutral ───────────────────────────────────────────────
    ("idk maybe try flanking next time?",              "uncertain/strategic"),
    ("let's push left and cover right flank",          "strategic intent"),

    # ── Edge cases ────────────────────────────────────────────────────────
    ("",                                               "empty input"),
    ("!!! ???",                                        "no words, just punctuation"),
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(val: float, width: int = 20) -> str:
    """ASCII bar chart for a 0-1 value."""
    filled = int(val * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {val:.2f}"


def _sentiment_bar(score: float, width: int = 20) -> str:
    """ASCII bar for -1 to +1 sentiment score (center = 0)."""
    half = width // 2
    if score >= 0:
        pos  = int(score * half)
        return "-" * half + "+" * pos + "." * (half - pos) + f"  {score:+.2f}"
    else:
        neg  = int(-score * half)
        return "." * (half - neg) + "-" * neg + "-" * half + f"  {score:+.2f}"


def print_full_result(result: dict, show_model: bool = True) -> None:
    """Pretty-print a single transformer analysis result."""
    sep = "-" * 64
    proc = result.get("preprocessing", {})

    print(sep)
    print(f"  ORIGINAL : {proc.get('original_text', result)!r}")
    if proc.get("slang_replacements"):
        repl = proc["slang_replacements"]
        sample = list(repl.items())[:4]
        print(f"  SLANG    : {dict(sample)}")
    if "repeated_chars" in proc.get("normalizations", []):
        print(f"  REPEAT   : collapsed elongated chars")
    print(f"  CLEANED  : {proc.get('cleaned_text', '')!r}")
    print()

    score = result["sentiment_score"]
    label = result["sentiment_label"].upper()
    print(f"  SENTIMENT  {label:<12}  {_sentiment_bar(score)}")
    print(f"  EMOTION    {result['emotion_label'].upper():<12}  conf={result['confidence']:.2f}")

    emotions = result.get("emotion_scores", {})
    for emo, val in emotions.items():
        print(f"    {emo:<14} {_bar(val)}")

    if result.get("sarcasm_detected"):
        print(f"  SARCASM  [!]  score={result['sarcasm_score']:.2f}  (sentiment adjusted)")

    if show_model:
        print(f"  MODEL    : {result['model_used']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment & Emotion Demo")
    parser.add_argument("--bert",    action="store_true",
                        help="Use BERT/transformer model (downloads on first run)")
    parser.add_argument("--json",    action="store_true",
                        help="Output results as JSON instead of pretty-print")
    parser.add_argument("--compare", action="store_true",
                        help="Show rule-based vs transformer comparison report")
    parser.add_argument("--text",    type=str, default=None,
                        help="Analyze a single custom message and exit")
    args = parser.parse_args()

    # ── Single custom message mode ─────────────────────────────────────────
    if args.text:
        analyzer = TransformerSentimentAnalyzer(
            preprocess=True,
            sarcasm_adjust=True,
        )
        result = analyzer.analyze(args.text)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_full_result(result)
        return

    # ── Full demo suite ────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  GAMING CHAT — Sentiment & Emotion Detection Demo")
    print("  Transformer-Enhanced NLP Module")
    print("=" * 64)

    all_results: list[dict] = []

    if args.compare:
        # Comparison mode: rule-based vs transformer
        cmp = SentimentComparison(use_transformer=args.bert)
        for text, label in DEMO_MESSAGES:
            if not text:
                continue
            cmp_result = cmp.compare(text)
            print(cmp.report(cmp_result))
            print()
        return

    # Standard analysis
    analyzer = TransformerSentimentAnalyzer(
        model_name=TransformerSentimentAnalyzer.DEFAULT_MODEL if args.bert
                   else TransformerSentimentAnalyzer.DEFAULT_MODEL,
        emotion_mode="rule",
        preprocess=True,
        sarcasm_adjust=True,
    )

    if not args.bert:
        # Force rule-based polarity by not loading model
        analyzer._loaded = True   # skip model loading
        analyzer._sentiment_pipe = None

    print(f"\n  Mode: {'BERT/Transformer' if args.bert else 'Rule-Based (no model download)'}\n")

    for text, label in DEMO_MESSAGES:
        result = analyzer.analyze(text)
        all_results.append(result)

        if args.json:
            continue  # accumulate, print at end

        print(f"  [{label}]")
        print_full_result(result, show_model=False)

    if args.json:
        print(json.dumps([
            {"label": label, "result": result}
            for (_, label), result in zip(DEMO_MESSAGES, all_results)
        ], indent=2, ensure_ascii=False))
        return

    # ── Summary statistics ─────────────────────────────────────────────────
    print("=" * 64)
    print("  SUMMARY STATISTICS")
    print("─" * 64)
    scores   = [r["sentiment_score"]   for r in all_results]
    sarcasms = [r for r in all_results if r.get("sarcasm_detected")]
    emotions = [r["emotion_label"]     for r in all_results]

    print(f"  Total messages analyzed : {len(all_results)}")
    print(f"  Avg sentiment score     : {sum(scores)/max(len(scores),1):+.3f}")
    print(f"  Sarcasm detected in     : {len(sarcasms)} messages")
    print(f"  Emotion distribution    :")
    from collections import Counter
    emo_counts = Counter(emotions)
    for emo, cnt in emo_counts.most_common():
        pct = cnt / len(emotions) * 100
        print(f"    {emo:<14} {cnt:>2} ({pct:.0f}%)")

    print("=" * 64)
    print()
    print("  To use in your code:")
    print()
    print("    from nlp.transformer_sentiment import TransformerSentimentAnalyzer")
    print("    analyzer = TransformerSentimentAnalyzer()")
    print("    result   = analyzer.analyze('your gaming chat message here')")
    print("    print(result)  # JSON-serialisable dict")
    print()
    print("  To use BERT model (download required):")
    print("    pip install transformers torch")
    print("    Then run:  python -m nlp.demo_inference --bert")
    print()


if __name__ == "__main__":
    main()
