"""
nlp/sentiment_comparison.py
----------------------------
Side-by-side comparison of:
  (A) Traditional rule-based sentiment analysis  (utils/nlp_extractor.py)
  (B) Transformer-based sentiment analysis       (nlp/transformer_sentiment.py)

Shows WHERE and WHY the transformer outperforms simple lexicon matching,
especially on:
  - Slang and abbreviations
  - Negation of positive/negative words
  - Context-dependent meaning
  - Sarcasm
  - Informal gaming language

Output:
  {
    "input": str,
    "rule_based": {
        "polarity": float,
        "sentiment_label": str,
        "anger": float,
        "frustration": float,
        "confidence": float,
    },
    "transformer": {
        "sentiment_score": float,
        "sentiment_label": str,
        "emotion_label": str,
        "emotion_scores": dict,
        "sarcasm_detected": bool,
        "model_used": str,
    },
    "delta": {
        "polarity_shift": float,     # transformer - rule_based polarity
        "label_agreement": bool,
        "improvement_notes": list,   # human-readable explanations
    }
  }

Usage:
    from nlp.sentiment_comparison import SentimentComparison

    cmp = SentimentComparison()
    result = cmp.compare("wtf that noob RUINED everything!!")
    print(cmp.report(result))
"""

from __future__ import annotations

import json
from typing import Dict, List

from utils.nlp_extractor import NLPExtractor
from nlp.transformer_sentiment import TransformerSentimentAnalyzer
from nlp.gaming_preprocessor import GamingPreprocessor


# ---------------------------------------------------------------------------
# Helper: map polarity float to label (shared for both systems)
# ---------------------------------------------------------------------------
def _label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Improvement Detector
# ---------------------------------------------------------------------------

def _detect_improvements(
    text: str,
    preprocessed: str,
    rule_polarity: float,
    trans_score: float,
    trans_result: Dict,
) -> List[str]:
    """
    Return a list of human-readable notes about where the transformer
    adds value over the rule-based system.
    """
    notes: List[str] = []

    # 1. Slang was handled
    slang = trans_result["preprocessing"].get("slang_replacements", {})
    if slang:
        pairs = ", ".join(f'"{k}"->"{v}"' for k, v in list(slang.items())[:3])
        notes.append(f"Slang normalized before model input: {pairs}")

    # 2. Repeated characters were collapsed
    if "repeated_chars" in trans_result["preprocessing"].get("normalizations", []):
        notes.append("Repeated character elongation collapsed (e.g., 'SOOOO' -> 'SOO')")

    # 3. Sarcasm detected - rule-based would miss this
    if trans_result.get("sarcasm_detected"):
        score = trans_result["sarcasm_score"]
        notes.append(
            f"Sarcasm detected (score={score:.2f}): sentiment score adjusted/inverted. "
            f"Rule-based cannot detect sarcasm."
        )

    # 4. Label disagreement -> transformer likely more contextual
    rule_label  = _label(rule_polarity)
    trans_label = trans_result["sentiment_label"]
    if rule_label != trans_label:
        notes.append(
            f"Sentiment label mismatch: rule-based='{rule_label}', "
            f"transformer='{trans_label}'. "
            f"Transformer uses contextual representation, not just keyword presence."
        )

    # 5. Negation handling difference
    import re
    has_neg = bool(re.search(r"\b(not|dont|can't|never|didn't|isn't)\b", text, re.IGNORECASE))
    if has_neg:
        notes.append(
            "Negation detected. Transformer attends over full context window "
            "vs rule-based local negation window (only next word)."
        )

    # 6. Transformer adds granular emotion (rule-based has none in original)
    emotions = trans_result.get("emotion_scores", {})
    top_emotion = max(emotions, key=emotions.get) if emotions else "neutral"
    if top_emotion != "neutral":
        notes.append(
            f"Granular emotion detected: '{top_emotion}' "
            f"(rule-based only outputs binary anger/frustration scores, "
            f"not a unified 4-class label)."
        )

    # 7. Polarity shift magnitude
    delta = abs(trans_score - rule_polarity)
    if delta > 0.25:
        notes.append(
            f"Significant polarity shift: |delta|={delta:.2f}. "
            f"Transformer better captures overall message tone from context."
        )

    if not notes:
        notes.append("Both systems agree - message is clear and unambiguous.")

    return notes


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SentimentComparison:
    """
    Runs both rule-based and transformer-based analysis on the same input
    and produces a structured comparison report.

    Args:
        transformer_model: HuggingFace model ID to use for transformer side.
        use_transformer:   Set False to skip model loading (comparison will
                           use rule-based on both sides with different configs).
    """

    def __init__(
        self,
        transformer_model: str = TransformerSentimentAnalyzer.DEFAULT_MODEL,
        use_transformer: bool = True,
    ) -> None:
        self._rule   = NLPExtractor(mode="rule_based")
        self._trans  = TransformerSentimentAnalyzer(
            model_name=transformer_model,
            emotion_mode="rule",
            preprocess=True,
            sarcasm_adjust=True,
        ) if use_transformer else None
        self._prep   = GamingPreprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, text: str) -> Dict:
        """
        Run both systems and return comparison dict.

        Args:
            text: Raw chat message.

        Returns:
            Structured comparison dict (see module docstring).
        """
        if not text or not isinstance(text, str):
            return {}

        # Rule-based pass
        rule_feats    = self._rule.extract(text)
        rule_polarity = float(rule_feats["polarity"])
        rule_result   = {
            "polarity":        rule_polarity,
            "sentiment_label": _label(rule_polarity),
            "anger":           rule_feats["anger"],
            "frustration":     rule_feats["frustration"],
            "confidence":      rule_feats["confidence"],
            "intensity":       rule_feats["intensity"],
        }

        # Transformer / preprocessed pass
        if self._trans:
            trans_full   = self._trans.analyze(text)
            preprocessed = trans_full["preprocessing"]["cleaned_text"]
        else:
            # Fallback: rule-based with preprocessing applied
            prep_result  = self._prep.process(text)
            preprocessed = prep_result["cleaned_text"]
            trans_feats  = self._rule.extract(preprocessed)
            trans_full   = {
                "sentiment_score": trans_feats["polarity"],
                "sentiment_label": _label(trans_feats["polarity"]),
                "emotion_label":   "neutral",
                "emotion_scores":  {
                    "anger":       trans_feats["anger"],
                    "frustration": trans_feats["frustration"],
                    "joy":         0.0,
                    "neutral":     0.5,
                },
                "sarcasm_detected":  prep_result["sarcasm_detected"],
                "sarcasm_score":     prep_result["sarcasm_score"],
                "model_used":        "rule_based_preprocessed_fallback",
                "preprocessing": {
                    "original_text":      text,
                    "cleaned_text":       preprocessed,
                    "slang_replacements": prep_result["slang_replacements"],
                    "normalizations":     prep_result["normalizations"],
                },
            }

        trans_score = float(trans_full["sentiment_score"])

        transformer_result = {
            "sentiment_score": trans_score,
            "sentiment_label": trans_full["sentiment_label"],
            "emotion_label":   trans_full["emotion_label"],
            "emotion_scores":  trans_full["emotion_scores"],
            "sarcasm_detected": trans_full.get("sarcasm_detected", False),
            "sarcasm_score":    trans_full.get("sarcasm_score", 0.0),
            "model_used":       trans_full.get("model_used", "unknown"),
        }

        # Delta & improvements
        polarity_shift  = round(trans_score - rule_polarity, 4)
        label_agreement = rule_result["sentiment_label"] == transformer_result["sentiment_label"]

        improvement_notes = _detect_improvements(
            text, preprocessed, rule_polarity, trans_score, trans_full
        )

        return {
            "input":       text,
            "rule_based":  rule_result,
            "transformer": transformer_result,
            "delta": {
                "polarity_shift":    polarity_shift,
                "label_agreement":   label_agreement,
                "improvement_notes": improvement_notes,
            },
        }

    def compare_batch(self, messages: List[str]) -> List[Dict]:
        """Run comparison on a list of messages."""
        return [self.compare(msg) for msg in messages]

    @staticmethod
    def report(result: Dict, width: int = 70) -> str:
        """
        Pretty-print a comparison result as human-readable text.

        Args:
            result:  Single comparison dict from compare().
            width:   Column width for formatting.
        """
        if not result:
            return "(empty comparison result)"

        sep   = "-" * width
        lines: List[str] = []
        lines.append("=" * width)
        lines.append(f"  INPUT: {result['input']!r}")
        lines.append(sep)

        rb = result["rule_based"]
        tr = result["transformer"]
        d  = result["delta"]

        lines.append(f"  {'METRIC':<28}  {'RULE-BASED':>14}  {'TRANSFORMER':>14}")
        lines.append(sep)
        lines.append(f"  {'Polarity Score':<28}  {rb['polarity']:>+14.3f}  {tr['sentiment_score']:>+14.3f}")
        lines.append(f"  {'Sentiment Label':<28}  {rb['sentiment_label']:>14}  {tr['sentiment_label']:>14}")
        lines.append(f"  {'Anger Score':<28}  {rb['anger']:>14.3f}  {tr['emotion_scores'].get('anger', 0):>14.3f}")
        lines.append(f"  {'Frustration Score':<28}  {rb['frustration']:>14.3f}  {tr['emotion_scores'].get('frustration', 0):>14.3f}")
        lines.append(f"  {'Joy Score':<28}  {'N/A':>14}  {tr['emotion_scores'].get('joy', 0):>14.3f}")
        lines.append(f"  {'Emotion Label':<28}  {'N/A':>14}  {tr['emotion_label']:>14}")
        lines.append(f"  {'Sarcasm Detected':<28}  {'N/A':>14}  {str(tr['sarcasm_detected']):>14}")
        lines.append(sep)
        lines.append(f"  Polarity Shift (Trans - Rule): {d['polarity_shift']:+.3f}")
        lines.append(f"  Label Agreement              : {d['label_agreement']}")
        lines.append(sep)
        lines.append("  CONTEXTUAL IMPROVEMENTS:")

        for note in d["improvement_notes"]:
            # Word-wrap long notes
            words = note.split()
            line  = "    * "
            chunk: List[str] = []
            for w in words:
                chunk.append(w)
                if len(" ".join(chunk)) > width - 8:
                    lines.append(line + " ".join(chunk[:-1]))
                    line  = "      "
                    chunk = [w]
            lines.append(line + " ".join(chunk))

        lines.append("=" * width)
        return "\n".join(lines)

    def to_json(self, text: str, indent: int = 2) -> str:
        """Compare and return JSON string."""
        return json.dumps(self.compare(text), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    GAMING_MESSAGES = [
        "wtf that noob RUINED the whole game!!",
        "oh great another lag spike smh",
        "gg nice clutch everyone!",
        "SOOOO close yet I still lost fml",
        "yeah right like that was fair lmao",
        "not bad, almost nailed it",
        "definitely not tilted rn lol",
        "omg that was amazing I love this game",
        "idk maybe try flanking next time",
        "this trash game is so laggy I quit",
    ]

    cmp = SentimentComparison(use_transformer=False)

    print("\n" + "=" * 70)
    print("  Sentiment Comparison: Traditional vs Transformer-Enhanced")
    print("=" * 70)

    for msg in GAMING_MESSAGES:
        result = cmp.compare(msg)
        print(cmp.report(result))
        print()
