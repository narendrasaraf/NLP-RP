"""
utils/nlp_extractor.py
----------------------
Canonical NLP feature extraction module for the Adaptive Shooter system.

This is the SINGLE SOURCE OF TRUTH for NLP feature extraction.
All other modules (backend/nlp_module.py, app/core/nlp.py,
utils/emotion_analyzer.py) should delegate here or be replaced by this.

Output schema (guaranteed, never None):
    {
        "polarity":            float  ∈ [-1.0, +1.0]
        "intensity":           float  ∈ [ 0.0,  1.0]
        "anger":               float  ∈ [ 0.0,  1.0]
        "frustration":         float  ∈ [ 0.0,  1.0]
        "confidence":          float  ∈ [ 0.0,  1.0]
        "exclamation_count":   int    ≥ 0
        "uppercase_ratio":     float  ∈ [ 0.0,  1.0]
        "negative_word_ratio": float  ∈ [ 0.0,  1.0]
    }

Modes:
    RULE_BASED  (default) — zero dependencies, runs in microseconds.
    PRETRAINED            — plugs in a HuggingFace pipeline for polarity;
                            all other signals remain rule-based.
                            Enable via: NLPExtractor(mode="pretrained")

Design principles:
    - Each signal is computed by an isolated private method → easy to swap.
    - Empty / None input → safe defaults, no crash.
    - All outputs are rounded and clamped → downstream math never sees NaN.
    - Stateless: no rolling windows here (temporal logic lives in preprocessing.py).

Usage:
    from utils.nlp_extractor import NLPExtractor

    extractor = NLPExtractor()                        # rule-based
    feats = extractor.extract("this is IMPOSSIBLE!!")

    # or pretrained mode (requires: pip install transformers torch)
    extractor = NLPExtractor(mode="pretrained")
    feats = extractor.extract("I finally got it, amazing!")
"""

from __future__ import annotations

import re
import math
from typing import Dict, Literal, Optional


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
NLPFeatures = Dict[str, float | int]

# ---------------------------------------------------------------------------
# Lexicons  (game-domain specific, tuned for shooter chat patterns)
# ---------------------------------------------------------------------------

# ── Polarity ────────────────────────────────────────────────────────────────
_POS_WORDS: frozenset[str] = frozenset({
    "good", "great", "awesome", "amazing", "love", "happy", "excellent",
    "fantastic", "yes", "perfect", "thanks", "thank", "glad", "yay",
    "nice", "cool", "win", "easy", "fun", "challenge", "better", "finally",
    "nailed", "got", "close", "improving",
})

_NEG_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "awful", "hate", "sad", "angry", "stupid", "no",
    "worst", "fail", "wrong", "sucks", "poor", "impossible", "broken",
    "trash", "quit", "rage", "garbage", "unfair", "useless", "bug",
    "hard", "difficult", "frustrating", "annoying", "stuck", "died",
    "laggy", "cheating", "ridiculous", "losing", "failed", "missed", "nope",
    "rigged",
})

# Amplifiers — shift polarity further toward extremes
_STRONG_NEG: frozenset[str] = frozenset({
    "trash", "quit", "garbage", "broken", "hate", "worst", "rage", "useless",
})
_STRONG_POS: frozenset[str] = frozenset({
    "amazing", "perfect", "awesome", "love", "excellent", "fantastic",
})

# ── Modifiers ───────────────────────────────────────────────────────────────
_NEGATIONS: frozenset[str] = frozenset({
    "not", "don't", "dont", "can't", "cant", "cannot", "never",
    "didn't", "didnt", "isn't", "isnt", "aren't", "arent", "won't", "wont",
})
_INTENSIFIERS: frozenset[str] = frozenset({
    "very", "really", "extremely", "absolutely", "so", "too",
    "completely", "totally", "super", "fucking",
})

# ── Emotion-specific ────────────────────────────────────────────────────────
_ANGER_WORDS: frozenset[str] = frozenset({
    "angry", "furious", "mad", "hate", "stupid", "idiot", "rage", "hell",
    "wtf", "damn", "bullshit", "fuck", "shit", "pissed", "annoying",
    "trash", "garbage", "quit",
})

_FRUSTRATION_WORDS: frozenset[str] = frozenset({
    "ugh", "annoyed", "stuck", "broken", "why", "tired", "sigh", "again",
    "confusing", "hard", "error", "failing", "crash", "crashing", "bug",
    "impossible", "unfair", "laggy", "difficult", "frustrating",
})

# ── Confidence ──────────────────────────────────────────────────────────────
_CONFIDENCE_WORDS: frozenset[str] = frozenset({
    "definitely", "absolutely", "sure", "certain", "guarantee", "clear",
    "obvious", "know", "exactly", "always", "100%", "easy",
})
_UNCERTAINTY_WORDS: frozenset[str] = frozenset({
    "maybe", "perhaps", "guess", "think", "probably", "might", "could",
    "unsure", "sometimes", "possibly", "idk", "hopefully",
})


# ---------------------------------------------------------------------------
# Safe output defaults (returned when text is empty)
# ---------------------------------------------------------------------------
_EMPTY_FEATURES: NLPFeatures = {
    "polarity":            0.0,
    "intensity":           0.0,
    "anger":               0.0,
    "frustration":         0.0,
    "confidence":          0.5,  # neutral baseline
    "exclamation_count":   0,
    "uppercase_ratio":     0.0,
    "negative_word_ratio": 0.0,
}


# ---------------------------------------------------------------------------
# NLPExtractor — main class
# ---------------------------------------------------------------------------

class NLPExtractor:
    """
    Stateless NLP feature extractor.

    Args:
        mode: "rule_based" (default) or "pretrained".
              In "pretrained" mode, polarity is computed via a HuggingFace
              sentiment pipeline. All other features remain rule-based.
        model_name: HuggingFace model identifier used in pretrained mode.
                    Defaults to a lightweight Twitter-tuned RoBERTa model.
    """

    def __init__(
        self,
        mode: Literal["rule_based", "pretrained"] = "rule_based",
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ) -> None:
        self._mode = mode
        self._pipeline = None  # lazy-loaded on first call

        if mode == "pretrained":
            self._model_name = model_name
            self._load_pretrained_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str | None) -> NLPFeatures:
        """
        Extract NLP features from a chat string.

        Args:
            text: Raw player chat message. None or empty string → safe defaults.

        Returns:
            NLPFeatures dict with exactly 8 keys. All values clamped and rounded.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return dict(_EMPTY_FEATURES)

        text = text.strip()

        # ── Tokenization ──────────────────────────────────────────────────
        tokens      = _tokenize(text)          # lowercase word tokens
        raw_words   = text.split()             # preserves casing (for caps)
        total_words = max(len(tokens), 1)      # prevent div-by-zero

        # ── Format signals (punctuation + casing) ─────────────────────────
        exclamation_count = text.count("!")
        question_count    = text.count("?")
        uppercase_ratio   = _uppercase_ratio(text, raw_words)

        # ── Lexical scanning (single pass) ────────────────────────────────
        (
            score_pos, score_neg,
            score_anger, score_frust,
            score_conf, score_unconf,
            negative_word_count,
        ) = self._scan_tokens(tokens)

        # ── Polarity ─────────────────────────────────────────────────────
        if self._mode == "pretrained" and self._pipeline is not None:
            polarity = self._pretrained_polarity(text)
        else:
            polarity = _rule_polarity(score_pos, score_neg, tokens)

        # ── Intensity ─────────────────────────────────────────────────────
        intensity = _compute_intensity(
            text, polarity, uppercase_ratio,
            exclamation_count, total_words,
            score_pos, score_neg, score_anger, score_frust,
        )

        # ── Emotion probabilities ─────────────────────────────────────────
        caps_count = sum(1 for w in raw_words if w.isupper() and len(w) > 1)

        anger = _emotion_prob(
            score_anger,
            format_boost=caps_count * 0.10 + exclamation_count * 0.10,
            total_words=total_words,
        )
        frustration = _emotion_prob(
            score_frust,
            format_boost=caps_count * 0.05 + question_count * 0.10
                         + exclamation_count * 0.05,
            total_words=total_words,
        )

        # ── Confidence ────────────────────────────────────────────────────
        confidence = _compute_confidence(score_conf, score_unconf, total_words)

        # ── Surface ratios ────────────────────────────────────────────────
        negative_word_ratio = round(negative_word_count / total_words, 4)

        return {
            "polarity":            _clamp(round(polarity,            4), -1.0, 1.0),
            "intensity":           _clamp(round(intensity,           4),  0.0, 1.0),
            "anger":               _clamp(round(anger,               4),  0.0, 1.0),
            "frustration":         _clamp(round(frustration,         4),  0.0, 1.0),
            "confidence":          _clamp(round(confidence,          4),  0.0, 1.0),
            "exclamation_count":   exclamation_count,
            "uppercase_ratio":     _clamp(round(uppercase_ratio,     4),  0.0, 1.0),
            "negative_word_ratio": _clamp(round(negative_word_ratio, 4),  0.0, 1.0),
        }

    # ------------------------------------------------------------------
    # Internal: Single-pass lexical scanner
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_tokens(
        tokens: list[str],
    ) -> tuple[float, float, float, float, float, float, int]:
        """
        Scan token list in one pass, accumulating scores for all categories.

        Handles negation (flips valence of next content word)
        and intensifiers (multiplies the next content word's weight).

        Returns:
            (score_pos, score_neg, score_anger, score_frust,
             score_conf, score_unconf, negative_word_count)
        """
        score_pos = score_neg = 0.0
        score_anger = score_frust = 0.0
        score_conf = score_unconf = 0.0
        negative_word_count = 0

        intensity_mult = 1.0
        is_negated     = False

        for word in tokens:
            # ── Modifier words (do not score themselves) ─────────────────
            if word in _NEGATIONS:
                is_negated = True
                continue
            if word in _INTENSIFIERS:
                intensity_mult += 0.5
                continue

            val = 1.0 * intensity_mult
            if is_negated:
                val *= -0.5   # negation flips and dampens

            # ── Polarity ──────────────────────────────────────────────────
            if word in _POS_WORDS:
                if is_negated:
                    score_neg += abs(val)
                    negative_word_count += 1
                else:
                    score_pos += val
            elif word in _NEG_WORDS:
                if is_negated:
                    score_pos += abs(val) * 0.5   # "not bad" → mildly positive
                else:
                    score_neg += val
                    negative_word_count += 1

            # ── Anger ─────────────────────────────────────────────────────
            if word in _ANGER_WORDS and not is_negated:
                score_anger += val
                negative_word_count += 1

            # ── Frustration ───────────────────────────────────────────────
            if word in _FRUSTRATION_WORDS and not is_negated:
                score_frust += val
                negative_word_count += 1

            # ── Confidence ────────────────────────────────────────────────
            if word in _CONFIDENCE_WORDS:
                if is_negated:
                    score_unconf += val
                else:
                    score_conf += val
            if word in _UNCERTAINTY_WORDS:
                if is_negated:
                    score_conf += val * 0.5
                else:
                    score_unconf += val

            # ── Reset modifiers after each content word ────────────────────
            is_negated     = False
            intensity_mult = 1.0

        return (
            score_pos, score_neg,
            score_anger, score_frust,
            score_conf, score_unconf,
            negative_word_count,
        )

    # ------------------------------------------------------------------
    # Internal: Pretrained model (optional, lazy-loaded)
    # ------------------------------------------------------------------

    def _load_pretrained_model(self) -> None:
        """
        Lazy-load a HuggingFace sentiment pipeline.
        Falls back to rule-based silently if transformers is not installed.
        """
        try:
            from transformers import pipeline   # type: ignore[import]
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                truncation=True,
                max_length=128,
            )
            print(f"[NLPExtractor] Pretrained model loaded: {self._model_name}")
        except ImportError:
            print(
                "[NLPExtractor] WARNING: 'transformers' not installed. "
                "Falling back to rule-based mode. "
                "Install with: pip install transformers torch"
            )
            self._mode = "rule_based"

    def _pretrained_polarity(self, text: str) -> float:
        """
        Map HuggingFace label → [-1, +1].

        The cardiffnlp model returns labels: LABEL_0=negative,
        LABEL_1=neutral, LABEL_2=positive.
        """
        try:
            result = self._pipeline(text)[0]
            label  = result["label"].upper()
            score  = float(result["score"])
            if "POSITIVE" in label or label == "LABEL_2":
                return score
            elif "NEGATIVE" in label or label == "LABEL_0":
                return -score
            else:
                return 0.0   # neutral
        except Exception as e:
            print(f"[NLPExtractor] Pretrained inference failed: {e}. Using rule-based.")
            return 0.0


# ---------------------------------------------------------------------------
# Pure helper functions  (no class state — easy to unit-test)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, strips punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


def _uppercase_ratio(text: str, raw_words: list[str]) -> float:
    """
    Ratio of ALL-CAPS words (≥2 alpha chars) to total words.
    Single-char caps (like 'I') are excluded to avoid inflation.
    """
    caps = sum(
        1 for w in raw_words
        if w.isupper() and len(re.sub(r"[^A-Z]", "", w)) > 1
    )
    return caps / max(len(raw_words), 1)


def _rule_polarity(
    score_pos: float,
    score_neg: float,
    tokens: list[str],
) -> float:
    """
    Smooth polarity via Laplace-smoothed ratio, then amplify for strong words.
    Formula: (pos - neg) / (pos + neg + 2)   (LaPlace smoothing → avoids ±1 extremes)
    """
    raw = (score_pos - score_neg) / (score_pos + score_neg + 2.0)

    # Strong word amplification
    if any(t in _STRONG_NEG for t in tokens):
        raw = min(raw - 0.3, -0.3)
    if any(t in _STRONG_POS for t in tokens):
        raw = max(raw + 0.2,  0.2)

    return max(-1.0, min(1.0, raw))


def _compute_intensity(
    text: str,
    polarity: float,
    uppercase_ratio: float,
    exclamation_count: int,
    total_words: int,
    score_pos: float,
    score_neg: float,
    score_anger: float,
    score_frust: float,
) -> float:
    """
    Intensity = punctuation/casing arousal + lexical emotional density.

    Signal sources (weighted):
      1. Uppercase word ratio       → shouting / high arousal
      2. Exclamation mark density   → expressiveness
      3. Repeated char sequences    → "noooo", "whyyy"
      4. Lexical emotional density  → how many charged words per token

    Dampened by proximity to neutral polarity (neutral text = low intensity).
    """
    # Format-based arousal
    excl_density   = min(exclamation_count / max(total_words, 1), 1.0)
    has_repeated   = 1.0 if re.search(r"(.)\1{2,}", text) else 0.0

    format_score = (
        0.35 * uppercase_ratio
      + 0.25 * excl_density
      + 0.20 * has_repeated
    )

    # Lexical emotional density
    lexical_weight = (score_pos + score_neg + score_anger + score_frust) / total_words
    lexical_score  = min(lexical_weight * 0.20, 0.20)   # capped contribution

    raw = format_score + lexical_score

    # Neutral text gets dampened: multiply by (0.3 + 0.7 * |polarity|)
    raw *= (0.3 + 0.7 * abs(polarity))

    return max(0.0, min(1.0, raw))


def _emotion_prob(
    score: float,
    format_boost: float,
    total_words: int,
) -> float:
    """
    Convert raw lexical score + format signals into a probability ∈ [0, 1].

    Uses a simple scaled ratio with a format boost from caps/exclamations.
    """
    if score <= 0:
        return max(0.0, format_boost * 0.5)
    norm_denom = math.ceil(total_words / 3)   # adaptive denominator
    raw = (score / norm_denom) * 1.5 + format_boost
    return max(0.0, min(1.0, raw))


def _compute_confidence(
    score_conf: float,
    score_unconf: float,
    total_words: int,
) -> float:
    """
    Confidence ∈ [0, 1] centered at 0.5 (neutral baseline).

    Positive confidence words push above 0.5; uncertainty words push below.
    """
    conf_diff = score_conf - score_unconf
    adjusted  = 0.5 + (conf_diff / total_words) * 1.5
    return max(0.0, min(1.0, adjusted))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Module-level singleton (drop-in replacement for emotion_analyzer instance)
# ---------------------------------------------------------------------------

#: Default rule-based extractor — import this directly throughout the system.
nlp_extractor = NLPExtractor(mode="rule_based")


# ---------------------------------------------------------------------------
# Self-test  (run: python -m utils.nlp_extractor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    TEST_CASES = [
        ("",                               "empty input"),
        ("gg nice round",                  "positive / engaged"),
        ("this is IMPOSSIBLE!!",           "frustration + caps + exclamation"),
        ("I QUIT this trash is unfair!!!",  "high anger + rage"),
        ("maybe I should try a different strategy", "uncertain / strategic"),
        ("not bad, almost got it",         "negated positive"),
        ("noooooo why does this keep happening", "repeated chars + frustration"),
        ("definitely gonna win this",      "high confidence"),
    ]

    extractor = NLPExtractor()

    print("=" * 72)
    print("  NLPExtractor — Self Test  (rule_based mode)")
    print("=" * 72)

    for text, label in TEST_CASES:
        feats = extractor.extract(text)
        print(f"\n[{label}]")
        print(f"  Input   : {text!r}")
        print(f"  polarity={feats['polarity']:+.3f}  "
              f"intensity={feats['intensity']:.3f}  "
              f"anger={feats['anger']:.3f}  "
              f"frustration={feats['frustration']:.3f}  "
              f"confidence={feats['confidence']:.3f}")
        print(f"  exclamations={feats['exclamation_count']}  "
              f"uppercase_ratio={feats['uppercase_ratio']:.3f}  "
              f"neg_word_ratio={feats['negative_word_ratio']:.3f}")
