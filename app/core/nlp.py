"""
app/core/nlp.py
---------------
NLP feature extraction module.

Extracts the following affective signals from player-generated text:
  P(t)  -- Emotional Polarity       ∈ [-1, +1]
  I(t)  -- Emotional Intensity      ∈ [ 0,  1]
  G(t)  -- Intent-Outcome Gap       ∈ [ 0,  1]   (cross-modal, passed in)
  Δs(t) -- Sentiment Trend          ∈  R         (sliding window derivative)

In production: replace the rule-based lexicon with a fine-tuned
RoBERTa / VADER model. The interface is identical — only the
_compute_polarity() and _compute_intensity() methods change.

Design principle: if no text is available (silent session), the module
returns the API-provided pre-computed values or safe defaults. The system
degrades gracefully to telemetry-only mode.
"""

import re
import math
from collections import deque
from typing import Dict, Tuple, Optional

from app.api.schemas import NLPInput
from app.core.config import settings


# ---------------------------------------------------------------------------
# Lightweight rule-based lexicon (demo / no-dependency mode)
# Replace with transformer model for production.
# ---------------------------------------------------------------------------

_NEGATIVE_WORDS = {
    # Frustration / anger
    "impossible", "broken", "trash", "quit", "rage", "garbage",
    "unfair", "rigged", "stupid", "hate", "awful", "terrible",
    "worst", "useless", "bug", "laggy", "cheating", "ridiculous",
    # Mild negative
    "hard", "difficult", "frustrating", "annoying", "stuck",
    "losing", "failed", "missed", "died", "nope",
}

_POSITIVE_WORDS = {
    # Engagement
    "nice", "cool", "great", "love", "amazing", "perfect",
    "good", "win", "yes", "awesome", "nailed", "easy", "fun",
    # Challenge-positive
    "challenge", "better", "improving", "finally", "got", "close",
}

_INTENSITY_MARKERS = {
    "caps_ratio",   # proportion of uppercase characters
    "exclamation",  # count of ! marks
    "repetition",   # repeated characters like "nooooo"
    "expletive",    # strong negative words
}

_STRONG_NEGATIVE = {"trash", "quit", "garbage", "broken", "hate", "worst", "rage"}
_STRONG_POSITIVE = {"amazing", "perfect", "awesome", "love"}


class NLPExtractor:
    """
    Stateful NLP feature extractor for a single session.
    Maintains sentiment trend history (sliding window over P(t)).
    """

    def __init__(self, trend_window: int = 5):
        self._polarity_history: deque = deque(maxlen=trend_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, nlp_input: NLPInput) -> Dict[str, float]:
        """
        Extract all NLP features from a NLPInput snapshot.

        Returns dict with keys: polarity, intensity, sentiment_trend, intent_gap
        """
        text = nlp_input.text.strip()

        if text:
            polarity  = self._compute_polarity(text)
            intensity = self._compute_intensity(text, polarity)
        else:
            # Fallback to pre-computed values provided by caller
            polarity  = nlp_input.polarity
            intensity = nlp_input.intensity

        # Sentiment trend Δs(t)
        self._polarity_history.append(polarity)
        trend = self._compute_trend()

        return {
            "polarity":       round(polarity,  4),
            "intensity":      round(intensity, 4),
            "sentiment_trend": round(trend,    4),
            "intent_gap":     round(nlp_input.intent_gap, 4),
        }

    def reset(self):
        self._polarity_history.clear()

    # ------------------------------------------------------------------
    # Polarity estimation (rule-based lexicon — replace with BERT)
    # ------------------------------------------------------------------

    def _compute_polarity(self, text: str) -> float:
        """
        Estimate sentiment polarity ∈ [-1, +1].

        Production replacement:
            from transformers import pipeline
            pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            result = pipe(text)[0]
            return result['score'] if result['label']=='POSITIVE' else -result['score']
        """
        tokens = self._tokenize(text)
        neg_count = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        pos_count = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        total = neg_count + pos_count

        if total == 0:
            return 0.0   # Neutral / no sentiment signal

        # Raw polarity score
        raw = (pos_count - neg_count) / total

        # Strong word amplification
        if any(t in _STRONG_NEGATIVE for t in tokens):
            raw = min(raw - 0.3, -0.3)
        if any(t in _STRONG_POSITIVE for t in tokens):
            raw = max(raw + 0.2, 0.2)

        return max(-1.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Intensity estimation
    # ------------------------------------------------------------------

    def _compute_intensity(self, text: str, polarity: float) -> float:
        """
        Estimate emotional intensity ∈ [0, 1].

        Intensity signals:
          1. Uppercase ratio (SHOUTING → high arousal)
          2. Exclamation/question mark density
          3. Repeated character sequences (noooo, whyyy)
          4. Presence of strong-sentiment words
          5. Scaled by absolute polarity (neutral text = low intensity)
        """
        if not text:
            return 0.0

        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio  = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        excl_density = text.count("!") / max(len(text.split()), 1)
        has_repeated = 1.0 if re.search(r"(.)\1{2,}", text) else 0.0
        strong_word  = 1.0 if any(
            t in _STRONG_NEGATIVE or t in _STRONG_POSITIVE
            for t in self._tokenize(text)
        ) else 0.0

        raw = (
            0.35 * caps_ratio
          + 0.25 * min(excl_density, 1.0)
          + 0.20 * has_repeated
          + 0.20 * strong_word
        )

        # Neutral-polarity text gets dampened intensity
        raw *= (0.3 + 0.7 * abs(polarity))
        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Sentiment trend
    # ------------------------------------------------------------------

    def _compute_trend(self) -> float:
        """
        Δs(t) = linear slope of polarity over the history window.
        Negative slope = declining sentiment = early frustration signal.
        """
        h = list(self._polarity_history)
        n = len(h)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        cov = sum((i - x_mean) * h[i] for i in range(n))
        var = sum((i - x_mean) ** 2 for i in range(n))
        return cov / var if var > 0 else 0.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------------------------
# Session-level registry
# ---------------------------------------------------------------------------

class NLPRegistry:
    """One NLPExtractor per active session."""

    def __init__(self):
        self._extractors: Dict[str, NLPExtractor] = {}

    def get_or_create(self, session_id: str) -> NLPExtractor:
        if session_id not in self._extractors:
            self._extractors[session_id] = NLPExtractor()
        return self._extractors[session_id]

    def drop_session(self, session_id: str):
        self._extractors.pop(session_id, None)


nlp_registry = NLPRegistry()
