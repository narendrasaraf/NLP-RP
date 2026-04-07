"""
nlp/transformer_sentiment.py
-----------------------------
Transformer-based Sentiment & Emotion Detection Module for Gaming Chat.

Architecture:
  - Sentiment backbone : HuggingFace pipeline wrapping a pre-trained model
                         (default: nlptown/bert-base-multilingual-uncased-sentiment
                          which produces 1-5 star ratings → mapped to [-1, +1])
  - Emotion classifier : Lightweight secondary classifiers
                         Option A (default): Rule-based heuristics (zero deps)
                         Option B:           HuggingFace emotion pipeline
                                             (j-hartmann/emotion-english-distilroberta-base)
  - Preprocessor       : GamingPreprocessor (slang, repeats, sarcasm)

Output schema:
  {
    "sentiment_score":   float  ∈ [-1.0, +1.0],   # -1=negative, 0=neutral, +1=positive
    "sentiment_label":   str    ∈ {negative, neutral, positive},
    "emotion_label":     str    ∈ {anger, frustration, joy, neutral},
    "emotion_scores":    {anger: float, frustration: float, joy: float, neutral: float},
    "confidence":        float  ∈ [0.0, 1.0],
    "sarcasm_detected":  bool,
    "sarcasm_score":     float  ∈ [0.0, 1.0],
    "model_used":        str,
    "preprocessing": {
        "original_text":      str,
        "cleaned_text":       str,
        "slang_replacements": dict,
        "normalizations":     list,
    }
  }

Model options (pass ``model_name`` to constructor):
  "nlptown/bert-base-multilingual-uncased-sentiment"   ← default (1-5 stars)
  "distilbert-base-uncased-finetuned-sst-2-english"    ← lightweight POSITIVE/NEGATIVE
  "cardiffnlp/twitter-roberta-base-sentiment-latest"   ← Twitter-tuned RoBERTa
  "ProsusAI/finbert"                                   ← finance (replace if needed)

Usage:
    from nlp.transformer_sentiment import TransformerSentimentAnalyzer

    analyzer = TransformerSentimentAnalyzer()          # lazy-loads on first call
    result   = analyzer.analyze("wtf that noob RUINED the game!!")
    print(result)   # JSON-serialisable dict
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, Literal, Optional

from nlp.gaming_preprocessor import GamingPreprocessor


# ---------------------------------------------------------------------------
# Emotion label constants
# ---------------------------------------------------------------------------
EMOTION_LABELS = ("anger", "frustration", "joy", "neutral")


# ---------------------------------------------------------------------------
# Sentiment model label mappings
# ---------------------------------------------------------------------------

# nlptown 1-5 star → polarity score
_STAR_TO_SCORE: Dict[str, float] = {
    "1 star":  -1.00,
    "2 stars": -0.50,
    "3 stars":  0.00,
    "4 stars":  0.60,
    "5 stars":  1.00,
}

# DistilBERT / SST-2 / Cardiff RoBERTa label → polarity
_BINARY_LABEL_MAP: Dict[str, float] = {
    "POSITIVE": 1.0,
    "NEGATIVE": -1.0,
    "LABEL_0":  -1.0,
    "LABEL_1":   0.0,
    "LABEL_2":   1.0,
    "POS":       1.0,
    "NEG":      -1.0,
    "NEU":       0.0,
}


# ---------------------------------------------------------------------------
# Rule-based emotion classifier (no extra deps)
# ---------------------------------------------------------------------------

# Lexicon sets for four target classes (kept minimal & fast)
_ANGER_LEXICON = frozenset({
    "angry", "furious", "mad", "rage", "hate", "stupid", "idiot", "wtf",
    "damn", "hell", "ruined", "ruin", "bullshit", "pissed", "quit",
    "trash", "garbage", "terrible", "betrayed", "cheater", "aimbot",
})
_FRUSTRATION_LEXICON = frozenset({
    "ugh", "stuck", "again", "impossible", "unfair", "laggy", "lag",
    "bug", "broken", "why", "failed", "miss", "missed", "losing", "lost",
    "sigh", "hard", "difficult", "frustrating", "annoying", "crash",
    "tilted", "feeder", "dying", "dies",
})
_JOY_LEXICON = frozenset({
    "great", "amazing", "awesome", "perfect", "love", "fantastic", "happy",
    "gg", "nice", "won", "win", "clutch", "carry", "excellent", "beautiful",
    "fun", "exciting", "yes", "finally", "nailed", "yay", "glhf",
})


def _rule_emotion(tokens: list[str], sentiment_score: float) -> Dict[str, float]:
    """
    Compute soft emotion scores for 4 classes via lexicon match.

    Scores are *independent* (can co-occur), then normalised to sum=1.
    """
    score: Dict[str, float] = {"anger": 0.0, "frustration": 0.0, "joy": 0.0, "neutral": 0.2}

    for tok in tokens:
        if tok in _ANGER_LEXICON:
            score["anger"] += 1.0
        if tok in _FRUSTRATION_LEXICON:
            score["frustration"] += 0.8
        if tok in _JOY_LEXICON:
            score["joy"] += 0.8

    # Sentiment score bias: strong negative → boost anger/frustration
    if sentiment_score < -0.4:
        score["anger"] += 0.5
        score["frustration"] += 0.3
    elif sentiment_score > 0.4:
        score["joy"] += 0.5
        score["neutral"] -= 0.1

    total = max(sum(score.values()), 1.0)
    return {k: round(max(0.0, min(1.0, v / total)), 4) for k, v in score.items()}


def _pick_emotion_label(emotion_scores: Dict[str, float]) -> str:
    return max(emotion_scores, key=emotion_scores.get)


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------
def _softmax(values: Dict[str, float]) -> Dict[str, float]:
    exps = {k: math.exp(v * 5) for k, v in values.items()}  # *5 sharpens the distribution
    total = sum(exps.values())
    return {k: round(v / total, 4) for k, v in exps.items()}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TransformerSentimentAnalyzer:
    """
    Real-time sentiment & emotion analyser backed by a transformer model.

    Args:
        model_name:      HuggingFace model ID for the sentiment backbone.
        emotion_mode:    "rule" (default, zero-dep) or "transformer"
                         (loads j-hartmann/emotion-english-distilroberta-base).
        device:          -1 = CPU (default), 0+ = GPU id.
        preprocess:      If True (default), apply GamingPreprocessor first.
        sarcasm_adjust:  If True (default), invert/dampen sentiment when
                         sarcasm is detected (score >= 0.40).
    """

    DEFAULT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        emotion_mode: Literal["rule", "transformer"] = "rule",
        device: int = -1,
        preprocess: bool = True,
        sarcasm_adjust: bool = True,
    ) -> None:
        self.model_name     = model_name
        self.emotion_mode   = emotion_mode
        self.device         = device
        self.preprocess     = preprocess
        self.sarcasm_adjust = sarcasm_adjust

        self._sentiment_pipe  = None   # lazy-loaded
        self._emotion_pipe    = None   # lazy-loaded (if emotion_mode == "transformer")
        self._preprocessor    = GamingPreprocessor() if preprocess else None
        self._loaded          = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """
        Analyze a raw chat message.

        Args:
            text: Raw player chat string.

        Returns:
            JSON-serialisable dict with the full result schema (see module docstring).
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result(text or "")

        # ── Lazy-load model on first call ─────────────────────────────────
        if not self._loaded:
            self._load_models()

        # ── Preprocessing ─────────────────────────────────────────────────
        if self._preprocessor:
            prep_result = self._preprocessor.process(text)
            input_text   = prep_result["cleaned_text"]
            sarcasm_score = prep_result["sarcasm_score"]
            sarcasm_detected = prep_result["sarcasm_detected"]
        else:
            input_text   = text.lower().strip()
            prep_result  = {
                "original_text": text, "cleaned_text": input_text,
                "slang_replacements": {}, "normalizations": [],
            }
            sarcasm_score    = 0.0
            sarcasm_detected = False

        # ── Sentiment inference ───────────────────────────────────────────
        raw_score, raw_confidence = self._get_sentiment(input_text)

        # ── Sarcasm adjustment ────────────────────────────────────────────
        # If sarcasm detected with high confidence, invert the sentiment score
        if self.sarcasm_adjust and sarcasm_detected:
            # Partial inversion weighted by sarcasm confidence
            alpha = min(1.0, sarcasm_score)
            raw_score = raw_score * (1 - alpha) + (-raw_score) * alpha

        # Clamp
        sentiment_score = float(round(max(-1.0, min(1.0, raw_score)), 4))

        # ── Sentiment label ───────────────────────────────────────────────
        sentiment_label = (
            "positive" if sentiment_score > 0.15
            else "negative" if sentiment_score < -0.15
            else "neutral"
        )

        # ── Emotion classification ────────────────────────────────────────
        tokens = re.findall(r"\b\w+\b", input_text.lower())

        if self.emotion_mode == "transformer" and self._emotion_pipe is not None:
            emotion_scores = self._get_transformer_emotions(input_text)
        else:
            emotion_scores = _rule_emotion(tokens, sentiment_score)

        # Apply softmax for well-calibrated distribution
        emotion_scores = _softmax(emotion_scores)
        emotion_label  = _pick_emotion_label(emotion_scores)

        return {
            "sentiment_score":   sentiment_score,
            "sentiment_label":   sentiment_label,
            "emotion_label":     emotion_label,
            "emotion_scores":    emotion_scores,
            "confidence":        round(raw_confidence, 4),
            "sarcasm_detected":  sarcasm_detected,
            "sarcasm_score":     round(sarcasm_score, 4),
            "model_used":        self.model_name if self._sentiment_pipe else "rule_based_fallback",
            "preprocessing": {
                "original_text":      text,
                "cleaned_text":       input_text,
                "slang_replacements": prep_result.get("slang_replacements", {}),
                "normalizations":     prep_result.get("normalizations", []),
            },
        }

    def analyze_batch(self, messages: list[str]) -> list[Dict]:
        """Analyze a list of messages. Returns list of result dicts."""
        return [self.analyze(msg) for msg in messages]

    def to_json(self, text: str, indent: int = 2) -> str:
        """Convenience: analyze and return JSON string."""
        return json.dumps(self.analyze(text), indent=indent, ensure_ascii=False)

    def reset_context(self) -> None:
        """Reset the preprocessor's context window (call between players/sessions)."""
        if self._preprocessor:
            self._preprocessor.reset_context()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        self._loaded = True  # mark as attempted regardless of outcome
        try:
            from transformers import pipeline  # type: ignore[import]
            print(f"[TransformerSentiment] Loading sentiment model: {self.model_name} ...")
            self._sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=128,
            )
            print(f"[TransformerSentiment] Model loaded successfully.")

            if self.emotion_mode == "transformer":
                emo_model = "j-hartmann/emotion-english-distilroberta-base"
                print(f"[TransformerSentiment] Loading emotion model: {emo_model} ...")
                self._emotion_pipe = pipeline(
                    "text-classification",
                    model=emo_model,
                    device=self.device,
                    top_k=None,
                    truncation=True,
                    max_length=128,
                )
                print(f"[TransformerSentiment] Emotion model loaded.")

        except ImportError:
            print(
                "[TransformerSentiment] WARNING: 'transformers' not installed. "
                "Falling back to rule-based sentiment.\n"
                "  Install: pip install transformers torch"
            )
        except Exception as exc:
            print(f"[TransformerSentiment] Model load error: {exc}. Using rule-based fallback.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _get_sentiment(self, text: str) -> tuple[float, float]:
        """
        Returns (sentiment_score ∈ [-1,+1], confidence ∈ [0,1]).
        Falls back to rule-based if pipeline is unavailable.
        """
        if self._sentiment_pipe is None:
            return self._rule_sentiment(text)

        try:
            result     = self._sentiment_pipe(text)[0]
            label      = result["label"]
            confidence = float(result["score"])

            # nlptown star rating model
            if label in _STAR_TO_SCORE:
                score = _STAR_TO_SCORE[label]
                # Weight by confidence: partially toward zero for low-confidence preds
                score = score * (0.5 + 0.5 * confidence)
                return score, confidence

            # Binary / ternary models
            upper_label = label.upper()
            if upper_label in _BINARY_LABEL_MAP:
                score = _BINARY_LABEL_MAP[upper_label] * confidence
                return score, confidence

            # Unknown label → fallback
            return self._rule_sentiment(text)

        except Exception as exc:
            print(f"[TransformerSentiment] Inference error: {exc}")
            return self._rule_sentiment(text)

    @staticmethod
    def _rule_sentiment(text: str) -> tuple[float, float]:
        """
        Minimal rule-based polarity as fallback.
        Returns (score, confidence=0.5).
        """
        from utils.nlp_extractor import NLPExtractor
        extractor = NLPExtractor(mode="rule_based")
        feats = extractor.extract(text)
        return feats["polarity"], 0.5

    def _get_transformer_emotions(self, text: str) -> Dict[str, float]:
        """
        Map j-hartmann model output to our 4-label schema.
        j-hartmann labels: anger, disgust, fear, joy, neutral, sadness, surprise
        """
        try:
            raw = self._emotion_pipe(text)[0]  # list of {label, score}
            mapping = {"anger": 0.0, "frustration": 0.0, "joy": 0.0, "neutral": 0.0}
            for item in raw:
                lbl   = item["label"].lower()
                score = float(item["score"])
                if lbl == "anger":
                    mapping["anger"] += score
                elif lbl in ("disgust", "fear"):
                    mapping["frustration"] += score * 0.7
                elif lbl == "sadness":
                    mapping["frustration"] += score * 0.5
                elif lbl == "joy":
                    mapping["joy"] += score
                elif lbl == "surprise":
                    mapping["joy"] += score * 0.3
                elif lbl == "neutral":
                    mapping["neutral"] += score
            return mapping
        except Exception as exc:
            print(f"[TransformerSentiment] Emotion pipeline error: {exc}")
            return {"anger": 0.0, "frustration": 0.0, "joy": 0.0, "neutral": 1.0}

    # ------------------------------------------------------------------
    # Empty result guard
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(text: str) -> Dict:
        return {
            "sentiment_score":   0.0,
            "sentiment_label":   "neutral",
            "emotion_label":     "neutral",
            "emotion_scores":    {"anger": 0.0, "frustration": 0.0, "joy": 0.0, "neutral": 1.0},
            "confidence":        0.0,
            "sarcasm_detected":  False,
            "sarcasm_score":     0.0,
            "model_used":        "none",
            "preprocessing": {
                "original_text":      text,
                "cleaned_text":       text,
                "slang_replacements": {},
                "normalizations":     [],
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton (rule-based by default, zero startup cost)
# ---------------------------------------------------------------------------
#: Use this for fast imports across the system.
#: Replace with TransformerSentimentAnalyzer(model_name=...) if you want BERT.
sentiment_analyzer = TransformerSentimentAnalyzer(
    model_name=TransformerSentimentAnalyzer.DEFAULT_MODEL,
    emotion_mode="rule",
    preprocess=True,
    sarcasm_adjust=True,
)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEST_CASES = [
        ("noob ruined the game wtf!!",              "anger/frustration"),
        ("SOOOO close again fml",                   "frustration + repeated char"),
        ("gg nice round everyone",                  "joy/positive"),
        ("yeah right so skilled lol",               "sarcasm"),
        ("omg this lag is IMPOSSIBLE!!",            "frustration + slang"),
        ("I QUIT this trash is rigged af!!!",       "high anger + slang"),
        ("clutch play nice one pro move",           "joy/positive"),
        ("not bad, almost had it",                  "negated positive"),
        ("idk maybe try a flank?",                  "uncertainty/strategic"),
        ("",                                        "empty input"),
    ]

    print("=" * 72)
    print("  TransformerSentimentAnalyzer — Self Test (rule_based fallback)")
    print("=" * 72)

    analyzer = TransformerSentimentAnalyzer(preprocess=True, sarcasm_adjust=True)

    for text, label in TEST_CASES:
        result = analyzer.analyze(text)
        print(f"\n[{label}]")
        print(f"  Input     : {text!r}")
        print(f"  Cleaned   : {result['preprocessing']['cleaned_text']!r}")
        if result["preprocessing"]["slang_replacements"]:
            print(f"  Slang     : {result['preprocessing']['slang_replacements']}")
        print(f"  Sentiment : {result['sentiment_label']} ({result['sentiment_score']:+.3f})  "
              f"conf={result['confidence']:.2f}")
        print(f"  Emotion   : {result['emotion_label']}  "
              f"scores={result['emotion_scores']}")
        if result["sarcasm_detected"]:
            print(f"  Sarcasm   : DETECTED (score={result['sarcasm_score']:.2f})")
        print(f"  Model     : {result['model_used']}")
