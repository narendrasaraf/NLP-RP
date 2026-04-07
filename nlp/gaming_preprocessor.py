"""
nlp/gaming_preprocessor.py
---------------------------
Pre-processing pipeline specifically tuned for informal gaming chat.

Stages (applied in order):
  1. Unicode normalisation
  2. Repeated character collapsing  (e.g. "sooooo" -> "so")
  3. Slang normalisation            (e.g. "noob"   -> "newbie", "wtf" -> "what the heck")
  4. Sarcasm probability scoring    (context-pattern analysis)

The preprocessor is stateless except for a rolling ``context_window`` used
by the sarcasm detector.  Reset it by calling ``reset_context()``.

Usage:
    from nlp.gaming_preprocessor import GamingPreprocessor

    prep = GamingPreprocessor()
    result = prep.process("omg this is SOOOO easy lmao nice one")
    # result = {
    #   "cleaned_text": "oh my god this is so easy laughing out loud nice one",
    #   "original_text": "omg this is SOOOO easy lmao nice one",
    #   "sarcasm_score": 0.72,
    #   "sarcasm_detected": True,
    #   "normalizations": ["repeated_chars", "slang"],
    #   "slang_replacements": {"omg": "oh my god", "lmao": "laughing out loud"},
    # }
"""

from __future__ import annotations

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Slang / abbreviation dictionary  (gaming + internet + profanity euphemisms)
# ---------------------------------------------------------------------------
_SLANG_MAP: Dict[str, str] = {
    # ── Profanity / frustration abbreviations ───────────────────────────────
    "wtf":    "what the heck",
    "wth":    "what the heck",
    "omg":    "oh my god",
    "omfg":   "oh my god",
    "stfu":   "shut up",
    "gtfo":   "get out",
    "ffs":    "for crying out loud",
    "fml":    "my life sucks",
    "bs":     "nonsense",
    "af":     "as heck",
    "af":     "very much",

    # ── Skill / performance slang ────────────────────────────────────────────
    "noob":   "newbie",
    "n00b":   "newbie",
    "nub":    "newbie",
    "newb":   "newbie",
    "pro":    "professional",
    "tryhard":"try-hard player",
    "smurf":  "high-skill disguised player",
    "toxic":  "hostile",
    "troll":  "disruptive player",
    "griefing":"sabotaging teammates",
    "ratting":"hiding to survive",
    "camping":"staying in one spot",
    "rush":   "charge aggressively",
    "bot":    "automated player",
    "ez":     "easy",
    "gg":     "good game",
    "ggwp":   "good game well played",
    "glhf":   "good luck have fun",
    "wp":     "well played",

    # ── Reaction / emotion tokens ────────────────────────────────────────────
    "lol":    "laughing",
    "lmao":   "laughing out loud",
    "lmfao":  "laughing very hard",
    "rofl":   "rolling on the floor laughing",
    "xd":     "laughing face",
    "kek":    "laughing",
    "bruh":   "come on",
    "smh":    "shaking my head",
    "ngl":    "not going to lie",
    "tbh":    "to be honest",
    "imo":    "in my opinion",
    "imho":   "honestly",
    "irl":    "in real life",
    "idk":    "I do not know",
    "idc":    "I do not care",
    "nvm":    "never mind",
    "rn":     "right now",
    "tho":    "though",
    "bc":     "because",
    "cuz":    "because",
    "pls":    "please",
    "plz":    "please",
    "thx":    "thanks",
    "ty":     "thank you",
    "np":     "no problem",
    "yw":     "you are welcome",
    "gonna":  "going to",
    "wanna":  "want to",
    "gotta":  "got to",
    "kinda":  "kind of",
    "sorta":  "sort of",
    "ur":     "your",
    "u":      "you",
    "r":      "are",
    "b4":     "before",
    "gr8":    "great",
    "h8":     "hate",
    "l8r":    "later",
    "omw":    "on my way",
    "bbl":    "be back later",
    "brb":    "be right back",
    "afk":    "away from keyboard",
    "gg ez":  "easy game",

    # ── Game-specific terms normalized to sentiment-carrying words ───────────
    "aimbot":    "cheating with aim assist",
    "lag":       "connection delay",
    "lagging":   "experiencing connection delay",
    "laggy":     "experiencing connection delay",
    "lag spike":  "sudden connection delay",
    "trash":     "terrible",
    "garbage":   "terrible",
    "busted":    "very powerful",
    "op":        "overpowered",
    "nerf":      "weaken",
    "buff":      "strengthen",
    "meta":      "optimal strategy",
    "pwned":     "dominated",
    "rekt":      "defeated badly",
    "owned":     "defeated badly",
    "clutch":    "impressive under pressure",
    "choke":     "fail under pressure",
    "diff":      "skill difference",
    "carry":     "leading the team to victory",
    "feed":      "dying repeatedly helping enemy",
    "feeder":    "player dying repeatedly",
    "afk":       "away from keyboard inactive",
    "cap":       "false statement",
    "no cap":    "honestly truthfully",
    "salty":     "upset and bitter",
    "tilted":    "frustrated and playing poorly",
    "toxic":     "aggressively hostile",
    "flame":     "verbally attack",
    "flaming":   "verbally attacking",
    "rage quit": "angrily leaving the game",
}

# Sorted by descending token length so longer phrases match first
_SLANG_PHRASES = sorted(_SLANG_MAP.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Sarcasm context patterns
# ---------------------------------------------------------------------------
# Each rule: (regex pattern, weight contribution 0-1)
_SARCASM_PATTERNS: List[tuple[re.Pattern, float]] = [
    # "oh great / oh how wonderful / oh fantastic" — feigned approval
    (re.compile(r"\boh\s+(great|fantastic|wonderful|perfect|sure|yeah|right)\b", re.IGNORECASE), 0.4),
    # "nice / great / awesome job" immediately after negative context
    (re.compile(r"\b(nice|great|awesome|amazing|wonderful)\b.{0,20}\b(failed?|lost|died?|rekt|terrible|horrible)\b", re.IGNORECASE), 0.45),
    # Positive word + "lol/lmao/xd" combo (dismissive praise)
    (re.compile(r"\b(great|amazing|awesome|perfect|easy|pro|genius)\b.{0,30}(lol|lmao|lmfao|xd|kek)\b", re.IGNORECASE), 0.35),
    # "sure" / "definitely" / "totally" with strong ironic markers
    (re.compile(r"\b(sure|totally|definitely|absolutely)\b.{0,25}\b(not|never|nope|no way)\b", re.IGNORECASE), 0.5),
    # "yeah right" / "as if" patterns
    (re.compile(r"\b(yeah\s+right|as\s+if|sure\s+buddy|oh\s+really)\b", re.IGNORECASE), 0.55),
    # Quotes around praise: '"great job"' — implicit mockery
    (re.compile(r'"(great|amazing|perfect|excellent|pro)\s*(job|play|move|strat)?"', re.IGNORECASE), 0.4),
    # "wow .. so .. [positive]" — ironic exaggeration pattern
    (re.compile(r"\bwow\b.{0,30}\bso\b.{0,20}\b(smart|skilled?|good|great|talented)\b", re.IGNORECASE), 0.45),
    # "thanks for the [negative outcome]" — sarcastic gratitude
    (re.compile(r"\bthanks?\b.{0,20}\b(nothing|dying|loss|ruining|killing|feeding)\b", re.IGNORECASE), 0.5),
    # Emoji sarcasm markers (text-form)
    (re.compile(r"\b(clap|slow\s?clap|7\/7|10\/10)\b", re.IGNORECASE), 0.3),
]

# Extra weight if ALL_CAPS is detected (shouting → heightens sarcasm probability)
_CAPS_SARCASM_BOOST = 0.15


# ---------------------------------------------------------------------------
# Repeated character pattern
# ---------------------------------------------------------------------------
# Collapse 3+ consecutive identical chars to TWO (preserves "oo" in "good")
_REPEAT_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GamingPreprocessor:
    """
    Context-aware preprocessing pipeline for gaming chat text.

    Args:
        context_window_size: Number of recent messages retained for
                             cross-message sarcasm detection (default 5).
        slang_map: Custom slang dictionary that *overrides* the built-in
                   one for specified keys (merged, not replaced).
    """

    def __init__(
        self,
        context_window_size: int = 5,
        slang_map: Dict[str, str] | None = None,
    ) -> None:
        self._context: List[str] = []
        self._window_size = context_window_size

        # Merge custom slang on top of defaults
        self._slang = dict(_SLANG_MAP)
        if slang_map:
            self._slang.update(slang_map)

        # Rebuild sorted keys after merge
        self._phrases = sorted(self._slang.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> Dict:
        """
        Full preprocessing pipeline.

        Args:
            text: Raw player chat message.

        Returns:
            {
              "cleaned_text":       str,
              "original_text":      str,
              "sarcasm_score":      float ∈ [0.0, 1.0],
              "sarcasm_detected":   bool,
              "normalizations":     list[str],  # stages that fired
              "slang_replacements": dict,        # {original_token: replacement}
            }
        """
        if not text or not isinstance(text, str):
            return self._empty_result(text or "")

        original = text
        normalizations: List[str] = []
        slang_replacements: Dict[str, str] = {}

        # Stage 1: Unicode → ASCII approximation
        cleaned = self._unicode_normalise(text)

        # Stage 2: Collapse repeated characters
        collapsed = self._collapse_repeats(cleaned)
        if collapsed != cleaned:
            normalizations.append("repeated_chars")
        cleaned = collapsed

        # Stage 3: Slang normalisation
        normalized, replacements = self._normalize_slang(cleaned)
        if replacements:
            normalizations.append("slang")
            slang_replacements = replacements
        cleaned = normalized

        # Stage 4: Sarcasm scoring (uses original + context window)
        sarcasm_score = self._score_sarcasm(original, cleaned)

        # Update context window (use cleaned text to help future sarcasm checks)
        self._update_context(cleaned)

        return {
            "cleaned_text":       cleaned.strip(),
            "original_text":      original,
            "sarcasm_score":      round(sarcasm_score, 4),
            "sarcasm_detected":   sarcasm_score >= 0.40,
            "normalizations":     normalizations,
            "slang_replacements": slang_replacements,
        }

    def reset_context(self) -> None:
        """Clear rolling context window (call between sessions/players)."""
        self._context.clear()

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    @staticmethod
    def _unicode_normalise(text: str) -> str:
        """Strip zero-width chars, normalize curly quotes, etc."""
        text = text.replace("\u200b", "").replace("\u200c", "")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return text

    @staticmethod
    def _collapse_repeats(text: str) -> str:
        """
        Collapse 3+ repeated characters to 2 (preserves expressive elongation
        as a binary flag without destroying sentiment-carrying words).

        Examples:
            "sooooo" -> "soo"  (still different from "so", carries emphasis)
            "noooo"  -> "noo"
            "yessss" -> "yess"
            "!!!!"   -> "!!"   (punctuation also collapsed)
        """
        return _REPEAT_PATTERN.sub(r"\1\1", text)

    def _normalize_slang(self, text: str) -> tuple[str, Dict[str, str]]:
        """
        Replace slang tokens/phrases with their expanded forms.

        Multi-word phrases are checked first (longest-match).
        Single tokens are then matched word-by-word.
        """
        replacements: Dict[str, str] = {}

        # Multi-word phrase pass (iterate over sorted phrase list)
        for phrase in self._phrases:
            if " " not in phrase:
                continue  # handle single words in next pass
            if phrase.lower() in text.lower():
                expanded = self._slang[phrase]
                # Case-insensitive replacement
                new_text = re.sub(
                    re.escape(phrase),
                    expanded,
                    text,
                    flags=re.IGNORECASE,
                )
                if new_text != text:
                    replacements[phrase] = expanded
                    text = new_text

        # Single-token pass
        tokens = text.split()
        new_tokens: List[str] = []
        for tok in tokens:
            clean_tok = tok.strip(".,!?;:\"'").lower()
            if clean_tok in self._slang:
                expanded = self._slang[clean_tok]
                replacements[clean_tok] = expanded
                # Preserve trailing punctuation
                suffix = re.sub(r"\w", "", tok[-1]) if tok[-1] in ".,!?;:" else ""
                new_tokens.append(expanded + suffix)
            else:
                new_tokens.append(tok)

        return " ".join(new_tokens), replacements

    def _score_sarcasm(self, original: str, cleaned: str) -> float:
        """
        Compute sarcasm probability via pattern matching + context signals.

        Algorithm:
          1. Match each sarcasm pattern against the message+cleaned text.
          2. Accumulate evidence weights (capped to prevent runaway).
          3. Add ALL-CAPS boost if text has excessive capitalisation.
          4. Cross-message signal: if previous message was very negative
             and current message is very positive → contrast sarcasm flag.
          5. Sigmoid-squash the raw score to [0, 1].
        """
        combined = f"{original} {cleaned}"
        raw_score = 0.0

        for pattern, weight in _SARCASM_PATTERNS:
            if pattern.search(combined):
                raw_score += weight

        # ALL-CAPS boost
        if original.upper() == original and any(c.isalpha() for c in original):
            raw_score += _CAPS_SARCASM_BOOST

        # Cross-message contrast (previous very negative → current positive = sarcasm)
        if self._context:
            prev_lower = self._context[-1].lower()
            prev_has_neg = any(
                w in prev_lower
                for w in ("failed", "lost", "died", "terrible", "awful", "rekt", "trash")
            )
            curr_has_pos = any(
                w in cleaned.lower()
                for w in ("great", "nice", "amazing", "awesome", "well done", "good job")
            )
            if prev_has_neg and curr_has_pos:
                raw_score += 0.30

        # Sigmoid squash: s(x) = 1 / (1 + e^(-3*x+1))  — inflects at ~0.33
        import math
        score = 1.0 / (1.0 + math.exp(-3.0 * raw_score + 1.0))

        return float(min(1.0, max(0.0, score)))

    def _update_context(self, cleaned: str) -> None:
        self._context.append(cleaned)
        if len(self._context) > self._window_size:
            self._context.pop(0)

    @staticmethod
    def _empty_result(original: str) -> Dict:
        return {
            "cleaned_text":       original,
            "original_text":      original,
            "sarcasm_score":      0.0,
            "sarcasm_detected":   False,
            "normalizations":     [],
            "slang_replacements": {},
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
gaming_preprocessor = GamingPreprocessor()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    prep = GamingPreprocessor()

    TEST_MSGS = [
        "omg this noob is SOOOO bad wtf",
        "gg ez lmao nice one bruh",
        "I QUIT this trash is rigged af!!",
        "oh great another lag spike smh",
        "yeah right bro so skilled lol",
        "soooo close yet another loss fml",
        "noooo why does this always happen wtf!!!",
        "nice job feeding the entire game lmao",
        "definitely not tilted rn lol",
        "bruh this is the best game ever (after dying 10 times)",
    ]

    print("=" * 70)
    print("  GamingPreprocessor — Self Test")
    print("=" * 70)

    for msg in TEST_MSGS:
        result = prep.process(msg)
        print(f"\n  ORIG   : {result['original_text']}")
        print(f"  CLEAN  : {result['cleaned_text']}")
        print(f"  STAGES : {result['normalizations']}")
        print(f"  SARCASM: {result['sarcasm_score']:.2f} => {result['sarcasm_detected']}")
        if result["slang_replacements"]:
            print(f"  SLANG  : {result['slang_replacements']}")
