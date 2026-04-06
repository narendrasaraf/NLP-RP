import re
import math

class EmotionAnalyzer:
    """
    A lightweight, rule-based real-time sentiment and emotion analyzer.
    Requires no external dependencies and runs extremely fast.
    """
    def __init__(self):
        # Base Lexicons
        self.positive_words = {'good', 'great', 'awesome', 'amazing', 'love', 'happy', 'excellent', 'fantastic', 'yes', 'perfect', 'thanks', 'thank', 'glad', 'yay', 'nice', 'cool', 'win', 'easy', 'fun', 'challenge', 'better', 'finally'}
        self.negative_words = {'bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'stupid', 'no', 'worst', 'fail', 'wrong', 'sucks', 'poor', 'impossible', 'broken', 'trash', 'quit', 'rage', 'garbage', 'unfair', 'useless', 'bug', 'hard', 'difficult', 'frustrating', 'annoying', 'stuck', 'died', 'laggy', 'cheating', 'ridiculous', 'losing', 'failed', 'missed', 'nope'}
        
        # Modifiers
        self.negations = {'not', "don't", 'dont', "can't", 'cant', 'cannot', 'never', "didn't", 'didnt', "isn't", 'isnt', "aren't", 'arent', "won't", 'wont'}
        self.intensifiers = {'very', 'really', 'extremely', 'absolutely', 'so', 'too', 'completely', 'totally', 'fucking', 'super'}
        
        # Specific Emotion Lexicons
        self.anger_words = {'angry', 'furious', 'mad', 'hate', 'stupid', 'idiot', 'rage', 'hell', 'wtf', 'damn', 'bullshit', 'fuck', 'shit', 'pissed', 'annoying', 'trash', 'garbage', 'quit'}
        self.frustration_words = {'ugh', 'annoyed', 'stuck', 'broken', 'why', 'tired', 'sigh', 'again', 'confusing', 'hard', 'error', 'failing', 'crash', 'crashing', 'bug', 'impossible', 'unfair', 'laggy', 'difficult', 'frustrating'}
        
        # Confidence Lexicons
        self.confidence_words = {'definitely', 'absolutely', 'sure', 'certain', 'guarantee', 'clear', 'obvious', 'know', 'exactly', 'always', '100%', 'easy'}
        self.uncertainty_words = {'maybe', 'perhaps', 'guess', 'think', 'probably', 'might', 'could', 'unsure', 'sometimes', 'possibly', 'idk', 'hopefully'}

    def analyze(self, text: str) -> dict:
        """
        Analyzes text and returns:
        - polarity: -1.0 to 1.0 (Negative to Positive)
        - intensity: 0.0 to 1.0 (Calm to Highly Intense)
        - anger: 0.0 to 1.0 (Probability/Confidence of Anger)
        - frustration: 0.0 to 1.0 (Probability/Confidence of Frustration)
        - confidence: 0.0 to 1.0 (How confident the speaker sounds)
        """
        if not text or not isinstance(text, str):
            return {"polarity": 0.0, "intensity": 0.0, "anger": 0.0, "frustration": 0.0, "confidence": 0.5}

        # Tokenize ignoring basic punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        raw_words = text.split()
        total_words = len(words) if len(words) > 0 else 1
        
        # 1. Feature Extraction: Intensity markers (Punctuation & Caps)
        caps_count = sum(1 for w in raw_words if w.isupper() and len(re.sub(r'[^A-Z]', '', w)) > 1)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Punctuation/Caps contribution to intensity
        base_intensity = min(1.0, (caps_count * 0.15) + (exclamation_count * 0.2) + (question_count * 0.1))

        # 2. Variables for lexical scoring
        score_pos, score_neg = 0.0, 0.0
        score_anger, score_frust = 0.0, 0.0
        score_conf, score_unconf = 0.0, 0.0
        
        intensity_multiplier = 1.0
        is_negated = False
        
        for word in words:
            # Handle modifiers
            if word in self.negations:
                is_negated = True
                continue
            if word in self.intensifiers:
                intensity_multiplier += 0.5
                continue
                
            # Current word value taking modifiers into account
            val = 1.0 * intensity_multiplier
            if is_negated:
                val *= -0.5 # Negation flips polarity and dampens intensity slightly
                
            # Polarity Scoring
            if word in self.positive_words:
                if is_negated: score_neg += abs(val)
                else: score_pos += val
            elif word in self.negative_words:
                if is_negated: score_pos += abs(val) * 0.5 # "not bad" is mildly positive
                else: score_neg += val
                
            # Emotion Scoring
            if word in self.anger_words:
                score_anger += val if not is_negated else 0
            if word in self.frustration_words:
                score_frust += val if not is_negated else 0
                
            # Confidence Scoring
            if word in self.confidence_words:
                if is_negated: score_unconf += val
                else: score_conf += val
            if word in self.uncertainty_words:
                if is_negated: score_conf += val * 0.5
                else: score_unconf += val
                
            # Reset modifiers after applying them to a content word
            is_negated = False
            intensity_multiplier = 1.0

        # 3. Final Calculations
        
        # Polarity: Normalized between -1 and 1
        raw_polarity = (score_pos - score_neg) / (score_pos + score_neg + 2.0)
        
        # Intensity: Combines base text formatting intensity + lexical emotional weight
        lexical_weight = (score_pos + score_neg + score_anger + score_frust) / total_words
        intensity = min(1.0, base_intensity + lexical_weight)
        
        # Anger & Frustration Probabilities
        def calc_prob(score, format_boost=0.0):
            if score <= 0: return format_boost * 0.5
            raw = (score / math.ceil(total_words/3)) * 1.5 + format_boost
            return min(1.0, max(0.0, raw))

        anger_prob = calc_prob(score_anger, (caps_count * 0.1 + exclamation_count * 0.1))
        frust_prob = calc_prob(score_frust, (caps_count * 0.05 + question_count * 0.1 + exclamation_count * 0.05))
        
        # Confidence: Base 0.5 (Neutral). Adjusted by confidence vs uncertainty words
        base_confidence = 0.5
        conf_diff = score_conf - score_unconf
        confidence_prob = max(0.0, min(1.0, base_confidence + (conf_diff / total_words) * 1.5))

        return {
            "polarity": float(round(raw_polarity, 3)),
            "intensity": float(round(intensity, 3)),
            "anger": float(round(anger_prob, 3)),
            "frustration": float(round(frust_prob, 3)),
            "confidence": float(round(confidence_prob, 3))
        }

# Singleton instance
emotion_analyzer = EmotionAnalyzer()
