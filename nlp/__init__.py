"""
nlp/__init__.py
----------------
Package initializer for the NLP subpackage.

Exposes the three main entry points:
  - TransformerSentimentAnalyzer  : full transformer-backed analyzer
  - GamingPreprocessor            : gaming chat normalizer
  - SentimentComparison           : rule vs transformer contrast tool

Quick usage:
    from nlp import TransformerSentimentAnalyzer
    analyzer = TransformerSentimentAnalyzer()
    result   = analyzer.analyze("omg this lag is SOOOO bad wtf")
"""

from nlp.gaming_preprocessor    import GamingPreprocessor, gaming_preprocessor
from nlp.transformer_sentiment   import TransformerSentimentAnalyzer, sentiment_analyzer
from nlp.sentiment_comparison    import SentimentComparison

__all__ = [
    "GamingPreprocessor",
    "gaming_preprocessor",
    "TransformerSentimentAnalyzer",
    "sentiment_analyzer",
    "SentimentComparison",
]
