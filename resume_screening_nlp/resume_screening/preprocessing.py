"""
Text preprocessing for resume / job-description comparison.

Steps: lowercase → remove punctuation → tokenize → remove English stopwords → rejoin.

Uses scikit-learn's built-in English stopword set (no NLTK downloads) so Streamlit Cloud
deploys stay reliable — NLTK punkt/stopwords downloads often break cold starts on Cloud.
"""

from __future__ import annotations

import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# sklearn uses a frozenset of lowercase tokens
_STOP = set(ENGLISH_STOP_WORDS)


def _tokenize_simple(text: str) -> list[str]:
    """Alphanumeric tokens (handles resumes without external tokenizer data)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def preprocess_text(text: str) -> str:
    """
    Normalize text for TF-IDF / overlap-based features.

    - Lowercase
    - Remove punctuation
    - Tokenize (regex)
    - Remove English stopwords (same family as sklearn's vectorizers)
    - Collapse whitespace
    """
    if not text or not str(text).strip():
        return ""

    raw = str(text).lower()
    cleaned = raw.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = _tokenize_simple(cleaned)
    filtered = [t for t in tokens if t not in _STOP and len(t) > 1]

    return " ".join(filtered)


def preprocess_for_highlight(text: str) -> list[str]:
    """
    Return lowercase tokens without stopwords (for keyword overlap), not joined.
    """
    processed = preprocess_text(text)
    return processed.split() if processed else []
