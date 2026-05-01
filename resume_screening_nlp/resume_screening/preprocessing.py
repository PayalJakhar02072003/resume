"""
Text preprocessing for resume / job-description comparison.

Steps: lowercase → remove punctuation → tokenize → remove English stopwords → rejoin.
"""

from __future__ import annotations

import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _ensure_nltk_stopwords() -> None:
    """Download NLTK stopwords once (safe to call multiple times)."""
    import nltk

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # Punkt tokenizer data (name varies slightly across NLTK versions)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def preprocess_text(text: str) -> str:
    """
    Normalize text for TF-IDF / overlap-based features.

    - Lowercase
    - Remove punctuation (keep intra-word apostrophes stripped with word tokens)
    - Remove NLTK English stopwords
    - Collapse whitespace
    """
    if not text or not str(text).strip():
        return ""

    _ensure_nltk_stopwords()
    raw = str(text).lower()
    # Replace punctuation with spaces (keeps word boundaries clear)
    cleaned = raw.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = word_tokenize(cleaned)
    stops = set(stopwords.words("english"))
    # Keep short tokens that are still technical (e.g., "c", "r") — optional filter
    filtered = [t for t in tokens if t not in stops and len(t) > 1]

    return " ".join(filtered)


def preprocess_for_highlight(text: str) -> list[str]:
    """
    Return lowercase tokens without stopwords (for keyword overlap), not joined.
    """
    processed = preprocess_text(text)
    return processed.split() if processed else []
