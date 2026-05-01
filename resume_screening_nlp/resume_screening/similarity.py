"""
TF-IDF vectorization + cosine similarity between job description and each resume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarities(
    job_description_processed: str,
    resumes_processed: list[str],
) -> np.ndarray:
    """
    Fit TF-IDF on [job_description] + all resumes (shared vocabulary), then
    return cosine similarity of the job row to each resume row (shape: n_resumes,).

    Preprocessed strings should already be lowercase / cleaned per your pipeline.
    """
    if not job_description_processed.strip():
        raise ValueError("Job description is empty after preprocessing.")

    n = len(resumes_processed)
    if n == 0:
        return np.array([], dtype=float)

    corpus = [job_description_processed] + list(resumes_processed)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    jd_vec = matrix[0:1]
    resume_vecs = matrix[1:]
    sims = cosine_similarity(jd_vec, resume_vecs)[0]
    return np.clip(sims, 0.0, 1.0)


def build_ranking_dataframe(
    filenames: list[str],
    scores: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame with rank, filename, score (percentage)."""
    df = pd.DataFrame({"filename": filenames, "score": scores})
    df["score_pct"] = (df["score"] * 100).round(2)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
