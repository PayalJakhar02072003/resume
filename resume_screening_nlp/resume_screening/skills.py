"""
Skill extraction using a curated lexicon (good for student projects: no ML model required).

Extend SKILL_LEXICON for your domain (data science, web dev, etc.).
"""

from __future__ import annotations

import re
from typing import Iterable

# Curated list — order does not matter; matching is case-insensitive.
SKILL_LEXICON: tuple[str, ...] = (
    # Languages & core
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "go",
    "rust",
    "ruby",
    "php",
    "scala",
    "r",
    "sql",
    "html",
    "css",
    # Data & ML
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "opencv",
    "xgboost",
    "lightgbm",
    "statistics",
    "a/b testing",
    "ab testing",
    "data analysis",
    "data engineering",
    "etl",
    "spark",
    "hadoop",
    "kafka",
    "airflow",
    "dbt",
    # MLOps / cloud
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "mlflow",
    "ci/cd",
    "github actions",
    # Web / APIs
    "fastapi",
    "flask",
    "django",
    "react",
    "node.js",
    "nodejs",
    "rest api",
    "graphql",
    # Databases
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "snowflake",
    "elasticsearch",
)


def extract_skills(text: str, lexicon: Iterable[str] = SKILL_LEXICON) -> list[str]:
    """
    Find skills from lexicon that appear in the resume text.

    Multi-word skills are matched first to avoid partial duplicates.
    Returns a de-duplicated list preserving lexicon order.
    """
    if not text or not str(text).strip():
        return []

    lower = str(text).lower()
    # Sort lexicon by length descending so "machine learning" wins over "learning"
    sorted_skills = sorted({s.lower().strip() for s in lexicon}, key=len, reverse=True)
    found: list[str] = []
    seen: set[str] = set()

    for skill in sorted_skills:
        # Word-boundary style check for single tokens; phrase uses substring with spaces/punctuation boundaries
        if " " in skill or "/" in skill or "." in skill:
            pattern = re.escape(skill)
            if re.search(pattern, lower):
                key = skill
                if key not in seen:
                    seen.add(key)
                    found.append(skill)
        else:
            pat = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
            if re.search(pat, lower):
                if skill not in seen:
                    seen.add(skill)
                    found.append(skill)

    # Re-sort found by appearance order in original lexicon for stable UI
    order = {s: i for i, s in enumerate(SKILL_LEXICON)}
    found.sort(key=lambda s: order.get(s.lower(), 999))
    return found
