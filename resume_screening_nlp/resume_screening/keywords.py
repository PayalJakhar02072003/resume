"""
Keyword overlap between job description and resume (for Streamlit highlighting).

Uses the same preprocessing idea: significant tokens in JD that also appear in resume.
"""

from __future__ import annotations

from .preprocessing import preprocess_for_highlight


def matching_keywords(job_text: str, resume_text: str, max_terms: int = 40) -> list[str]:
    """
    Return sorted unique tokens present in both documents (after stopword removal).

    Limited to `max_terms` for readability in the UI.
    """
    jd_tokens = set(preprocess_for_highlight(job_text))
    res_tokens = set(preprocess_for_highlight(resume_text))
    common = sorted(jd_tokens & res_tokens)
    return common[:max_terms]


def highlight_resume_html(resume_text: str, keywords: set[str]) -> str:
    """
    Wrap occurrences of keywords in <mark> for st.markdown(unsafe_allow_html=True).

    Simple token-based highlight on word boundaries (case-insensitive).
    """
    import html
    import re

    if not resume_text.strip():
        return "<p><em>Empty text</em></p>"

    escaped = html.escape(resume_text)
    if not keywords:
        return f"<p style='white-space: pre-wrap;'>{escaped}</p>"

    # Longest keywords first to prefer multi-word where applicable
    kws = sorted({k for k in keywords if k}, key=len, reverse=True)
    pattern = "|".join(re.escape(k) for k in kws)
    if not pattern:
        return f"<p style='white-space: pre-wrap;'>{escaped}</p>"

    def repl(m: re.Match[str]) -> str:
        return f"<mark style='background-color:#fff3a3;padding:0 2px;'>{m.group(0)}</mark>"

    highlighted = re.sub(f"(?i)(?<![a-z0-9])({pattern})(?![a-z0-9])", repl, escaped)
    return f"<div style='white-space: pre-wrap; font-size:0.95rem;'>{highlighted}</div>"
