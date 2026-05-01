"""
Resume Screening System — Streamlit entrypoint.

Run from this folder (uses port 8502 via .streamlit/config.toml if 8501 is busy):
    streamlit run app.py

Override once: streamlit run app.py --server.port 8510

Deploy on Streamlit Community Cloud: set main file to app.py and Python 3.10+.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from resume_screening.keywords import highlight_resume_html, matching_keywords
from resume_screening.pdf_utils import extract_text_from_pdf
from resume_screening.preprocessing import preprocess_text
from resume_screening.similarity import build_ranking_dataframe, compute_tfidf_similarities
from resume_screening.skills import extract_skills


st.set_page_config(
    page_title="Resume Screener (NLP)",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def nltk_bootstrap() -> None:
    """Warm up NLTK data downloads once per session."""
    preprocess_text("init stopwords and punkt")


def read_uploaded_resume(uploaded_file) -> tuple[str, str]:
    """Return (raw_text, filename). Supports .pdf and .txt."""
    name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    if name.lower().endswith(".pdf"):
        return extract_text_from_pdf(raw_bytes), name
    return raw_bytes.decode("utf-8", errors="replace"), name


def main() -> None:
    st.title("📄 Resume Screening (TF-IDF + Cosine Similarity)")
    st.markdown(
        "Upload multiple resumes, paste a **job description**, and get **scores**, "
        "**ranking**, **extracted skills**, and optional **keyword highlights**."
    )

    nltk_bootstrap()

    col_left, col_right = st.columns((1, 1))
    with col_left:
        job_description = st.text_area(
            "Job description",
            height=220,
            placeholder="Paste the full job description here...",
        )
    with col_right:
        uploads = st.file_uploader(
            "Resumes (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

    run = st.button("Analyze resumes", type="primary")

    if not run:
        st.info("Enter a job description, upload one or more resumes, then click **Analyze resumes**.")
        return

    if not job_description.strip():
        st.warning("Please provide a non-empty job description.")
        return
    if not uploads:
        st.warning("Please upload at least one resume.")
        return

    # Load raw text from uploads
    raw_resumes: list[str] = []
    names: list[str] = []
    for f in uploads:
        text, name = read_uploaded_resume(f)
        raw_resumes.append(text)
        names.append(name)

    jd_proc = preprocess_text(job_description)
    res_proc = [preprocess_text(t) for t in raw_resumes]

    try:
        scores = compute_tfidf_similarities(jd_proc, res_proc)
    except ValueError as e:
        st.error(str(e))
        return

    ranking = build_ranking_dataframe(names, scores)

    st.subheader("Matching scores & ranking")
    st.dataframe(
        ranking[["rank", "filename", "score_pct"]].rename(
            columns={"score_pct": "match_score_%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart (Streamlit native chart)
    chart_df = ranking.sort_values("rank")[["filename", "score_pct"]].set_index("filename")
    st.bar_chart(chart_df)

    st.subheader("Extracted skills (lexicon-based)")
    skill_rows = []
    for name, raw in zip(names, raw_resumes, strict=True):
        skills = extract_skills(raw)
        skill_rows.append({"filename": name, "skills": ", ".join(skills) if skills else "—"})
    st.dataframe(pd.DataFrame(skill_rows), use_container_width=True, hide_index=True)

    st.subheader("Keyword overlap & highlights")
    st.caption("Terms that appear in both the job description and each resume (after stopword removal).")

    for name, raw in zip(names, raw_resumes, strict=True):
        kws = matching_keywords(job_description, raw)
        with st.expander(f"🔎 {name}", expanded=False):
            st.write("**Matching keywords:**", ", ".join(kws) if kws else "—")
            kw_set = set(kws)
            st.markdown(highlight_resume_html(raw, kw_set), unsafe_allow_html=True)

    # Optional: show sklearn cosine matrix snippet for learning (small footnote)
    with st.expander("Technical note (for reports)"):
        st.write(
            "- **Vectorizer:** TF-IDF with unigrams and bigrams, `sublinear_tf=True`.\n"
            "- **Similarity:** Cosine similarity between the job vector and each resume vector.\n"
            "- **Skills:** Dictionary match (extend list in `resume_screening/skills.py`)."
        )


if __name__ == "__main__":
    main()
