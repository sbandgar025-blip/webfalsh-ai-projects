import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening & Ranking System")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Multiple Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if uploaded_files and job_description:

    results = []

    for file in uploaded_files:
        resume_text = extract_text(file)

        documents = [job_description, resume_text]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        score = similarity[0][0] * 100

        # Accept / Reject rule
        if score >= 60:
            status = "Accepted"
        else:
            status = "Rejected"

        results.append({
            "Resume Name": file.name,
            "Match Score (%)": round(score,2),
            "Status": status
        })

    df = pd.DataFrame(results)

    df = df.sort_values(by="Match Score (%)", ascending=False)

    st.subheader("Candidate Ranking")

    st.dataframe(df)

    st.subheader("Top Candidate")

    st.success(df.iloc[0]["Resume Name"])