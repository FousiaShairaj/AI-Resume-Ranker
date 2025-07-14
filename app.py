import streamlit as st
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_text(file):
    if file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + " "
        return text
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        return ' '.join([para.text for para in doc.paragraphs])
    elif file.name.endswith('.txt'):
        return str(file.read(), 'utf-8')
    return ""

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def calculate_similarity(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer().fit_transform(documents)
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return score

st.set_page_config(page_title="AI Resume Ranker", layout="centered")

st.title("🤖 AI Resume Ranker ")

st.sidebar.header("Upload Files")

resumes = {}
job_description_text = ""
feedback_data = []

jd_file = st.sidebar.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])
if jd_file:
    job_description_text = extract_text(jd_file)
    st.sidebar.success("Job Description Uploaded Successfully!")

uploaded_files = st.sidebar.file_uploader("Upload Resumes (Multiple Allowed)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
for file in uploaded_files:
    resumes[file.name] = extract_text(file)

if st.sidebar.button("Rank Resumes"):
    if not job_description_text or not resumes:
        st.warning("Please upload Job Description and at least one Resume.")
    else:
        st.subheader("📊 Resume Ranking Results")
        cleaned_jd = clean_text(job_description_text)
        results = []
        for name, text in resumes.items():
            cleaned_resume = clean_text(text)
            score = calculate_similarity(cleaned_resume, cleaned_jd)
            results.append((name, score))
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        for name, score in ranked:
            st.write(f"**{name}** — Similarity Score: `{score:.2f}%`")
        st.success("Ranking Completed!")

st.subheader("💡 Recruiter Feedback")
if len(resumes) > 0:
    selected_resume = st.selectbox("Select Resume for Feedback", list(resumes.keys()))
    feedback = st.text_input("Enter Feedback for Selected Resume")
    if st.button("Submit Feedback"):
        feedback_data.append({'resume': selected_resume, 'feedback': feedback})
        st.success("Feedback Saved!")

if st.button("View All Feedback"):
    if feedback_data:
        for record in feedback_data:
            st.write(f"**{record['resume']}** — Feedback: {record['feedback']}")
    else:
        st.info("No Feedback Submitted Yet.")
