import streamlit as st
from helper import extract_text, clean_text, calculate_similarity

st.title("💼 Recruiter Portal")

username = st.text_input("Recruiter Username")
password = st.text_input("Password", type="password")

if st.button("Login as Recruiter"):
    st.session_state['user'] = username
    st.success(f"Welcome Recruiter {username}!")

jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])
if jd_file:
    jd_text = extract_text(jd_file)
    st.text_area("Job Description Text", jd_text, height=200)

resume_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if jd_file and resume_files:
        jd_clean = clean_text(jd_text)
        for res in resume_files:
            res_text = extract_text(res)
            res_clean = clean_text(res_text)
            score = calculate_similarity(res_clean, jd_clean)
            st.write(f"**{res.name}** — Similarity Score: `{score:.2f}%`")
    else:
        st.warning("Please upload JD and resumes first.")