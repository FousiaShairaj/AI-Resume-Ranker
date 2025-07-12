import streamlit as st
from helper import extract_text

st.title("📄 Candidate Portal")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login as Candidate"):
    st.session_state['user'] = username
    st.success(f"Welcome {username}!")

resume_file = st.file_uploader("Upload Your Resume", type=['pdf', 'docx', 'txt'])
if resume_file:
    text = extract_text(resume_file)
    st.success("Resume uploaded and text extracted!")
    st.text_area("Extracted Resume Text", value=text[:500] + "...", height=200)