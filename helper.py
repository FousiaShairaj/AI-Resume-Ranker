import PyPDF2
import docx
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

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
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def calculate_similarity(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer().fit_transform(documents)
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return score