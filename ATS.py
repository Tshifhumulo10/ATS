from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')

st.title("Application Tracking System")

uploaded_CV  = st.text_area("Paste your CV here:", height=100)
uploaded_job = st.text_area("Paste  job description:",height=100)
button_clicked = st.button("Score", key="predict_button", kwargs={"style": "background-color: red; color: red"})

if button_clicked:
    def ATS(uploaded_CV, uploaded_job):
        
        if uploaded_CV is None or uploaded_job is None:
            return "Please insert text in both fields."
        else:
       
            text_cv= [uploaded_CV]
            text_job= [uploaded_job]

            # Fit and transform the vectorizer on individual documents
            tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
            tfidf_cv = tfidf.fit_transform(text_job)
            tfidf_job = tfidf.transform(text_cv)

            # Calculate cosine similarity
            ats = cosine_similarity(tfidf_cv, tfidf_job)[0][0]
            ats *= 100

            work_tokens = set(word_tokenize(uploaded_job.lower()))
            cv_tokens = set(word_tokenize(uploaded_CV.lower()))

            stop_words = set(stopwords.words('english'))
            work_tokens = work_tokens.difference(stop_words)
            cv_tokens = cv_tokens.difference(stop_words)
        
            punctuation = set(string.punctuation)
            work_tokens = work_tokens.difference(punctuation)
            cv_tokens = cv_tokens.difference(punctuation)

            missing_words = work_tokens.difference(cv_tokens)
            
            return ats, missing_words
        
    ats, missing words = ATS(uploaded_CV, uploaded_job)
    st.subheader("SCORE")
    st.write(ats)
    st.subheader("Missing Words")
    st.write(missing_words)
    
    
   
