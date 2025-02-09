import streamlit as st
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load NLP model (English)
nlp = spacy.load("en_core_web_sm")

# Predefined list of common skills
SKILLS_DB = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "Data Analysis",
    "Power BI", "Tableau", "Excel", "TensorFlow", "PyTorch", "NLP",
    "Statistics", "Data Visualization", "AWS", "Big Data", "R", "ETL",
    "Data Engineering", "Time Series Analysis", "Kubernetes", "Docker"
]

# ---------------------------- Streamlit UI Configuration ---------------------------- #
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stButton>button { 
            background-color: #4CAF50; 
            color: white; 
            font-size: 16px; 
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- Sidebar ---------------------------- #
with st.sidebar:
    st.image("my_logo.png", width=300)  # Add your logo
    st.title("📂 Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.write("💡 **Tip:** Upload your resume and paste a job description for analysis.")

# ---------------------------- Main Content ---------------------------- #
st.title("📄 AI Resume Analyzer")
st.write("Analyze your resume against a job description and get insights.")

# ---------------------------- File Handling ---------------------------- #
if uploaded_file:
    with st.spinner("Extracting text from resume..."):
        pdf_reader = PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text + " "

        st.subheader("📜 Extracted Resume Text")
        st.write(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

    # ---------------------------- Word Cloud ---------------------------- #
    def generate_wordcloud(text):
        return WordCloud(width=800, height=400, background_color="white").generate(text)

    st.subheader("🌥 Resume Word Cloud")
    if resume_text.strip():
        wordcloud = generate_wordcloud(resume_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # ---------------------------- Job Description Input ---------------------------- #
    st.subheader("📌 Enter Job Description")
    job_desc = st.text_area("Paste the job description here:", height=150)

    if job_desc:
        # ---------------------------- Similarity Analysis ---------------------------- #
        st.subheader("📊 Resume Match Score")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity_score = cosine_similarity(tfidf_matrix)[0, 1] * 100  # Convert to percentage

        # Display Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity_score,
            title={"text": "Match Percentage"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "blue"}}
        ))
        st.plotly_chart(fig)

        # ---------------------------- Skills Matching Using NLP ---------------------------- #
        st.subheader("🎯 Skills Match Analysis")

        def extract_skills(text):
            """ Extract skills from text using NLP """
            doc = nlp(text)
            extracted_skills = set()
            for token in doc:
                if token.text in SKILLS_DB:
                    extracted_skills.add(token.text)
            return list(extracted_skills)

        # Extract skills from resume and job description
        extracted_resume_skills = extract_skills(resume_text)
        extracted_job_skills = extract_skills(job_desc)

        # Find matched and missing skills
        matched_skills = list(set(extracted_resume_skills) & set(extracted_job_skills))
        missing_skills = list(set(extracted_job_skills) - set(extracted_resume_skills))

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")
        with col2:
            st.warning(f"❌ Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}")

        # ---------------------------- Skills Match Chart ---------------------------- #
        df = pd.DataFrame({
            "Skill": extracted_resume_skills + missing_skills,
            "Matched": [1] * len(extracted_resume_skills) + [0] * len(missing_skills)
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df, x="Skill", y="Matched", hue="Matched", palette=["red", "green"])
        plt.xticks(rotation=45)
        plt.xlabel("Skills")
        plt.ylabel("Match (1 = Yes, 0 = No)")
        st.pyplot(fig)

    else:
        st.info("📌 Please enter a job description for analysis.")

else:
    st.info("📂 Please upload a resume to start analysis.")
