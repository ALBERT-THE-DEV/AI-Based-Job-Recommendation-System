# app.py

import streamlit as st
import pdfplumber
from model_jobrec import JobRecommender
import os
import gdown
import zipfile

st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("AI-Based Job Recommendation System")
st.write("Upload your resume (PDF) or enter your skills to get personalized job recommendations")

# Sidebar Info
with st.sidebar:
    st.header("About This App")
    st.markdown("""
**App Purpose:**  
Recommends top AI/ML and software jobs based on your resume or skills.

**Model:**  
Uses SentenceTransformer (all-MiniLM-L6-v2) to encode job descriptions and resume text, then computes cosine similarity for recommendations.

**Dataset:**  
300+ curated US-based jobs including roles like Machine Learning Engineer, Data Scientist, NLP Engineer, etc.

**Tips for Resume Upload:**  
- Use text-based PDFs, not scanned images.  
- Include relevant skills and experience.  
- Manual text input works for short summaries or keywords.
""")

# Download and unzip trained model if not exists
if not os.path.exists("trained_model/jobs_embedded.csv"):
    st.info("Downloading trained model from Google Drive...")
    file_id = "1K_LzTD1OH5MbVHaFtPSICR_lacUGtZnK"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "trained_model.zip"
    gdown.download(url, output, quiet=False)

    os.makedirs("trained_model", exist_ok=True)

    # Flatten zip extraction to trained_model/
    with zipfile.ZipFile(output, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Skip directories
            if member.endswith("/"):
                continue
            # Remove first folder in path (if exists)
            filename = "/".join(member.split("/")[1:])
            if not filename:
                continue
            target_path = os.path.join("trained_model", filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    st.success("Trained model downloaded and extracted!")

# Load Model
@st.cache_resource
def load_model():
    return JobRecommender(
        model_path="trained_model/sbert_job_model",
        data_path="trained_model/jobs_embedded.csv"
    )

recommender = load_model()

# User Input
st.subheader("Upload or Enter Resume Details")
option = st.radio("Choose your input method:", ("Upload Resume (PDF)", "Enter Text/Skills Manually"))

resume_text = ""

if option == "Upload Resume (PDF)":
    uploaded_file = st.file_uploader("Upload your resume file", type=["pdf"])
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                resume_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            st.success("Resume text extracted successfully!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

elif option == "Enter Text/Skills Manually":
    resume_text = st.text_area("Enter your resume text or skills here:", height=200)

# Recommendations
if st.button("Find Matching Jobs", help="Click to obtain job recommendations"):
    if resume_text.strip():
        with st.spinner("Analyzing your profile..."):
            recs = recommender.recommend_jobs(resume_text)

        if recs.empty:
            st.warning("No closely matching jobs found in our dataset.")
        else:
            st.success("Here are your top job matches:")

            # Display jobs in column layout with color-coded similarity
            for _, row in recs.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"🏢 **{row.title}** at **{row.company}**")
                    st.markdown(f"📍 Location: {row.location}")
                    skills_short = ", ".join(row.skills.split(",")[:5])
                    st.markdown(f"💡 Skills Required: {skills_short}")
                with col2:
                    sim = row.similarity
                    if sim >= 0.7:
                        color = "green"
                    elif sim >= 0.5:
                        color = "orange"
                    else:
                        color = "red"
                    st.markdown(f"<span style='color:{color};font-weight:bold'>Score: {sim:.2f}</span>", unsafe_allow_html=True)

                st.markdown("---")
    else:
        st.warning("Please provide your resume text or upload a valid PDF file.")




