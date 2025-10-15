import streamlit as st
import pdfplumber
from model_jobrec import JobRecommender
import os
import gdown
import zipfile

st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("AI-Based Job Recommendation System")
st.write("Upload your resume (PDF) or enter your skills to get personalized job recommendations")

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    **App Purpose:**  
    Recommends top AI/ML and software jobs based on your resume or skills.

    **Model:**  
    Uses SentenceTransformer (all-MiniLM-L6-v2) to encode jobs and resume text, then computes cosine similarity.

    **Dataset:**  
    300+ curated US-based jobs including roles like Machine Learning Engineer, Data Scientist, NLP Engineer, etc.

    **Tips for Resume Upload:**  
    - Use text-based PDFs, not scanned images.  
    - Include relevant skills and experience.  
    - Manual text input works for short summaries or keywords.
    """)

# Download and extract trained model if not exists
if not os.path.exists("trained_model"):
    st.info("Downloading trained model from Google Drive...")
    file_id = "1K_LzTD1OH5MbVHaFtPSICR_lacUGtZnK"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "trained_model.zip"
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    st.success("Trained model downloaded and extracted!")

# Detect correct trained_model folder
def find_trained_folder(base="trained_model"):
    """Return folder containing CSV, embeddings, and sbert_job_model."""
    required_items = ["jobs_embedded.csv", "job_embeddings.npy", "sbert_job_model"]

    # Check base folder
    if all(os.path.exists(os.path.join(base, item)) for item in required_items):
        return base

    # Check first-level subfolders
    subfolders = [f.path for f in os.scandir(base) if f.is_dir()]
    for folder in subfolders:
        if all(os.path.exists(os.path.join(folder, item)) for item in required_items):
            return folder

    return None

trained_folder = find_trained_folder()
if not trained_folder:
    st.error("Cannot find jobs CSV, embeddings, or model folder inside trained_model!")
    st.stop()

st.write(f"✅ Using trained model folder: {trained_folder}")

# Load model
@st.cache_resource
def load_model():
    return JobRecommender(
        model_path=os.path.join(trained_folder, "sbert_job_model"),
        data_path=os.path.join(trained_folder, "jobs_embedded.csv"),
        embeddings_path=os.path.join(trained_folder, "job_embeddings.npy")
    )

recommender = load_model()

# User input
st.subheader("Upload or Enter Resume Details")
option = st.radio("Choose your input method:", ("Upload Resume (PDF)", "Enter Text/Skills Manually"))

resume_text = ""
if option == "Upload Resume (PDF)":
    uploaded_file = st.file_uploader("Upload your resume file", type=["pdf"])
    if uploaded_file:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                resume_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            st.success("Resume text extracted successfully!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
elif option == "Enter Text/Skills Manually":
    resume_text = st.text_area("Enter your resume text or skills here:", height=200)

# Recommendations
if st.button("Find Matching Jobs"):
    if resume_text.strip():
        with st.spinner("Analyzing your profile..."):
            recs = recommender.recommend_jobs(resume_text)

        if recs.empty:
            st.warning("No closely matching jobs found in our dataset.")
        else:
            st.success("Here are your top job matches:")
            for _, row in recs.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row.title}** at **{row.company}**")
                    st.markdown(f"Location: {row.location}")
                    skills_short = ", ".join(row.skills.split(",")[:5])
                    st.markdown(f"Skills Required: {skills_short}")
                with col2:
                    sim = row.similarity
                    color = "green" if sim >= 0.7 else "orange" if sim >= 0.5 else "red"
                    st.markdown(f"<span style='color:{color};font-weight:bold'>Score: {sim:.2f}</span>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.warning("Please provide your resume text or upload a valid PDF file.")

