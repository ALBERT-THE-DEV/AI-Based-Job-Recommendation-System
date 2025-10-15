# model_jobrec.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

class JobRecommender:
    def __init__(self, model_path="trained_model/sbert_job_model",
                 data_path="trained_model/jobs_embedded.csv",
                 embeddings_path="trained_model/job_embeddings.npy"):

        # Check if CSV exists
        if not os.path.exists(data_path):
            st.error(f"Jobs CSV not found at: {data_path}")
            st.stop()

        # Check if model folder exists
        if not os.path.exists(model_path):
            st.error(f"SentenceTransformer model not found at: {model_path}")
            st.stop()

        # Check if embeddings exist
        if not os.path.exists(embeddings_path):
            st.error(f"Job embeddings not found at: {embeddings_path}")
            st.stop()

        st.info("Loading dataset and model...")
        self.data = pd.read_csv(data_path)
        self.model = SentenceTransformer(model_path)
        self.job_embeddings = np.load(embeddings_path)

    def recommend_jobs(self, resume_text, top_k=5, similarity_threshold=0.3):
        # Embed user input
        resume_emb = self.model.encode([resume_text])

        # Compute similarity
        similarities = cosine_similarity(resume_emb, self.job_embeddings)[0]

        df_sim = self.data.copy()
        df_sim['similarity'] = similarities

        # Sort and deduplicate by title + company
        df_sim = df_sim.sort_values(by='similarity', ascending=False)
        df_sim = df_sim.drop_duplicates(subset=['title', 'company'], keep='first')

        # Filter by similarity threshold
        df_sim = df_sim[df_sim['similarity'] >= similarity_threshold]

        return df_sim.head(top_k)[['title','company','location','skills','similarity']]


