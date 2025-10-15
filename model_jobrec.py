import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class JobRecommender:
    def __init__(self, model_path="trained_model/sbert_job_model",
                 data_path="trained_model/jobs_embedded.csv"):
        print("Loading dataset and model...")
        self.data = pd.read_csv(data_path)
        self.model = SentenceTransformer(model_path)
        self.job_embeddings = np.load("trained_model/job_embeddings.npy")

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

