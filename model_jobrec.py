import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class JobRecommender:
    def __init__(self, model_path, data_path, embeddings_path):
        print("Loading dataset and model...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model folder not found: {model_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        self.data = pd.read_csv(data_path)
        self.model = SentenceTransformer(model_path)
        self.job_embeddings = np.load(embeddings_path)

    def recommend_jobs(self, resume_text, top_k=5, similarity_threshold=0.3):
        resume_emb = self.model.encode([resume_text])
        similarities = cosine_similarity(resume_emb, self.job_embeddings)[0]

        df_sim = self.data.copy()
        df_sim['similarity'] = similarities

        # Sort, deduplicate, filter by similarity
        df_sim = df_sim.sort_values(by='similarity', ascending=False)
        df_sim = df_sim.drop_duplicates(subset=['title', 'company'], keep='first')
        df_sim = df_sim[df_sim['similarity'] >= similarity_threshold]

        return df_sim.head(top_k)[['title','company','location','skills','similarity']]

