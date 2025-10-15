import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class JobRecommender:
    def __init__(self, model_path="trained_model/sbert_job_model",
                 data_path="trained_model/jobs_embedded.csv"):
        print("Loading dataset and model...")
        
        # Load job dataset
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")
        self.data = pd.read_csv(data_path)
        
        # Load SentenceTransformer model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model folder not found: {model_path}")
        self.model = SentenceTransformer(model_path)
        
        # Load embeddings
        embeddings_file = os.path.join(os.path.dirname(data_path), "job_embeddings.npy")
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        self.job_embeddings = np.load(embeddings_file)

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
