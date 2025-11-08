# AI-Based-Job-Recommendation-System
This is a learning project that showcases an AI-powered job recommendation system built with SentenceTransformer and locally hosted on Streamlit. The web app recommends relevant AI/ML and software engineering jobs based on your resume text or skills input, matching your profile with the closest job descriptions using semantic similarity.

Project Overview
----------------
The system uses SentenceTransformer (all-MiniLM-L6-v2) to convert job descriptions and resumes into embeddings and computes cosine similarity to find the most relevant matches. It was trained on a synthetic dataset of over 300 U.S.-based tech jobs, including roles like Machine Learning Engineer, Data Scientist, NLP Engineer, Software Developer, AI Researcher etc..

Workflow
-------------
-Input: Resume uploaded as PDF or skills entered manually via the Streamlit app.

-Model Architecture: SentenceTransformer (all-MiniLM-L6-v2).

-Output: Top job matches with similarity scores, locations, and key skills required.

Files in Repository
-------------------
-app.py: Streamlit web application for local usage

-model_jobrec.py: Core model for job embedding and recommendation

-requirements.txt: Required packages to be installed

Screenshots
-----------
1. Webapp interface
![App Interface](screenshots/interface.png)

2. Resume Upload & Prediction
![Resume Prediction Example](screenshots/prediction1.png)

3. Text Upload & Prediction
![Text Prediction Example](screenshots/prediction2.png)

Acknowledgments
---------------
-Model: SentenceTransformers - all-MiniLM-L6-v2

-Framework: Streamlit for web interface

-Dataset: Curated from multiple AI/ML job listings

Results
-------
The recommendation system provides highly relevant job matches, achieving an average similarity score of 0.40 across the dataset, with top matches often exceeding 0.70.

Model Performance
-----------------
The SentenceTransformer-based model achieved an average cosine similarity of 0.405 across all job embeddings, with values ranging from 0.091 (minimum) to 1.000 (maximum). For sample resumes, the top 5 job matches consistently reached similarity scores around 0.70, reflecting strong alignment between candidate profiles and job descriptions. 
