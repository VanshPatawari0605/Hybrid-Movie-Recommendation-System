🎬 Hybrid Movie Recommendation System
https://hybrid-movie-recommendation-system-peytp5pylv5ux3brexl9tk.streamlit.app/

🔍 Overview
This project builds a hybrid-style recommendation system by combining:
- Text feature engineering from movie metadata
- TF-IDF vectorization
- Cosine similarity-based ranking
- Streamlit web deployment
The system processes genres, keywords, cast, director, and overview text to generate meaningful similarity scores between movies.

🧠 How It Works

1. Feature Engineering
   - Extracts genres, keywords, cast (top 3), and director
   - Cleans and merges metadata into a unified "tags" column

2. Vectorization
   - Applies TF-IDF on processed tags
   - Converts movies into high-dimensional feature vectors

3. Similarity Computation
   - Uses cosine similarity to measure content similarity
   - Returns top 10 most similar movies

4. Deployment
   - Built with Streamlit
   - Optimized using caching to avoid recomputation
   - Deployed on Streamlit Cloud

---

 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF
- Cosine Similarity

---

📂 Project Structure
