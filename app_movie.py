import streamlit as st
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")
# Load Data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
movies['cast'] = movies['cast'].apply(lambda x: x[:3])

movies['overview'] = movies['overview'].astype(str)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

def remove_space(L):
    return [i.replace(" ", "") for i in L]

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(remove_space)
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

new_df = movies[['movie_id','title','tags']]

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)

    return recommended_movies


# Streamlit UI
st.title("🎬 Hybrid Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie:",
    new_df['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Top Recommendations:")
    for movie in recommendations:
        st.write(movie)