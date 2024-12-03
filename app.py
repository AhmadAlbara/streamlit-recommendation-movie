import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Set page configuration as the first command
st.set_page_config(page_title="Movie Recommendation System")

# Load preprocessed data
@st.cache_data
def load_data():
    data = pd.read_csv('imdb_movies.csv')
    data['genre'] = data['genre'].fillna('')
    data['crew'] = data['crew'].fillna('')
    data['overview'] = data['overview'].fillna('')
    data['combined_features'] = (
        data['genre'] + " " + data['overview'] + " " + data['crew']
    )
    
    data['combined_features'] = data['combined_features'].astype(str)
    return data

movies_data = load_data()

# Prepare the recommendation model
@st.cache_resource
def prepare_model(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(reduced_tfidf_matrix)
    return nn_model, reduced_tfidf_matrix, svd, tfidf_vectorizer

nn_model, reduced_tfidf_matrix, svd, tfidf_vectorizer = prepare_model(movies_data)

# Recommendation function with similarity scores
def get_recommendations_ann(title, nn_model, data, tfidf_matrix, svd, tfidf_vectorizer):
    indices = pd.Series(data.index, index=data['names']).drop_duplicates()
    idx = indices.get(title, None)
    
    if idx is None:
        return [], []
    
    query_vector = tfidf_vectorizer.transform([str(data['combined_features'].iloc[idx])])
    query_vector_svd = svd.transform(query_vector)  # Apply SVD to reduce dimensions
    
    distances, indices = nn_model.kneighbors(query_vector_svd, n_neighbors=11)
    recommended_indices = indices[0][1:]
    similarities = 1 - distances[0][1:]  # Convert distances to similarity scores
    recommended_movies = data['names'].iloc[recommended_indices].tolist()
    return recommended_movies, similarities

# Streamlit UI
st.title("**Movie Recommendation System**")
st.write("""
    Welcome to the **Movie Recommendation System**! 🎥✨
    This system uses advanced AI algorithms to help you find movies similar to the one you like.
    Just choose a movie, and let the magic happen! 🔮
""")

movie_name = st.selectbox("Select a movie:", options=movies_data['names'].unique(), index=0)

if movie_name:
    recommendations, similarities = get_recommendations_ann(movie_name, nn_model, movies_data, reduced_tfidf_matrix, svd, tfidf_vectorizer)
    if not recommendations:
        st.error("Movie not found. Please try another title.")
    else:
        similarities_percentage = [f"{similarity * 100:.2f}%" for similarity in similarities]

        result_df = pd.DataFrame({
            "No": range(1, len(recommendations) + 1),
            "Recommended Movie": recommendations,
            "Similarity (%)": similarities_percentage
        })
        result_df = result_df.sort_values(by="Similarity (%)", ascending=False)

        result_df.set_index("No", inplace=True)

        st.write("### Recommended Movies Based on Your Selection: 🎬🎉")
        st.dataframe(result_df, use_container_width=True)

st.markdown("""
    ---
    ### Credits 🌟
    This system is built with **Streamlit** and **Scikit-learn**.
    The recommendation engine is powered by **TF-IDF**, **SVD**, and **Nearest Neighbors**.
    Developed by Ahmad Albara & Yanti Puspita Sari 💻.
    Reach out for any questions or collaborations! 🤝
""")
