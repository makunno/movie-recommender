import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movies_data = pd.read_csv("movies.csv")
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data[selected_features].agg(' '.join, axis=1)

# Vectorize and compute similarity
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):
    movie_name = movie_name.lower()
    list_of_all_titles = movies_data['title'].str.lower().tolist()
    if movie_name not in list_of_all_titles:
        return "Movie not found in database."
    
    index = list_of_all_titles.index(movie_name)
    similarity_scores = list(enumerate(similarity[index]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in sorted_similar_movies:
        recommendations.append(movies_data.iloc[i[0]].title)
    
    return recommendations

# Gradio UI
iface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Textbox(label="Enter a Movie Name"),
    outputs=gr.Textbox(label="Top 5 Recommendations"),
    title="Movie Recommendation System",
    description="Enter a movie you like and get 5 similar movie suggestions!"
)

iface.launch()
