import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from rapidfuzz import process
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
    list_of_titles = movies_data['title'].str.lower().tolist()

    # Use rapidfuzz to find best match
    best_match = process.extractOne(movie_name, list_of_titles, score_cutoff=60)

    if not best_match:
        return "❌ Movie not found in database. Try typing more of the title."

    match_title = best_match[0]
    index = list_of_titles.index(match_title)

    similarity_scores = list(enumerate(similarity[index]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = [movies_data.iloc[i[0]].title for i in sorted_similar_movies]

    return f"✅ Showing recommendations based on: {movies_data.iloc[index].title}\n\n" + "\n".join(recommendations)




# Gradio UI
iface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Textbox(label="Enter a Movie Name"),
    outputs=gr.Textbox(label="Top 5 Recommendations"),
    title="Movie Recommendation System",
    description="Enter a movie you like and get 5 similar movie suggestions!"
)

iface.launch()
