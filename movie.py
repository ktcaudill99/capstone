#!/usr/bin/python3
import pandas as pd
import ast
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    movies_df = load_movies_df()
    # Step 5: Train a recommendation model
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Adjust 'max_features' as needed
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])
    # Example usage
    user_input_title = input("Enter a movie title to get recommendations: ")
    recommendations = get_recommendations(user_input_title, movies_df, tfidf_matrix)

    if recommendations is not None and not recommendations.empty:
        print(f"\nMovies recommended based on '{user_input_title}':")
        for idx, title in enumerate(recommendations, start=1):
            print(f"{idx}. {title}")
    else:
        print("Movie title not found in the dataset or no recommendations available.")

def load_movies_df():
    movies_df_filename = 'movies_df.pkl'
    if os.path.isfile(movies_df_filename):
        with open(movies_df_filename, 'rb') as fin:
            movies_df = pickle.load(fin)
        print("[+] Loaded movies_df from pickeled data")
    else:
        # Load datasets
        movies_df = pd.read_csv('archive/movies_metadata.csv', low_memory=True, usecols=['id', 'original_title', 'genres'])
        keywords_df = pd.read_csv('archive/keywords.csv', usecols=['id', 'keywords'])
        credits_df = pd.read_csv('archive/credits.csv', usecols=['id', 'cast', 'crew'])
        print("[+] Loaded data files")

        # Convert 'id' to string in all DataFrames
        movies_df['id'] = movies_df['id'].astype(str)
        keywords_df['id'] = keywords_df['id'].astype(str)
        credits_df['id'] = credits_df['id'].astype(str)

        # Merge datasets on movie ID
        movies_df = movies_df.merge(keywords_df, on='id').merge(credits_df, on='id')
        print("[+] Merged data sets")
        # Process meta columns
        movies_df['genres'] = movies_df['genres'].apply(parse_json_list)
        movies_df['keywords'] = movies_df['keywords'].apply(parse_json_list)
        movies_df['cast'] = movies_df['cast'].apply(parse_json_list)
        movies_df['crew'] = movies_df['crew'].apply(parse_json_list)
        # Extract relevant features from the dataset
        # Extract relevant features (assuming 'title' and 'genre' are columns in your dataset)
        # combine title and genre into a single string for each movie
        #movies_df['combined_features'] = movies_df['original_title'] + ' ' + movies_df['genres']
        # Feature Engineering
        movies_df['combined_features'] = (movies_df['original_title'] + ' ' +
                                        movies_df['genres'] + ' ' +
                                        movies_df['keywords'] + ' ' +
                                        movies_df['cast'] + ' ' +
                                        movies_df['crew'])
        print("[+] Data set preprocessing complete")
        with open(movies_df_filename, 'wb') as fout:
            pickle.dump(movies_df, fout)
        print("[+] Pickled movies_df")
    return movies_df

# Step 2: Preprocess the data
# Handle missing values, remove duplicates, and transform categorical variables

def parse_json_list(str):
    parsed_list = ast.literal_eval(str)
    return ' '.join([item['name'] for item in parsed_list])

# Step 4: Split the data
# Step 6: Evaluate the model
# Step 7: Make recommendations
def get_recommendations(title, movies_df, tfidf_matrix):
    # Handle case where title is not found
    if title not in movies_df['original_title'].values:
        return "Title not found in dataset."
    # Get the index of the movie that matches the title
    idx = movies_df.index[movies_df['original_title'] == title].tolist()[0]
    # Get the pairwise similarity scores of all movies with that movie
    cos_sim = cosine_similarity(tfidf_matrix.getrow(idx), tfidf_matrix)
    sim_movie_idx = cos_sim.argsort()[0][-11:]
    sim_movie_idx = np.flip(sim_movie_idx)[1:]
    # Return the top 10 most similar movies
    return movies_df['original_title'].iloc[sim_movie_idx]

# Example usage
# movie_title = "The Matrix"  # Replace with a movie title in your dataset
# recommendations = get_recommendations(movie_title)
# print("Movies recommended based on", movie_title, ":\n", recommendations)

if __name__ == "__main__":
    main()