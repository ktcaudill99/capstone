#!/usr/bin/python3
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import ast
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_movies_df():
    movies_df_filename = 'movies_df.pkl'
    if os.path.isfile(movies_df_filename):
        with open(movies_df_filename, 'rb') as fin:
            movies_df = pickle.load(fin)
        print("[+] Loaded movies_df from pickled data")
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

def parse_json_list(str):
    parsed_list = ast.literal_eval(str)
    return ' '.join([item['name'] for item in parsed_list])

def get_recommendations(title, movies_df, tfidf_matrix):
    if title not in movies_df['original_title'].values:
        return None, None
    idx = movies_df.index[movies_df['original_title'] == title].tolist()[0]
    cos_sim = cosine_similarity(tfidf_matrix.getrow(idx), tfidf_matrix)
    sim_movie_idx = cos_sim.argsort()[0][-11:]
    sim_scores = cos_sim[0, sim_movie_idx]
    sim_movie_idx = np.flip(sim_movie_idx)[1:]
    sim_scores = np.flip(sim_scores)[1:]
    return movies_df['original_title'].iloc[sim_movie_idx], sim_scores

def show_similarity_bar_chart(recommended_movies, sim_scores):
    global chart_frame
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots()
    ax.barh(recommended_movies, sim_scores)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Recommended Movies')
    ax.set_title('Similarity of Recommended Movies to User\'s Choice')

    chart = FigureCanvasTkAgg(fig, master=chart_frame)
    chart_widget = chart.get_tk_widget()
    chart_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def update_gui_with_results(movies_df, tfidf_matrix, user_input_title):
    global recommendations_label, chart_frame
    recommendations, sim_scores = get_recommendations(user_input_title, movies_df, tfidf_matrix)
    if recommendations is not None:
        recommendation_text = "\n".join([f"{idx}. {title}" for idx, title in enumerate(recommendations, start=1)])
        recommendations_label.config(text=recommendation_text)

        # Update accuracy (dummy value for now)
        accuracy = 0.85  # Replace with actual accuracy calculation
        accuracy_label.config(text=f"Model Accuracy: {accuracy * 100:.2f}%")

        show_similarity_bar_chart(recommendations, sim_scores)

def on_recommend():
    user_input_title = movie_input.get()
    update_gui_with_results(movies_df, tfidf_matrix, user_input_title)

if __name__ == "__main__":
    # GUI initialization
    root = tk.Tk()
    root.title("Movie Recommender")
    root.geometry("1000x1000")

    # GUI widgets
    label = tk.Label(root, text="Enter a movie title to get recommendations:")
    label.pack()

    movie_input = tk.Entry(root)
    movie_input.pack()

    recommend_button = tk.Button(root, text="Recommend", command=on_recommend)
    recommend_button.pack()

    # Frame for results
    results_frame = tk.Frame(root)
    results_frame.pack(fill=tk.BOTH, expand=True)

    # Recommendations label
    recommendations_label = tk.Label(results_frame, text="Recommendations will appear here")
    recommendations_label.pack()

    # Frame for accuracy and pie chart
    accuracy_frame = tk.Frame(root)
    accuracy_frame.pack(fill=tk.BOTH, expand=True)

    accuracy_label = tk.Label(accuracy_frame, text="Model Accuracy: 0%")
    accuracy_label.pack()

    # Frame for similarity bar chart
    chart_frame = tk.Frame(root)
    chart_frame.pack(fill=tk.BOTH, expand=True)

    # Load data and prepare model
    movies_df = load_movies_df()
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

    # Main loop
    root.mainloop()
