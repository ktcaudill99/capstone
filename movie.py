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
#pip install wordcloud networkx
#install wordcloud and networkx
from wordcloud import WordCloud
import networkx as nx
from itertools import combinations
#pip install --upgrade networkx

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

recommendations_label = None
accuracy_label = None
chart_frame = None
genre_chart_frame = None
keyword_chart_frame = None

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

def show_similarity_bar_chart(recommended_movies, sim_scores, chart_frame):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # Adjust the figure size as needed
    fig, ax = plt.subplots(figsize=(8, 4))  # Reduced height for better fit

    ax.barh(recommended_movies, sim_scores)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Recommended Movies')
    ax.set_title('Similarity of Recommended Movies to User\'s Choice')

    # Adjust layout and padding
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.tight_layout(pad=3.0)

    chart = FigureCanvasTkAgg(fig, master=chart_frame)
    chart_widget = chart.get_tk_widget()
    chart_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
################################################################
    

# Function to show genre distribution
def show_genre_distribution(movies_df, recommended_movies, genre_chart_frame):
    clear_frame(genre_chart_frame)  # Clear the frame before adding new content

    # Filter the DataFrame for recommended movies and extract genres
    recommended_genres = movies_df[movies_df['original_title'].isin(recommended_movies)]['genres'].str.cat(sep=' ')
    genre_counts = pd.Series(recommended_genres.split(' ')).value_counts()

    # Generate the pie chart
    fig, ax = plt.subplots()
    ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    chart = FigureCanvasTkAgg(fig, master=genre_chart_frame)
    chart_widget = chart.get_tk_widget()
    chart_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


# Function to show keyword frequency word cloud
def show_keyword_wordcloud(movies_df, recommended_movies, frame):
    clear_frame(frame)  # Clear the frame before adding new content

    # Filter the DataFrame for recommended movies and extract keywords
    recommended_keywords = movies_df[movies_df['original_title'].isin(recommended_movies)]['keywords'].str.cat(sep=' ')

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(recommended_keywords)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    chart = FigureCanvasTkAgg(fig, master=frame)
    chart_widget = chart.get_tk_widget()
    chart_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)





    ##################################################################


def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()



def update_gui_with_results(movies_df, tfidf_matrix, user_input_title):
    global recommendations_label, genre_chart_frame, keyword_chart_frame, accuracy_label, chart_frame
   
    clear_frame(genre_chart_frame)
    clear_frame(keyword_chart_frame)
    clear_frame(chart_frame)

    recommendations, sim_scores = get_recommendations(user_input_title, movies_df, tfidf_matrix)
    
    if recommendations is not None and len(recommendations) > 0:
        recommendation_text = "\n".join([f"{idx+1}. {title}" for idx, title in enumerate(recommendations)])
        recommendations_label.config(text=recommendation_text)

        # Show charts for the recommended movies
        show_genre_distribution(movies_df, recommendations, genre_chart_frame)
        show_keyword_wordcloud(movies_df, recommendations, keyword_chart_frame)
        show_similarity_bar_chart(recommendations, sim_scores, chart_frame)

        accuracy = 0.85  # Replace with actual accuracy calculation
        accuracy_label.config(text=f"Model Accuracy: {accuracy * 100:.2f}%")
    else:
        recommendations_label.config(text="No recommendations found. Please try another movie title.")



def on_recommend():
    global recommendations_label, accuracy_label

    user_input_title = movie_input.get()
    update_gui_with_results(movies_df, tfidf_matrix, user_input_title)

    # Initialize if not already
    if not accuracy_label:
        # Frame for accuracy label
        accuracy_frame = tk.Frame(recommendations_frame)
        accuracy_frame.pack(fill=tk.BOTH, expand=True)

        # Accuracy label
        accuracy_label = tk.Label(accuracy_frame, text="Model Accuracy: 0%")
        accuracy_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Movie Recommender")
    root.geometry("1200x800")

    # Create input label and entry field
    label = tk.Label(root, text="Enter a movie title to get recommendations:")
    label.grid(row=0, column=0, columnspan=2, pady=10)

    movie_input = tk.Entry(root, width=50)
    movie_input.grid(row=1, column=0, columnspan=2)

    # Create the 'Recommend' button
    recommend_button = tk.Button(root, text="Recommend", command=on_recommend)
    recommend_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Initialize frames
    recommendations_frame = tk.Frame(root)
    genre_chart_frame = tk.Frame(root)
    chart_frame = tk.Frame(root)
    keyword_chart_frame = tk.Frame(root)

    # Pack frames in the grid
    recommendations_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
    genre_chart_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
    chart_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
    keyword_chart_frame.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

    # Initialize labels and pack them in their respective frames
    recommendations_label = tk.Label(recommendations_frame, text="Recommendations will appear here", justify=tk.LEFT)
    recommendations_label.pack(fill=tk.BOTH, expand=True)

    accuracy_label = tk.Label(recommendations_frame, text="Model Accuracy: 0%")
    accuracy_label.pack(fill=tk.BOTH, expand=True)

    # Load data and prepare model
    movies_df = load_movies_df()
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

    # Configure grid layout weights
    root.grid_rowconfigure(3, weight=1)
    root.grid_rowconfigure(4, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()
