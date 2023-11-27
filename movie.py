import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
movies_df = pd.read_csv('archive/movies_metadata.csv', low_memory=True, usecols=['id', 'original_title', 'genres'])
keywords_df = pd.read_csv('archive/keywords.csv', usecols=['id', 'keywords'])
credits_df = pd.read_csv('archive/credits.csv', usecols=['id', 'cast', 'crew'])

# Convert 'id' to string in all DataFrames
movies_df['id'] = movies_df['id'].astype(str)
keywords_df['id'] = keywords_df['id'].astype(str)
credits_df['id'] = credits_df['id'].astype(str)

# Merge datasets on movie ID
movies_df = movies_df.merge(keywords_df, on='id').merge(credits_df, on='id')


# Step 2: Preprocess the data
# Handle missing values, remove duplicates, and transform categorical variables
def parse_genres(genres_str):
    try:
        genres_list = json.loads(genres_str.replace("'", "\""))
        return ' '.join([genre['name'] for genre in genres_list])
    except:
        return ''

# Process 'genres' column
movies_df['genres'] = movies_df['genres'].apply(parse_genres)

# Limit the Data Size (Use a subset of your data)
movies_df = movies_df.sample(frac=.75)  # Adjust 'frac' to your needs


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

# Step 4: Split the data

# Step 5: Train a recommendation model
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust 'max_features' as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

# Compute cosine similarity between samples in the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix)


# Step 6: Evaluate the model


# Step 7: Make recommendations
# TODO: Provide a user ID or movie ID as input and generate recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    import pdb; pdb.set_trace()
    # Handle case where title is not found
    if title not in movies_df['original_title'].values:
        return "Title not found in dataset."
    
    # Get the index of the movie that matches the title
    idx = movies_df.index[movies_df['original_title'] == title].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_df['original_title'].iloc[movie_indices]

# Example usage
# movie_title = "The Matrix"  # Replace with a movie title in your dataset
# recommendations = get_recommendations(movie_title)
# print("Movies recommended based on", movie_title, ":\n", recommendations)

# Example usage
user_input_title = input("Enter a movie title to get recommendations: ")
recommendations = get_recommendations(user_input_title)

if recommendations is not None and not recommendations.empty:
    print(f"\nMovies recommended based on '{user_input_title}':")
    for idx, title in enumerate(recommendations, start=1):
        print(f"{idx}. {title}")
else:
    print("Movie title not found in the dataset or no recommendations available.")