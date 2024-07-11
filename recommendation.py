import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def map_genres_to_emotions(genres, emotion_genre_mapping):
    emotions = set()
    for genre in genres.split(', '):
        for emotion, mapped_genres in emotion_genre_mapping.items():
            if genre in mapped_genres:
                emotions.add(emotion)
    return ', '.join(emotions)

# Load the movie dataset
file_path = 'mymoviedb.csv'
movies_df = pd.read_csv(file_path, lineterminator='\n')

# Define the emotion-genre mapping
emotion_genre_mapping = {
    'Happy': ['Animation', 'Family', 'Comedy', 'Romance'],
    'Sad': ['Drama', 'Romance', 'War'],
    'Angry': ['Action', 'Thriller', 'Crime'],
    'Surprised': ['Mystery', 'Science Fiction', 'Fantasy'],
    'Fearful': ['Horror', 'Thriller', 'Mystery'],
    'Disgusted': ['Horror', 'Thriller', 'Crime']
}

# Add Emotion and Features columns to the dataset
movies_df['Emotion'] = movies_df['Genre'].apply(lambda x: map_genres_to_emotions(x, emotion_genre_mapping))
movies_df['Features'] = movies_df['Genre'] + ', ' + movies_df['Emotion']

# Create a TF-IDF vectorizer and fit_transform on the Features column
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Features'])

# Save the TF-IDF vectorizer, TF-IDF matrix, and the movies DataFrame
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

movies_df.to_pickle('movies_df.pkl')
