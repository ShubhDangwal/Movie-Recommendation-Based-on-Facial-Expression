import streamlit as st
import cv2
import time
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import random
from keras.models import load_model
import os
import pickle

model_path = 'emotion_model.h5'

# Check if model file exists, if not, print error and exit
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print("Error")
    exit()

# Load the face cascade for emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the TF-IDF vectorizer, TF-IDF matrix, and movies DataFrame
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

movies_df = pd.read_pickle('movies_df.pkl')

dark_theme = """
    <style>
        body {
            color: white;
            background-color: #1E1E1E; /* Dark background color */
        }
        .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
            color: black; /* Text color for input elements */
            background-color: #D3D3D3; /* Background color for input elements */
        }
        /* Add more specific styling as needed for different Streamlit elements */
    </style>
"""

def select_best5(recommendations):
    max_val = recommendations.max()
    movie_list = []

    for movie in recommendations.index:
        if (recommendations[movie] == max_val).any():
            movie_list.append(movie)
    return movie_list

def recommended_movies(emotion):
    def recommend_movies_based_on_emotion(user_emotion, movies_df):
        # Transform user emotion into a TF-IDF vector
        user_vector = tfidf_vectorizer.transform([user_emotion])

        # Calculate cosine similarity between user emotion and movies
        similarity_scores = linear_kernel(user_vector, tfidf_matrix).flatten()

        # Rank movies based on similarity scores
        rankings = pd.Series(similarity_scores, index=movies_df['Title']).sort_values(ascending=False)

        return rankings

    recommendations = recommend_movies_based_on_emotion(emotion, movies_df)
    top5 = select_best5(recommendations)
    return top5

# Display the HTML code to set the dark theme
st.markdown(dark_theme, unsafe_allow_html=True)
selected = st.button("Open Camera")
if selected:
    text = st.text("Detecting Emotion...")
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while time.time() - start_time < 7:
        # Capture the current frame
        ret, frame = cap.read()
        # Show the webcam feed
        cv2.imshow('Adjust Your Face', frame)
        # Small delay to allow for key press detection
        cv2.waitKey(1)
    text.empty()
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotion = ""
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]
        # Resize the face ROI to (48,48)
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image so each pixel lies b/w 0 to 1
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    st.image(frame)
    st.text(f"Detected Emotion: {emotion}")

    top5 = recommended_movies(emotion)
    random.shuffle(top5)
    st.title("Here are few recommended movies for you!")
    for i in range(5):
        st.text(top5[i])
