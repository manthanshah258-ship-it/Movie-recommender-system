import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# load + build similarity
@st.cache_data
def load_data():
    movie = pickle.load(open('movie_dict.pkl', mode='rb'))
    data = pd.DataFrame(movie)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray().astype("float32")
    similarity = cosine_similarity(vectors)

    return data, similarity


data, similarity = load_data()


# recommendation function
def recommend(movie):
    recommended_movies = []

    movie_index = data[data['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movie_list:
        recommended_movies.append(data.iloc[i[0]].title)

    return recommended_movies


# streamlit web-app
st.title('Movie Recommendation Model')

selected_movie = st.selectbox(
    "Which movie would you like to get recommendations ??",
    list(data['title'].values)
)

btn = st.button('Recommend')

if btn:
    movies_list = recommend(selected_movie)

    for i in movies_list:
        st.write(i)
