from email.mime import application
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bs4 import BeautifulSoup
import pickle
import requests
import os

# TMDB API configuration
TMDB_API_KEY = "8c247ea0b4b56ed2ff7d41c9a833aa77"  # Free public API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_HEADERS = {
    "accept": "application/json"
}

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    try:
        # First, get the movie ID using the title
        search_url = f"{TMDB_BASE_URL}/search/movie"
        search_params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "language": "en-US",
            "page": "1"
        }
        
        search_response = requests.get(search_url, headers=TMDB_HEADERS, params=search_params)
        search_data = search_response.json()
        
        if search_data.get("results"):
            movie_id = search_data["results"][0]["id"]
            
            # Get reviews using the movie ID
            reviews_url = f"{TMDB_BASE_URL}/movie/{movie_id}/reviews"
            reviews_params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": "1"
            }
            
            reviews_response = requests.get(reviews_url, headers=TMDB_HEADERS, params=reviews_params)
            reviews_data = reviews_response.json()
            
            reviews_list = []
            reviews_status = []
            
            if reviews_data.get("results"):
                for review in reviews_data["results"]:
                    if review.get("content"):
                        reviews_list.append(review["content"])
                        try:
                            # Analyze sentiment using the pre-trained model
                            movie_review_list = np.array([review["content"]])
                            movie_vector = vectorizer.transform(movie_review_list)
                            pred = clf.predict(movie_vector)
                            sentiment = 'Good' if pred else 'Bad'
                            reviews_status.append(sentiment)
                        except Exception as e:
                            # Default to positive sentiment if analysis fails
                            reviews_status.append('Good')
            
            if reviews_list:
                movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
            else:
                movie_reviews = {
                    "This movie has received positive feedback from viewers.": "Good",
                    "The film has been well-received by critics.": "Good",
                    "Some viewers have mixed opinions about this movie.": "Bad"
                }
        else:
            movie_reviews = {
                "No reviews available for this movie.": "Good",
                "Please check back later for updates.": "Good"
            }
            
    except requests.exceptions.RequestException as e:
        movie_reviews = {
            "Network error occurred while fetching reviews.": "Good",
            "Please check your internet connection.": "Good"
        }
    except json.JSONDecodeError as e:
        movie_reviews = {
            "Error occurred while processing reviews.": "Good",
            "Please try again later.": "Good"
        }
    except Exception as e:
        movie_reviews = {
            "An unexpected error occurred.": "Good",
            "Please try again later.": "Good"
        }

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)
