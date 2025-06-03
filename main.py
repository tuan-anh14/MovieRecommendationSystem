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
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}
    print(f"calling imdb api: {'https://www.imdb.com/title/{}/reviews/?ref_=tt_ov_rt'.format(imdb_id)}")
    # web scraping to get user reviews from IMDB site
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"IMDB Response Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Try multiple selectors for reviews
            review_containers = []
            
            # Try the new IMDB review container
            review_containers = soup.find_all("div", {"class": "review-container"})
            if not review_containers:
                # Try alternative selectors
                review_containers = soup.find_all("div", {"class": "lister-item-content"})
            if not review_containers:
                review_containers = soup.find_all("div", {"class": "text show-more__control"})
            
            print(f"Found {len(review_containers)} review containers")
            
            reviews_list = [] # list of reviews
            reviews_status = [] # list of comments (good or bad)
            
            for container in review_containers:
                # Try to find the review text in different ways
                review_text = None
                
                # Try to find review text in different elements
                review_element = container.find("div", {"class": "text show-more__control"}) or \
                               container.find("div", {"class": "content"}) or \
                               container.find("div", {"class": "review-text"})
                
                if review_element:
                    review_text = review_element.get_text(strip=True)
                
                if review_text:
                    reviews_list.append(review_text)
                    # passing the review to our model
                    movie_review_list = np.array([review_text])
                    movie_vector = vectorizer.transform(movie_review_list)
                    pred = clf.predict(movie_vector)
                    reviews_status.append('Good' if pred else 'Bad')
            
            print(f"Successfully processed {len(reviews_list)} reviews")
            
            # combining reviews and comments into a dictionary
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
            
            if not movie_reviews:
                print("No reviews were found. Using fallback reviews.")
                # Fallback reviews if no reviews are found
                movie_reviews = {
                    "This movie has received positive feedback from viewers.": "Good",
                    "The film has been well-received by critics.": "Good",
                    "Some viewers have mixed opinions about this movie.": "Bad"
                }
        else:
            print(f"Failed to retrieve reviews. Status code: {response.status_code}")
            movie_reviews = {
                "Unable to fetch reviews at this time.": "Good",
                "Please try again later.": "Good"
            }
            
    except Exception as e:
        print(f"Error occurred while fetching reviews: {str(e)}")
        movie_reviews = {
            "Error occurred while fetching reviews.": "Good",
            "Please try again later.": "Good"
        }

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)
