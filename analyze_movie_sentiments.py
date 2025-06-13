import pandas as pd
import requests
import pickle
import time
from collections import defaultdict
import numpy as np

# Load model sentiment
with open('nlp_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

# TMDB API config
TMDB_API_KEY = '4f7c3ea95b05ea5a83b661924a0c10ee'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Đọc danh sách phim
movies_df = pd.read_csv('main_data.csv')
movie_titles = movies_df['movie_title'].drop_duplicates().tolist()

sentiment_stats = {}

for idx, title in enumerate(movie_titles):
    print(f'[{idx+1}/{len(movie_titles)}] {title}')
    # Tìm movie_id từ TMDB
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': title,
        'language': 'en-US',
        'page': 1
    }
    try:
        resp = requests.get(search_url, params=params)
        data = resp.json()
        if not data.get('results'):
            continue
        movie_id = data['results'][0]['id']
    except Exception as e:
        print(f'  Lỗi tìm movie_id: {e}')
        continue
    # Lấy review
    reviews_url = f"{TMDB_BASE_URL}/movie/{movie_id}/reviews"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US',
        'page': 1
    }
    try:
        resp = requests.get(reviews_url, params=params)
        reviews_data = resp.json()
        reviews = [r['content'] for r in reviews_data.get('results', []) if r.get('content')]
        reviews = reviews[:5]  # Lấy tối đa 5 review
    except Exception as e:
        print(f'  Lỗi lấy review: {e}')
        continue
    if not reviews:
        continue
    # Phân tích cảm xúc
    preds = sentiment_model.predict(reviews)
    total = len(preds)
    pos = np.sum(np.array(preds) == 2)
    neu = np.sum(np.array(preds) == 1)
    neg = np.sum(np.array(preds) == 0)
    sentiment_stats[title] = {
        'total_reviews': total,
        'positive': int(pos),
        'neutral': int(neu),
        'negative': int(neg),
        'pos_ratio': float(pos)/total,
        'neu_ratio': float(neu)/total,
        'neg_ratio': float(neg)/total
    }
    time.sleep(0.3)  # Tránh spam API

# Lưu kết quả
with open('movie_sentiment_stats.pkl', 'wb') as f:
    pickle.dump(sentiment_stats, f)

print('Đã lưu xong movie_sentiment_stats.pkl!') 