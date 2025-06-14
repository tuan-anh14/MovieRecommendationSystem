"""
Tiền xử lý dữ liệu phim - Giai đoạn 2
Xử lý dữ liệu phim năm 2017 từ TMDB API và kết hợp với dữ liệu cũ

Kỹ thuật tiền xử lý được sử dụng:
1. Đã học trên lớp:
   - Merge/Join dữ liệu từ nhiều nguồn
   - Xử lý dữ liệu JSON (AST literal evaluation)
   - Text processing và feature extraction
   - Data type conversion

2. Chưa học trên lớp:
   - API data extraction (Nguồn: TMDB API documentation)
   - JSON parsing với ast.literal_eval (Nguồn: Python AST documentation)
   - Complex string manipulation cho cast/crew data

Kỹ thuật trích chọn đặc trưng:
1. Dữ liệu dạng bảng:
   - Trích xuất thông tin từ nested JSON structures
   - Feature engineering từ cast và crew data
   - Date/time feature extraction

2. Mã hóa dữ liệu phi cấu trúc:
   - Chuyển đổi JSON objects thành structured features
   - Text normalization cho genre và cast names

Giải thích tại sao cần dùng:
- Merge data: Kết hợp thông tin từ nhiều nguồn để có dataset phong phú hơn
- JSON parsing: Xử lý dữ liệu API trả về dạng nested structures
- Feature extraction: Tạo ra các đặc trưng có ý nghĩa từ raw data
"""

import pandas as pd
import numpy as np
import ast
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_tmdb_data():
    """Load dữ liệu từ TMDB"""
    logging.info("Đọc dữ liệu từ TMDB...")
    
    try:
        # Đọc dữ liệu credits và metadata
        credits = pd.read_csv('../datasets/credits.csv')
        meta = pd.read_csv('../datasets/movie_metadata.csv')
        
        logging.info(f"Credits data: {credits.shape}")
        logging.info(f"Metadata: {meta.shape}")
        
        return credits, meta
    
    except FileNotFoundError as e:
        logging.error(f"Không tìm thấy file: {e}")
        # Tạo dữ liệu mẫu nếu không có file thực
        return create_sample_data()

def create_sample_data():
    """Tạo dữ liệu mẫu cho demo"""
    logging.info("Tạo dữ liệu mẫu...")
    
    # Dữ liệu credits mẫu
    credits = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'cast': [
            '[{"name": "Tom Hanks", "character": "Forrest"}, {"name": "Robin Wright", "character": "Jenny"}]',
            '[{"name": "Leonardo DiCaprio", "character": "Jack"}, {"name": "Kate Winslet", "character": "Rose"}]',
            '[{"name": "Robert Downey Jr.", "character": "Tony Stark"}, {"name": "Gwyneth Paltrow", "character": "Pepper"}]',
            '[{"name": "Christian Bale", "character": "Batman"}, {"name": "Heath Ledger", "character": "Joker"}]',
            '[{"name": "Will Smith", "character": "Neo"}, {"name": "Laurence Fishburne", "character": "Morpheus"}]'
        ],
        'crew': [
            '[{"name": "Robert Zemeckis", "job": "Director"}, {"name": "Winston Groom", "job": "Writer"}]',
            '[{"name": "James Cameron", "job": "Director"}, {"name": "James Cameron", "job": "Writer"}]',
            '[{"name": "Jon Favreau", "job": "Director"}, {"name": "Mark Fergus", "job": "Writer"}]',
            '[{"name": "Christopher Nolan", "job": "Director"}, {"name": "Jonathan Nolan", "job": "Writer"}]',
            '[{"name": "Lana Wachowski", "job": "Director"}, {"name": "Lilly Wachowski", "job": "Director"}]'
        ]
    })
    
    # Dữ liệu metadata mẫu
    meta = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'title': ['Forrest Gump', 'Titanic', 'Iron Man', 'The Dark Knight', 'The Matrix'],
        'release_date': ['1994-07-06', '1997-12-19', '2008-05-02', '2008-07-18', '1999-03-31'],
        'genres': [
            '[{"name": "Drama"}, {"name": "Romance"}]',
            '[{"name": "Romance"}, {"name": "Drama"}]',
            '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Science Fiction"}]',
            '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]',
            '[{"name": "Action"}, {"name": "Science Fiction"}]'
        ]
    })
    
    return credits, meta

def extract_genres_list(genres_data):
    """Trích xuất danh sách thể loại"""
    
    def make_genres_list(x):
        """Chuyển đổi genres từ JSON sang string"""
        if not x or not isinstance(x, list):
            return np.NaN
        
        gen = []
        for genre in x:
            if isinstance(genre, dict) and 'name' in genre:
                # Xử lý Science Fiction đặc biệt
                if genre['name'] == 'Science Fiction':
                    gen.append('Sci-Fi')
                else:
                    gen.append(genre['name'])
        
        return ' '.join(gen) if gen else np.NaN
    
    return genres_data.apply(make_genres_list)

def extract_cast_info(cast_data):
    """Trích xuất thông tin diễn viên"""
    
    def get_actor(cast_list, position):
        """Lấy diễn viên ở vị trí cụ thể"""
        if not cast_list or not isinstance(cast_list, list) or len(cast_list) <= position:
            return np.NaN
        
        actor = cast_list[position]
        if isinstance(actor, dict) and 'name' in actor:
            return actor['name']
        return np.NaN
    
    # Trích xuất 3 diễn viên chính
    actor_1 = cast_data.apply(lambda x: get_actor(x, 0))
    actor_2 = cast_data.apply(lambda x: get_actor(x, 1))
    actor_3 = cast_data.apply(lambda x: get_actor(x, 2))
    
    return actor_1, actor_2, actor_3

def extract_director_info(crew_data):
    """Trích xuất thông tin đạo diễn"""
    
    def get_directors(crew_list):
        """Lấy danh sách đạo diễn"""
        if not crew_list or not isinstance(crew_list, list):
            return np.NaN
        
        directors = []
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                directors.append(person.get('name', ''))
        
        return ' '.join(directors) if directors else np.NaN
    
    return crew_data.apply(get_directors)

def main():
    """Hàm chính cho preprocessing giai đoạn 2"""
    logging.info("Bắt đầu preprocessing giai đoạn 2...")
    
    # Load dữ liệu
    credits, meta = load_tmdb_data()
    
    # Xử lý ngày phát hành
    meta['release_date'] = pd.to_datetime(meta['release_date'], errors='coerce')
    meta['year'] = meta['release_date'].dt.year
    
    # Lọc phim năm 2017 (hoặc sử dụng tất cả nếu không có)
    recent_movies = meta.loc[meta.year == 2017, ['genres', 'id', 'title', 'year']].copy()
    
    if recent_movies.empty:
        logging.warning("Không có phim năm 2017, sử dụng tất cả phim")
        recent_movies = meta[['genres', 'id', 'title', 'year']].copy()
    
    recent_movies['id'] = recent_movies['id'].astype(int)
    
    # Merge dữ liệu
    merged_data = pd.merge(recent_movies, credits, on='id', how='inner')
    
    # Parse JSON columns
    json_columns = ['genres', 'cast', 'crew']
    for col in json_columns:
        if col in merged_data.columns:
            try:
                merged_data[col] = merged_data[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
            except (ValueError, SyntaxError):
                merged_data[col] = merged_data[col].apply(lambda x: [] if pd.notna(x) else [])
    
    # Trích xuất features
    merged_data['genres_list'] = extract_genres_list(merged_data['genres'])
    merged_data['actor_1_name'], merged_data['actor_2_name'], merged_data['actor_3_name'] = extract_cast_info(merged_data['cast'])
    merged_data['director_name'] = extract_director_info(merged_data['crew'])
    
    # Chọn và làm sạch dữ liệu
    final_columns = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'genres_list', 'title']
    movie_data = merged_data[final_columns].copy()
    movie_data = movie_data.dropna(how='any')
    
    # Rename và normalize
    movie_data = movie_data.rename(columns={'genres_list': 'genres', 'title': 'movie_title'})
    movie_data['movie_title'] = movie_data['movie_title'].str.lower()
    
    # Tạo combined features
    movie_data['comb'] = (
        movie_data['actor_1_name'].astype(str) + ' ' +
        movie_data['actor_2_name'].astype(str) + ' ' +
        movie_data['actor_3_name'].astype(str) + ' ' +
        movie_data['director_name'].astype(str) + ' ' +
        movie_data['genres'].astype(str)
    )
    
    # Kết hợp với dữ liệu cũ nếu có
    try:
        old_data = pd.read_csv('../datasets/movies_stage1.csv')
        if 'comb' not in old_data.columns:
            old_data['comb'] = (
                old_data['actor_1_name'].astype(str) + ' ' +
                old_data['actor_2_name'].astype(str) + ' ' +
                old_data['actor_3_name'].astype(str) + ' ' +
                old_data['director_name'].astype(str) + ' ' +
                old_data['genres'].astype(str)
            )
        
        final_data = pd.concat([old_data, movie_data], ignore_index=True)
        final_data = final_data.drop_duplicates(subset=['movie_title'], keep='first')
    except FileNotFoundError:
        final_data = movie_data
    
    # Lưu dữ liệu
    output_path = '../datasets/movies_stage2.csv'
    final_data.to_csv(output_path, index=False)
    logging.info(f"Đã lưu dữ liệu vào {output_path}")
    logging.info(f"Kích thước cuối: {final_data.shape}")
    
    print("Sample data:")
    print(final_data.head())

if __name__ == '__main__':
    main() 