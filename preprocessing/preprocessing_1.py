"""
Tiền xử lý dữ liệu phim - Giai đoạn 1
Xử lý dữ liệu từ movie_metadata.csv với các kỹ thuật tiền xử lý cơ bản

Kỹ thuật tiền xử lý được sử dụng:
1. Đã học trên lớp:
   - Xử lý giá trị thiếu (missing values) bằng thay thế với giá trị mặc định
   - Làm sạch văn bản (text cleaning): lowercase, loại bỏ ký tự đặc biệt
   - Chuẩn hóa dữ liệu văn bản
   
2. Chưa học trên lớp:
   - String manipulation với regex patterns (Nguồn: Python re module documentation)
   - Text normalization techniques (Nguồn: NLTK documentation)

Giải thích tại sao cần dùng các biện pháp tiền xử lý:
- Xử lý giá trị thiếu: Đảm bảo mô hình không bị lỗi khi gặp dữ liệu rỗng
- Làm sạch văn bản: Chuẩn hóa định dạng để tăng độ chính xác khi so sánh
- Kết hợp thông tin: Tạo ra đặc trưng tổng hợp từ nhiều cột để tăng hiệu quả recommendation
"""

import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_movie_data(file_path):
    """Load movie metadata"""
    logging.info("Đọc dữ liệu phim từ file...")
    data = pd.read_csv(file_path)
    logging.info(f"Đã đọc {data.shape[0]} dòng, {data.shape[1]} cột")
    return data

def explore_data(data):
    """Khám phá dữ liệu cơ bản"""
    logging.info("Khám phá dữ liệu:")
    logging.info(f"Kích thước dữ liệu: {data.shape}")
    logging.info(f"Các cột: {list(data.columns)}")
    
    # Kiểm tra phân bố năm phim
    if 'title_year' in data.columns:
        year_counts = data['title_year'].value_counts(dropna=False).sort_index()
        logging.info(f"Phân bố năm: {year_counts.head()}")
        
        # Vẽ biểu đồ phân bố năm
        plt.figure(figsize=(12, 6))
        year_counts.plot(kind='bar')
        plt.title('Phân bố phim theo năm')
        plt.xlabel('Năm')
        plt.ylabel('Số lượng phim')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('year_distribution.png')
        logging.info("Đã lưu biểu đồ phân bố năm vào year_distribution.png")
    
    return data

def select_features(data):
    """Chọn các đặc trưng cần thiết cho recommendation"""
    logging.info("Chọn các đặc trưng quan trọng...")
    
    # Các cột cần thiết cho recommendation
    required_columns = ['director_name', 'actor_1_name', 'actor_2_name', 
                       'actor_3_name', 'genres', 'movie_title']
    
    # Kiểm tra xem các cột có tồn tại không
    available_columns = [col for col in required_columns if col in data.columns]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.warning(f"Các cột bị thiếu: {missing_columns}")
    
    # Chọn dữ liệu với các cột có sẵn
    selected_data = data[available_columns].copy()
    logging.info(f"Đã chọn {len(available_columns)} cột đặc trưng")
    
    return selected_data

def handle_missing_values(data):
    """Xử lý giá trị thiếu"""
    logging.info("Xử lý giá trị thiếu...")
    
    # Hiển thị số lượng giá trị thiếu trước khi xử lý
    missing_before = data.isnull().sum()
    logging.info(f"Giá trị thiếu trước khi xử lý:\n{missing_before}")
    
    # Thay thế giá trị thiếu cho các cột actor và director
    actor_director_columns = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
    for col in actor_director_columns:
        if col in data.columns:
            data[col] = data[col].replace(np.nan, 'unknown')
            data[col] = data[col].fillna('unknown')
    
    # Hiển thị số lượng giá trị thiếu sau khi xử lý
    missing_after = data.isnull().sum()
    logging.info(f"Giá trị thiếu sau khi xử lý:\n{missing_after}")
    
    return data

def preprocess_genres(data):
    """Tiền xử lý cột genres"""
    logging.info("Tiền xử lý cột genres...")
    
    if 'genres' in data.columns:
        # Thay thế dấu | bằng khoảng trắng để tách các thể loại
        data['genres'] = data['genres'].str.replace('|', ' ')
        data['genres'] = data['genres'].fillna('unknown')
        logging.info("Đã xử lý cột genres")
    
    return data

def clean_movie_titles(data):
    """Làm sạch tên phim"""
    logging.info("Làm sạch tên phim...")
    
    if 'movie_title' in data.columns:
        # Chuyển về chữ thường
        data['movie_title'] = data['movie_title'].str.lower()
        
        # Loại bỏ ký tự null terminating ở cuối (nếu có)
        data['movie_title'] = data['movie_title'].apply(lambda x: x[:-1] if isinstance(x, str) and x.endswith('\xa0') else x)
        
        # Loại bỏ khoảng trắng thừa
        data['movie_title'] = data['movie_title'].str.strip()
        
        logging.info("Đã làm sạch tên phim")
    
    return data

def create_combined_features(data):
    """Tạo đặc trưng tổng hợp"""
    logging.info("Tạo đặc trưng tổng hợp...")
    
    # Tạo cột 'comb' kết hợp tất cả thông tin quan trọng
    feature_columns = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name', 'genres']
    available_features = [col for col in feature_columns if col in data.columns]
    
    if available_features:
        data['comb'] = data[available_features].apply(
            lambda row: ' '.join([str(val) if pd.notna(val) else 'unknown' for val in row]), axis=1
        )
        logging.info("Đã tạo đặc trưng tổng hợp 'comb'")
    
    return data

def validate_data(data):
    """Kiểm tra tính hợp lệ của dữ liệu"""
    logging.info("Kiểm tra tính hợp lệ của dữ liệu...")
    
    # Kiểm tra dữ liệu trống
    if data.empty:
        logging.error("Dữ liệu trống!")
        return False
    
    # Kiểm tra các cột bắt buộc
    required_columns = ['movie_title']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        logging.error(f"Thiếu các cột bắt buộc: {missing_required}")
        return False
    
    # Kiểm tra dữ liệu duplicate
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        logging.warning(f"Tìm thấy {duplicates} dòng dữ liệu trùng lặp")
    
    logging.info("Dữ liệu hợp lệ")
    return True

def save_processed_data(data, output_path):
    """Lưu dữ liệu đã xử lý"""
    logging.info(f"Lưu dữ liệu đã xử lý vào {output_path}...")
    data.to_csv(output_path, index=False)
    logging.info("Đã lưu thành công")

def main():
    """Hàm chính thực hiện tiền xử lý giai đoạn 1"""
    try:
        # Đọc dữ liệu
        data = load_movie_data('../datasets/movie_metadata.csv')
        
        # Khám phá dữ liệu
        data = explore_data(data)
        
        # Chọn đặc trưng
        data = select_features(data)
        
        # Xử lý giá trị thiếu
        data = handle_missing_values(data)
        
        # Tiền xử lý genres
        data = preprocess_genres(data)
        
        # Làm sạch tên phim
        data = clean_movie_titles(data)
        
        # Tạo đặc trưng tổng hợp
        data = create_combined_features(data)
        
        # Kiểm tra tính hợp lệ
        if validate_data(data):
            # Lưu dữ liệu
            save_processed_data(data, '../datasets/movies_stage1.csv')
            
            # Hiển thị mẫu dữ liệu
            logging.info("\nMẫu dữ liệu đã xử lý:")
            print(data.head())
            
            logging.info(f"\nKích thước dữ liệu cuối: {data.shape}")
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình xử lý: {str(e)}")
        raise

if __name__ == '__main__':
    main() 