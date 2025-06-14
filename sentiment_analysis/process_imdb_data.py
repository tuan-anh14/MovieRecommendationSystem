import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IMDBDataProcessor:
    def __init__(self):
        """Khởi tạo bộ xử lý dữ liệu IMDB"""
        # Tải các tài nguyên NLTK cần thiết
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Khởi tạo các công cụ
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Thêm các từ dừng đặc biệt cho phim
        self.custom_stopwords = {
            'movie', 'film', 'watch', 'watched', 'watching',
            'cinema', 'cinematic', 'director', 'actor', 'actress',
            'scene', 'scenes', 'plot', 'story', 'character', 'characters'
        }
        self.stop_words.update(self.custom_stopwords)
    
    def clean_text(self, text):
        """Làm sạch văn bản"""
        if not isinstance(text, str):
            return ''
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Loại bỏ ký tự đặc biệt và số
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization và lemmatization
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) 
                 for word in tokens if word not in self.stop_words]
        
        # Ghép lại thành câu
        text = ' '.join(tokens)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def convert_sentiment(self, score):
        """Chuyển đổi điểm đánh giá thành 3 lớp cảm xúc"""
        if score <= 0.3:
            return 0  # Tiêu cực
        elif score <= 0.7:
            return 1  # Trung lập
        else:
            return 2  # Tích cực
    
    def extract_features(self, text):
        """Trích xuất đặc trưng từ văn bản"""
        # Phân tích cảm xúc với TextBlob
        blob = TextBlob(text)
        
        features = {
            'sentiment_score': blob.sentiment.polarity,
            'subjectivity_score': blob.sentiment.subjectivity,
            'num_words': len(text.split()),
            'num_sentences': len(blob.sentences),
            'avg_word_length': np.mean([len(word) for word in text.split()])
        }
        
        return features
    
    def process_data(self, input_file, output_dir):
        """Xử lý dữ liệu IMDB"""
        logging.info("Đang đọc dữ liệu IMDB...")
        df = pd.read_csv(input_file)
        
        # Đổi tên cột nếu cần
        if 'review' not in df.columns:
            df = df.rename(columns={'text': 'review'})
        
        logging.info("Đang làm sạch văn bản...")
        df['cleaned_text'] = df['review'].apply(self.clean_text)
        
        logging.info("Đang trích xuất đặc trưng...")
        features = df['cleaned_text'].apply(self.extract_features)
        feature_df = pd.DataFrame(features.tolist())
        df = pd.concat([df, feature_df], axis=1)
        
        # Chuyển đổi sentiment
        df['sentiment'] = df['sentiment_score'].apply(self.convert_sentiment)
        
        # Phân chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'],
            df['sentiment'],
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment']
        )
        
        # Lưu dữ liệu đã xử lý
        train_data = pd.DataFrame({
            'text': X_train,
            'sentiment': y_train
        })
        test_data = pd.DataFrame({
            'text': X_test,
            'sentiment': y_test
        })
        
        # Tạo thư mục output nếu chưa tồn tại
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu dữ liệu
        train_data.to_csv(f'{output_dir}/processed_train.csv', index=False)
        test_data.to_csv(f'{output_dir}/processed_test.csv', index=False)
        
        # Lưu thống kê
        stats = {
            'total_reviews': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_review_length': df['cleaned_text'].str.len().mean(),
            'feature_stats': feature_df.describe().to_dict()
        }
        
        import json
        with open(f'{output_dir}/processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        logging.info(f"""
        Xử lý dữ liệu hoàn tất!
        - Dữ liệu đã xử lý được lưu trong {output_dir}/
        - Thống kê được lưu trong {output_dir}/processing_stats.json
        """)

def main():
    processor = IMDBDataProcessor()
    processor.process_data(
        input_file='../datasets/IMDB Dataset.csv',
        output_dir='../datasets/processed'
    )

if __name__ == '__main__':
    main() 