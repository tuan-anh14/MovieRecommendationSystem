import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
import re
from typing import Tuple, List
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import requests
import os
import gzip
import shutil

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_imdb_dataset():
    """Tải IMDB Dataset"""
    logging.info("Đang tải IMDB Dataset...")
    
    # URL của dataset
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    
    # Tải file
    response = requests.get(url, stream=True)
    with open("aclImdb_v1.tar.gz", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Giải nén
    import tarfile
    with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
        tar.extractall()
    
    logging.info("Đã tải xong IMDB Dataset")

def load_imdb_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Đọc dữ liệu từ IMDB Dataset"""
    logging.info("Đang đọc dữ liệu IMDB...")
    
    # Đường dẫn đến thư mục dữ liệu
    base_path = "aclImdb"
    
    # Đọc dữ liệu train
    train_pos = []
    train_neg = []
    train_neu = []  # Sẽ tạo từ dữ liệu có sẵn
    
    # Đọc positive reviews
    pos_path = os.path.join(base_path, "train", "pos")
    for file in os.listdir(pos_path):
        with open(os.path.join(pos_path, file), 'r', encoding='utf-8') as f:
            train_pos.append(f.read())
    
    # Đọc negative reviews
    neg_path = os.path.join(base_path, "train", "neg")
    for file in os.listdir(neg_path):
        with open(os.path.join(neg_path, file), 'r', encoding='utf-8') as f:
            train_neg.append(f.read())
    
    # Tạo neutral reviews từ positive và negative
    # Lấy một số review ngắn và trung tính
    for review in train_pos + train_neg:
        if len(review.split()) < 50:  # Review ngắn
            train_neu.append(review)
    
    # Giới hạn số lượng để cân bằng
    min_size = min(len(train_pos), len(train_neg), len(train_neu))
    train_pos = train_pos[:min_size]
    train_neg = train_neg[:min_size]
    train_neu = train_neu[:min_size]
    
    # Tạo DataFrame
    data = pd.DataFrame({
        'text': train_pos + train_neg + train_neu,
        'label': [2]*min_size + [0]*min_size + [1]*min_size  # 2: positive, 0: negative, 1: neutral
    })
    
    return data['text'], data['label']

def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản"""
    if not isinstance(text, str):
        return ''
    
    # Chuyển về chữ thường
    text = text.lower()
    
    # Xóa HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Xóa các ký tự đặc biệt và số
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Xóa khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text

def create_model_pipeline() -> Pipeline:
    """Tạo pipeline xử lý dữ liệu và model"""
    return Pipeline([
        ('vectorizer', TfidfVectorizer(
            use_idf=True,
            lowercase=True,
            strip_accents='ascii',
            stop_words=stopwords.words('english'),
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

def optimize_hyperparameters(X: pd.Series, y: pd.Series) -> dict:
    """Tối ưu hyperparameter với Optuna"""
    def objective(trial):
        params = {
            'max_features': trial.suggest_int('max_features', 1000, 10000),
            'ngram_range': trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (1, 3)]),
            'C': trial.suggest_loguniform('C', 1e-5, 1e5),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
        
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                use_idf=True,
                lowercase=True,
                strip_accents='ascii',
                stop_words=stopwords.words('english'),
                max_features=params['max_features'],
                ngram_range=params['ngram_range']
            )),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(
                C=params['C'],
                class_weight=params['class_weight'],
                max_iter=1000,
                random_state=42
            ))
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    return study.best_params

def train_and_evaluate(X: pd.Series, y: pd.Series) -> Tuple[Pipeline, dict]:
    """Train và đánh giá model"""
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tối ưu hyperparameter
    logging.info("Đang tối ưu hyperparameter...")
    best_params = optimize_hyperparameters(X_train, y_train)
    logging.info(f"Best parameters: {best_params}")
    
    # Tạo và train model
    logging.info("Đang train model...")
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logging.info(f"\nAccuracy: {accuracy:.2f}")
    logging.info("\nClassification Report:")
    logging.info(report)
    logging.info("\nConfusion Matrix:")
    logging.info(conf_matrix)
    
    return pipeline, {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'best_params': best_params
    }

def main():
    # Download stopwords nếu chưa có
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Tải và load dữ liệu IMDB
    if not os.path.exists("aclImdb"):
        download_imdb_dataset()
    
    X, y = load_imdb_data()
    
    # Tiền xử lý văn bản
    X = X.apply(preprocess_text)
    
    # Train và đánh giá model
    model, results = train_and_evaluate(X, y)
    
    # Lưu model và vectorizer
    logging.info("Đang lưu model...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu kết quả đánh giá
    with open('sentiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logging.info("Hoàn thành!")

if __name__ == '__main__':
    main() 