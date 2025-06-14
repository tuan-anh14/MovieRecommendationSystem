import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import logging
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    """Load và chuẩn bị dữ liệu"""
    logging.info("Đang đọc dữ liệu...")
    train_data = pd.read_csv('../datasets/processed/processed_train.csv')
    
    # Chỉ giữ lại class 0 (negative) và 1 (neutral)
    train_data = train_data[train_data['sentiment'].isin([0, 1])]
    
    # Phân tích class distribution
    class_dist = train_data['sentiment'].value_counts()
    logging.info("\nPhân bố dữ liệu theo class:")
    logging.info(class_dist)
    
    # Lấy 50% dữ liệu để giảm thời gian train nhưng vẫn đảm bảo độ chính xác
    sample_size = int(len(train_data) * 0.5)
    train_data = train_data.sample(n=sample_size, random_state=42)
    
    X = train_data['text']
    y = train_data['sentiment']
    
    logging.info(f"\nKích thước dữ liệu sau khi lấy mẫu: {len(train_data)}")
    logging.info("Phân bố dữ liệu sau khi lấy mẫu:")
    logging.info(train_data['sentiment'].value_counts())
    
    return X, y

def create_model_pipeline(model_type='lr'):
    """Tạo pipeline cho mô hình với nhiều lựa chọn"""
    vectorizer = TfidfVectorizer(
        max_features=8000, # Giới hạn vocabulary size
        ngram_range=(1, 2), # Unigrams và bigrams
        min_df=2, # Loại bỏ từ xuất hiện < 2 lần
        max_df=0.95 # Loại bỏ từ xuất hiện > 95% documents
    )
    
    if model_type == 'lr':
        classifier = LogisticRegression(
            C=1.0,
            max_iter=500,
            solver='saga',
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'svm':
        classifier = SVC(
            C=1.0,
            kernel='linear',
            probability=True,
            random_state=42
        )
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('classifier', classifier)
    ])
    
    return pipeline

def plot_roc_curves(y_test, y_score, model_name):
    """Vẽ ROC curves cho binary classification"""
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'../models/roc_curve_{model_name}.png')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """Đánh giá mô hình với nhiều metrics"""
    logging.info(f"\nĐánh giá mô hình {model_name}:")
    
    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Các metrics cơ bản
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'../models/confusion_matrix_{model_name}.png')
    plt.close()
    
    # Các metrics nâng cao
    metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'average_precision': average_precision_score(y_test, y_pred_proba[:, 1])
    }
    
    print("\nMetrics nâng cao:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Vẽ ROC curve
    plot_roc_curves(y_test, y_pred_proba, model_name)
    
    return metrics

def cross_validate_model(pipeline, X, y, model_name):
    """Thực hiện cross-validation với StratifiedKFold"""
    logging.info(f"\nCross-validation cho {model_name}:")
    
    # Sử dụng StratifiedKFold để giữ tỷ lệ class trong mỗi fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Thực hiện cross-validation với nhiều metrics
    cv_scores = {
        'f1_macro': cross_val_score(pipeline, X, y, cv=skf, scoring='f1_macro'),
        'precision_macro': cross_val_score(pipeline, X, y, cv=skf, scoring='precision_macro'),
        'recall_macro': cross_val_score(pipeline, X, y, cv=skf, scoring='recall_macro')
    }
    
    print("\nCross-validation scores:")
    for metric, scores in cv_scores.items():
        print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_scores

def tune_hyperparameters(pipeline, X_train, y_train, model_type):
    """Tối ưu hyperparameters cho mô hình"""
    logging.info(f"\nTối ưu hyperparameters cho {model_type}...")
    
    if model_type == 'lr':
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'vectorizer__max_features': [5000, 8000]
        }
    elif model_type == 'rf':
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15],
            'vectorizer__max_features': [5000, 8000]
        }
    elif model_type == 'svm':
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'vectorizer__max_features': [5000, 8000]
        }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # Giảm số fold để tăng tốc
        scoring='f1_macro',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, vectorizer, model_name, metrics, cv_scores):
    """Lưu mô hình và kết quả"""
    os.makedirs('../models', exist_ok=True)
    
    # Lưu mô hình
    with open(f'../models/sentiment_model_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu vectorizer
    with open(f'../models/tfidf_vectorizer_{model_name}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Lưu metrics và cv_scores
    results = {
        'metrics': metrics,
        'cv_scores': cv_scores
    }
    with open(f'../models/results_{model_name}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Đã lưu mô hình {model_name} và kết quả vào thư mục models/")

def main():
    # Load dữ liệu
    X, y = load_data()
    
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Danh sách các mô hình cần thử
    models = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'svm': 'SVM'
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = None
    
    # Thử nghiệm các mô hình
    for model_type, model_name in models.items():
        logging.info(f"\nThử nghiệm mô hình {model_name}...")
        start_time = time.time()
        
        # Tạo pipeline
        pipeline = create_model_pipeline(model_type)
        
        # Tối ưu hyperparameters
        best_pipeline = tune_hyperparameters(pipeline, X_train, y_train, model_type)
        
        # Cross-validation
        cv_scores = cross_validate_model(best_pipeline, X_train, y_train, model_name)
        
        # Đánh giá trên test set
        metrics = evaluate_model(best_pipeline, X_test, y_test, model_name)
        
        # Lưu mô hình
        save_model(best_pipeline, best_pipeline.named_steps['vectorizer'], model_type, metrics, cv_scores)
        
        # Cập nhật best model
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_model = best_pipeline
            best_model_name = model_type
        
        train_time = time.time() - start_time
        logging.info(f"Thời gian huấn luyện {model_name}: {train_time:.2f} giây")
    
    logging.info(f"\nMô hình tốt nhất: {models[best_model_name]} với F1-score: {best_f1:.4f}")

if __name__ == '__main__':
    main() 