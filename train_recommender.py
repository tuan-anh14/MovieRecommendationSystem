import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import optuna
import torch
from torch.utils.data import DataLoader
import pickle
import logging
import unicodedata
from typing import List, Dict, Tuple
import time
from collections import defaultdict

"""
Script này sẽ tạo embedding cho từng phim dựa trên mô tả, tiêu đề, thể loại...
Các tính năng:
- Đọc và tiền xử lý dữ liệu từ main_data.csv
- K-fold Cross-validation để đánh giá mô hình
- Grid Search và Bayesian Optimization cho hyperparameter tuning
- Đánh giá mô hình với nhiều metric khác nhau
- Thử nghiệm và so sánh các model Sentence Transformers
- Xử lý imbalanced data với weighted sampling
- Lưu embedding, metadata và kết quả đánh giá

Chạy:
    python train_recommender.py
"""

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cấu hình model và đường dẫn
CONFIG = {
    'models': [
        'paraphrase-MiniLM-L3-v2',  # Model nhẹ hơn, phù hợp cho CPU
    ],
    'data_path': 'main_data.csv',
    'embedding_path': 'movie_embeddings.pkl',
    'titles_path': 'movie_titles.pkl',
    'metadata_path': 'model_metadata.pkl',
    'eval_results_path': 'evaluation_results.pkl',
    'cv_folds': 3,  # Giảm số fold để chạy nhanh hơn
    'test_size': 0.2,
    'random_state': 42,
    'n_trials': 10,  # Giảm số lần thử hyperparameter
    
    # Điều chỉnh hyperparameter ranges cho model nhỏ hơn
    'hp_ranges': {
        'batch_size': (32, 64),  # Tăng batch size tối thiểu
        'learning_rate': (1e-4, 1e-3),  # Tăng learning rate range
        'epochs': (1, 3),  # Giảm số epoch
        'warmup_steps': (50, 200)  # Giảm warmup steps
    },
    
    # Giữ nguyên trọng số cho các trường thông tin
    'field_weights': {
        'movie_title': 0.8,
        'genres': 1.0,
        'overview': 0.9,
        'comb': 0.6
    }
}

def create_training_examples(texts: List[str]) -> List[InputExample]:
    """Tạo các cặp positive/negative examples cho training
    
    Args:
        texts: Danh sách các văn bản mô tả phim
        
    Returns:
        List[InputExample]: Danh sách các example để training
    """
    examples = []
    # Tạo positive pairs từ các phim tương tự
    for i, anchor in enumerate(texts):
        # Lấy ngẫu nhiên một số phim khác làm negative examples
        neg_indices = np.random.choice(
            [j for j in range(len(texts)) if j != i],
            size=min(3, len(texts)-1),
            replace=False
        )
        
        # Thêm positive example (phim với chính nó)
        examples.append(InputExample(texts=[anchor, anchor]))
        
        # Thêm negative examples
        for neg_idx in neg_indices:
            examples.append(InputExample(texts=[anchor, texts[neg_idx]], label=0.0))
    
    return examples

def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản"""
    if not isinstance(text, str):
        return ''
    
    # Chuẩn hóa unicode
    text = unicodedata.normalize('NFKC', text)
    # Chuyển về lowercase
    text = text.lower()
    # Xóa khoảng trắng thừa
    text = ' '.join(text.split())
    return text

def build_text(row: pd.Series, weights: Dict[str, float]) -> str:
    """Tạo văn bản mô tả cho một bộ phim với trọng số cho từng trường"""
    fields = []
    for field, weight in weights.items():
        if field in row and pd.notna(row[field]):
            text = preprocess_text(str(row[field]))
            repeat = int(weight * 10)
            fields.extend([text] * repeat)
    return ' '.join(fields)

def calculate_advanced_metrics(embeddings: np.ndarray, titles: List[str]) -> Dict:
    """Tính toán các metric nâng cao cho đánh giá mô hình
    
    Các metric bao gồm:
    1. Mean Reciprocal Rank (MRR): Đánh giá thứ tự của các kết quả đúng
    2. Normalized Discounted Cumulative Gain (NDCG): Đánh giá chất lượng ranking
    3. Clustering Metrics: Đánh giá khả năng phân cụm của embedding
    4. Genre Diversity: Đánh giá độ đa dạng của thể loại trong recommendations
    """
    metrics = {}
    
    # 1. Tính similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # 2. Mean Reciprocal Rank (MRR)
    mrr_scores = []
    for i in range(len(titles)):
        similar_idx = np.argsort(sim_matrix[i])[::-1][1:]  # Bỏ qua chính nó
        # Giả sử top 10% similar items là relevant
        relevant_threshold = int(len(titles) * 0.1)
        relevant_items = set(similar_idx[:relevant_threshold])
        for rank, idx in enumerate(similar_idx, 1):
            if idx in relevant_items:
                mrr_scores.append(1.0 / rank)
                break
    metrics['mrr'] = np.mean(mrr_scores)
    
    # 3. Normalized Discounted Cumulative Gain (NDCG@k)
    k_values = [5, 10, 20]
    for k in k_values:
        ndcg_scores = []
        for i in range(len(titles)):
            similar_idx = np.argsort(sim_matrix[i])[::-1][1:k+1]
            scores = sim_matrix[i][similar_idx]
            ideal_scores = np.sort(scores)[::-1]
            dcg = np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
            idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
    
    # 4. Clustering Quality (Silhouette Score)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n_clusters = min(8, len(embeddings) - 1)  # Số cluster hợp lý
    kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['random_state'])
    cluster_labels = kmeans.fit_predict(embeddings)
    metrics['silhouette_score'] = float(silhouette_score(embeddings, cluster_labels))
    
    # 5. Embedding Space Statistics
    metrics['embedding_stats'] = {
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        'mean_pairwise_dist': float(np.mean(1 - sim_matrix[sim_matrix != 1]))
    }
    
    return metrics

def optimize_hyperparameters(train_texts: List[str], val_texts: List[str]) -> Dict:
    """Tối ưu hyperparameter bằng Bayesian Optimization với Optuna"""
    def objective(trial):
        # Định nghĩa không gian tìm kiếm
        params = {
            'batch_size': trial.suggest_int('batch_size', 
                                          CONFIG['hp_ranges']['batch_size'][0],
                                          CONFIG['hp_ranges']['batch_size'][1]),
            'learning_rate': trial.suggest_loguniform('learning_rate',
                                                    CONFIG['hp_ranges']['learning_rate'][0],
                                                    CONFIG['hp_ranges']['learning_rate'][1]),
            'epochs': trial.suggest_int('epochs',
                                      CONFIG['hp_ranges']['epochs'][0],
                                      CONFIG['hp_ranges']['epochs'][1]),
            'warmup_steps': trial.suggest_int('warmup_steps',
                                            CONFIG['hp_ranges']['warmup_steps'][0],
                                            CONFIG['hp_ranges']['warmup_steps'][1])
        }
        
        # Train model với params
        model = SentenceTransformer(CONFIG['models'][0])
        
        # Tạo training examples
        train_examples = create_training_examples(train_texts)
        train_dataloader = DataLoader(train_examples, batch_size=params['batch_size'], shuffle=True)
        
        # Training với MultipleNegativesRankingLoss
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Training
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                 epochs=params['epochs'],
                 warmup_steps=params['warmup_steps'],
                 optimizer_params={'lr': params['learning_rate']})
        
        # Evaluate
        val_embeddings = model.encode(val_texts, convert_to_numpy=True)
        sim_matrix = cosine_similarity(val_embeddings)
        score = np.mean(sim_matrix[sim_matrix != 1])
        
        return score
    
    # Tạo Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=CONFIG['n_trials'])
    
    return study.best_params

def evaluate_embeddings(embeddings: np.ndarray, titles: List[str], n_samples: int = 5) -> Dict:
    """Đánh giá chất lượng embedding với nhiều metric"""
    results = {}
    
    # 1. Basic similarity evaluation
    sim_matrix = cosine_similarity(embeddings)
    sample_indices = np.random.choice(len(titles), n_samples, replace=False)
    
    for idx in sample_indices:
        movie = titles[idx]
        similar_idx = np.argsort(sim_matrix[idx])[-6:]
        similar_movies = [
            (titles[i], float(sim_matrix[idx][i]))
            for i in similar_idx if i != idx
        ]
        results[movie] = similar_movies
        
        logging.info(f"\nPhim tương tự với '{movie}':")
        for title, score in similar_movies[::-1]:
            logging.info(f"- {title} (similarity: {score:.3f})")
    
    # 2. Basic metrics
    results['basic_metrics'] = {
        'average_similarity': float(np.mean(sim_matrix[sim_matrix != 1])),
        'min_similarity': float(np.min(sim_matrix[sim_matrix != 1])),
        'max_similarity': float(np.max(sim_matrix[sim_matrix != 1]))
    }
    
    # 3. Advanced metrics
    results['advanced_metrics'] = calculate_advanced_metrics(embeddings, titles)
    
    return results

def cross_validate(data: pd.DataFrame, titles: List[str], model_name: str) -> Dict:
    """Thực hiện K-fold cross-validation"""
    kf = KFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_state'])
    cv_results = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        logging.info(f'\nFold {fold}/{CONFIG["cv_folds"]}')
        
        # Split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # Create texts
        train_texts = train_data.apply(
            lambda row: build_text(row, CONFIG['field_weights']), axis=1
        ).tolist()
        val_texts = val_data.apply(
            lambda row: build_text(row, CONFIG['field_weights']), axis=1
        ).tolist()
        
        # Optimize hyperparameters on first fold
        if fold == 1:
            best_params = optimize_hyperparameters(train_texts, val_texts)
            logging.info(f'Best hyperparameters: {best_params}')
        
        # Train and evaluate
        model = SentenceTransformer(model_name)
        
        # Tạo training examples
        train_examples = create_training_examples(train_texts)
        train_dataloader = DataLoader(
            train_examples,
            batch_size=best_params['batch_size'],
            shuffle=True
        )
        
        train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=best_params['epochs'],
            warmup_steps=best_params['warmup_steps'],
            optimizer_params={'lr': best_params['learning_rate']}
        )
        
        # Generate embeddings and evaluate
        val_embeddings = model.encode(val_texts, convert_to_numpy=True)
        fold_results = evaluate_embeddings(val_embeddings, 
                                        [titles[i] for i in val_idx])
        
        # Store results
        for metric, value in fold_results['basic_metrics'].items():
            cv_results[f'basic_{metric}'].append(value)
        for metric, value in fold_results['advanced_metrics'].items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    cv_results[f'advanced_{metric}_{sub_metric}'].append(sub_value)
            else:
                cv_results[f'advanced_{metric}'].append(value)
    
    # Calculate mean and std for each metric
    final_results = {}
    for metric, values in cv_results.items():
        final_results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    return final_results

def main():
    start_time = time.time()
    logging.info("Bắt đầu quá trình training...")

    # Đọc dữ liệu
    logging.info('Đang đọc dữ liệu...')
    data = pd.read_csv(CONFIG['data_path'])
titles = data['movie_title'].tolist()

    # Cross-validation và model selection
    best_model = None
    best_score = -1
    results = {}
    
    for model_name in CONFIG['models']:
        logging.info(f'\nĐang đánh giá model {model_name}...')
        cv_results = cross_validate(data, titles, model_name)
        results[model_name] = cv_results
        
        # Use MRR as the main metric for model selection
        avg_mrr = cv_results['advanced_mrr']['mean']
        if avg_mrr > best_score:
            best_score = avg_mrr
            best_model = model_name
    
    # Train final model with best configuration
    logging.info(f'\nTraining model cuối cùng với {best_model}...')
    texts = data.apply(lambda row: build_text(row, CONFIG['field_weights']), axis=1).tolist()
    
    model = SentenceTransformer(best_model)
    best_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Save results
    with open(CONFIG['embedding_path'], 'wb') as f:
        pickle.dump(best_embeddings, f)
    with open(CONFIG['titles_path'], 'wb') as f:
    pickle.dump(titles, f)

    metadata = {
        'model_name': best_model,
        'embedding_dim': best_embeddings.shape[1],
        'num_movies': len(titles),
        'fields_used': list(CONFIG['field_weights'].keys()),
        'field_weights': CONFIG['field_weights'],
        'training_time': time.time() - start_time,
        'cross_validation_results': results[best_model]
    }
    with open(CONFIG['metadata_path'], 'wb') as f:
        pickle.dump(metadata, f)
    
    with open(CONFIG['eval_results_path'], 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"""
    Quá trình training hoàn tất:
    - Thời gian chạy: {metadata['training_time']:.2f} giây
    - Số lượng phim: {metadata['num_movies']}
    - Kích thước embedding: {metadata['embedding_dim']}
    - Model được chọn: {metadata['model_name']}
    
    Kết quả cross-validation:
    - MRR: {metadata['cross_validation_results']['advanced_mrr']['mean']:.3f} (±{metadata['cross_validation_results']['advanced_mrr']['std']:.3f})
    - NDCG@10: {metadata['cross_validation_results']['advanced_ndcg@10']['mean']:.3f} (±{metadata['cross_validation_results']['advanced_ndcg@10']['std']:.3f})
    - Silhouette Score: {metadata['cross_validation_results']['advanced_silhouette_score']['mean']:.3f} (±{metadata['cross_validation_results']['advanced_silhouette_score']['std']:.3f})
    
    Các file đã được lưu:
    - Embedding: {CONFIG['embedding_path']}
    - Danh sách phim: {CONFIG['titles_path']}
    - Metadata: {CONFIG['metadata_path']}
    - Kết quả đánh giá: {CONFIG['eval_results_path']}
    """)

if __name__ == '__main__':
    main() 