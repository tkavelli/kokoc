# scripts/ensemble_model.py

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, ndcg_score
from scripts.neural_network_model import TransformerRecommenderModel, RecommenderDataset
from scripts.graph_model import GATRecommenderModel, GraphDataset
import torch
from torch.utils.data import DataLoader
from scipy.optimize import minimize
import logging
import yaml
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple

def load_config(config_path: str = 'configs/config.yml') -> Dict[str, Any]:
    """Загружает конфигурационный файл."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def configure_logging(log_config: Dict[str, Any]) -> logging.Logger:
    """Настраивает логирование."""
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_dir = log_config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "ensemble_model.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

config = load_config()
logger = configure_logging(config['logging'])

def load_models() -> Tuple[LGBMClassifier, CatBoostClassifier, XGBClassifier, TransformerRecommenderModel, GATRecommenderModel]:
    """Загружает обученные модели."""
    try:
        lgb_model = LGBMClassifier()
        lgb_model.booster_ = lgb_model._Booster = lgb_model._Booster.load_model('models/lightgbm_model.txt')
        
        cat_model = CatBoostClassifier()
        cat_model.load_model('models/catboost_model.cbm')
        
        xgb_model = XGBClassifier()
        xgb_model.load_model('models/xgboost_model.json')
        
        nn_model = TransformerRecommenderModel.load_from_checkpoint('models/transformer_model.ckpt')
        
        gcn_model = GATRecommenderModel.load_from_checkpoint('models/gat_model.ckpt')
        
        logger.info("Все модели успешно загружены.")
        return lgb_model, cat_model, xgb_model, nn_model, gcn_model
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {str(e)}")
        raise

def get_model_predictions(models: Tuple[Any, ...], X: pd.DataFrame, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Получает предсказания от всех моделей."""
    lgb_model, cat_model, xgb_model, nn_model, gcn_model = models
    preds = {}

    # Предсказания градиентных бустингов
    preds['lgb'] = lgb_model.predict_proba(X)[:, 1]
    preds['cat'] = cat_model.predict_proba(X)[:, 1]
    preds['xgb'] = xgb_model.predict_proba(X)[:, 1]
    
    # Предсказания нейронной сети
    nn_dataset = RecommenderDataset(dataset)
    nn_loader = DataLoader(nn_dataset, batch_size=512, shuffle=False)
    nn_preds = []
    nn_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model.to(device)
    with torch.no_grad():
        for batch in nn_loader:
            logits = nn_model(batch['user_id'].to(device), batch['sequence'].to(device))
            preds_batch = torch.sigmoid(logits)
            nn_preds.extend(preds_batch.cpu().numpy())
    preds['nn'] = np.array(nn_preds)
    
    # Предсказания графовой модели
    gcn_dataset = GraphDataset(dataset)
    gcn_loader = DataLoader(gcn_dataset, batch_size=1, shuffle=False)
    gcn_preds = []
    gcn_model.eval()
    gcn_model.to(device)
    with torch.no_grad():
        for batch in gcn_loader:
            g, labels = batch
            logits = gcn_model(g[0], g[0].ndata['feat'])
            preds_batch = torch.sigmoid(logits)
            gcn_preds.extend(preds_batch.cpu().numpy())
    preds['gcn'] = np.array(gcn_preds)
    
    return preds

def ensemble_predictions(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Комбинирует предсказания моделей с заданными весами."""
    total_weight = sum(weights.values())
    ensemble_pred = sum(preds[model_name] * weight for model_name, weight in weights.items()) / total_weight
    return ensemble_pred

def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 25) -> float:
    """Вычисляет метрику Recall@K."""
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true) / np.sum(y_true)

def optimize_weights(models: Tuple[Any, ...], X_val: pd.DataFrame, y_val: pd.Series, val_dataset: pd.DataFrame) -> Dict[str, float]:
    """Оптимизирует веса моделей в ансамбле для максимизации Recall@25 на валидационной выборке."""
    logger.info("Оптимизация весов моделей в ансамбле...")
    preds = get_model_predictions(models, X_val, val_dataset)
    model_names = list(preds.keys())

    def objective(weights):
        weights_dict = dict(zip(model_names, weights))
        ensemble_pred = ensemble_predictions(preds, weights_dict)
        recall = recall_at_k(y_val.values, ensemble_pred, k=25)
        return -recall  # Минус, так как мы минимизируем

    # Начальные веса
    initial_weights = np.ones(len(model_names)) / len(model_names)

    # Ограничения: веса >= 0 и сумма весов = 1
    constraints = (
        {'type': 'eq', 'fun': lambda w: 1 - sum(w)},
    )
    bounds = [(0, 1)] * len(model_names)

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = dict(zip(model_names, result.x))
        logger.info(f"Оптимальные веса: {optimized_weights}")
        return optimized_weights
    else:
        logger.warning("Оптимизация весов не удалась, будут использованы равные веса.")
        return dict(zip(model_names, initial_weights))

def evaluate_ensemble(models: Tuple[Any, ...], X_val: pd.DataFrame, y_val: pd.Series, val_dataset: pd.DataFrame, weights: Dict[str, float] = None) -> Dict[str, float]:
    """Оценивает производительность ансамбля."""
    if weights is None:
        weights = {model_name: 1/len(models) for model_name in ['lgb', 'cat', 'xgb', 'nn', 'gcn']}
    preds = get_model_predictions(models, X_val, val_dataset)
    ensemble_pred = ensemble_predictions(preds, weights)
    
    metrics = {
        'recall@25': recall_at_k(y_val.values, ensemble_pred, k=25),
        'ndcg@25': ndcg_score(y_val.values.reshape(1, -1), ensemble_pred.reshape(1, -1), k=25),
        'auc': roc_auc_score(y_val, ensemble_pred)
    }
    
    for metric_name, metric_value in metrics.items():
        logger.info(f'Ensemble {metric_name.upper()}: {metric_value:.4f}')
        mlflow.log_metric(f'ensemble_{metric_name}', metric_value)
    
    return metrics

def save_ensemble_predictions(models: Tuple[Any, ...], X_test: pd.DataFrame, test_dataset: pd.DataFrame, weights: Dict[str, float]):
    """Сохраняет предсказания ансамбля."""
    preds = get_model_predictions(models, X_test, test_dataset)
    ensemble_pred = ensemble_predictions(preds, weights)
    
    # Получаем топ-25 продуктов для каждого пользователя
    test_dataset['prediction'] = ensemble_pred
    top_25_products = test_dataset.groupby('user_id').apply(
        lambda x: x.nlargest(25, 'prediction')['product_id'].tolist()
    ).reset_index()
    top_25_products.columns = ['user_id', 'products']
    
    output_path = config['output']['predictions_dir']
    os.makedirs(output_path, exist_ok=True)
    top_25_products.to_csv(os.path.join(output_path, 'ensemble_predictions.csv'), index=False)
    logger.info(f"Ensemble predictions saved to {output_path}/ensemble_predictions.csv")

def visualize_ensemble_results(metrics: Dict[str, float], weights: Dict[str, float]):
    """Визуализирует результаты ансамбля."""
    # Метрики
    fig_metrics = go.Figure(data=[go.Bar(x=list(metrics.keys()), y=list(metrics.values()))])
    fig_metrics.update_layout(title='Ensemble Metrics', xaxis_title='Metric', yaxis_title='Value')
    fig_metrics.write_html(os.path.join(config['output']['plots_dir'], 'ensemble_metrics.html'))

    # Веса моделей
    fig_weights = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()))])
    fig_weights.update_layout(title='Model Weights in Ensemble')
    fig_weights.write_html(os.path.join(config['output']['plots_dir'], 'ensemble_weights.html'))

    logger.info("Ensemble visualizations saved.")

def main():
    try:
        # Загрузка моделей и данных
        models = load_models()
        val_data = pd.read_parquet(config['data_paths']['processed']['test_data'])
        X_val = val_data[config['preprocessing']['feature_columns']]
        y_val = val_data['is_purchase']

        # Оптимизация весов
        optimized_weights = optimize_weights(models, X_val, y_val, val_data)

        # Оценка ансамбля с оптимизированными весами
        metrics = evaluate_ensemble(models, X_val, y_val, val_data, weights=optimized_weights)

        # Визуализация результатов
        visualize_ensemble_results(metrics, optimized_weights)

        # Сохранение предсказаний ансамбля
        test_data = pd.read_parquet(config['data_paths']['processed']['test_data'])
        X_test = test_data[config['preprocessing']['feature_columns']]
        save_ensemble_predictions(models, X_test, test_data, optimized_weights)

        logger.info("Ensemble model processing completed successfully.")
    except Exception as e:
        logger.error(f"Error in ensemble model processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()