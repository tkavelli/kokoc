# scripts/gradient_boosting_models.py

import os
import joblib
import mlflow
import mlflow.lightgbm
import mlflow.catboost
import mlflow.xgboost
import optuna
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import logging
import yaml
from concurrent.futures import ProcessPoolExecutor

def load_config(config_path: str = 'configs/config.yml') -> Dict[str, Any]:
    """Загружает конфигурационный файл."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
logger = logging.getLogger(__name__)

mlflow.set_experiment("recommender_system_gradient_boosting")
mlflow.lightgbm.autolog()
mlflow.catboost.autolog()
mlflow.xgboost.autolog()

def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 25) -> float:
    """Вычисляет метрику Recall@K."""
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true) / np.sum(y_true)

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict[str, Any] = None) -> LGBMClassifier:
    """Обучает модель LightGBM."""
    if params is None:
        params = config['models']['gradient_boosting']['lgbm']
    
    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lambda y_true, y_pred: ('recall@25', recall_at_k(y_true, y_pred, k=25), True),
        early_stopping_rounds=10,
        verbose=100
    )
    logger.info("Модель LightGBM обучена.")
    return model

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict[str, Any] = None) -> CatBoostClassifier:
    """Обучает модель CatBoost."""
    if params is None:
        params = config['models']['gradient_boosting']['catboost']
    
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=10,
        verbose=100,
        use_best_model=True
    )
    logger.info("Модель CatBoost обучена.")
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict[str, Any] = None) -> XGBClassifier:
    """Обучает модель XGBoost."""
    if params is None:
        params = config['models']['gradient_boosting']['xgboost']
    
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lambda y_true, y_pred: ('recall@25', recall_at_k(y_true, y_pred, k=25), True),
        early_stopping_rounds=10,
        verbose=100
    )
    logger.info("Модель XGBoost обучена.")
    return model

def objective(trial: optuna.Trial, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
    """Целевая функция для оптимизации гиперпараметров."""
    if model_type == 'lightgbm':
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        params.update(config['models']['gradient_boosting']['lgbm'])
        model = LGBMClassifier(**params)
    elif model_type == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        }
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
        params.update(config['models']['gradient_boosting']['catboost'])
        model = CatBoostClassifier(**params, verbose=False)
    elif model_type == 'xgboost':
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        }
        params.update(config['models']['gradient_boosting']['xgboost'])
        model = XGBClassifier(**params)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    return recall_at_k(y_val.values, y_pred, k=25)

def optimize_model(model_type: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """Оптимизирует гиперпараметры модели."""
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_type, X, y), n_trials=n_trials)

    logger.info(f"Лучшие гиперпараметры {model_type.upper()}: {study.best_params}")
    logger.info(f"Лучший Recall@25 {model_type.upper()}: {study.best_value}")

    mlflow.log_params(study.best_params)
    mlflow.log_metric(f'{model_type}_best_recall@25', study.best_value)

    joblib.dump(study, f'models/{model_type}_study.pkl')

    return study.best_params

def parallel_optimize_models(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
    """Параллельно оптимизирует гиперпараметры для всех моделей."""
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(optimize_model, model_type, X, y, n_trials): model_type
            for model_type in ['lightgbm', 'catboost', 'xgboost']
        }
        results = {}
        for future in futures:
            model_type = futures[future]
            results[model_type] = future.result()
    return results

def compare_models(models: Dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Сравнивает модели и возвращает их Recall@25."""
    results = {}
    for name, model in models.items():
        preds = model.predict_proba(X_val)[:, 1]
        recall = recall_at_k(y_val.values, preds, k=25)
        results[name] = recall
        logger.info(f"{name} Recall@25: {recall:.4f}")
        mlflow.log_metric(f'{name}_recall@25', recall)

    best_model = max(results, key=results.get)
    logger.info(f"\nЛучшая модель: {best_model} с Recall@25: {results[best_model]:.4f}")
    return results

def plot_feature_importance(model: Any, feature_names: List[str], model_name: str):
    """Визуализирует важность признаков с помощью Plotly."""
    if isinstance(model, LGBMClassifier):
        importance = model.booster_.feature_importance(importance_type='gain')
    elif isinstance(model, CatBoostClassifier):
        importance = model.get_feature_importance()
    elif isinstance(model, XGBClassifier):
        importance = model.feature_importances_
    else:
        raise ValueError("Неподдерживаемый тип модели")

    sorted_idx = np.argsort(importance)
    sorted_features = [feature_names[i] for i in sorted_idx]

    fig = go.Figure(go.Bar(
        x=importance[sorted_idx],
        y=sorted_features,
        orientation='h'
    ))
    fig.update_layout(
        title=f'{model_name} - Важность признаков',
        xaxis_title='Важность',
        yaxis_title='Признаки'
    )
    fig.write_html(f'outputs/plots/{model_name}_feature_importance.html')
    logger.info(f"График важности признаков сохранен: outputs/plots/{model_name}_feature_importance.html")

def main():
    try:
        # Загрузка данных
        train_data = pd.read_parquet(config['data_paths']['processed']['train_data_resampled'])
        test_data = pd.read_parquet(config['data_paths']['processed']['test_data'])

        # Подготовка признаков и целевой переменной
        features = config['preprocessing']['feature_columns']
        X = train_data[features]
        y = train_data['is_purchase']

        # Масштабирование признаков
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=config['split']['validation_size'], random_state=config['split']['random_state'], stratify=y
        )
        logger.info("Данные разделены на обучающую и валидационную выборки.")

        # Параллельная оптимизация гиперпараметров
        best_params = parallel_optimize_models(X_train, y_train, n_trials=100)

        # Обучение моделей с оптимальными параметрами
        models = {
            'LightGBM': train_lightgbm(X_train, y_train, X_val, y_val, best_params['lightgbm']),
            'CatBoost': train_catboost(X_train, y_train, X_val, y_val, best_params['catboost']),
            'XGBoost': train_xgboost(X_train, y_train, X_val, y_val, best_params['xgboost'])
        }

        # Сравнение моделей
        compare_models(models, X_val, y_val)

        # Визуализация важности признаков
        for name, model in models.items():
            plot_feature_importance(model, features, name)

        # Сохранение моделей и скейлера
        os.makedirs('models', exist_ok=True)
        for name, model in models.items():
            if name == 'LightGBM':
                model.booster_.save_model(f'models/{name.lower()}_model.txt')
            elif name == 'CatBoost':
                model.save_model(f'models/{name.lower()}_model.cbm')
            elif name == 'XGBoost':
                model.save_model(f'models/{name.lower()}_model.json')
        joblib.dump(scaler, 'models/scaler.pkl')
        logger.info("Модели и скейлер сохранены в директории 'models/'")

        print("Модели обучены, оценены и сохранены в директории 'models/'")
        print("Графики важности признаков сохранены в директории 'outputs/plots/'")
    except Exception as e:
        logger.error(f"Ошибка в процессе обучения моделей: {str(e)}")
        raise

if __name__ == "__main__":
    main()