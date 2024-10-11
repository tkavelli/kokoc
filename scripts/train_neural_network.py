# scripts/data_preparation.py

import os
import sys
import time
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils import FlexibleLabelEncoder
from scripts.config_loader import ConfigLoader

config_loader = ConfigLoader()
config = config_loader.load_config('config.yml')

logging.basicConfig(level=logging.INFO, format=config['logging']['format'])
logger = logging.getLogger(__name__)

# Загрузка конфигураций MLflow из mlflow.yml
mlflow_config = config['mlflow']
mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
mlflow.set_experiment(mlflow_config['experiment_name'])

def safe_load_data() -> pd.DataFrame:
    try:
        catalog = pd.read_parquet(config['data_paths']['raw']['stokman_catalog_preprocessed'])
        actions = pd.read_parquet(config['data_paths']['raw']['train_actions'])
        vectors = np.load(config['data_paths']['raw']['vectors'], allow_pickle=True)
        embeddings = vectors['arr_0']
        logger.info(f"Catalog: {catalog.shape}, Actions: {actions.shape}, Vectors: {embeddings.shape}")
        return catalog, actions, embeddings
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_actions(actions: pd.DataFrame) -> pd.DataFrame:
    actions['date'] = pd.to_datetime(actions['date'], format=config['preprocessing']['date_format'])
    actions = actions.explode('products')
    actions.rename(columns={'products': 'product_id'}, inplace=True)
    actions.dropna(subset=['product_id'], inplace=True)
    actions['product_id'] = actions['product_id'].astype(int)
    relevant_actions = actions[actions['action'].isin(config['preprocessing']['relevant_actions'])]
    logger.info(f"Relevant actions: {relevant_actions.shape}")
    return relevant_actions

def generate_title_embeddings(data: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    # Предполагаем, что каждый номер в 'title' соответствует индексу эмбеддинга слова
    title_embeddings = []
    for idx, row in data.iterrows():
        try:
            title_codes = [int(code) for code in str(row['title']).split()]
            word_embeddings = embeddings[title_codes]  # Получаем эмбеддинги слов
            # Агрегируем эмбеддинги слов, например, путем усреднения
            title_embedding = np.mean(word_embeddings, axis=0)
            title_embeddings.append(title_embedding)
        except Exception as e:
            logger.error(f"Error processing title at index {idx}: {e}")
            title_embeddings.append(np.zeros(embeddings.shape[1]))
    data['title_embedding'] = title_embeddings
    return data

def merge_data(actions: pd.DataFrame, catalog: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    data = actions.merge(catalog, on='product_id', how='left')
    logger.info(f"Data shape after merging: {data.shape}")
    logger.info(f"Missing values after merging: {data.isna().sum()}")

    # Генерируем эмбеддинги названий товаров
    data = generate_title_embeddings(data, embeddings)
    logger.info("Title embeddings generated.")
    return data

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data['is_purchase'] = (data['action'] == 5).astype(int)
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    logger.info(f"Data shape after feature creation: {data.shape}")
    return data

def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    le_user = FlexibleLabelEncoder()
    le_product = FlexibleLabelEncoder()

    data['user_id_encoded'] = le_user.fit_transform(data['user_id'])
    data['product_id_encoded'] = le_product.fit_transform(data['product_id'])

    os.makedirs(os.path.dirname(config['data_paths']['interim']['encoders']['user']), exist_ok=True)
    joblib.dump(le_user, config['data_paths']['interim']['encoders']['user'])
    joblib.dump(le_product, config['data_paths']['interim']['encoders']['product'])
    logger.info("Encoders saved.")
    return data

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data,
        test_size=config['split']['test_size'],
        random_state=config['split']['random_state'],
        stratify=data['is_purchase']
    )
    logger.info(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
    return train_data, test_data

def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = config['preprocessing']['feature_columns']
    X = data[feature_columns].copy()

    # Распаковываем эмбеддинги в отдельные столбцы
    if 'title_embedding' in X.columns:
        title_embeddings = np.stack(X['title_embedding'].values)
        embedding_dim = title_embeddings.shape[1]
        embedding_columns = [f'title_emb_{i}' for i in range(embedding_dim)]
        title_embeddings_df = pd.DataFrame(title_embeddings, columns=embedding_columns)
        X = X.drop('title_embedding', axis=1)
        X = pd.concat([X.reset_index(drop=True), title_embeddings_df], axis=1)
    else:
        logger.warning("Title embeddings not found in data.")

    y = data['is_purchase']
    return X, y

def handle_missing_values_and_scale(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    logger.info(f"Missing values before imputation: {X.isna().sum()}")
    mlflow.log_param("missing_values_before_imputation", X.isna().sum().to_dict())

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    logger.info(f"Missing values after imputation: {pd.DataFrame(X_imputed, columns=X.columns).isna().sum()}")
    mlflow.log_param("missing_values_after_imputation", pd.DataFrame(X_imputed, columns=X.columns).isna().sum().to_dict())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    os.makedirs(os.path.dirname(config['data_paths']['interim']['imputer']), exist_ok=True)
    joblib.dump(imputer, config['data_paths']['interim']['imputer'])
    joblib.dump(scaler, config['data_paths']['interim']['scaler'])
    logger.info("Imputer and scaler saved.")

    mlflow.log_param("imputer_strategy", imputer.strategy)
    mlflow.log_param("scaler_mean", scaler.mean_.tolist())
    mlflow.log_param("scaler_scale", scaler.scale_.tolist())

    return X_scaled, scaler

def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Class distribution before SMOTE: {np.bincount(y)}")
    mlflow.log_param("class_distribution_before_smote", np.bincount(y).tolist())

    smote = SMOTE(random_state=config['smote']['random_state'])
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")
    mlflow.log_param("class_distribution_after_smote", np.bincount(y_resampled).tolist())

    return X_resampled, y_resampled

def save_processed_data(X_train: np.ndarray, y_train: np.ndarray, train_data: pd.DataFrame, test_data: pd.DataFrame):
    processed_dir = config['data_paths']['processed']
    os.makedirs(processed_dir, exist_ok=True)

    # Save resampled data for gradient boosting models
    train_data_resampled = pd.DataFrame(X_train)
    train_data_resampled['is_purchase'] = y_train
    train_data_resampled.to_parquet(config['data_paths']['processed']['train_data_resampled'])

    # Save original train data for neural networks (without SMOTE)
    train_data.to_parquet(config['data_paths']['processed']['train_data_nn'])

    # Save test data
    test_data.to_parquet(config['data_paths']['processed']['test_data'])

    logger.info("Processed data saved.")

def main():
    with mlflow.start_run():
        catalog, actions, embeddings = safe_load_data()

        relevant_actions = preprocess_actions(actions)
        data = merge_data(relevant_actions, catalog, embeddings)

        data = create_features(data)
        data = encode_categorical(data)

        train_data, test_data = split_data(data)

        X_train, y_train = prepare_features(train_data)
        X_train_scaled, scaler = handle_missing_values_and_scale(X_train)

        X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train.values)

        save_processed_data(X_train_resampled, y_train_resampled, train_data, test_data)

        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("num_samples", X_train.shape[0])
        mlflow.log_metric("class_balance", y_train.mean())

        logger.info("Data preparation completed successfully.")

if __name__ == "__main__":
    main()