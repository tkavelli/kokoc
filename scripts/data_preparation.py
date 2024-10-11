# scripts/data_preparation.py

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils import FlexibleLabelEncoder, generate_embeddings
from scripts.config_loader import ConfigLoader

config_loader = ConfigLoader()
config = config_loader.load_config('config.yml')

logging.basicConfig(level=logging.INFO, format=config['logging']['format'])
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

def safe_load_data():
    try:
        catalog = pd.read_parquet(config['data_paths']['raw']['stokman_catalog_preprocessed'])
        actions = pd.read_parquet(config['data_paths']['raw']['train_actions'])
        vectors = np.load(config['data_paths']['raw']['vectors'], allow_pickle=True)
        catalog_vector_mapping = pd.read_parquet(config['data_paths']['raw']['catalog_vector_mapping'])
        logger.info(f"Data loaded successfully. Shapes: Catalog: {catalog.shape}, Actions: {actions.shape}, Vectors: {vectors['arr_0'].shape}, Mapping: {catalog_vector_mapping.shape}")
        return catalog, actions, vectors['arr_0'], catalog_vector_mapping
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_actions(actions):
    actions['date'] = pd.to_datetime(actions['date'], format=config['preprocessing']['date_format'])
    actions = actions.explode('products')
    actions.rename(columns={'products': 'product_id'}, inplace=True)
    actions.dropna(subset=['product_id'], inplace=True)
    actions['product_id'] = actions['product_id'].astype(int)
    relevant_actions = actions[actions['action'].isin(config['preprocessing']['relevant_actions'])]
    logger.info(f"Preprocessed actions shape: {relevant_actions.shape}")
    return relevant_actions

def merge_data(actions, catalog, embeddings, catalog_vector_mapping):
    data = actions.merge(catalog, on='product_id', how='left')
    data = data.merge(catalog_vector_mapping, on='product_id', how='left')
    data['title_embedding'] = generate_embeddings(data['vector_id'], embeddings)
    logger.info(f"Merged data shape: {data.shape}")
    return data

def create_features(data):
    data['is_purchase'] = (data['action'] == 5).astype(int)
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    logger.info(f"Data shape after feature creation: {data.shape}")
    return data

def encode_categorical(data):
    le_user = FlexibleLabelEncoder()
    le_product = FlexibleLabelEncoder()

    data['user_id_encoded'] = le_user.fit_transform(data['user_id'])
    data['product_id_encoded'] = le_product.fit_transform(data['product_id'])

    encoder_dir = os.path.dirname(config['data_paths']['interim']['encoders']['user'])
    os.makedirs(encoder_dir, exist_ok=True)
    le_user.save(config['data_paths']['interim']['encoders']['user'])
    le_product.save(config['data_paths']['interim']['encoders']['product'])
    logger.info(f"Encoders saved to {encoder_dir}")
    return data

def split_data(data):
    train_data, test_data = train_test_split(
        data,
        test_size=config['split']['test_size'],
        random_state=config['split']['random_state'],
        stratify=data['is_purchase']
    )
    logger.info(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
    return train_data, test_data

def prepare_features(data):
    feature_columns = config['preprocessing']['feature_columns']
    X = data[feature_columns].copy()
    
    title_embeddings = np.stack(data['title_embedding'].values)
    embedding_columns = [f'title_emb_{i}' for i in range(title_embeddings.shape[1])]
    X = pd.concat([X, pd.DataFrame(title_embeddings, columns=embedding_columns)], axis=1)
    
    y = data['is_purchase']
    return X, y

def handle_missing_values_and_scale(X):
    logger.info(f"Missing values before imputation: {X.isna().sum()}")
    mlflow.log_param("missing_values_before_imputation", X.isna().sum().to_dict())

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    logger.info(f"Missing values after imputation: {pd.DataFrame(X_imputed, columns=X.columns).isna().sum()}")
    mlflow.log_param("missing_values_after_imputation", pd.DataFrame(X_imputed, columns=X.columns).isna().sum().to_dict())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    interim_dir = os.path.dirname(config['data_paths']['interim']['imputer'])
    os.makedirs(interim_dir, exist_ok=True)
    joblib.dump(imputer, config['data_paths']['interim']['imputer'])
    joblib.dump(scaler, config['data_paths']['interim']['scaler'])
    logger.info(f"Imputer and scaler saved to {interim_dir}")

    mlflow.log_param("imputer_strategy", imputer.strategy)
    mlflow.log_param("scaler_mean", scaler.mean_.tolist())
    mlflow.log_param("scaler_scale", scaler.scale_.tolist())

    return X_scaled, scaler

def apply_smote(X, y):
    logger.info(f"Class distribution before SMOTE: {np.bincount(y)}")
    mlflow.log_param("class_distribution_before_smote", np.bincount(y).tolist())

    plots_dir = config['output']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Not Purchase', 'Purchase'], y=np.bincount(y))
    plt.title("Class Distribution Before SMOTE")
    plt.savefig(os.path.join(plots_dir, 'class_distribution_before_smote.png'))
    plt.close()

    smote = SMOTE(random_state=config['smote']['random_state'])
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")
    mlflow.log_param("class_distribution_after_smote", np.bincount(y_resampled).tolist())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Not Purchase', 'Purchase'], y=np.bincount(y_resampled))
    plt.title("Class Distribution After SMOTE")
    plt.savefig(os.path.join(plots_dir, 'class_distribution_after_smote.png'))
    plt.close()

    logger.info(f"SMOTE plots saved to {plots_dir}")
    return X_resampled, y_resampled

def save_processed_data(X_train, y_train, train_data, test_data):
    processed_dir = config['data_paths']['processed']
    os.makedirs(processed_dir, exist_ok=True)

    train_data_resampled = pd.DataFrame(X_train)
    train_data_resampled['is_purchase'] = y_train
    train_data_resampled.to_parquet(config['data_paths']['processed']['train_data_resampled'])

    train_data.to_parquet(config['data_paths']['processed']['train_data_nn'])

    test_data.to_parquet(config['data_paths']['processed']['test_data'])

    logger.info(f"Processed data saved to {processed_dir}")

def main():
    with mlflow.start_run():
        catalog, actions, embeddings, catalog_vector_mapping = safe_load_data()

        relevant_actions = preprocess_actions(actions)
        data = merge_data(relevant_actions, catalog, embeddings, catalog_vector_mapping)

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