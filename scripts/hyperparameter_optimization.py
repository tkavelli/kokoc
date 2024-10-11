# scripts/hyperparameter_optimization.py

import optuna
import os
import logging
from scripts.config_loader import ConfigLoader
import mlflow

# Загрузка конфигурации
config_loader = ConfigLoader()
config = config_loader.load_config('config.yml')
mlflow_config = config['mlflow']
mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
mlflow.set_experiment(mlflow_config['experiment_name'])

logger = logging.getLogger(__name__)

def create_or_load_study(study_name, storage='sqlite:///optuna.db', direction='maximize'):
    """
    Создает новое исследование или загружает существующее.
    """
    study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, load_if_exists=True)
    logger.info(f"Study '{study_name}' loaded.")
    return study

def optimize_hyperparameters(model_name, n_trials=100, study_name=None):
    """
    Оптимизирует гиперпараметры для заданной модели.
    """
    logger.info(f"Starting hyperparameter optimization for {model_name}")

    model_config = config['models'][model_name]
    param_space = model_config.get('hyperparameters', {})

    def objective(trial):
        # Suggest hyperparameters
        params = {}
        for param, space in param_space.items():
            if space['type'] == 'categorical':
                value = trial.suggest_categorical(param, space['values'])
            elif space['type'] == 'int':
                value = trial.suggest_int(param, space['low'], space['high'])
            elif space['type'] == 'float':
                value = trial.suggest_float(param, space['low'], space['high'], log=space.get('log', False))
            else:
                raise ValueError(f"Unsupported parameter type: {space['type']}")
            params[param] = value

        # Update model configuration with suggested parameters
        model_config_updated = model_config.copy()
        model_config_updated.update(params)

        # Train and evaluate the model
        # Здесь должен быть код обучения и оценки модели
        metric = simulate_training_and_evaluation(model_config_updated)

        # Log parameters and metric to MLflow
        mlflow.log_params(params)
        mlflow.log_metric('metric', metric)

        return metric

    study = create_or_load_study(study_name or f"study_{model_name}")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Update the model configuration with the best parameters
    model_config.update(best_params)
    config_loader.save_config(config, 'config.yml')

    logger.info(f"Optimization completed for {model_name}. Best value: {best_value}")
    logger.info(f"Best parameters: {best_params}")

    return best_params, best_value

def simulate_training_and_evaluation(model_config):
    """
    Симулирует обучение и оценку.
    Замените это на реальный код обучения и оценки.
    """
    import random
    return random.uniform(0.5, 1.0)

if __name__ == "__main__":
    # Пример использования
    optimize_hyperparameters('advanced_transformer', n_trials=50, study_name='transformer_study')