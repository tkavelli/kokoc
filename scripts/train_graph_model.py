# train_graph_model.py

import os
import sys
import torch
import logging
import mlflow
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from datetime import datetime
from tqdm import tqdm

# Добавляем путь к utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils import GraphDataset, load_data
from scripts.graph_model import GATRecommenderModel

# Загружаем параметры из конфигурационных файлов
def load_config(config_path: str = 'configs/config.yml'):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке конфигурационного файла: {str(e)}")

def configure_logging(log_config: dict) -> logging.Logger:
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_dir = log_config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train_graph_model.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train(model, dataloader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc='Training', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        predictions = model(batch)
        labels = batch.y.to(device)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info(f'Training loss: {avg_loss:.4f}')
    return avg_loss

def evaluate(model, dataloader, device, logger):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            batch = batch.to(device)
            predictions = model(batch)
            labels = batch.y.to(device)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    recall = recall_score(all_labels, all_predictions > 0.5, average='micro')
    logger.info(f'Recall: {recall:.4f}')
    return recall

def main():
    # Загрузка конфигурации
    config = load_config('configs/config.yml')
    global logger
    logger = configure_logging(config['logging'])

    try:
        logger.info("Загрузка данных для обучения...")
        data_config = load_config('configs/data_paths.yml')
        data_paths = data_config['data_paths']

        train_data = load_data(data_paths['processed_train'])
        test_data = load_data(data_paths['processed_test'])

        # Создаем графовые датасеты
        train_dataset = GraphDataset(train_data)
        test_dataset = GraphDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GATRecommenderModel(
            in_feats=config['models']['gcn']['in_feats'],
            hidden_size=config['models']['gcn']['hidden_size'],
            num_classes=config['models']['gcn']['num_classes'],
            dropout=config['models']['gcn']['dropout']
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate_gcn'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_scheduler']['step_size'], gamma=config['training']['lr_scheduler']['gamma'])

        logger.info("Начало обучения...")
        mlflow.set_experiment('recommender_system_graph')
        with mlflow.start_run(run_name=f"train_gat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(config['training'])
            mlflow.log_params(config['models']['gcn'])

            for epoch in range(config['training']['max_epochs']):
                logger.info(f"Эпоха {epoch + 1}/{config['training']['max_epochs']}")
                train_loss = train(model, train_loader, criterion, optimizer, device, logger)
                recall = evaluate(model, test_loader, device, logger)
                scheduler.step()

                mlflow.log_metric('train_loss', train_loss, step=epoch)
                mlflow.log_metric('recall', recall, step=epoch)

            mlflow.pytorch.log_model(model, "gat_model")

        logger.info("Обучение завершено успешно.")

    except Exception as e:
        logger.error(f"Ошибка в процессе обучения: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()