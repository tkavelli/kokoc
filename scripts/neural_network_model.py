# scripts/neural_network_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torch.utils.data import DataLoader
import logging
import os

from scripts.config_loader import ConfigLoader
from scripts.utils import pad_sequences

config_loader = ConfigLoader()
config = config_loader.load_config('config.yml')

logging.basicConfig(level=logging.INFO, format=config['logging']['format'])
logger = logging.getLogger(__name__)

class TransformerRecommenderModel(pl.LightningModule):
    def __init__(self, num_users, num_products, embedding_dim, n_heads, n_layers, dropout, learning_rate, lr_scheduler_params, warmup_epochs):
        super(TransformerRecommenderModel, self).__init__()
        self.save_hyperparameters()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.product_embedding = nn.Embedding(num_products + 1, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.lr_scheduler_params = lr_scheduler_params
        self.warmup_epochs = warmup_epochs

    def forward(self, user_id, sequence):
        user_emb = self.user_embedding(user_id).unsqueeze(1)
        seq_emb = self.product_embedding(sequence)
        transformer_input = torch.cat([user_emb, seq_emb], dim=1)
        transformer_output = self.transformer_encoder(transformer_input)
        output = transformer_output[:, 0, :]
        output = self.dropout(output)
        logits = self.fc(output).squeeze(1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['user_id'], batch['sequence'])
        labels = batch['label']
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, **self.lr_scheduler_params)
        return [optimizer], [scheduler]

class RecommenderDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_sequence_length=10):
        self.data = data
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.data.iloc[idx]['user_id_encoded']
        product_id = self.data.iloc[idx]['product_id_encoded']
        label = self.data.iloc[idx]['is_purchase']
        sequence = [product_id]  # Для простоты используем только текущий продукт
        sequence = pad_sequences([sequence], self.max_sequence_length)[0]
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_neural_network():
    train_data = pd.read_parquet(config['data_paths']['processed']['train_data_nn'])
    test_data = pd.read_parquet(config['data_paths']['processed']['test_data'])

    num_users = train_data['user_id_encoded'].max() + 1
    num_products = train_data['product_id_encoded'].max() + 1

    train_dataset = RecommenderDataset(train_data)
    test_dataset = RecommenderDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = TransformerRecommenderModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=config['models']['advanced_transformer']['embedding_dim'],
        n_heads=config['models']['advanced_transformer']['n_heads'],
        n_layers=config['models']['advanced_transformer']['n_layers'],
        dropout=config['models']['advanced_transformer']['dropout'],
        learning_rate=config['models']['advanced_transformer']['learning_rate'],
        lr_scheduler_params=config['models']['advanced_transformer']['lr_scheduler_params'],
        warmup_epochs=config['models']['advanced_transformer']['warmup_epochs']
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger('logs/', name='TransformerModel'),
        log_every_n_steps=50
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(config['data_paths']['models']['transformer'])
    logger.info("Neural network training completed and model saved.")

if __name__ == "__main__":
    train_neural_network()