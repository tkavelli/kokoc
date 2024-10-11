# scripts/graph_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
from dgl.nn import GATConv
import pandas as pd
import logging
import os
import yaml
from torch.utils.data import DataLoader

from scripts.config_loader import ConfigLoader
from scripts.utils import FlexibleLabelEncoder, build_bipartite_graph

config_loader = ConfigLoader()
config = config_loader.load_config('config.yml')

logging.basicConfig(level=logging.INFO, format=config['logging']['format'])
logger = logging.getLogger(__name__)

class GATRecommenderModel(pl.LightningModule):
    def __init__(self, in_feats, hidden_size, num_heads, num_layers, num_classes, dropout, learning_rate):
        super(GATRecommenderModel, self).__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feats, hidden_size, num_heads[0], dropout, dropout))
        for l in range(1, num_layers-1):
            self.gat_layers.append(GATConv(hidden_size * num_heads[l-1], hidden_size, num_heads[l], dropout, dropout))
        self.gat_layers.append(GATConv(hidden_size * num_heads[-2], num_classes, num_heads[-1], dropout, dropout))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, g, features):
        h = features
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](g, h).flatten(1)
            h = F.elu(h)
        h = self.gat_layers[-1](g, h).mean(1)
        return h.squeeze()

    def training_step(self, batch, batch_idx):
        g, features, labels = batch
        logits = self(g, features)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        g, features, labels = batch
        logits = self(g, features)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss, 'preds': preds, 'targets': labels}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.user_ids = data['user_id_encoded'].values
        self.product_ids = data['product_id_encoded'].values
        self.labels = data['is_purchase'].values.astype(float)
        self.features = data.drop(columns=['user_id_encoded', 'product_id_encoded', 'is_purchase']).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        product_id = self.product_ids[idx]
        label = self.labels[idx]
        feature = torch.tensor(self.features[idx], dtype=torch.float32)

        # Создаем граф с одним ребром
        g = dgl.graph(([user_id], [product_id]))
        return g, feature, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    graphs, features, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    features = torch.stack(features)
    labels = torch.stack(labels)
    return batched_graph, features, labels

def train_graph_model():
    train_data = pd.read_parquet(config['data_paths']['processed']['train_data_nn'])
    test_data = pd.read_parquet(config['data_paths']['processed']['test_data'])

    train_dataset = GraphDataset(train_data)
    test_dataset = GraphDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = GATRecommenderModel(
        in_feats=train_dataset.features.shape[1],
        hidden_size=config['models']['advanced_gat']['hidden_size'],
        num_heads=config['models']['advanced_gat']['num_heads'],
        num_layers=config['models']['advanced_gat']['num_layers'],
        num_classes=1,
        dropout=config['models']['advanced_gat']['dropout'],
        learning_rate=config['models']['advanced_gat']['learning_rate']
    )

    trainer = pl.Trainer(
        max_epochs=config['models']['advanced_gat']['max_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger('logs/', name='GATModel'),
        log_every_n_steps=50
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(config['data_paths']['models']['gat'])
    logger.info("Graph model training completed and model saved.")

if __name__ == "__main__":
    train_graph_model()