# scripts/utils.py

import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import dgl

class FlexibleLabelEncoder:
    """
    Расширенный LabelEncoder, который обрабатывает невиданные категории во время трансформации,
    присваивая им значение по умолчанию.
    """
    def __init__(self, unknown_value=None):
        self.encoder = LabelEncoder()
        self.classes_ = None
        self.class_to_index = {}
        self.unknown_value = unknown_value

    def fit(self, data):
        self.encoder.fit(data)
        self.classes_ = set(self.encoder.classes_)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.encoder.classes_)}
        if self.unknown_value is None:
            self.unknown_value = len(self.classes_)
        return self

    def transform(self, data):
        data_encoded = []
        for x in data:
            if x in self.class_to_index:
                data_encoded.append(self.class_to_index[x])
            else:
                data_encoded.append(self.unknown_value)
        return np.array(data_encoded)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        joblib.dump({
            'encoder': self.encoder,
            'unknown_value': self.unknown_value,
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.encoder = data['encoder']
        self.unknown_value = data['unknown_value']
        self.classes_ = set(self.encoder.classes_)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.encoder.classes_)}

def build_bipartite_graph(user_ids, product_ids):
    """
    Создает двудольный граф на основе пользователей и товаров.
    """
    edges_src = torch.tensor(user_ids, dtype=torch.int64)
    edges_dst = torch.tensor(product_ids, dtype=torch.int64)
    graph = dgl.graph((edges_src, edges_dst))
    graph = dgl.to_bidirected(graph)
    return graph

def pad_sequences(sequences, max_length, padding_value=0):
    """
    Дополняет последовательности до максимальной длины.
    """
    padded_sequences = np.full((len(sequences), max_length), padding_value)
    for idx, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded_sequences[idx, :length] = seq[:length]
    return padded_sequences

# Additional functions that were added in the revised version
def recall_at_k(y_true, y_pred, k=25):
    """
    Вычисляет метрику Recall@K.
    """
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true) / np.sum(y_true)

def generate_embeddings(title_codes, embeddings):
    """
    Генерирует эмбеддинги для заголовков на основе кодов слов.
    """
    title_embeddings = []
    for codes in title_codes:
        if isinstance(codes, str):
            codes = [int(code) for code in codes.split()]
        word_embeddings = embeddings[codes]
        title_embeddings.append(np.mean(word_embeddings, axis=0))
    return np.array(title_embeddings)