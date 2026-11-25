"""Representación del texto y arquitecturas.

Flujo de un lote:

    DataFrame -> RequirementsDataset -> DataLoader(collate_fn=collator) -> {"inputs", "labels"} -> modelo -> logits

- CNN y BiLSTM: el texto se pasa a minúsculas, se divide en palabras y se convierte en índices de
  un vocabulario construido solo con la partición de entrenamiento. Todas las secuencias se
  rellenan hasta la longitud del requisito de entrenamiento más largo, para no truncar ninguno.
- BETO, RoBERTa-BNE y DistilBETO: su tokenizador de subpalabras, con un máximo de 128 tokens.

Cada modelo tiene una salida sigmoide por etiqueta de la tarea: 1 en binary y 4 en subcategories.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import HF_REVISIONS, HF_SOURCES, MAX_LENGTH, ROBERTA_MIRROR, TEXT_COL, WORD_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- dataset


class RequirementsDataset(Dataset):
    """Pares (texto, vector de etiquetas) de una partición."""

    def __init__(self, frame: pd.DataFrame, labels: list[str]):
        self.ids = frame["ID"].astype(str).tolist()
        self.texts = frame[TEXT_COL].astype(str).tolist()
        if set(labels) <= set(frame.columns):
            self.labels = frame[labels].to_numpy(dtype=np.float32)
        else:  # filas sin etiqueta (p. ej. los requisitos sin seguridad que predice la tarea 2)
            self.labels = np.zeros((len(frame), len(labels)), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], self.labels[i]


# --------------------------------------------------------------------------- CNN y BiLSTM: palabras

PAD_IDX, UNK_IDX = 0, 1
WORD_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:['’-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)?")


def tokenize(text: str) -> list[str]:
    return WORD_PATTERN.findall(str(text).lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Palabra -> índice, de la más frecuente a la menos frecuente."""
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for token, _ in counts.most_common():
        vocab[token] = len(vocab)
    return vocab


class WordCollator:
    """Convierte un lote de textos en índices de palabras rellenados con <PAD>."""

    def __init__(self, train_texts: list[str]):
        self.vocab = build_vocab(train_texts)
        # longitud del requisito de entrenamiento más largo (al menos 5, el kernel más ancho de la CNN)
        self.max_len = max(5, max(len(tokenize(text)) for text in train_texts))

    def __call__(self, batch):
        texts, labels = zip(*batch)
        sequences = [[self.vocab.get(token, UNK_IDX) for token in tokenize(text)] or [UNK_IDX]
                     for text in texts]
        # un requisito de prueba más largo que todos los de entrenamiento ensancha su propio lote
        width = max(self.max_len, max(map(len, sequences)))
        input_ids = torch.full((len(sequences), width), PAD_IDX, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence)
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        inputs = {"input_ids": input_ids, "lengths": lengths}
        return {"inputs": inputs, "labels": torch.tensor(np.stack(labels))}


class TextCNN(nn.Module):
    """Embedding -> Conv1d -> ReLU -> max pooling -> dropout -> lineal."""

    def __init__(self, config: dict, vocab_size: int, n_labels: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["embedding_dim"], padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(config["embedding_dim"], config["num_filters"], config["kernel_size"])
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(config["num_filters"], n_labels)

    def forward(self, input_ids, lengths=None):
        x = self.embedding(input_ids)            # (lote, palabras, embedding)
        x = x.transpose(1, 2)                    # (lote, embedding, palabras) para Conv1d
        x = torch.relu(self.conv(x))             # (lote, filtros, posiciones)
        x = x.amax(dim=2)                        # max pooling sobre las posiciones
        return self.classifier(self.dropout(x))  # logits (lote, etiquetas)


class BiLSTMClassifier(nn.Module):
    """Embedding -> LSTM bidireccional -> últimos estados de ambas direcciones -> dropout -> lineal."""

    def __init__(self, config: dict, vocab_size: int, n_labels: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["embedding_dim"], padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(config["embedding_dim"], config["hidden_units"],
                            num_layers=config["num_layers"], batch_first=True, bidirectional=True,
                            dropout=config["dropout"] if config["num_layers"] > 1 else 0.0)
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(2 * config["hidden_units"], n_labels)

    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids)
        # pack_padded_sequence hace que la LSTM ignore el relleno
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        last = torch.cat((hidden[-2], hidden[-1]), dim=1)  # última capa: adelante y atrás
        return self.classifier(self.dropout(last))


# --------------------------------------------------------------------------- Transformers: subpalabras


@lru_cache(maxsize=None)
def transformer_source(model_name: str) -> str:
    """Repositorio de Hugging Face del modelo (RoBERTa-BNE cae al espejo si el oficial no tiene pesos)."""
    source = HF_SOURCES[model_name]
    if model_name != "roberta_bne":
        return source
    from huggingface_hub import list_repo_files
    try:
        if "config.json" in list_repo_files(source):
            return source
    except Exception:
        pass
    return ROBERTA_MIRROR


class TransformerCollator:
    """Tokeniza un lote con el tokenizador del modelo, rellenando hasta el texto más largo del lote."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.source = transformer_source(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.source, revision=HF_REVISIONS.get(self.source))

    def __call__(self, batch):
        texts, labels = zip(*batch)
        inputs = self.tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH, padding=True,
                                return_tensors="pt")
        return {"inputs": dict(inputs), "labels": torch.tensor(np.stack(labels))}


class TransformerClassifier(nn.Module):
    """Modelo preentrenado con una cabeza de clasificación; devuelve solo los logits."""

    def __init__(self, config: dict, source: str, n_labels: int):
        super().__init__()
        from transformers import AutoConfig, AutoModelForSequenceClassification, logging
        logging.set_verbosity_error()  # oculta el aviso de "cabeza recién inicializada" de cada fold
        revision = HF_REVISIONS.get(source)
        hf_config = AutoConfig.from_pretrained(source, revision=revision, num_labels=n_labels,
                                               problem_type="multi_label_classification")
        # Solo la cabeza toma el dropout ajustado; el codificador conserva el de su preentrenamiento
        # (BETO 0.1, RoBERTa-BNE 0.0, DistilBETO 0.1). DistilBERT lo lee de seq_classif_dropout.
        if hf_config.model_type == "distilbert":
            hf_config.seq_classif_dropout = config["dropout"]
        else:
            hf_config.classifier_dropout = config["dropout"]
        self.hf = AutoModelForSequenceClassification.from_pretrained(source, revision=revision, config=hf_config)

    def forward(self, **inputs):
        return self.hf(**inputs).logits


# --------------------------------------------------------------------------- fábrica


def build_collator(model_name: str, train_frame: pd.DataFrame):
    if model_name in WORD_MODELS:
        return WordCollator(train_frame[TEXT_COL].astype(str).tolist())
    return TransformerCollator(model_name)


def build_model(model_name: str, config: dict, n_labels: int, collator) -> nn.Module:
    if model_name == "cnn":
        model = TextCNN(config, len(collator.vocab), n_labels)
    elif model_name == "bilstm":
        model = BiLSTMClassifier(config, len(collator.vocab), n_labels)
    else:
        model = TransformerClassifier(config, collator.source, n_labels)
    return model.to(DEVICE)
