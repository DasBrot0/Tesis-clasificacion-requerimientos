"""The four architectures and the text encoders that feed them.

CNN and BiLSTM use a word vocabulary built only from the training partition; BETO and
RoBERTa-BNE use their pretrained subword tokenizers (no stemming, no stopword removal).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import HF_REVISIONS, HF_SOURCES, LABEL_COLS, MAX_LENGTH, ROBERTA_MIRROR, TEXT_COL, WORD_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:['’-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)?")
PAD_IDX, UNK_IDX = 0, 1


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocab(texts: pd.Series) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for token, _ in counts.most_common():
        vocab.setdefault(token, len(vocab))
    return vocab


@dataclass
class EncodedData:
    """A partition encoded once and kept on DEVICE; the datasets are small enough for that."""

    ids: list[str]
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.ids)

    def batches(self, batch_size: int, generator: torch.Generator | None = None) -> Iterator[dict]:
        """Yield batches; shuffled when a generator is given."""
        order = torch.randperm(len(self), generator=generator) if generator is not None else torch.arange(len(self))
        for start in range(0, len(self), batch_size):
            index = order[start:start + batch_size].to(DEVICE)
            batch = {key: value[index] for key, value in self.inputs.items()}
            if "attention_mask" in batch:  # dynamic padding: trim to the longest text of the batch
                width = int(batch["attention_mask"].sum(dim=1).max())
                batch = {key: value[:, :width] for key, value in batch.items()}
            yield {"inputs": batch, "labels": self.labels[index]}


def _labels(frame: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(frame[LABEL_COLS].to_numpy(dtype=np.float32), device=DEVICE)


class WordEncoder:
    """Vocabulary and padding length come exclusively from the training partition."""

    def __init__(self, train_frame: pd.DataFrame):
        self.vocab = build_vocab(train_frame[TEXT_COL])
        lengths = np.array([max(1, len(tokenize(text))) for text in train_frame[TEXT_COL]])
        self.max_len = max(5, min(256, int(np.ceil(np.percentile(lengths, 95)))))

    def encode(self, frame: pd.DataFrame) -> EncodedData:
        input_ids = np.full((len(frame), self.max_len), PAD_IDX, dtype=np.int64)
        lengths = np.empty(len(frame), dtype=np.int64)
        for row, text in enumerate(frame[TEXT_COL]):
            ids = [self.vocab.get(token, UNK_IDX) for token in tokenize(text)][:self.max_len] or [UNK_IDX]
            input_ids[row, :len(ids)] = ids
            lengths[row] = len(ids)
        inputs = {"input_ids": torch.tensor(input_ids, device=DEVICE), "lengths": torch.tensor(lengths, device=DEVICE)}
        return EncodedData(frame["ID"].astype(str).tolist(), inputs, _labels(frame))


class TransformerEncoder:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.source = transformer_source(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.source, revision=HF_REVISIONS.get(self.source))

    def encode(self, frame: pd.DataFrame) -> EncodedData:
        encoded = self.tokenizer(frame[TEXT_COL].astype(str).tolist(), truncation=True, padding=True,
                                 max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in encoded.items()}
        return EncodedData(frame["ID"].astype(str).tolist(), inputs, _labels(frame))


def build_encoder(model_name: str, train_frame: pd.DataFrame) -> WordEncoder | TransformerEncoder:
    return WordEncoder(train_frame) if model_name in WORD_MODELS else TransformerEncoder(model_name)


class TextCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["embedding_dim"], padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(config["embedding_dim"], config["num_filters"], config["kernel_size"])
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(config["num_filters"], len(LABEL_COLS))

    def forward(self, input_ids, lengths=None):
        x = self.embedding(input_ids).transpose(1, 2)
        x = torch.relu(self.conv(x)).amax(dim=2)
        return self.classifier(self.dropout(x))


class BiLSTMClassifier(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["embedding_dim"], padding_idx=PAD_IDX)
        lstm_dropout = config["dropout"] if config["num_layers"] > 1 else 0.0
        self.lstm = nn.LSTM(config["embedding_dim"], config["hidden_units"],
                            num_layers=config["num_layers"], batch_first=True,
                            bidirectional=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(config["hidden_units"] * 2, len(LABEL_COLS))

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))


@lru_cache(maxsize=None)
def transformer_source(model_name: str) -> str:
    if model_name != "roberta_bne":
        return HF_SOURCES[model_name]
    from huggingface_hub import list_repo_files
    try:
        if "config.json" in set(list_repo_files(HF_SOURCES[model_name])):
            return HF_SOURCES[model_name]
    except Exception:
        pass
    print(f"AVISO: se usa el mirror explícito {ROBERTA_MIRROR} porque el repositorio BNE no publica pesos.")
    return ROBERTA_MIRROR


def build_transformer(config, source):
    from transformers import AutoConfig, AutoModelForSequenceClassification, logging
    logging.set_verbosity_error()  # silence the "newly initialized classifier" report of every fold
    revision = HF_REVISIONS.get(source)
    hf_config = AutoConfig.from_pretrained(source, revision=revision, num_labels=len(LABEL_COLS),
                                            problem_type="multi_label_classification")
    # Only the classification head takes the tuned dropout. The encoder keeps the dropout it was
    # pretrained with (BETO 0.1, RoBERTa-BNE 0.0): RoBERTa-BNE stops learning with 0.1 or more.
    hf_config.classifier_dropout = config["dropout"]
    return AutoModelForSequenceClassification.from_pretrained(source, revision=revision, config=hf_config)


def build_model(model_name: str, config: dict, encoder) -> nn.Module:
    if model_name == "cnn":
        model = TextCNN(config, len(encoder.vocab))
    elif model_name == "bilstm":
        model = BiLSTMClassifier(config, len(encoder.vocab))
    else:
        model = build_transformer(config, encoder.source)
    return model.to(DEVICE)


def forward(model: nn.Module, model_name: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if model_name in WORD_MODELS:
        return model(inputs["input_ids"], inputs["lengths"])
    return model(**inputs).logits
