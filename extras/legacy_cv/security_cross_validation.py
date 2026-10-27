"""Leakage-safe 5-fold training pipeline for all four thesis models.

The frozen test split is never constructed as a DataLoader during CV. Thresholds
come only from out-of-fold predictions; the final model is then trained once on
the complete train+old-validation pool and may be evaluated once on frozen test.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from security_multilabel_common import (
    CV_FOLDS_PATH, DATA_PATH, LABEL_COLS, N_SPLITS, SEED, SPLIT_PATH, TEXT_COL,
    fold_distribution_frame, get_cv_pool_and_test, get_or_create_cv_fold_manifest,
    is_better, load_security_dataset, multilabel_metrics, prediction_frame,
    save_json, seed_everything, select_per_label_thresholds,
)
from security_train_augmentation import augment_balanced_subcategories

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS = Path("artifacts")
CHECKPOINTS = Path("checkpoints")
TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:['’-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)?")
PAD_IDX, UNK_IDX = 0, 1
METRIC_NAMES = [
    "macro_f1", "weighted_f1", "micro_f1", "macro_precision",
    "security_recall", "subset_accuracy", "hamming_loss",
]

CONFIGS = {
    "cnn": {
        "embedding_dim": 64, "num_filters": 100, "kernel_size": 3,
        "batch_size": 32, "epochs": 50, "dropout": 0.2,
        "learning_rate": 0.001, "optimizer": "Adam", "threshold": 0.50,
    },
    "bilstm": {
        "embedding_dim": 100, "hidden_units": 64, "num_layers": 1,
        "batch_size": 32, "epochs": 30, "dropout": 0.2,
        "learning_rate": 0.001, "optimizer": "Adam", "threshold": 0.50,
    },
    "beto": {
        "learning_rate": 2e-5, "batch_size": 16, "epochs": 5, "dropout": 0.1,
        "weight_decay": 0.01, "max_length": 128, "optimizer": "AdamW", "threshold": 0.50,
    },
    "roberta_bne": {
        "learning_rate": 2e-5, "batch_size": 16, "epochs": 5, "dropout": 0.1,
        "weight_decay": 0.01, "max_length": 128, "optimizer": "AdamW", "threshold": 0.50,
    },
}

SEARCH_TRIALS = {
    "cnn": [
        CONFIGS["cnn"],
        {**CONFIGS["cnn"], "embedding_dim": 100, "num_filters": 128, "kernel_size": 4,
         "batch_size": 16, "dropout": 0.3, "learning_rate": 0.0005, "threshold": 0.45},
        {**CONFIGS["cnn"], "embedding_dim": 128, "kernel_size": 5, "epochs": 60,
         "dropout": 0.5, "learning_rate": 0.005, "threshold": 0.55},
    ],
    "bilstm": [
        CONFIGS["bilstm"],
        {**CONFIGS["bilstm"], "embedding_dim": 64, "hidden_units": 32, "num_layers": 2,
         "batch_size": 16, "epochs": 20, "dropout": 0.3, "learning_rate": 0.0005,
         "threshold": 0.45},
        {**CONFIGS["bilstm"], "embedding_dim": 128, "hidden_units": 128,
         "batch_size": 50, "epochs": 50, "dropout": 0.5, "learning_rate": 0.002,
         "threshold": 0.55},
    ],
    "beto": [],
    "roberta_bne": [],
}
SEARCH_TRIALS["beto"] = [CONFIGS["beto"],
    {**CONFIGS["beto"], "learning_rate": 1e-5, "batch_size": 8, "epochs": 10,
     "dropout": 0.2, "weight_decay": 1e-4, "max_length": 256, "threshold": 0.45},
    {**CONFIGS["beto"], "learning_rate": 5e-5, "epochs": 15, "dropout": 0.5,
     "weight_decay": 0.1, "threshold": 0.55}]
SEARCH_TRIALS["roberta_bne"] = [dict(item) for item in SEARCH_TRIALS["beto"]]

MODEL_KEYS = {
    "cnn": "cnn_security", "bilstm": "bilstm_security",
    "beto": "beto_security", "roberta_bne": "roberta_bne_security",
}
MODEL_NAMES = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "roberta_bne": "PlanTL-GOB-ES/roberta-base-bne",
}
ROBERTA_MIRROR = "Buzzeitor/roberta-base-bne"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocab(texts: pd.Series) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for token, _ in counts.most_common():
        vocab.setdefault(token, len(vocab))
    return vocab


class WordDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int], max_len: int):
        self.frame, self.vocab, self.max_len = frame.reset_index(drop=True), vocab, max_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        ids = [self.vocab.get(token, UNK_IDX) for token in tokenize(row[TEXT_COL])][:self.max_len] or [UNK_IDX]
        length = len(ids)
        ids += [PAD_IDX] * (self.max_len - length)
        return {"id": str(row["ID"]), "input_ids": torch.tensor(ids),
                "length": torch.tensor(length),
                "labels": torch.tensor(row[LABEL_COLS].to_numpy(dtype=np.float32))}


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


def criterion_for(frame: pd.DataFrame, moderate_pos_weight: bool):
    if not moderate_pos_weight:
        return nn.BCEWithLogitsLoss()
    positives = frame[LABEL_COLS].sum().to_numpy(dtype=float)
    weights = np.minimum(np.sqrt((len(frame) - positives) / np.maximum(positives, 1.0)), 10.0)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))


def evaluate_word(model, loader, criterion, threshold):
    model.eval(); ids, truths, probs, total = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(DEVICE)
            logits = model(batch["input_ids"].to(DEVICE), batch["length"].to(DEVICE))
            total += criterion(logits, labels).item() * len(labels)
            ids.extend(batch["id"]); truths.append(labels.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
    y_true, y_prob = np.vstack(truths), np.vstack(probs)
    metrics = multilabel_metrics(y_true, y_prob, threshold)
    metrics.update({"loss": total / len(loader.dataset), "ids": ids, "y_true": y_true, "y_prob": y_prob})
    return metrics


def train_word_fold(model_name, config, train_frame, validation_frame, fold, trial,
                    use_augmentation, moderate_pos_weight):
    seed_everything(SEED + trial * 100 + fold)
    original_train_ids = set(train_frame["ID"].astype(str))
    if use_augmentation:
        train_frame = augment_balanced_subcategories(train_frame, output_path=None,
                                                      seed=SEED + trial * 100 + fold)
        sources = set(train_frame.loc[train_frame["Is_augmented"], "Augmentation_source_ID"].astype(str))
        if not sources <= original_train_ids:
            raise AssertionError("Augmentation usó una fuente fuera del train del fold.")
    if set(train_frame["ID"].astype(str)) & set(validation_frame["ID"].astype(str)):
        raise AssertionError("Train y validation del fold comparten IDs.")
    vocab = build_vocab(train_frame[TEXT_COL])
    lengths = np.array([max(1, len(tokenize(text))) for text in train_frame[TEXT_COL]])
    max_len = max(5, min(256, int(np.ceil(np.percentile(lengths, 95)))))
    generator = torch.Generator().manual_seed(SEED + fold)
    train_loader = DataLoader(WordDataset(train_frame, vocab, max_len), batch_size=config["batch_size"],
                              shuffle=True, generator=generator)
    val_loader = DataLoader(WordDataset(validation_frame, vocab, max_len),
                            batch_size=config["batch_size"], shuffle=False)
    cls = TextCNN if model_name == "cnn" else BiLSTMClassifier
    model = cls(config, len(vocab)).to(DEVICE)
    criterion = criterion_for(train_frame, moderate_pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, best_state, history = None, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train(); total = 0.0
        for batch in train_loader:
            labels = batch["labels"].to(DEVICE); optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(DEVICE), batch["length"].to(DEVICE))
            loss = criterion(logits, labels); loss.backward()
            if model_name == "bilstm": nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += loss.item() * len(labels)
        current = evaluate_word(model, val_loader, criterion, config["threshold"])
        history.append({"trial_id": trial, "fold": fold, "epoch": epoch,
                        "train_loss": total / len(train_loader.dataset), "val_loss": current["loss"],
                        "val_macro_f1": current["macro_f1"],
                        "val_weighted_f1": current["weighted_f1"],
                        "val_security_recall": current["security_recall"]})
        if is_better(current, best):
            best = {"macro_f1": current["macro_f1"], "security_recall": current["security_recall"],
                    "loss": current["loss"], "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    metrics = evaluate_word(model, val_loader, criterion, config["threshold"])
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics, history, int(best["epoch"])


class TransformerDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame, self.tokenizer, self.max_length = frame.reset_index(drop=True), tokenizer, max_length

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(str(row[TEXT_COL]), truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(row[LABEL_COLS].to_numpy(dtype=np.float32))
        item["id"] = str(row["ID"])
        return item


def transformer_source(model_name):
    if model_name != "roberta_bne": return MODEL_NAMES[model_name]
    from huggingface_hub import list_repo_files
    try:
        if "config.json" in set(list_repo_files(MODEL_NAMES[model_name])):
            return MODEL_NAMES[model_name]
    except Exception:
        pass
    print(f"AVISO: se usa el mirror explícito {ROBERTA_MIRROR} porque el repositorio BNE no publica pesos.")
    return ROBERTA_MIRROR


def build_transformer(config, source):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    hf_config = AutoConfig.from_pretrained(source, num_labels=len(LABEL_COLS),
                                            problem_type="multi_label_classification")
    hf_config.hidden_dropout_prob = config["dropout"]
    hf_config.attention_probs_dropout_prob = config["dropout"]
    hf_config.classifier_dropout = config["dropout"]
    return AutoModelForSequenceClassification.from_pretrained(source, config=hf_config)


def forward_transformer(model, batch):
    inputs = {key: value.to(DEVICE) for key, value in batch.items() if key not in {"id", "labels"}}
    return model(**inputs).logits


def evaluate_transformer(model, loader, criterion, threshold):
    model.eval(); ids, truths, probs, total = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(DEVICE); logits = forward_transformer(model, batch)
            total += criterion(logits, labels).item() * len(labels)
            ids.extend(batch["id"]); truths.append(labels.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
    y_true, y_prob = np.vstack(truths), np.vstack(probs)
    metrics = multilabel_metrics(y_true, y_prob, threshold)
    metrics.update({"loss": total / len(loader.dataset), "ids": ids, "y_true": y_true, "y_prob": y_prob})
    return metrics


def train_transformer_fold(model_name, config, train_frame, validation_frame, fold, trial,
                           use_augmentation, moderate_pos_weight):
    from transformers import AutoTokenizer
    seed_everything(SEED + trial * 100 + fold)
    original_train_ids = set(train_frame["ID"].astype(str))
    if use_augmentation:
        train_frame = augment_balanced_subcategories(train_frame, output_path=None,
                                                      seed=SEED + trial * 100 + fold)
        sources = set(train_frame.loc[train_frame["Is_augmented"], "Augmentation_source_ID"].astype(str))
        if not sources <= original_train_ids:
            raise AssertionError("Augmentation usó una fuente fuera del train del fold.")
    source = transformer_source(model_name); tokenizer = AutoTokenizer.from_pretrained(source)
    train_loader = DataLoader(TransformerDataset(train_frame, tokenizer, config["max_length"]),
                              batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(SEED + fold))
    val_loader = DataLoader(TransformerDataset(validation_frame, tokenizer, config["max_length"]),
                            batch_size=config["batch_size"], shuffle=False)
    model = build_transformer(config, source).to(DEVICE)
    criterion = criterion_for(train_frame, moderate_pos_weight)
    optimizer_cls = torch.optim.AdamW if config["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])
    best, best_state, history = None, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train(); total = 0.0
        for batch in train_loader:
            labels = batch["labels"].to(DEVICE); optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward_transformer(model, batch), labels)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += loss.item() * len(labels)
        current = evaluate_transformer(model, val_loader, criterion, config["threshold"])
        history.append({"trial_id": trial, "fold": fold, "epoch": epoch,
                        "train_loss": total / len(train_loader.dataset), "val_loss": current["loss"],
                        "val_macro_f1": current["macro_f1"],
                        "val_weighted_f1": current["weighted_f1"],
                        "val_security_recall": current["security_recall"]})
        if is_better(current, best):
            best = {"macro_f1": current["macro_f1"], "security_recall": current["security_recall"],
                    "loss": current["loss"], "epoch": epoch}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = evaluate_transformer(model, val_loader, criterion, config["threshold"])
    del model, best_state; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics, history, int(best["epoch"])


def verify_frozen_test(test_frame: pd.DataFrame):
    frozen_path = ARTIFACTS / "security_test_frozen_seed42.csv"
    digest_path = ARTIFACTS / "security_test_frozen_seed42.sha256"
    if not frozen_path.exists() or not digest_path.exists():
        raise FileNotFoundError("Faltan el CSV o checksum del test congelado.")
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0]
    actual_digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise AssertionError("El archivo del test congelado no coincide con su SHA-256.")
    frozen = pd.read_csv(frozen_path, encoding="utf-8-sig")
    if set(frozen["ID"].astype(str)) != set(test_frame["ID"].astype(str)):
        raise AssertionError("Los IDs del test actual no coinciden con el test congelado.")
    return actual_digest


def fold_frames(cv_pool, manifest, fold):
    val_ids = set(manifest.loc[manifest["fold"].astype(int).eq(fold), "ID"].astype(str))
    is_val = cv_pool["ID"].astype(str).isin(val_ids)
    train_frame, validation_frame = cv_pool[~is_val].copy(), cv_pool[is_val].copy()
    if train_frame["_text_key"].isin(set(validation_frame["_text_key"])).any():
        raise AssertionError("Leakage de texto entre train y validation.")
    return train_frame, validation_frame


def aggregate_rows(fold_rows):
    aggregate = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        aggregate[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return aggregate


def final_train_and_test(model_name, config, epochs, cv_pool, test_frame, thresholds,
                         use_augmentation, moderate_pos_weight, model_key):
    """Train without validation/early stopping; touch test only after training ends."""
    seed_everything(SEED)
    train_frame = cv_pool.copy()
    if use_augmentation:
        train_frame = augment_balanced_subcategories(train_frame, output_path=None, seed=SEED)
    final_config = {**config, "epochs": int(epochs)}
    checkpoint_dir = CHECKPOINTS / model_key / "final_cv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_sentinel = ARTIFACTS / f"{model_key}_final_test_consumed.json"
    criterion = criterion_for(train_frame, moderate_pos_weight)
    if model_name in {"cnn", "bilstm"}:
        vocab = build_vocab(train_frame[TEXT_COL])
        lengths = np.array([max(1, len(tokenize(text))) for text in train_frame[TEXT_COL]])
        max_len = max(5, min(256, int(np.ceil(np.percentile(lengths, 95)))))
        cls = TextCNN if model_name == "cnn" else BiLSTMClassifier
        model = cls(final_config, len(vocab)).to(DEVICE)
        train_loader = DataLoader(WordDataset(train_frame, vocab, max_len), batch_size=config["batch_size"],
                                  shuffle=True, generator=torch.Generator().manual_seed(SEED))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                labels = batch["labels"].to(DEVICE); optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch["input_ids"].to(DEVICE), batch["length"].to(DEVICE)), labels)
                loss.backward()
                if model_name == "bilstm": nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        torch.save({"model_state_dict": model.state_dict(), "config": final_config, "vocab": vocab,
                    "max_len": max_len, "thresholds": thresholds.tolist(), "label_cols": LABEL_COLS,
                    "seed": SEED}, checkpoint_dir / "model.pt")
        # Test is instantiated for the first time only after final training is complete.
        save_json(test_sentinel, {"status": "evaluation_started", "frozen_test_rows": len(test_frame),
                                  "seed": SEED})
        test_loader = DataLoader(WordDataset(test_frame, vocab, max_len), batch_size=config["batch_size"])
        return evaluate_word(model, test_loader, criterion, thresholds)
    from transformers import AutoTokenizer
    source = transformer_source(model_name); tokenizer = AutoTokenizer.from_pretrained(source)
    model = build_transformer(final_config, source).to(DEVICE)
    train_loader = DataLoader(TransformerDataset(train_frame, tokenizer, config["max_length"]),
                              batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    optimizer_cls = torch.optim.AdamW if config["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch["labels"].to(DEVICE); optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward_transformer(model, batch), labels)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.save_pretrained(checkpoint_dir); tokenizer.save_pretrained(checkpoint_dir)
    save_json(checkpoint_dir / "metrics_config.json", {"config": final_config,
              "thresholds_by_label": dict(zip(LABEL_COLS, thresholds.tolist())), "seed": SEED,
              "model_source_used": source})
    save_json(test_sentinel, {"status": "evaluation_started", "frozen_test_rows": len(test_frame),
                              "seed": SEED})
    test_loader = DataLoader(TransformerDataset(test_frame, tokenizer, config["max_length"]),
                             batch_size=config["batch_size"], shuffle=False)
    return evaluate_transformer(model, test_loader, criterion, thresholds)


def update_comparison_table():
    names = {"cnn_security": "CNN", "bilstm_security": "BiLSTM",
             "beto_security": "BETO", "roberta_bne_security": "RoBERTa-BNE"}
    rows = []
    for key, display_name in names.items():
        path = ARTIFACTS / f"{key}_cv_summary.json"
        if not path.exists(): continue
        report = json.loads(path.read_text(encoding="utf-8")); cv = report["cv_aggregate"]
        test = report.get("final_test") or {}
        fmt = lambda name: f"{cv[name]['mean']:.4f} ± {cv[name]['std']:.4f}"
        rows.append({"Modelo": display_name, "CV Macro-F1": fmt("macro_f1"),
                     "CV Weighted-F1": fmt("weighted_f1"),
                     "CV Recall Security": fmt("security_recall"),
                     "Test Macro-F1": test.get("macro_f1"),
                     "Test Recall Security": test.get("security_recall")})
    pd.DataFrame(rows).to_csv(ARTIFACTS / "model_comparison_cv.csv", index=False, encoding="utf-8-sig")


def run(args):
    seed_everything(SEED); ARTIFACTS.mkdir(exist_ok=True)
    data = load_security_dataset(DATA_PATH, allow_provisional_labels=args.allow_provisional_labels)
    cv_pool, test_frame, split_manifest = get_cv_pool_and_test(data, SPLIT_PATH)
    test_digest = verify_frozen_test(test_frame)
    fold_manifest = get_or_create_cv_fold_manifest(cv_pool, CV_FOLDS_PATH, N_SPLITS, SEED)
    fold_distribution_frame(cv_pool, fold_manifest).to_csv(
        ARTIFACTS / "security_cv_fold_label_distribution.csv", index=False, encoding="utf-8-sig")
    print(f"Pool CV={len(cv_pool)} | test congelado={len(test_frame)} | SHA-256={test_digest}")
    print("Test no participa en CV; folds:", fold_manifest["fold"].value_counts().sort_index().to_dict())
    if args.protocol_only:
        print("Protocolo validado para modelos:", ", ".join(CONFIGS))
        return

    model_key = MODEL_KEYS[args.model]
    test_sentinel = ARTIFACTS / f"{model_key}_final_test_consumed.json"
    if args.final_test and test_sentinel.exists():
        raise RuntimeError(
            f"El test final de {args.model} ya fue consumido ({test_sentinel}). "
            "No se permite una segunda evaluación accidental."
        )
    configs = SEARCH_TRIALS[args.model] if args.hparam_search else [CONFIGS[args.model]]
    trial_records, histories, trial_outputs = [], [], {}
    fold_runner = train_word_fold if args.model in {"cnn", "bilstm"} else train_transformer_fold
    for trial, config in enumerate(configs):
        outputs, base_fold_rows = [], []
        print(f"\n{args.model}: trial {trial}, config={config}")
        for fold in range(N_SPLITS):
            train_frame, validation_frame = fold_frames(cv_pool, fold_manifest, fold)
            metrics, history, best_epoch = fold_runner(
                args.model, config.copy(), train_frame, validation_frame, fold, trial,
                args.use_augmentation, args.moderate_pos_weight)
            histories.extend(history)
            outputs.append({"fold": fold, "metrics": metrics, "best_epoch": best_epoch})
            row = {"trial_id": trial, "fold": fold, "best_epoch": best_epoch,
                   **{name: metrics[name] for name in METRIC_NAMES}}
            base_fold_rows.append(row)
            print(f"fold={fold+1}/{N_SPLITS} macro_f1={metrics['macro_f1']:.4f} "
                  f"weighted_f1={metrics['weighted_f1']:.4f} security_recall={metrics['security_recall']:.4f}")
        trial_outputs[trial] = outputs
        aggregate = aggregate_rows(base_fold_rows)
        trial_records.append({"trial_id": trial, **config,
                              "cv_macro_f1_mean": aggregate["macro_f1"]["mean"],
                              "cv_macro_f1_std": aggregate["macro_f1"]["std"],
                              "cv_security_recall_mean": aggregate["security_recall"]["mean"],
                              "mean_best_epoch": float(np.mean([o["best_epoch"] for o in outputs]))})
    trials = pd.DataFrame(trial_records).sort_values(
        ["cv_macro_f1_mean", "cv_security_recall_mean"], ascending=False, kind="stable").reset_index(drop=True)
    winner_id = int(trials.iloc[0]["trial_id"]); winning_config = dict(configs[winner_id])
    outputs = trial_outputs[winner_id]

    oof_parts = []
    for output in outputs:
        metrics = output["metrics"]
        part = pd.DataFrame({"ID": metrics["ids"], "fold": output["fold"]})
        for index, label in enumerate(LABEL_COLS):
            part[f"{label}_true"] = metrics["y_true"][:, index]
            part[f"{label}_prob"] = metrics["y_prob"][:, index]
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != len(cv_pool) or oof["ID"].astype(str).duplicated().any() or set(oof["ID"].astype(str)) != set(cv_pool["ID"].astype(str)):
        raise AssertionError("Las predicciones OOF no cubren exactamente una vez todo el pool de CV.")
    true_cols = [f"{label}_true" for label in LABEL_COLS]
    prob_cols = [f"{label}_prob" for label in LABEL_COLS]
    thresholds, threshold_search = select_per_label_thresholds(
        oof[true_cols].to_numpy(dtype=int), oof[prob_cols].to_numpy(dtype=float))

    fold_rows, label_rows = [], []
    for fold in range(N_SPLITS):
        part = oof[oof["fold"].eq(fold)]
        metrics = multilabel_metrics(part[true_cols].to_numpy(), part[prob_cols].to_numpy(), thresholds)
        fold_rows.append({"fold": fold, "best_epoch": outputs[fold]["best_epoch"],
                          **{name: metrics[name] for name in METRIC_NAMES}})
        labels = metrics["per_label"].copy(); labels.insert(0, "fold", fold)
        label_rows.extend(labels.to_dict("records"))
        for index, label in enumerate(LABEL_COLS):
            oof.loc[oof["fold"].eq(fold), f"{label}_pred"] = metrics["y_pred"][:, index]
    aggregate = aggregate_rows(fold_rows)
    label_frame = pd.DataFrame(label_rows)
    label_aggregate = (
        label_frame.groupby("label", sort=False)[["precision", "recall", "f1"]]
        .agg(["mean", "std"]).reset_index()
    )
    label_aggregate.columns = [
        "label" if column[0] == "label" else f"{column[0]}_{column[1]}"
        for column in label_aggregate.columns
    ]
    for name in METRIC_NAMES:
        print(f"{name}: {aggregate[name]['mean']:.4f} ± {aggregate[name]['std']:.4f}")

    prefix = ARTIFACTS / model_key
    pd.DataFrame(histories).to_csv(f"{prefix}_cv_history.csv", index=False, encoding="utf-8-sig")
    trials.to_csv(f"{prefix}_cv_trials.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(f"{prefix}_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    label_frame.to_csv(f"{prefix}_cv_fold_metrics_by_label.csv", index=False, encoding="utf-8-sig")
    label_aggregate.to_csv(f"{prefix}_cv_label_metrics_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(f"{prefix}_oof_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_search.to_csv(f"{prefix}_oof_threshold_search.csv", index=False, encoding="utf-8-sig")
    final_epochs = max(1, int(round(np.mean([item["best_epoch"] for item in outputs]))))
    report: dict[str, Any] = {
        "model": args.model, "model_key": model_key, "seed": SEED, "n_splits": N_SPLITS,
        "cv_pool_size": len(cv_pool), "frozen_test_size": len(test_frame),
        "frozen_test_sha256": test_digest, "winning_config": winning_config,
        "final_training_epochs": final_epochs,
        "thresholds_by_label": dict(zip(LABEL_COLS, thresholds.tolist())),
        "cv_folds": fold_rows, "cv_aggregate": aggregate,
        "cv_per_label_aggregate": label_aggregate.to_dict("records"), "final_test": None,
        "augmentation": bool(args.use_augmentation),
    }
    if args.final_test:
        test_metrics = final_train_and_test(
            args.model, winning_config, final_epochs, cv_pool, test_frame, thresholds,
            args.use_augmentation, args.moderate_pos_weight, model_key)
        prediction_frame(test_metrics["ids"], test_metrics).to_csv(
            f"{prefix}_final_test_predictions.csv", index=False, encoding="utf-8-sig")
        test_metrics["per_label"].to_csv(
            f"{prefix}_final_test_metrics_by_label.csv", index=False, encoding="utf-8-sig")
        report["final_test"] = {name: test_metrics[name] for name in METRIC_NAMES}
        report["final_test"]["loss"] = test_metrics["loss"]
        save_json(Path(f"{prefix}_final_test_metrics.json"), report["final_test"])
        save_json(test_sentinel, {"status": "evaluation_completed", "frozen_test_rows": len(test_frame),
                                  "seed": SEED, "metrics_path": f"{prefix}_final_test_metrics.json"})
        print("Final Test Results:", report["final_test"])
    else:
        print("Test continúa reservado. Use --final-test una sola vez cuando la configuración esté fijada.")
    save_json(Path(f"{prefix}_cv_summary.json"), report)
    save_json(Path(f"{prefix}_winning_config.json"), {
        "config": winning_config, "final_training_epochs": final_epochs,
        "thresholds_by_label": report["thresholds_by_label"], "selected_from": "OOF only"})
    update_comparison_table()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CONFIGS), default="cnn")
    parser.add_argument("--hparam-search", action="store_true")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--moderate-pos-weight", action="store_true")
    parser.add_argument("--final-test", action="store_true",
                        help="Entrena con todo el pool y evalúa una sola vez el test congelado.")
    parser.add_argument("--protocol-only", action="store_true",
                        help="Valida splits, checksum, folds y registro de los cuatro modelos sin entrenar.")
    parser.add_argument("--allow-provisional-labels", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
