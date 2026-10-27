"""One training routine shared by the CV folds and by the final model."""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import DEFAULT_THRESHOLD, EARLY_STOPPING_PATIENCE, LABEL_COLS, SEED, WORD_MODELS
from .metrics import is_better, multilabel_metrics
from .models import DEVICE, EncodedData, build_encoder, build_model, forward

# Required by cuBLAS for deterministic GPU runs; must be set before the first CUDA call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def criterion_for(frame: pd.DataFrame, moderate_pos_weight: bool) -> nn.Module:
    """Plain BCE, or BCE weighted by min(sqrt(N_neg/N_pos), 10) computed on the training frame."""
    if not moderate_pos_weight:
        return nn.BCEWithLogitsLoss()
    positives = frame[LABEL_COLS].sum().to_numpy(dtype=float)
    weights = np.minimum(np.sqrt((len(frame) - positives) / np.maximum(positives, 1.0)), 10.0)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))


def build_optimizer(model_name: str, config: dict, model: nn.Module, steps_per_epoch: int):
    if model_name in WORD_MODELS:
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"]), None
    optimizer_cls = torch.optim.AdamW if config["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])
    if not config["warmup_ratio"]:
        return optimizer, None  # constant learning rate, as in the original notebooks
    from transformers import get_linear_schedule_with_warmup
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(round(config["warmup_ratio"] * total_steps)), total_steps)
    return optimizer, scheduler


def predict(model: nn.Module, model_name: str, data: EncodedData, criterion: nn.Module,
            batch_size: int) -> tuple[np.ndarray, float]:
    """Return sigmoid probabilities and the mean loss over `data`."""
    model.eval()
    probs, total = [], 0.0
    use_amp = DEVICE.type == "cuda" and model_name not in WORD_MODELS
    with torch.no_grad():
        for batch in data.batches(batch_size):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                logits = forward(model, model_name, batch["inputs"])
            logits = logits.float()
            total += criterion(logits, batch["labels"]).item() * len(batch["labels"])
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs), total / len(data)


@dataclass
class FitResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    val_ids: list[str] = field(default_factory=list)
    val_true: np.ndarray | None = None
    val_prob: np.ndarray | None = None  # probabilities at the best epoch
    model: nn.Module | None = None
    encoder: Any = None
    criterion: nn.Module | None = None


def fit(model_name: str, config: dict, train_frame: pd.DataFrame, seed: int,
        validation_frame: pd.DataFrame | None = None, active: np.ndarray | None = None,
        stop_epoch: int | None = None, keep_model: bool = False) -> FitResult:
    """Train `model_name` with `config`.

    With a validation frame, every epoch is scored and the probabilities of the best epoch
    (macro-F1, then Security recall, then loss) are returned; transformers stop early after
    EARLY_STOPPING_PATIENCE epochs in which neither that score nor the loss improves. Without it (final model) training simply
    runs for `stop_epoch` epochs; the learning-rate schedule still spans config["epochs"].
    """
    seed_everything(seed)
    encoder = build_encoder(model_name, train_frame)
    train_data = encoder.encode(train_frame)
    val_data = encoder.encode(validation_frame) if validation_frame is not None else None
    model = build_model(model_name, config, encoder)
    criterion = criterion_for(train_frame, config["pos_weight"])
    batch_size = config["batch_size"]
    steps_per_epoch = int(np.ceil(len(train_data) / batch_size))
    optimizer, scheduler = build_optimizer(model_name, config, model, steps_per_epoch)
    use_amp = DEVICE.type == "cuda" and model_name not in WORD_MODELS
    scaler = torch.amp.GradScaler(enabled=use_amp)
    clip_gradients = model_name != "cnn"
    patience = None if model_name in WORD_MODELS else EARLY_STOPPING_PATIENCE
    generator = torch.Generator().manual_seed(seed)

    result = FitResult(encoder=encoder, criterion=criterion)
    if val_data is not None:
        result.val_ids, result.val_true = val_data.ids, val_data.labels.cpu().numpy()
    best, best_loss, best_loss_epoch = None, float("inf"), 0
    for epoch in range(1, (stop_epoch or config["epochs"]) + 1):
        model.train()
        total = 0.0
        for batch in train_data.batches(batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                logits = forward(model, model_name, batch["inputs"])
            loss = criterion(logits.float(), batch["labels"])
            scaler.scale(loss).backward()
            if clip_gradients:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(batch["labels"])
        row = {"epoch": epoch, "train_loss": total / len(train_data)}
        if val_data is not None:
            y_prob, val_loss = predict(model, model_name, val_data, criterion, batch_size)
            current = multilabel_metrics(result.val_true, y_prob, DEFAULT_THRESHOLD, active)
            row.update({"val_loss": val_loss, "val_macro_f1": current["macro_f1"],
                        "val_weighted_f1": current["weighted_f1"],
                        "val_security_recall": current["security_recall"]})
            current["loss"] = val_loss
            if is_better(current, best):
                best = {name: current[name] for name in ("macro_f1", "security_recall", "loss")}
                result.best_epoch, result.val_prob = epoch, y_prob
            if val_loss < best_loss:
                best_loss, best_loss_epoch = val_loss, epoch
        result.history.append(row)
        # A model whose macro-F1 is still flat but whose loss keeps falling has not converged yet.
        if val_data is not None and patience and epoch - max(result.best_epoch, best_loss_epoch) >= patience:
            break

    if keep_model:
        result.model = model
    else:
        del model, optimizer, train_data, val_data
        release_memory()
    return result
