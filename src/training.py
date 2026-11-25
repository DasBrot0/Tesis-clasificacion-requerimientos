"""Entrenamiento de un modelo: el mismo procedimiento para los folds internos y para el modelo final.

    fit(...)
      ├─ prepara datos (collator + DataLoader), modelo, función de pérdida y optimizador
      └─ por cada época:
           train_one_epoch(...)       un recorrido por el entrenamiento
           predict(...)               probabilidades y pérdida de validación (si hay validación)
           parada temprana            si pasan `patience` épocas sin mejorar el objetivo ni la pérdida
"""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEFAULT_THRESHOLD, MAX_EPOCHS, PATIENCE, SEED, WORD_MODELS, get_task
from .metrics import compute_metrics, is_better
from .models import DEVICE, RequirementsDataset, build_collator, build_model

# cuBLAS lo necesita para que el entrenamiento en GPU sea determinista (antes de usar CUDA).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# --------------------------------------------------------------------------- utilidades


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_loader(frame: pd.DataFrame, labels: list[str], collator, batch_size: int,
                shuffle: bool = False, seed: int = SEED) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(RequirementsDataset(frame, labels), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, collate_fn=collator)


def make_loss(train_frame: pd.DataFrame, labels: list[str], use_pos_weight: bool) -> nn.Module:
    """Entropía cruzada binaria. Con pos_weight, el término positivo de cada etiqueta se pondera
    por sqrt(negativos / positivos) de la partición de entrenamiento."""
    if not use_pos_weight:
        return nn.BCEWithLogitsLoss()
    positives = train_frame[labels].sum().to_numpy(dtype=float)
    weights = np.sqrt((len(train_frame) - positives) / np.maximum(positives, 1.0))
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))


def make_optimizer(model_name: str, config: dict, model: nn.Module, total_steps: int):
    """Adam para CNN y BiLSTM; Adam o AdamW con calentamiento lineal opcional para los Transformers."""
    if model_name in WORD_MODELS:
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"]), None
    optimizer_class = torch.optim.AdamW if config["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
    if not config["warmup_ratio"]:
        return optimizer, None  # tasa de aprendizaje constante
    from transformers import get_linear_schedule_with_warmup
    warmup_steps = int(round(config["warmup_ratio"] * total_steps))
    return optimizer, get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def to_device(batch: dict) -> tuple[dict, torch.Tensor]:
    inputs = {name: tensor.to(DEVICE) for name, tensor in batch["inputs"].items()}
    return inputs, batch["labels"].to(DEVICE)


# --------------------------------------------------------------------------- una época


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, use_amp, clip_gradients) -> float:
    model.train()
    total_loss, n_rows = 0.0, 0
    for batch in loader:
        inputs, labels = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**inputs)
        loss = loss_fn(logits.float(), labels)
        scaler.scale(loss).backward()
        if clip_gradients:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(labels)
        n_rows += len(labels)
    return total_loss / n_rows


@torch.no_grad()
def predict(model, loader, loss_fn, use_amp: bool) -> tuple[np.ndarray, float]:
    """Probabilidades sigmoides (filas, etiquetas) y pérdida media sobre `loader`."""
    model.eval()
    probabilities, total_loss, n_rows = [], 0.0, 0
    for batch in loader:
        inputs, labels = to_device(batch)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**inputs).float()
        total_loss += loss_fn(logits, labels).item() * len(labels)
        n_rows += len(labels)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probabilities), total_loss / n_rows


# --------------------------------------------------------------------------- entrenamiento completo


@dataclass
class FitResult:
    history: list[dict] = field(default_factory=list)  # una fila por época
    best_epoch: int = 0
    val_ids: list[str] = field(default_factory=list)
    val_true: np.ndarray | None = None
    val_prob: np.ndarray | None = None   # probabilidades de validación en la mejor época
    model: nn.Module | None = None       # solo con keep_model=True
    collator: object = None
    loss_fn: nn.Module | None = None


def fit(task: str, model_name: str, config: dict, train_frame: pd.DataFrame, seed: int,
        val_frame: pd.DataFrame | None = None, active: np.ndarray | None = None,
        epochs: int | None = None, keep_model: bool = False) -> FitResult:
    """Entrena `model_name` con `config` sobre las etiquetas de `task`.

    Con `val_frame` (folds internos): evalúa cada época, se queda con las probabilidades de la mejor
    (objetivo de la tarea, luego su desempate y luego la pérdida) y aplica la parada temprana.
    Sin `val_frame` (modelo final de un fold externo): entrena exactamente `epochs` épocas.
    """
    spec = get_task(task)
    labels = list(spec.labels)
    max_epochs = MAX_EPOCHS[task][model_name]
    patience = PATIENCE[task][model_name]
    seed_everything(seed)

    # 1) datos
    collator = build_collator(model_name, train_frame)
    batch_size = config["batch_size"]
    train_loader = make_loader(train_frame, labels, collator, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_frame, labels, collator, batch_size) if val_frame is not None else None

    # 2) modelo, pérdida y optimizador
    model = build_model(model_name, config, len(labels), collator)
    loss_fn = make_loss(train_frame, labels, config["pos_weight"])
    optimizer, scheduler = make_optimizer(model_name, config, model, len(train_loader) * max_epochs)
    use_amp = DEVICE.type == "cuda" and model_name not in WORD_MODELS  # precisión mixta, por los 4 GB de VRAM
    scaler = torch.amp.GradScaler(enabled=use_amp)
    clip_gradients = model_name != "cnn"  # el notebook original de la CNN no recortaba gradientes

    result = FitResult(collator=collator, loss_fn=loss_fn)
    if val_frame is not None:
        result.val_ids = val_loader.dataset.ids
        result.val_true = val_loader.dataset.labels.astype(int)

    # 3) épocas
    best, best_loss, best_loss_epoch = None, float("inf"), 0
    for epoch in range(1, (epochs or max_epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler,
                                     use_amp, clip_gradients)
        row = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            val_prob, val_loss = predict(model, val_loader, loss_fn, use_amp)
            scores = compute_metrics(task, result.val_true, val_prob, DEFAULT_THRESHOLD, active)
            scores["loss"] = val_loss
            row.update({"val_loss": val_loss, **{f"val_{name}": scores[name] for name in spec.metrics}})
            if is_better(task, scores, best):
                best = scores
                result.best_epoch, result.val_prob = epoch, val_prob
            if val_loss < best_loss:
                best_loss, best_loss_epoch = val_loss, epoch
            # Parada temprana: ni el objetivo ni la pérdida mejoraron en las últimas `patience` épocas.
            if epoch - max(result.best_epoch, best_loss_epoch) >= patience:
                result.history.append(row)
                break
        result.history.append(row)

    if keep_model:
        result.model = model
    else:
        del model, optimizer
        release_memory()
    return result
