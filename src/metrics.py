"""Métricas de cada tarea, umbrales por etiqueta y utilidades para guardar resultados.

- binary:        F1, precisión y recall de la clase seguridad, y macro-F1 de las dos clases.
- subcategories: macro-F1, weighted-F1, micro-F1 y macro-recall de las subcategorías evaluadas
                 (las que tienen al menos MIN_POSITIVES_PER_LABEL positivos).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .config import THRESHOLD_CANDIDATES, get_task


def metric_names(task: str) -> list[str]:
    return list(get_task(task).metrics)


def per_label_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, ...]:
    """Precisión, recall, F1 y soporte de cada columna, tratando cada una como un problema binario."""
    scores = [precision_recall_fscore_support(y_true[:, j], y_pred[:, j], average="binary", zero_division=0)
              for j in range(y_true.shape[1])]
    precision, recall, f1, _ = (np.array(values, dtype=float) for values in zip(*scores))
    return precision, recall, f1, y_true.sum(axis=0).astype(int)


def compute_metrics(task: str, y_true, y_prob, thresholds=0.5, active=None) -> dict[str, Any]:
    """Métricas de la tarea con los umbrales dados (uno para todas las etiquetas o uno por etiqueta).

    Los promedios usan solo las etiquetas `active`; `per_label` informa todas.
    """
    labels = list(get_task(task).labels)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), (len(labels),))
    active = np.ones(len(labels), bool) if active is None else np.asarray(active, bool)
    y_pred = (y_prob >= thresholds).astype(int)
    precision, recall, f1, support = per_label_scores(y_true, y_pred)

    if task == "binary":
        summary = {
            "f1": f1[0], "precision": precision[0], "recall": recall[0],
            "f1_macro_2class": f1_score(y_true[:, 0], y_pred[:, 0], average="macro", labels=[0, 1],
                                        zero_division=0),
        }
    else:
        summary = {
            "macro_f1": f1[active].mean(),
            "weighted_f1": f1_score(y_true[:, active], y_pred[:, active], average="weighted", zero_division=0),
            "micro_f1": f1_score(y_true[:, active], y_pred[:, active], average="micro", zero_division=0),
            "macro_recall": recall[active].mean(),
        }
    per_label = pd.DataFrame({"label": labels, "precision": precision, "recall": recall, "f1": f1,
                              "support": support, "in_macro_average": active})
    return {**{name: float(value) for name, value in summary.items()},
            "per_label": per_label, "y_pred": y_pred, "threshold": thresholds.tolist()}


def is_better(task: str, current: dict, best: dict | None, tol: float = 1e-12) -> bool:
    """¿`current` supera a `best`? Primero el objetivo de la tarea, luego el desempate y luego la pérdida."""
    if best is None:
        return True
    spec = get_task(task)
    for name in (spec.objective, spec.tiebreak):
        if abs(current[name] - best[name]) > tol:
            return current[name] > best[name]
    return current["loss"] < best["loss"] - tol


def select_thresholds(task: str, y_true, y_prob, active, default: float = 0.5) -> tuple[np.ndarray, pd.DataFrame]:
    """Umbral de cada etiqueta, elegido solo con predicciones fuera de fold: el de mayor F1 y,
    a igualdad, el de mayor recall. Las etiquetas no evaluadas se quedan con el umbral por defecto."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows, chosen = [], []
    for j, label in enumerate(get_task(task).labels):
        candidates = []
        for threshold in THRESHOLD_CANDIDATES:
            pred = (y_prob[:, j] >= threshold).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true[:, j], pred, average="binary", zero_division=0)
            candidates.append({"label": label, "threshold": threshold, "precision": p, "recall": r, "f1": f})
        rows.extend(candidates)
        best = max(candidates, key=lambda c: (round(c["f1"], 12), round(c["recall"], 12), -c["threshold"]))
        chosen.append(best["threshold"] if active[j] else default)
    return np.array(chosen), pd.DataFrame(rows)


def prediction_frame(task: str, ids, y_true, y_prob, y_pred) -> pd.DataFrame:
    """Una fila por requisito: <etiqueta>_true, <etiqueta>_prob y <etiqueta>_pred."""
    frame = pd.DataFrame({"ID": list(ids)})
    for j, label in enumerate(get_task(task).labels):
        if y_true is not None:
            frame[f"{label}_true"] = np.asarray(y_true)[:, j].astype(int)
        frame[f"{label}_prob"] = y_prob[:, j]
        frame[f"{label}_pred"] = y_pred[:, j].astype(int)
    return frame


def score_predictions(task: str, pooled: pd.DataFrame, active) -> tuple[dict, list[dict]]:
    """(métricas de todas las predicciones juntas, métricas de cada fold externo) de un archivo de predicciones."""
    labels = get_task(task).labels
    names = metric_names(task)

    def score(part: pd.DataFrame) -> dict:
        y_true = part[[f"{label}_true" for label in labels]].to_numpy(dtype=int)
        y_pred = part[[f"{label}_pred" for label in labels]].to_numpy(dtype=float)
        return compute_metrics(task, y_true, y_pred, 0.5, active)

    folds = [{"outer": int(k), **{name: score(part)[name] for name in names}}
             for k, part in pooled.groupby("outer", sort=True)]
    return score(pooled), folds


def summarize_folds(task: str, fold_rows: list[dict]) -> dict[str, dict[str, float]]:
    """Media y desviación estándar de cada métrica entre los folds externos."""
    summary = {}
    for name in metric_names(task):
        values = np.array([row[name] for row in fold_rows], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return summary


def add_pooled_scores(task: str, report: dict, pooled: pd.DataFrame, active) -> None:
    """Escribe en `report` las métricas de las predicciones agrupadas y de cada fold externo."""
    names = metric_names(task)
    pooled_metrics, folds = score_predictions(task, pooled, active)
    report["pooled_outer_test"] = {name: pooled_metrics[name] for name in names}
    report["pooled_outer_test_per_label"] = pooled_metrics["per_label"].to_dict("records")
    report["outer_aggregate"] = summarize_folds(task, folds)
    for fold_report, fold in zip(report["outer_folds"], folds):
        fold_report["test"] = {**{name: fold[name] for name in names}, "loss": fold_report["test"]["loss"]}


# --------------------------------------------------------------------------- JSON


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
