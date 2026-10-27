"""Multilabel metrics, per-label threshold selection and small serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support

from .config import LABEL_COLS

METRIC_NAMES = [
    "macro_f1", "weighted_f1", "micro_f1", "macro_precision", "macro_recall",
    "security_recall", "subset_accuracy", "hamming_loss",
]


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | Iterable[float] = 0.50,
    active: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregates use only the `active` labels; `per_label` always reports the seven of them."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(LABEL_COLS):
        raise ValueError(f"Se esperaban matrices (n, {len(LABEL_COLS)}) con formas iguales.")
    thresholds = np.asarray(threshold, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.repeat(float(thresholds), len(LABEL_COLS))
    if thresholds.shape != (len(LABEL_COLS),):
        raise ValueError(f"threshold debe ser escalar o contener {len(LABEL_COLS)} valores.")
    active = np.ones(len(LABEL_COLS), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    true_active, pred_active = y_true[:, active], y_pred[:, active]
    return {
        "macro_f1": float(f1[active].mean()),
        "weighted_f1": float(f1_score(true_active, pred_active, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(true_active, pred_active, average="micro", zero_division=0)),
        "macro_precision": float(precision[active].mean()),
        "macro_recall": float(recall[active].mean()),
        "subset_accuracy": float(accuracy_score(true_active, pred_active)),
        "hamming_loss": float(hamming_loss(true_active, pred_active)),
        "security_recall": float(recall[0]),
        "per_label": pd.DataFrame({
            "label": LABEL_COLS, "precision": precision, "recall": recall, "f1": f1,
            "support": support, "in_macro_average": active,
        }),
        "y_pred": y_pred,
        "threshold": thresholds.tolist(),
    }


def select_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    active: np.ndarray,
    candidates: Iterable[float] = tuple(np.arange(0.40, 0.6001, 0.025)),
    default: float = 0.50,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Choose each label threshold on OOF predictions only, maximizing label F1 then recall.

    Labels that are not active keep the default: their handful of positives cannot support a choice.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows: list[dict[str, Any]] = []
    selected: list[float] = []
    for index, label in enumerate(LABEL_COLS):
        label_rows = []
        for candidate in candidates:
            pred = (y_prob[:, index] >= float(candidate)).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, index], pred, average="binary", zero_division=0
            )
            label_rows.append({
                "label": label,
                "threshold": float(candidate),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            })
        label_frame = pd.DataFrame(label_rows).sort_values(
            ["f1", "recall", "threshold"], ascending=[False, False, True], kind="stable"
        )
        selected.append(float(label_frame.iloc[0]["threshold"]) if active[index] else default)
        rows.extend(label_rows)
    return np.asarray(selected, dtype=float), pd.DataFrame(rows)


def is_better(current: dict[str, Any], best: dict[str, Any] | None, atol: float = 1e-12) -> bool:
    if best is None:
        return True
    if current["macro_f1"] > best["macro_f1"] + atol:
        return True
    if abs(current["macro_f1"] - best["macro_f1"]) > atol:
        return False
    if abs(current["security_recall"] - best["security_recall"]) > atol:
        return current["security_recall"] > best["security_recall"]
    # Full tie (typically a model that still predicts no positives): prefer the lower loss.
    return current["loss"] < best["loss"] - atol


def prediction_frame(ids: Iterable[Any], y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"ID": list(ids)})
    for j, label in enumerate(LABEL_COLS):
        frame[f"{label}_true"] = y_true[:, j].astype(int)
        frame[f"{label}_prob"] = y_prob[:, j]
        frame[f"{label}_pred"] = y_pred[:, j].astype(int)
    return frame


def aggregate_rows(fold_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        aggregate[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return aggregate


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
