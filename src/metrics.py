"""Metrics of each task, per-label threshold selection, the coherence rule (joint task only) and
small serialization helpers.

The task (src/config.py) fixes the labels, the reported metrics, the Optuna objective and the
tie-break:
- binary: f1, precision and recall of Security, plus f1_macro_2class (mean F1 of both classes);
- subcategories: macro_f1, weighted_f1, micro_f1 and macro_recall over the scored subcategories;
- joint (former single model): macro_f1, weighted_f1 and security_recall over the five labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .config import SUBCATEGORY_COLS, task_spec

# Metrics of the joint task (METODOLOGIA §4.3.1); task_spec(task).metrics for the others.
METRIC_NAMES = list(task_spec("joint").metrics)


def metric_names(task: str = "joint") -> list[str]:
    return list(task_spec(task).metrics)


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | Iterable[float] = 0.50,
    active: np.ndarray | None = None,
    task: str = "joint",
) -> dict[str, Any]:
    """Aggregates use only the `active` labels; `per_label` always reports every label."""
    labels = list(task_spec(task).labels)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(labels):
        raise ValueError(f"Se esperaban matrices (n, {len(labels)}) con formas iguales.")
    thresholds = np.asarray(threshold, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.repeat(float(thresholds), len(labels))
    if thresholds.shape != (len(labels),):
        raise ValueError(f"threshold debe ser escalar o contener {len(labels)} valores.")
    active = np.ones(len(labels), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    if len(labels) > 1:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)
    else:  # an (n, 1) matrix would be read as one binary target with two classes
        precision, recall, f1, support = _binary_prfs(y_true[:, 0], y_pred[:, 0])
    true_active, pred_active = y_true[:, active], y_pred[:, active]
    if task == "binary":
        summary = {
            "f1": float(f1[0]), "precision": float(precision[0]), "recall": float(recall[0]),
            "f1_macro_2class": float(f1_score(y_true[:, 0], y_pred[:, 0], average="macro",
                                              labels=[0, 1], zero_division=0)),
        }
    elif task == "subcategories":
        summary = {
            "macro_f1": float(f1[active].mean()),
            "weighted_f1": float(f1_score(true_active, pred_active, average="weighted", zero_division=0)),
            "micro_f1": float(f1_score(true_active, pred_active, average="micro", zero_division=0)),
            "macro_recall": float(recall[active].mean()),
        }
    else:
        summary = {
            "macro_f1": float(f1[active].mean()),
            "weighted_f1": float(f1_score(true_active, pred_active, average="weighted", zero_division=0)),
            "security_recall": float(recall[0]),
        }
    return {
        **summary,
        "per_label": pd.DataFrame({
            "label": labels, "precision": precision, "recall": recall, "f1": f1,
            "support": support, "in_macro_average": active,
        }),
        "y_pred": y_pred,
        "threshold": thresholds.tolist(),
    }


def _binary_prfs(y_true: np.ndarray, y_pred: np.ndarray):
    """precision_recall_fscore_support of the positive class only, as length-1 arrays."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return np.array([p]), np.array([r]), np.array([f]), np.array([int(y_true.sum())])


def select_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    active: np.ndarray,
    candidates: Iterable[float] = tuple(np.arange(0.40, 0.6001, 0.025)),
    default: float = 0.50,
    task: str = "joint",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Choose each label threshold on OOF predictions only, maximizing label F1 then recall.

    Labels that are not active keep the default: their handful of positives cannot support a choice.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows: list[dict[str, Any]] = []
    selected: list[float] = []
    for index, label in enumerate(task_spec(task).labels):
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


def is_better(current: dict[str, Any], best: dict[str, Any] | None, atol: float = 1e-12,
              task: str = "joint") -> bool:
    """Objective of the task first, then its tie-break, then the lower loss."""
    if best is None:
        return True
    spec = task_spec(task)
    objective, tiebreak = spec.objective, spec.tiebreak
    if current[objective] > best[objective] + atol:
        return True
    if abs(current[objective] - best[objective]) > atol:
        return False
    if abs(current[tiebreak] - best[tiebreak]) > atol:
        return current[tiebreak] > best[tiebreak]
    # Full tie (typically a model that still predicts no positives): prefer the lower loss.
    return current["loss"] < best["loss"] - atol


def prediction_frame(ids: Iterable[Any], y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
                     task: str = "joint") -> pd.DataFrame:
    frame = pd.DataFrame({"ID": list(ids)})
    for j, label in enumerate(task_spec(task).labels):
        if y_true is not None:
            frame[f"{label}_true"] = y_true[:, j].astype(int)
        frame[f"{label}_prob"] = y_prob[:, j]
        frame[f"{label}_pred"] = y_pred[:, j].astype(int)
    return frame


def aggregate_rows(fold_rows: list[dict[str, Any]], task: str = "joint") -> dict[str, dict[str, float]]:
    aggregate = {}
    for metric in metric_names(task):
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        aggregate[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return aggregate


# --------------------------------------------------------------------------- coherence rule

def apply_coherence(y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Security = 0 => every subcategory = 0 (METODOLOGIA §4.3.1).

    Applied once, to the final test predictions, after tuning and threshold selection: the
    subcategories are switched off rather than Security switched on (Patel et al., 2022; Yu et al.,
    2022). Returns the corrected predictions and the (n, subcategories) mask of switched-off cells.
    Only the joint task needs it: the binary and subcategory models are trained separately, and
    their combination is the cascade of src/cascade.py.
    """
    corrected = np.asarray(y_pred, dtype=int).copy()
    changed = (corrected[:, :1] == 0) & (corrected[:, 1:] == 1)
    corrected[:, 1:][changed] = 0
    return corrected, changed


def pooled_scores(pooled: pd.DataFrame, kind: str, active: np.ndarray, task: str = "joint") -> tuple[dict, list[dict]]:
    """(pooled metrics, per-outer-fold metrics) of the `kind` prediction columns."""
    labels = task_spec(task).labels
    names = metric_names(task)

    def score(part: pd.DataFrame) -> dict[str, Any]:
        y_true = part[[f"{l}_true" for l in labels]].to_numpy(dtype=int)
        y_pred = part[[f"{l}_{kind}" for l in labels]].to_numpy(dtype=float)
        return multilabel_metrics(y_true, y_pred, 0.5, active, task)
    folds = [{"outer": int(k), **{n: score(part)[n] for n in names}}
             for k, part in pooled.groupby("outer", sort=True)]
    return score(pooled), folds


def rescore_pooled(report: dict[str, Any], pooled: pd.DataFrame, active: np.ndarray,
                   task: str = "joint") -> pd.DataFrame:
    """Score a run's pooled outer test predictions and write the metrics into `report`.

    Joint task: the coherence rule is applied first. The uncorrected predictions are kept as
    `<label>_pred_raw` (and are the source when they exist, so this can be repeated); `<label>_pred`
    holds the corrected ones, which every report uses, and `report` gains the uncorrected metrics
    and the number of corrections. Binary and subcategory tasks: no correction, the predictions are
    scored as they are.
    """
    pooled = pooled.copy()
    names = metric_names(task)
    kinds = [("pred", "")]
    if task_spec(task).coherence:
        labels = task_spec(task).labels
        for label in labels:
            if f"{label}_pred_raw" not in pooled:
                pooled[f"{label}_pred_raw"] = pooled[f"{label}_pred"]
        raw = pooled[[f"{l}_pred_raw" for l in labels]].to_numpy(dtype=int)
        corrected, changed = apply_coherence(raw)
        for j, label in enumerate(labels):
            pooled[f"{label}_pred"] = corrected[:, j]
        kinds.append(("pred_raw", "_uncorrected"))
        report["coherence_corrections"] = {
            "requirements": int(changed.any(axis=1).sum()), "predictions": int(changed.sum()),
            "by_label": dict(zip(SUBCATEGORY_COLS, changed.sum(axis=0).astype(int).tolist())),
        }

    for kind, suffix in kinds:
        metrics, folds = pooled_scores(pooled, kind, active, task)
        report[f"pooled_outer_test{suffix}"] = {name: metrics[name] for name in names}
        report[f"pooled_outer_test_per_label{suffix}"] = metrics["per_label"].to_dict("records")
        report[f"outer_aggregate{suffix}"] = aggregate_rows(folds, task)
        for fold_report, fold in zip(report["outer_folds"], folds):
            fold_report[f"test{suffix}"] = {**{n: fold[n] for n in names},
                                            **({"loss": fold_report["test"]["loss"]} if not suffix else {})}
    return pooled


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
