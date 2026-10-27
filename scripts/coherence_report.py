#!/usr/bin/env python3
"""Coherencia jerárquica de las predicciones: una subcategoría solo puede ser 1 si Security es 1.

Los modelos predicen las 7 etiquetas de forma independiente, así que pueden decir "no es de
seguridad" y a la vez "Confidentiality". Para cada corrida terminada (artifacts/<dataset>/<modelo>/)
este script:

1. cuenta las predicciones incoherentes (Security_pred = 0 con alguna subcategoría en 1), en test
   y en las predicciones out-of-fold de la validación cruzada;
2. aplica la regla de coherencia (si Security_pred = 0, las subcategorías pasan a 0) y recalcula
   las métricas, para compararlas con las originales.

No reentrena ni modifica ningún artefacto: lee las predicciones guardadas.

Salida: artifacts/coherence_report.csv

Uso:  python scripts/coherence_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ARTIFACTS, LABEL_COLS, SUBCATEGORY_COLS  # noqa: E402
from src.metrics import multilabel_metrics  # noqa: E402

REPORTED = ["macro_f1", "weighted_f1", "micro_f1", "macro_precision", "macro_recall", "security_recall"]


def apply_coherence(y_pred: np.ndarray) -> np.ndarray:
    """Subcategorías en 0 cuando Security se predice 0."""
    coherent = y_pred.copy()
    coherent[coherent[:, 0] == 0, 1:] = 0
    return coherent


def evaluate(frame: pd.DataFrame, active: np.ndarray) -> dict:
    y_true = frame[[f"{label}_true" for label in LABEL_COLS]].to_numpy(dtype=int)
    y_pred = frame[[f"{label}_pred" for label in LABEL_COLS]].to_numpy(dtype=int)
    incoherent = (y_pred[:, 0] == 0) & (y_pred[:, 1:].sum(axis=1) > 0)
    # Las predicciones ya están binarizadas con los umbrales por etiqueta: umbral 0,5 sobre 0/1.
    before = multilabel_metrics(y_true, y_pred.astype(float), 0.5, active)
    after = multilabel_metrics(y_true, apply_coherence(y_pred).astype(float), 0.5, active)
    row = {"rows": len(frame), "incoherent_rows": int(incoherent.sum()),
           "incoherent_pct": float(incoherent.mean()),
           "incoherent_labels": int(y_pred[incoherent, 1:].sum())}
    for metric in REPORTED:
        row[f"{metric}_before"] = before[metric]
        row[f"{metric}_after"] = after[metric]
    return row


def main() -> None:
    rows = []
    for summary_path in sorted(ARTIFACTS.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        active = np.isin(LABEL_COLS, summary["labels_in_macro_average"])
        for split, name in (("test", "test_predictions.csv"), ("cv_oof", "oof_predictions.csv")):
            path = run_dir / name
            if not path.exists():
                continue
            frame = pd.read_csv(path, encoding="utf-8-sig")
            rows.append({"dataset": run_dir.parent.name, "model": run_dir.name, "split": split,
                         **evaluate(frame, active)})
    if not rows:
        print("No hay corridas terminadas en artifacts/.")
        return
    report = pd.DataFrame(rows)
    out = ARTIFACTS / "coherence_report.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    view = report[["dataset", "model", "split", "incoherent_rows", "incoherent_pct",
                   "macro_f1_before", "macro_f1_after", "security_recall_before"]]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(view.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
