"""Evaluación de extremo a extremo de la cascada binary -> subcategories (no entrena nada).

Para cada requisito real del dataset:

    Security        = predicción del modelo binario (su fold externo de prueba);
    subcategorías   = 0 si el modelo binario dice "no seguridad"; si dice "seguridad", la
                      predicción del modelo de subcategorías del mismo fold externo.

Como las dos tareas comparten los folds externos, el modelo de subcategorías del fold k nunca vio
los requisitos del fold de prueba k, tampoco los de no seguridad (src/pipeline.py los predice
aparte, en nonsecurity_predictions.csv). Las subcategorías se evalúan sobre todos los requisitos,
así que los errores de la etapa 1 bajan su rendimiento frente a la tarea 2 sola, que se evalúa
solo con los requisitos de seguridad reales (Binkhonain & Zhao, 2023).

La tarea binaria solo tiene c0_real; `condition` es la condición del modelo de subcategorías.
Salida: outputs/output_raw/cascade/<condición>/<dataset>/<binario>__<subcategorías>/.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import DATASETS, LABEL_COLS, MODELS, N_OUTER, NONSECURITY_CSV, OUTPUT_RAW, SUBCATEGORY_COLS, run_dir
from .data import load_full_table
from .metrics import compute_metrics, metric_names, save_json

CASCADE_ROOT = OUTPUT_RAW / "cascade"


def _read(path) -> pd.DataFrame | None:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str}) if path.exists() else None


def cascade_predictions(binary: pd.DataFrame, security: pd.DataFrame, nonsecurity: pd.DataFrame,
                        full: pd.DataFrame) -> pd.DataFrame:
    """Predicciones de las cinco etiquetas para cada requisito de la tabla completa."""
    truth = full.set_index("ID")[LABEL_COLS]
    binary = binary.set_index("ID")
    if set(binary.index) != set(truth.index):
        raise AssertionError("La tarea binaria no cubre exactamente el dataset completo.")
    pred_cols = [f"{label}_pred" for label in SUBCATEGORY_COLS]
    stage2 = pd.concat([security.set_index("ID")[pred_cols], nonsecurity.set_index("ID")[pred_cols]])
    if stage2.index.duplicated().any() or set(stage2.index) != set(truth.index):
        raise AssertionError("La tarea de subcategorías no predijo cada requisito exactamente una vez.")

    frame = pd.DataFrame(index=truth.index)
    frame["outer"] = binary["outer"]
    for label in LABEL_COLS:
        frame[f"{label}_true"] = truth[label]
    frame["Security_pred"] = binary["Security_pred"].astype(int)
    for label in SUBCATEGORY_COLS:
        frame[f"{label}_pred"] = stage2.loc[truth.index, f"{label}_pred"].astype(int) * frame["Security_pred"]
    return frame.reset_index()


def evaluate_cascade(condition: str, dataset: str, binary_model: str, subcategory_model: str) -> dict | None:
    """Métricas de una cascada, o None si falta alguna de las dos corridas."""
    binary_dir = run_dir("binary", "c0_real", dataset, binary_model)
    sub_dir = run_dir("subcategories", condition, dataset, subcategory_model)
    binary = _read(binary_dir / "pooled_outer_test_predictions.csv")
    security = _read(sub_dir / "pooled_outer_test_predictions.csv")
    others = [_read(sub_dir / f"outer_{k}" / NONSECURITY_CSV) for k in range(N_OUTER)]
    if binary is None or security is None or any(part is None for part in others):
        return None
    frame = cascade_predictions(binary, security, pd.concat(others, ignore_index=True), load_full_table(dataset))

    summary = json.loads((sub_dir / "summary.json").read_text(encoding="utf-8"))
    active = [label in summary["labels_in_macro_average"] for label in SUBCATEGORY_COLS]
    y_true = frame[[f"{label}_true" for label in SUBCATEGORY_COLS]].to_numpy()
    y_pred = frame[[f"{label}_pred" for label in SUBCATEGORY_COLS]].to_numpy(dtype=float)
    end_to_end = compute_metrics("subcategories", y_true, y_pred, 0.5, active)
    security_scores = compute_metrics("binary", frame[["Security_true"]].to_numpy(),
                                      frame[["Security_pred"]].to_numpy(dtype=float))

    out_dir = CASCADE_ROOT / condition / dataset / f"{binary_model}__{subcategory_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "cascade_predictions.csv", index=False, encoding="utf-8-sig")
    end_to_end["per_label"].to_csv(out_dir / "cascade_metrics_by_label.csv", index=False, encoding="utf-8-sig")
    report = {
        "condition": condition, "dataset": dataset, "binary_model": binary_model,
        "subcategory_model": subcategory_model, "rows": len(frame),
        "security": {name: security_scores[name] for name in metric_names("binary")},
        "end_to_end": {name: end_to_end[name] for name in metric_names("subcategories")},
        # tarea 2 sola (requisitos de seguridad reales), para medir cuánto arrastra la etapa 1
        "subcategories_only": summary["pooled_outer_test"],
        "per_label": end_to_end["per_label"].to_dict("records"),
    }
    save_json(out_dir / "summary.json", report)
    return report


def cascade_table(condition: str, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Una fila por (dataset, par de modelos) con las dos corridas terminadas.
    Por defecto, cada arquitectura consigo misma (CNN -> CNN, ...). [Decisión propia]"""
    pairs = pairs or [(m, m) for m in MODELS]
    rows = []
    for dataset in DATASETS:
        for binary_model, subcategory_model in pairs:
            report = evaluate_cascade(condition, dataset, binary_model, subcategory_model)
            if report is None:
                continue
            rows.append({
                "condition": condition, "dataset": dataset, "binary_model": binary_model,
                "subcategory_model": subcategory_model,
                **{f"security_{k}": v for k, v in report["security"].items()},
                **{f"end_to_end_{k}": v for k, v in report["end_to_end"].items()},
                "subcategories_only_macro_f1": report["subcategories_only"]["macro_f1"],
                **{f"f1 {r['label']}": r["f1"] for r in report["per_label"]},
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        path = CASCADE_ROOT / condition / "cascade_comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding="utf-8-sig")
    return table
