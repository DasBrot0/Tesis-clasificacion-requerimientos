"""End-to-end evaluation of the cascade binary -> subcategories (no training).

The two tasks are trained separately. To compare them with the former joint model, every real
requirement of a dataset gets one five-label prediction:

    Security        = prediction of the binary model (its pooled outer test prediction);
    subcategories   = 0 when the binary model says "not Security"; otherwise the prediction of the
                      subcategory model:
                        - Security requirements: their pooled outer test prediction (the subcategory
                          model of the outer fold in which they were tested);
                        - non-security requirements (false positives of the binary model): the
                          prediction of the subcategory model of outer fold k, where k is the
                          requirement's binary outer fold. The subcategory models never train on
                          non-security requirements, so this is leakage-free
                          (pipeline.NONSECURITY_CSV).

The five-label predictions are scored with the joint metrics (macro-F1, weighted-F1 and Security
recall over the same scored labels), so the result is comparable to the joint runs. The coherence
rule holds by construction. Output: outputs/output_raw/cascade/<condition>/<dataset>/<binary>__<subcat>/.

[Decisión propia, configurable] Pairing the same architecture in both stages (CNN -> CNN, ...) is
the default of scripts/evaluate_cascade.py; any pair can be requested.
"""

from __future__ import annotations

import pandas as pd

from .config import DATASETS, LABEL_COLS, MODELS, N_OUTER, OUTPUT_RAW, SUBCATEGORY_COLS, run_dir
from .data import active_label_mask, load_dataset
from .metrics import metric_names, multilabel_metrics, save_json

CASCADE_ROOT = OUTPUT_RAW / "cascade"
# Written by src/pipeline.py in each outer fold of the subcategory task.
NONSECURITY_CSV = "nonsecurity_predictions.csv"


def _pooled(condition: str, dataset: str, model: str, task: str) -> pd.DataFrame | None:
    path = run_dir(condition, dataset, model, task) / "pooled_outer_test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})


def _nonsecurity(condition: str, dataset: str, model: str) -> pd.DataFrame | None:
    out_dir = run_dir(condition, dataset, model, "subcategories")
    paths = [out_dir / f"outer_{k}" / NONSECURITY_CSV for k in range(N_OUTER)]
    if not all(p.exists() for p in paths):
        return None
    return pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"ID": str}) for p in paths], ignore_index=True)


def cascade_predictions(binary: pd.DataFrame, subcategories: pd.DataFrame, nonsecurity: pd.DataFrame,
                        full: pd.DataFrame) -> pd.DataFrame:
    """Five-label predictions of the cascade for every row of the full dataset table."""
    truth = full.set_index("ID")[LABEL_COLS].astype(int)
    binary = binary.set_index("ID")
    if set(binary.index) != set(truth.index):
        raise AssertionError("La tarea binaria no cubre exactamente el dataset completo.")
    if not binary["Security_true"].astype(int).equals(truth.loc[binary.index, "Security"]):
        raise AssertionError("Las etiquetas Security de la tarea binaria no coinciden con la tabla completa.")
    security_ids = set(truth.index[truth["Security"].eq(1)])
    sub = subcategories.set_index("ID")
    if not set(sub.index) <= security_ids:
        raise AssertionError("La tarea de subcategorías contiene requisitos sin Security.")
    # Non-security requirements: the subcategory model of the same outer fold as the binary task.
    folds = binary["outer"].astype(int)
    other = nonsecurity.assign(outer=nonsecurity["outer"].astype(int))
    other = other.merge(folds.rename("binary_outer"), left_on="ID", right_index=True)
    other = other[other["outer"].eq(other["binary_outer"])].set_index("ID")
    pred_cols = [f"{label}_pred" for label in SUBCATEGORY_COLS]
    stage2 = pd.concat([sub[pred_cols], other[pred_cols]])
    # Security requirements removed from the subcategory task (SUBCATEGORY_KEEP_EMPTY = False) have
    # no subcategory prediction: they count as "no subcategory".
    stage2 = stage2.reindex(truth.index).fillna(0).astype(int)

    frame = pd.DataFrame(index=truth.index)
    frame["binary_outer"] = folds.reindex(truth.index)
    for label in LABEL_COLS:
        frame[f"{label}_true"] = truth[label]
    frame["Security_pred"] = binary["Security_pred"].astype(int).reindex(truth.index)
    for label in SUBCATEGORY_COLS:
        frame[f"{label}_pred"] = stage2[f"{label}_pred"] * frame["Security_pred"]
    return frame.reset_index()


def evaluate_cascade(condition: str, dataset: str, binary_model: str, subcategory_model: str) -> dict | None:
    """Joint metrics of one cascade, or None when one of the two runs is missing."""
    binary = _pooled(condition, dataset, binary_model, "binary")
    subcategories = _pooled(condition, dataset, subcategory_model, "subcategories")
    nonsecurity = _nonsecurity(condition, dataset, subcategory_model)
    if binary is None or subcategories is None or nonsecurity is None:
        return None
    full = load_dataset(dataset, "joint")
    frame = cascade_predictions(binary, subcategories, nonsecurity, full)
    active = active_label_mask(full, "joint")
    y_true = frame[[f"{l}_true" for l in LABEL_COLS]].to_numpy(dtype=int)
    y_pred = frame[[f"{l}_pred" for l in LABEL_COLS]].to_numpy(dtype=float)
    metrics = multilabel_metrics(y_true, y_pred, 0.5, active, "joint")
    out_dir = CASCADE_ROOT / condition / dataset / f"{binary_model}__{subcategory_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "cascade_predictions.csv", index=False, encoding="utf-8-sig")
    metrics["per_label"].to_csv(out_dir / "cascade_metrics_by_label.csv", index=False, encoding="utf-8-sig")
    report = {"condition": condition, "dataset": dataset, "binary_model": binary_model,
              "subcategory_model": subcategory_model, "rows": len(frame),
              "pooled": {name: metrics[name] for name in metric_names("joint")},
              "per_label": metrics["per_label"].to_dict("records")}
    save_json(out_dir / "summary.json", report)
    return report


def cascade_table(condition: str, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """One row per (dataset, pair) with both runs finished; by default each model with itself."""
    pairs = pairs or [(m, m) for m in MODELS]
    rows = []
    for dataset in DATASETS:
        for binary_model, subcategory_model in pairs:
            report = evaluate_cascade(condition, dataset, binary_model, subcategory_model)
            if report is None:
                continue
            rows.append({"condition": condition, "dataset": dataset, "binary_model": binary_model,
                         "subcategory_model": subcategory_model, **report["pooled"],
                         **{f"f1 {r['label']}": r["f1"] for r in report["per_label"]}})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    path = CASCADE_ROOT / condition / "cascade_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return table
