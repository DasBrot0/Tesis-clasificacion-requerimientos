#!/usr/bin/env python3
"""G9 (PLAN_AUMENTO_DATOS.md): compara C0 (solo reales) y C1 (reales + sintéticos).

    python scripts/compare_conditions.py

Usa las predicciones externas agrupadas de cada (dataset, modelo) presente en las dos condiciones,
pareadas por requisito:
- McNemar exacto por etiqueta (aciertos de C0 frente a C1);
- intervalo de confianza del 95 % del delta macro-F1 por bootstrap pareado (1 000 remuestreos);
- Wilcoxon de rangos con signo del delta macro-F1 sobre todos los pares disponibles.

Salidas: outputs/output_raw/condition_comparison.csv y outputs/output_raw/condition_comparison_by_label.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_RAW, DATASETS, LABEL_COLS, MODELS, SEED, run_dir  # noqa: E402
from src.metrics import multilabel_metrics  # noqa: E402

N_BOOT = 1000


def load(condition: str, dataset: str, model: str):
    path = run_dir(condition, dataset, model) / "pooled_outer_test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str}).sort_values("ID").reset_index(drop=True)
    summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    active = np.array([label in summary["labels_in_macro_average"] for label in LABEL_COLS])
    return df, active


def macro_f1(y_true, y_pred, active) -> float:
    return multilabel_metrics(y_true, y_pred.astype(float), 0.5, active)["macro_f1"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows, label_rows = [], []
    for dataset in DATASETS:
        for model in MODELS:
            c0, c1 = load("c0_real", dataset, model), load("c1_augmented", dataset, model)
            if c0 is None or c1 is None:
                continue
            (p0, active), (p1, _) = c0, c1
            if not p0["ID"].equals(p1["ID"]):
                raise AssertionError(f"{dataset}/{model}: las condiciones no evaluaron los mismos requisitos")
            y = p0[[f"{l}_true" for l in LABEL_COLS]].to_numpy(int)
            y0 = p0[[f"{l}_pred" for l in LABEL_COLS]].to_numpy(int)
            y1 = p1[[f"{l}_pred" for l in LABEL_COLS]].to_numpy(int)
            f0, f1 = macro_f1(y, y0, active), macro_f1(y, y1, active)
            boots = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(y), len(y))
                boots.append(macro_f1(y[idx], y1[idx], active) - macro_f1(y[idx], y0[idx], active))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"dataset": dataset, "model": model, "macro_f1_c0": f0, "macro_f1_c1": f1,
                         "delta": f1 - f0, "ci95_low": lo, "ci95_high": hi})
            m0 = multilabel_metrics(y, y0.astype(float), 0.5, active)["per_label"]
            m1 = multilabel_metrics(y, y1.astype(float), 0.5, active)["per_label"]
            for j, label in enumerate(LABEL_COLS):
                right0, right1 = y0[:, j] == y[:, j], y1[:, j] == y[:, j]
                b, c = int((right0 & ~right1).sum()), int((~right0 & right1).sum())
                p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                label_rows.append({"dataset": dataset, "model": model, "label": label,
                                   "support": int(y[:, j].sum()), "f1_c0": m0.loc[j, "f1"], "f1_c1": m1.loc[j, "f1"],
                                   "only_c0_right": b, "only_c1_right": c, "mcnemar_exact_p": p,
                                   "in_macro_average": bool(active[j])})
    if not rows:
        raise SystemExit("No hay pares (dataset, modelo) entrenados en c0_real y c1_augmented.")
    table, labels = pd.DataFrame(rows), pd.DataFrame(label_rows)
    table.to_csv(OUTPUT_RAW / "condition_comparison.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(OUTPUT_RAW / "condition_comparison_by_label.csv", index=False, encoding="utf-8-sig")
    print(table.round(4).to_string(index=False))
    if len(table) >= 6 and table["delta"].ne(0).any():
        stat = wilcoxon(table["delta"])
        print(f"\nWilcoxon sobre {len(table)} pares: estadístico={stat.statistic:.1f}, p={stat.pvalue:.4f}")
    else:
        print(f"\nWilcoxon omitido: {len(table)} pares (hacen falta al menos 6).")


if __name__ == "__main__":
    main()
