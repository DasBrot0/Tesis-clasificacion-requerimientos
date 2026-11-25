#!/usr/bin/env python3
"""Verifica que las subcategorías de seguridad tengan positivos suficientes en cada dataset.

Escribe inputs/training/subcategory_coverage.csv y termina con código 1 si alguna etiqueta tiene
menos de MIN_POSITIVES_PER_LABEL positivos (se predice, pero no entra en los promedios).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATASET_NAMES, DATASETS, MIN_POSITIVES_PER_LABEL, SUBCATEGORY_COLS, TRAINING_DIR  # noqa: E402
from src.data import label_coverage, load_task_data  # noqa: E402

DEFINITIONS = {
    "Security": "security requirement (general)",
    "Confidentiality": "protecting information from unauthorized disclosure",
    "Integrity": "preventing unauthorized modification of data",
    "Authenticity": "verifying the identities of users, systems, and services",
    "Resistance": "withstanding, responding to, and recovering from attacks or malicious actions",
}


def main() -> int:
    parts = []
    for dataset in DATASETS:
        for task in ("binary", "subcategories"):
            coverage = label_coverage(load_task_data(dataset, task), task)
            coverage.insert(0, "dataset", DATASET_NAMES[dataset])
            parts.append(coverage)
    coverage = pd.concat(parts, ignore_index=True)
    coverage["definition"] = coverage["label"].map(DEFINITIONS)
    coverage.to_csv(TRAINING_DIR / "subcategory_coverage.csv", index=False, encoding="utf-8-sig")

    print(f"Positivos por etiqueta (mínimo para evaluarla: {MIN_POSITIVES_PER_LABEL})\n")
    print(coverage.pivot(index="label", columns="dataset", values="positives")
          .loc[["Security", *SUBCATEGORY_COLS]].to_string())
    problems = coverage[~coverage["scored"]]
    if problems.empty:
        print(f"\nOK: las {len(SUBCATEGORY_COLS)} subcategorías tienen cobertura suficiente en los tres datasets.")
        return 0
    print("\nEtiquetas sin cobertura suficiente:")
    for row in problems.itertuples():
        print(f"  {row.dataset}: {row.label} = {row.positives} positivos")
    return 1


if __name__ == "__main__":
    sys.exit(main())
